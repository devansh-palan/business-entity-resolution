#!/usr/bin/env python3
"""Leakage-safe validation folds for the Business Entity Resolution challenge.

Design (justification and measurements in eda/validation_design.md):

* The unit of scoring is the Source 1 entity, so an entity and every pair built from it live in one fold.
* Entities that would share candidates are kept together.  Connected components over the keys
  (country, normalised core name) and (country, normalised sorted address) are assigned to folds as a
  block, so an S2/S3 record that is a positive for a training entity is never a same-name or same-address
  candidate of a validation entity.  If the address key produced a giant component the build falls back
  to name-only components and says so in the summary.
* Folds are balanced per country by a deterministic greedy assignment of components, largest first,
  which also balances singleton share, match-count classes and source patterns (verified in the summary).
* The S2/S3 pool is never split: every fold is scored against the whole pool, exactly like the test set.
* Country is kept as a column so leave-one-country-out runs (train US, validate India and vice versa)
  can be made from the same table to estimate the drop on the unseen country.
* Nothing in this module reads the test files.

Artifacts written by ``build_folds`` into ``out_dir``:
    entity_folds.parquet     one row per S1 entity: fold, cluster, counts, pattern, matched ids
    positive_pairs.parquet   one row per true (S1, S2/S3) pair with its fold
    fold{f}_ground_truth.tsv official ground-truth format restricted to fold f (scoreable by evaluation.py)
    fold{f}_entity_ids.txt   the S1 ids of fold f (usable as ``evaluation.py --ids``)
    summary.json             per-fold statistics

Command line (run from code/business_entity_resolution):
    python src/validation.py build --train-dir ../../student_resource/dataset/train --out-dir ../../artifacts/validation
    python src/validation.py score --out-dir ../../artifacts/validation --pred output/dev_predictions.tsv --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev  # noqa: E402
from normalize import addr_sorted, name_core_sorted  # noqa: E402

N_FOLDS_DEFAULT = 5
K_CLASSES = ["0", "1", "2", "3+"]
PATTERNS = ["none", "S2 only", "S3 only", "both"]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, has_header=True, infer_schema_length=0, encoding="utf8")


def ground_truth_frame(path: str) -> pl.DataFrame:
    gt = read_tsv(path)
    if gt.columns != ["source1_entity_id", "matched_entity_ids"]:
        raise ValueError(f"{path}: unexpected columns {gt.columns}")
    gt = gt.with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("matched")
    ).drop("matched_entity_ids")
    gt = gt.with_columns(
        pl.col("matched").list.len().cast(pl.Int64).alias("n_true"),
        pl.col("matched").list.eval(pl.element().str.starts_with("S2-")).list.sum().cast(pl.Int64).alias("n_s2"),
    )
    gt = gt.with_columns((pl.col("n_true") - pl.col("n_s2")).alias("n_s3"))
    gt = gt.with_columns(
        pl.when(pl.col("n_true") == 0).then(pl.lit("0")).when(pl.col("n_true") == 1).then(pl.lit("1"))
          .when(pl.col("n_true") == 2).then(pl.lit("2")).otherwise(pl.lit("3+")).alias("k_class"),
        pl.when(pl.col("n_true") == 0).then(pl.lit("none")).when(pl.col("n_s3") == 0).then(pl.lit("S2 only"))
          .when(pl.col("n_s2") == 0).then(pl.lit("S3 only")).otherwise(pl.lit("both")).alias("pattern"),
    )
    return gt


# --------------------------------------------------------------------------- #
# Grouping: connected components over shared keys
# --------------------------------------------------------------------------- #
def _edges(frame: pl.DataFrame, key: str, offset: int, max_group: Optional[int] = None):
    """Entity->key edges for keys shared by at least two (and at most max_group) entities. Returns (rows, cols, n_keys)."""
    g = (frame.select("i", key).filter(pl.col(key) != "")
              .with_columns(pl.len().over(key).alias("sz")).filter(pl.col("sz") >= 2))
    if max_group is not None:
        g = g.filter(pl.col("sz") <= max_group)
    g = g.with_columns(pl.col(key).rank("dense").cast(pl.Int64).alias("kid"))
    if g.height == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
    return g["i"].to_numpy().astype(np.int64), (offset + g["kid"].to_numpy() - 1).astype(np.int64), int(g["kid"].max())


ADDR_MAX_GROUP_DEFAULT = 3   # measured on train: cap 3 -> largest component 6,531 entities; cap 5 -> 198,209; uncapped -> 263,844


def components(s1: pl.DataFrame, merge_address: bool = True, giant_limit: float = 0.01,
               addr_max_group: Optional[int] = ADDR_MAX_GROUP_DEFAULT) -> tuple[np.ndarray, dict]:
    """Cluster id per S1 row (0..C-1).  s1 needs entity_id, country, business_name, business_address.

    Name keys always merge.  Address keys merge only when shared by at most ``addr_max_group`` entities:
    popular addresses (business centres, 'Maison des Associations') would otherwise chain large name
    clusters into one giant component.
    """
    n = s1.height
    f = s1.with_row_index("i").with_columns(
        pl.concat_str([pl.col("country"), pl.lit("|"), name_core_sorted(pl.col("business_name"))]).alias("name_key"),
        pl.concat_str([pl.col("country"), pl.lit("|"), addr_sorted(pl.col("business_address"))]).alias("addr_key"),
        name_core_sorted(pl.col("business_name")).alias("_nk"),
        addr_sorted(pl.col("business_address")).alias("_ak"),
    ).with_columns(
        pl.when(pl.col("_nk") == "").then(pl.lit("")).otherwise(pl.col("name_key")).alias("name_key"),
        pl.when(pl.col("_ak") == "").then(pl.lit("")).otherwise(pl.col("addr_key")).alias("addr_key"),
    ).with_columns(pl.col("i").cast(pl.Int64))

    def run(use_address: bool):
        r1, c1, k1 = _edges(f, "name_key", n)
        rows, cols, total = [r1], [c1], n + k1
        if use_address:
            r2, c2, k2 = _edges(f, "addr_key", n + k1, max_group=addr_max_group)
            rows.append(r2); cols.append(c2); total += k2
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        a = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(total, total))
        _, labels = connected_components(a, directed=False)
        _, cl = np.unique(labels[:n], return_inverse=True)
        return cl

    cl = run(merge_address)
    sizes = np.bincount(cl)
    info = {"merge_address": merge_address, "addr_max_group": addr_max_group, "n_clusters": int(len(sizes)),
            "max_cluster": int(sizes.max()), "clusters_size_1": int((sizes == 1).sum()),
            "entities_in_clusters_ge_10": int(sizes[sizes >= 10].sum())}
    threshold = max(giant_limit * n, 100)   # absolute floor so tiny data sets never trigger the guard
    if merge_address and sizes.max() > threshold:
        info["fallback"] = f"address merge produced a component of {int(sizes.max())} entities (> {threshold:.0f}); using name-only components"
        cl = run(False)
        sizes = np.bincount(cl)
        info.update({"merge_address": False, "n_clusters": int(len(sizes)), "max_cluster": int(sizes.max()),
                     "clusters_size_1": int((sizes == 1).sum()), "entities_in_clusters_ge_10": int(sizes[sizes >= 10].sum())})
    return cl, info


def assign_folds(cluster: np.ndarray, country: np.ndarray, n_folds: int, seed: int = 0) -> np.ndarray:
    """Whole clusters go to one fold; per country, largest clusters first, each to the least-loaded fold."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(cluster), -1, dtype=np.int64)
    for c in np.unique(country):
        idx = np.flatnonzero(country == c)
        uniq, inv, counts = np.unique(cluster[idx], return_inverse=True, return_counts=True)
        order = np.lexsort((rng.random(len(uniq)), -counts))   # size descending, random among equals
        load = [0] * n_folds
        cl_fold = np.empty(len(uniq), dtype=np.int64)
        for j in order:
            f = min(range(n_folds), key=lambda t: (load[t], t))
            cl_fold[j] = f
            load[f] += int(counts[j])
        fold[idx] = cl_fold[inv]
    return fold


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_folds(train_dir: str, out_dir: str, n_folds: int = N_FOLDS_DEFAULT, merge_address: bool = True, seed: int = 0,
                addr_max_group: Optional[int] = ADDR_MAX_GROUP_DEFAULT) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    s1 = read_tsv(os.path.join(train_dir, "train_source1.tsv"))
    gt = ground_truth_frame(os.path.join(train_dir, "train_ground_truth.tsv"))
    if s1.height != gt.height or s1.join(gt, left_on="entity_id", right_on="source1_entity_id").height != s1.height:
        raise ValueError("train_source1.tsv and train_ground_truth.tsv do not describe the same entities")

    cluster, info = components(s1, merge_address=merge_address, addr_max_group=addr_max_group)
    fold = assign_folds(cluster, s1["country"].to_numpy(), n_folds, seed=seed)
    sizes = np.bincount(cluster)
    ent = s1.select("entity_id", "country").with_columns(
        pl.Series("cluster_id", cluster), pl.Series("cluster_size", sizes[cluster]), pl.Series("fold", fold)
    ).join(gt.rename({"source1_entity_id": "entity_id"}), on="entity_id", how="left")
    ent.write_parquet(os.path.join(out_dir, "entity_folds.parquet"))

    pairs = (ent.select("entity_id", "fold", "matched").explode("matched").drop_nulls("matched")
                .rename({"entity_id": "source1_entity_id", "matched": "matched_id"})
                .with_columns(pl.col("matched_id").str.slice(0, 2).alias("src")))
    pairs.write_parquet(os.path.join(out_dir, "positive_pairs.parquet"))
    if pairs["matched_id"].n_unique() != pairs.height:
        raise ValueError("a matched id appears under more than one S1 entity; the one-to-one assumption is broken")

    summary = {"n_entities": ent.height, "n_folds": n_folds, "seed": seed, "clusters": info, "folds": {}}
    for f in range(n_folds):
        sub = ent.filter(pl.col("fold") == f)
        mapping = dict(zip(sub["entity_id"].to_list(), sub["matched"].to_list()))
        ev.write_id_list_file(os.path.join(out_dir, f"fold{f}_ground_truth.tsv"), mapping)
        with open(os.path.join(out_dir, f"fold{f}_entity_ids.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(sub["entity_id"].to_list()) + "\n")
        summary["folds"][f] = fold_stats(sub)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary


def fold_stats(sub: pl.DataFrame) -> dict:
    n = sub.height
    share = lambda col, vals: {v: round(100 * sub.filter(pl.col(col) == v).height / n, 3) for v in vals}
    return {
        "entities": n,
        "positives": int(sub["n_true"].sum()),
        "country_pct": share("country", sorted(sub["country"].unique().to_list())),
        "k_class_pct": share("k_class", K_CLASSES),
        "pattern_pct": share("pattern", PATTERNS),
        "mean_n_true": round(float(sub["n_true"].mean()), 4),
        "mean_n_s2": round(float(sub["n_s2"].mean()), 4),
        "mean_n_s3": round(float(sub["n_s3"].mean()), 4),
        "max_cluster_size": int(sub["cluster_size"].max()),
        "pct_in_clusters_ge_10": round(100 * sub.filter(pl.col("cluster_size") >= 10).height / n, 3),
    }


# --------------------------------------------------------------------------- #
# Use
# --------------------------------------------------------------------------- #
def load_entities(out_dir: str, columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(out_dir, "entity_folds.parquet"), columns=columns)


def load_positive_pairs(out_dir: str) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(out_dir, "positive_pairs.parquet"))


def _select(ent: pl.DataFrame, folds: Optional[Iterable[int]] = None, country: Optional[str] = None,
            exclude_folds: Optional[Iterable[int]] = None) -> pl.DataFrame:
    if folds is not None:
        ent = ent.filter(pl.col("fold").is_in(list(folds)))
    if exclude_folds is not None:
        ent = ent.filter(~pl.col("fold").is_in(list(exclude_folds)))
    if country is not None:
        ent = ent.filter(pl.col("country") == country)
    return ent


def entity_ids(out_dir: str, folds: Optional[Iterable[int]] = None, country: Optional[str] = None,
               exclude_folds: Optional[Iterable[int]] = None) -> List[str]:
    """S1 ids of a fold, of everything but a fold (training side), or of one country (leave-one-country-out)."""
    return _select(load_entities(out_dir, ["entity_id", "fold", "country"]), folds, country, exclude_folds)["entity_id"].to_list()


def entity_truth(out_dir: str, folds: Optional[Iterable[int]] = None, country: Optional[str] = None,
                 exclude_folds: Optional[Iterable[int]] = None) -> Dict[str, List[str]]:
    """Entity-level ground truth {S1 id -> matched ids} for the selection, in the evaluator's format."""
    ent = _select(load_entities(out_dir, ["entity_id", "fold", "country", "matched"]), folds, country, exclude_folds)
    return dict(zip(ent["entity_id"].to_list(), ent["matched"].to_list()))


def label_pairs(pairs: pl.DataFrame, out_dir: str, s1_col: str = "source1_entity_id", cand_col: str = "candidate_id") -> pl.DataFrame:
    """Attach ``label`` (1 = true match, 0 otherwise) and ``fold`` to any (S1, candidate) pair table.

    The ground truth is complete, so every pair absent from it is a genuine negative.
    """
    pos = load_positive_pairs(out_dir).select(
        pl.col("source1_entity_id").alias(s1_col), pl.col("matched_id").alias(cand_col), pl.lit(1, dtype=pl.Int8).alias("label"))
    ent = load_entities(out_dir, ["entity_id", "fold"]).rename({"entity_id": s1_col})
    out = pairs.join(pos, on=[s1_col, cand_col], how="left").with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    if "fold" in out.columns:
        out = out.drop("fold")
    return out.join(ent, on=s1_col, how="left")


def score(pred: Mapping[str, Iterable[str]], out_dir: str, folds: Optional[Iterable[int]] = None, country: Optional[str] = None,
          exclude_folds: Optional[Iterable[int]] = None, strict: bool = False) -> dict:
    """Macro F0.5 on a selection of entities, with slices by country, match-count class and source pattern.

    ``strict=False`` scores a prediction that covers only the selected entities (missing rows = empty).
    """
    ent = _select(load_entities(out_dir, ["entity_id", "fold", "country", "matched", "k_class", "pattern", "n_true"]),
                  folds, country, exclude_folds)
    truth = dict(zip(ent["entity_id"].to_list(), ent["matched"].to_list()))
    res = ev.macro_f05(truth, pred, entity_ids=truth.keys(), strict=strict, keep_per_entity=True)
    per = pl.DataFrame({"entity_id": [s.entity_id for s in res.per_entity], "f05": [s.f05 for s in res.per_entity],
                        "n_pred": [s.n_pred for s in res.per_entity], "tp": [s.tp for s in res.per_entity],
                        "fp": [s.fp for s in res.per_entity], "fn": [s.fn for s in res.per_entity]})
    per = per.join(ent.select("entity_id", "country", "k_class", "pattern"), on="entity_id")

    def slice_by(col):
        g = per.group_by(col).agg(pl.len().alias("n"), pl.col("f05").mean().alias("f05"),
                                  (pl.col("n_pred") == 0).mean().alias("empty_rate")).sort(col)
        return {str(r[col]): {"n": int(r["n"]), "macro_f05": round(float(r["f05"]), 5), "empty_pred_rate": round(float(r["empty_rate"]), 4)}
                for r in g.to_dicts()}

    return {
        "macro_f05": res.macro_f05,
        "n_entities": res.n_entities,
        "singleton_f05": round(res.singleton_f05, 5),
        "non_singleton_f05": round(res.non_singleton_f05, 5),
        "pooled_precision": None if res.pooled_precision is None else round(res.pooled_precision, 5),
        "pooled_recall": None if res.pooled_recall is None else round(res.pooled_recall, 5),
        "empty_pred_rate": round(float((per["n_pred"] == 0).mean()), 4),
        "true_singleton_rate": round(float((ent["n_true"] == 0).mean()), 4),
        "by_country": slice_by("country"),
        "by_k_class": slice_by("k_class"),
        "by_pattern": slice_by("pattern"),
    }


def format_report(rep: dict) -> str:
    lines = [f"macro F0.5 = {rep['macro_f05']:.5f} on {rep['n_entities']:,} entities "
             f"(singletons {rep['singleton_f05']:.4f}, with matches {rep['non_singleton_f05']:.4f}; "
             f"empty predicted {rep['empty_pred_rate']:.2%} vs true {rep['true_singleton_rate']:.2%})"]
    for key in ("by_country", "by_k_class", "by_pattern"):
        lines.append("  " + key + ": " + ", ".join(f"{k}={v['macro_f05']:.4f} (n={v['n']:,})" for k, v in rep[key].items()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build or score leakage-safe validation folds.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--train-dir", required=True); b.add_argument("--out-dir", required=True)
    b.add_argument("--folds", type=int, default=N_FOLDS_DEFAULT); b.add_argument("--no-address-merge", action="store_true")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--addr-max-group", type=int, default=ADDR_MAX_GROUP_DEFAULT,
                   help="merge on an address only if at most this many entities share it")
    s = sub.add_parser("score"); s.add_argument("--out-dir", required=True); s.add_argument("--pred", required=True)
    s.add_argument("--fold", type=int, action="append", default=None); s.add_argument("--country", default=None)
    s.add_argument("--strict", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "build":
        summary = build_folds(a.train_dir, a.out_dir, n_folds=a.folds, merge_address=not a.no_address_merge, seed=a.seed,
                              addr_max_group=a.addr_max_group)
        print(json.dumps(summary, indent=1))
        return 0
    pred = ev.read_id_list_file(a.pred, strict=a.strict)
    rep = score(pred, a.out_dir, folds=a.fold, country=a.country, strict=a.strict)
    print(format_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
