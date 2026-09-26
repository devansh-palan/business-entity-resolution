"""Pairwise feature matrix: shapes, exact-match semantics, null handling, labels and entity-relative features."""
import json
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features as F  # noqa: E402
import preprocess as pp  # noqa: E402

S1 = [("S1-1", "Acme Tools Inc", "12 Main St, Springfield, IL", "US"),
      ("S1-2", "Zeta Foods Pvt Ltd", "40 MG Road, Bangalore, Karnataka", "India"),
      ("S1-3", "Blue River Dental", None, "US")]
POOL = [("S2-a", "ACME TOOLS", "12 Main Street, Springfield, IL", "US", "S2"),
        ("S2-b", "acmetools.com", "99 Elm St, Dayton, OH", "US", "S2"),
        ("S3-c", "Zeta Food", "40 M G Road, Bangalore", "India", "S3"),
        ("S3-d", "Blue River Dental Care", None, "US", "S3"),
        ("S2-e", "Omega Corp", "12 Main St, Springfield, IL", "US", "S2")]
PAIRS = [(0, 0), (0, 1), (0, 4), (1, 2), (2, 3), (2, 1)]
POSITIVES = [(0, 0), (1, 2), (2, 3)]


@pytest.fixture
def split(tmp_path):
    pp._init(None, None)
    s1 = pl.DataFrame(pp.process_records([r[0] for r in S1], [r[1] for r in S1], [r[2] for r in S1], [r[3] for r in S1], "S1"), schema=pp.SCHEMA)
    s1.write_parquet(tmp_path / "train_source1.parquet")
    for src in ("S2", "S3"):
        rows = [r for r in POOL if r[4] == src]
        df = pl.DataFrame(pp.process_records([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], src), schema=pp.SCHEMA)
        df.write_parquet(tmp_path / f"train_source{src[1]}.parquet")
    # p_idx positions follow the S2 file then the S3 file: S2-a 0, S2-b 1, S2-e 2, S3-c 3, S3-d 4
    remap = {0: 0, 1: 1, 4: 2, 2: 3, 3: 4}
    pl.DataFrame({"q_idx": [q for q, _ in PAIRS], "p_idx": [remap[p] for _, p in PAIRS]}).write_parquet(tmp_path / "pairs.parquet")
    pl.DataFrame({"q_idx": [q for q, _ in POSITIVES], "p_idx": [remap[p] for _, p in POSITIVES]}).write_parquet(tmp_path / "labels.parquet")
    return tmp_path, remap


def test_build_writes_labelled_feature_parts(split):
    tmp, remap = split
    meta = F.build(str(tmp), "train", str(tmp / "pairs.parquet"), str(tmp / "feat"), labels_path=str(tmp / "labels.parquet"), n_buckets=2, workers=1, log=lambda *a: None)
    parts = sorted(os.listdir(tmp / "feat"))
    assert parts == ["_pool_stats", "features_meta.json", "part_00.parquet", "part_01.parquet"]
    assert meta["stats"]["n_features"]["name_ngram"] > 0 and all(b["seconds"] >= 0 for b in meta["buckets"])
    df = pl.concat([pl.read_parquet(tmp / "feat" / p) for p in parts if p.endswith(".parquet")]).sort(["q_idx", "p_idx"])
    assert df.height == len(PAIRS) and meta["n_pairs"] == len(PAIRS)
    assert meta["n_features"] >= 100 and set(meta["features"]) == set(F.feature_columns(df))
    assert sum(len(v) for v in meta["groups"].values()) == meta["n_features"]
    assert df["label"].sum() == 3 and set(zip(df.filter(pl.col("label") == 1)["q_idx"].to_list(), df.filter(pl.col("label") == 1)["p_idx"].to_list())) == {(q, remap[p]) for q, p in POSITIVES}
    assert json.load(open(tmp / "feat" / "features_meta.json"))["n_features"] == meta["n_features"]
    # no feature ever equals the label trivially: every feature column is either non-constant or independent of q/p ids
    assert "label" not in meta["features"] and "q_idx" not in meta["features"]


def test_feature_semantics(split):
    tmp, remap = split
    F.build(str(tmp), "train", str(tmp / "pairs.parquet"), str(tmp / "feat"), labels_path=str(tmp / "labels.parquet"), n_buckets=1, workers=1, log=lambda *a: None, in_process=True)
    df = pl.read_parquet(tmp / "feat" / "part_00.parquet")
    row = lambda q, p: df.filter((pl.col("q_idx") == q) & (pl.col("p_idx") == remap[p])).to_dicts()[0]
    a = row(0, 0)     # Acme Tools Inc vs ACME TOOLS at the same address
    assert a["name_exact_core"] == 1 and a["name_exact_raw"] == 0 and a["name_lev_sim"] == pytest.approx(1.0) and a["house_eq"] == 1 and a["city_eq"] == 1
    assert a["src_s3"] == 0 and a["country_us"] == 1 and a["country_eq"] == 1 and a["legal_1"] == 1 and a["legal_2"] == 0
    assert a["cos_name_addr"] > 0.5 and a["cos_name_ngram"] > 0.9 and a["name_common_prefix"] == len("acme tools")
    b = row(0, 1)     # acmetools.com: domain-style pool name, different address
    assert b["p_domain_style"] == 1 and b["name_exact_compact"] == 1 and b["name_contains"] == 1 and b["house_conflict"] == 1 and b["city_conflict"] == 1
    assert b["ident_x_name"] == pytest.approx(b["name_partial"])
    e = row(0, 4)     # Omega Corp at Acme's address: address agrees, name does not
    assert e["addr_exact_norm"] == 1 and e["name_exact_core"] == 0 and e["name_lev_sim"] < 0.5
    assert e["n_exact_addr_q"] == 1 and a["addr_exact_norm"] == 0 and a["addr_tok_jaccard"] > 0.6     # 'street' is not folded to 'st' in addr_norm
    assert e["n_cand_q"] == 3 and e["n_cand_qs"] == 3 and a["rank_cos_q"] == 1 and a["is_top_cos_q"] == 1
    d = row(2, 3)     # both addresses missing
    assert d["addr_missing_1"] == 1 and d["addr_missing_2"] == 1 and d["addr_lev_sim"] is None and d["house_eq"] is None and d["addr_tok_jaccard"] is None
    assert d["name_contains"] == 1 and d["name_common_prefix"] == len("blue river dental")
    assert "house_lev_dist" not in df.columns      # the decoy block is off by default
    z = row(1, 2)     # Zeta Foods Pvt Ltd vs Zeta Food (India)
    assert z["country_us"] == 0 and z["legal_1"] == 1 and z["name_jw"] > 0.9 and z["name_tok_common"] == 1 and z["house_eq"] == 1
    assert df.select(pl.col("name_pool_freq_2")).to_series().max() > 0


def test_decoy_features(split, monkeypatch):
    tmp, remap = split
    monkeypatch.setattr(F, "DECOY_FEATURES", True)
    F.build(str(tmp), "train", str(tmp / "pairs.parquet"), str(tmp / "feat2"), labels_path=str(tmp / "labels.parquet"), n_buckets=1, workers=1, log=lambda *a: None, in_process=True)
    df = pl.read_parquet(tmp / "feat2" / "part_00.parquet")
    row = lambda q, p: df.filter((pl.col("q_idx") == q) & (pl.col("p_idx") == remap[p])).to_dicts()[0]
    a, b, e = row(0, 0), row(0, 1), row(0, 4)
    assert a["house_lev_dist"] == 0 and a["addr_same_ignoring_numbers"] == 0 and a["same_name_street_diff_house"] == 0 and a["has_name_house_twin_q"] == 1
    assert b["house_lev_dist"] == 2 and b["house_digit_dropped"] == 0 and b["addr_num_sym_diff"] == 2 and b["unit_one_side"] == 0
    assert e["addr_lev_dist"] == 0 and e["n_same_name_street_diff_house_q"] == 0 and e["addr_num_1_subset_of_2"] == 1
    meta = json.load(open(tmp / "feat2" / "features_meta.json"))
    assert {"house_lev_dist", "unit_lev_dist", "addr_lev_dist"} <= set(meta["groups"]["address"]) and "has_exact_twin_q" in meta["groups"]["entity_relative"]


def test_pool_rows_positions(split):
    tmp, _ = split
    files = F.pool_files(str(tmp), "train")
    sizes = F.pool_sizes(files)
    rows = F.load_pool_rows(files, sizes, np.array([4, 0, 2, 0], dtype=np.uint32), ["business_name"])
    assert rows.column("business_name").to_pylist() == ["Blue River Dental Care", "ACME TOOLS", "Omega Corp", "ACME TOOLS"]     # pair order, duplicates kept
