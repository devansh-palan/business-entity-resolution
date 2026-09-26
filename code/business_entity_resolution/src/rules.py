"""Deterministic matching rules on the preprocessed tables, and their entity-level evaluation.

A rule is an equality join between Source 1 and the Source 2/3 pool on ``country`` plus a set of key columns,
optionally followed by a pairwise predicate ("post filter") on parsed address components.  Every rule can be
evaluated raw or with the one-to-one refinement: the ground truth maps each Source 2/3 record to at most one
Source 1 entity (7,638,365 matched ids, all distinct), so when a rule links one Source 2/3 record to several
Source 1 entities at least all but one of those links are false merges; the refinement drops such records.

Evaluation is entity-level with the verified evaluator (evaluation.macro_f05): macro F0.5 over the Source 1
entities of the selected fold, singletons scoring 1.0 only when predicted empty.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev  # noqa: E402

NAME_KEYS = ("name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted")
POOL_COLUMNS = ["entity_id", "src", "country", *NAME_KEYS, "addr_norm", "addr_sorted", "house_number", "street", "street_type",
                "city_norm", "state", "state_recognised", "postal_code", "addr_numbers"]


@dataclass
class Rule:
    key: str
    title: str
    keys: Tuple[str, ...]                        # equality columns besides country
    post: Optional[str] = None                   # name of a pairwise predicate in POST_FILTERS
    name_union: bool = False                     # join once per name representation and take the union
    note: str = ""


# --------------------------------------------------------------------------- #
# Pairwise predicates on the joined frame (columns suffixed _1 for Source 1 and _2 for the pool)
# --------------------------------------------------------------------------- #
def _fuzzy_ge(a: Optional[str], b: Optional[str], thr: float) -> bool:
    return a is not None and b is not None and fuzz.ratio(a, b) >= thr


def street_agrees(df: pl.DataFrame) -> pl.Series:
    """Street names equal, or fuzzy ratio >= 85, or missing on at least one side."""
    s1, s2 = df["street_1"].to_list(), df["street_2"].to_list()
    return pl.Series([a is None or b is None or a == b or fuzz.ratio(a, b) >= 85 for a, b in zip(s1, s2)], dtype=pl.Boolean)


def city_agrees(df: pl.DataFrame) -> pl.Series:
    c1, c2 = df["city_norm_1"].to_list(), df["city_norm_2"].to_list()
    return pl.Series([a is None or b is None or a == b or fuzz.ratio(a, b) >= 85 for a, b in zip(c1, c2)], dtype=pl.Boolean)


def no_state_conflict(df: pl.DataFrame, compatible: Optional[Dict[str, Dict[str, Set[str]]]] = None) -> pl.Series:
    """No conflict when either state is missing or unrecognised, or both are recognised and equal/compatible."""
    both = df["state_recognised_1"] & df["state_recognised_2"] & df["state_1"].is_not_null() & df["state_2"].is_not_null()
    equal = df["state_1"] == df["state_2"]
    if compatible:
        comp = [(a is not None and b is not None and b in compatible.get(c, {}).get(a, set()))
                for a, b, c in zip(df["state_1"].to_list(), df["state_2"].to_list(), df["country"].to_list())]
        equal = equal | pl.Series(comp, dtype=pl.Boolean)
    return (~both) | equal


def strong_address(df: pl.DataFrame, compatible=None) -> pl.Series:
    return street_agrees(df) & city_agrees(df) & no_state_conflict(df, compatible)


def street_agrees_no_conflict(df: pl.DataFrame, compatible=None) -> pl.Series:
    return street_agrees(df) & no_state_conflict(df, compatible)


POST_FILTERS: Dict[str, Callable[..., pl.Series]] = {
    "strong_address": strong_address,
    "street_no_conflict": street_agrees_no_conflict,
    "no_state_conflict": no_state_conflict,
}


# --------------------------------------------------------------------------- #
# Rule catalogue
# --------------------------------------------------------------------------- #
RULES: List[Rule] = [
    Rule("R1", "exact normalised name + exact normalised address", ("name_core", "addr_norm"),
         note="name_core: legal forms, honorifics, designations removed; addr_norm: all cleaned address tokens in order"),
    Rule("R1s", "sorted name tokens + sorted address tokens", ("name_core_sorted", "addr_sorted"), note="order-insensitive variant of R1"),
    Rule("R6", "name + city + postal code", ("name_core", "city_norm", "postal_code"), note="postal codes exist in 0.004% of records"),
    Rule("R6c", "name + city + state (postal code unavailable)", ("name_core", "city_norm", "state"), note="nearest available form of rule 6"),
    Rule("R7", "name + house number + street", ("name_core", "house_number", "street")),
    Rule("R7s", "sorted name + house number + street", ("name_core_sorted", "house_number", "street")),
    Rule("R8", "very strong: name + house number, street/city agree, no state conflict", ("name_core", "house_number"), post="strong_address"),
    Rule("R8u", "R8 with any name representation (core, sorted, compact, transliterated)", ("house_number",), post="strong_address", name_union=True),
    # diagnostics that justify the candidate-signal / feature classification
    Rule("D1", "exact name only", ("name_core",)),
    Rule("D2", "name + state", ("name_core", "state")),
    Rule("D3", "name + house number", ("name_core", "house_number")),
    Rule("D4", "name + street", ("name_core", "street")),
    Rule("D5", "name + city", ("name_core", "city_norm")),
    Rule("D6", "exact address only", ("addr_norm",)),
    Rule("D7", "house number + street + city (no name)", ("house_number", "street", "city_norm")),
    Rule("D8", "any name representation + house number, no state conflict", ("house_number",), post="no_state_conflict", name_union=True),
]
RULE_BY_KEY = {r.key: r for r in RULES}


# --------------------------------------------------------------------------- #
# Applying a rule
# --------------------------------------------------------------------------- #
POST_COLUMNS = {
    "strong_address": ("street", "city_norm", "state", "state_recognised"),
    "street_no_conflict": ("street", "state", "state_recognised"),
    "no_state_conflict": ("state", "state_recognised"),
}


def _prep(df: pl.DataFrame, keys: Sequence[str], suffix: str, id_name: str, extra: Sequence[str]) -> pl.DataFrame:
    """Rows with every key present; id renamed, the columns a post filter needs suffixed, the rest dropped."""
    out = df.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in keys]))
    cols = ["entity_id", "country", *keys, *[c for c in extra if c not in keys]]
    return out.select(cols).rename({"entity_id": id_name, **{c: f"{c}{suffix}" for c in extra if c not in keys}})


def apply_rule(rule: Rule, s1: pl.DataFrame, pool: pl.DataFrame, compatible=None, keep_keys: bool = False) -> pl.DataFrame:
    """Return the pairs (source1_entity_id, matched_id) the rule links.

    ``keep_keys`` keeps the join-key columns (single-representation rules only), so the caller can count how
    many Source 1 entities share the key without materialising the join against all of Source 1.
    """
    key_sets = [(n, *rule.keys) for n in NAME_KEYS] if rule.name_union else [rule.keys]
    extra = POST_COLUMNS.get(rule.post, ()) if rule.post else ()
    parts = []
    for keys in key_sets:
        a = _prep(s1, keys, "_1", "source1_entity_id", extra)
        b = _prep(pool, keys, "_2", "matched_id", extra)
        joined = a.join(b, on=["country", *keys], how="inner")
        if rule.post and joined.height:
            joined = joined.filter(POST_FILTERS[rule.post](joined, compatible))
        cols = ["source1_entity_id", "matched_id"] + (["country", *keys] if keep_keys and not rule.name_union else [])
        parts.append(joined.select(cols))
    return pl.concat(parts).unique()


def one_to_one_by_key(pairs: pl.DataFrame, s1_all: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
    """One-to-one refinement for a pure equality rule: keep pairs whose key is carried by exactly one Source 1 entity.

    Equivalent to one_to_one() on the join against every Source 1 record, without materialising that join.
    """
    counts = (s1_all.filter(pl.all_horizontal([pl.col(k).is_not_null() for k in keys]))
                    .group_by(["country", *keys]).agg(pl.col("entity_id").n_unique().alias("n_s1_key")))
    out = pairs.join(counts, on=["country", *keys], how="left")
    return out.filter(pl.col("n_s1_key") == 1).select("source1_entity_id", "matched_id")


def one_to_one(pairs: pl.DataFrame) -> pl.DataFrame:
    """Keep only pool records the rule links to exactly one Source 1 entity."""
    n = pairs.group_by("matched_id").agg(pl.col("source1_entity_id").n_unique().alias("n_s1"))
    return pairs.join(n.filter(pl.col("n_s1") == 1).select("matched_id"), on="matched_id", how="inner")


def cap_predictions(pairs: pl.DataFrame, max_per_entity: int = 11) -> pl.DataFrame:
    """Drop the whole prediction of entities linked to more records than any true entity has (train cap 11)."""
    n = pairs.group_by("source1_entity_id").agg(pl.len().alias("n"))
    return pairs.join(n.filter(pl.col("n") <= max_per_entity).select("source1_entity_id"), on="source1_entity_id", how="inner")


# --------------------------------------------------------------------------- #
# Entity-level evaluation
# --------------------------------------------------------------------------- #
@dataclass
class RuleResult:
    key: str
    variant: str
    n_pairs: int
    tp: int
    fp: int
    fn: int
    pair_precision: float
    pair_recall: float
    entities_affected: int
    entities_with_false_merge: int
    entities_with_missed_match: int
    singleton_false_positives: int
    entities_fully_correct: int
    macro_f05: float
    delta_vs_empty: float
    by_country: Dict[str, float] = field(default_factory=dict)
    by_src_recall: Dict[str, float] = field(default_factory=dict)
    precision_by_n_pred: Dict[str, Tuple[int, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_pairs(pairs: pl.DataFrame, truth: Dict[str, List[str]], positives: pl.DataFrame, countries: Dict[str, str],
                   key: str = "", variant: str = "raw") -> RuleResult:
    """Score a pair table against the entity truth of one fold.

    ``truth``     : {S1 id -> true matched ids} for every entity of the fold (defines the scored set)
    ``positives`` : fold positives with columns source1_entity_id, matched_id, src
    """
    ids = set(truth)
    pairs = pairs.filter(pl.col("source1_entity_id").is_in(list(ids))).unique()
    labelled = pairs.join(positives.with_columns(pl.lit(1, dtype=pl.Int8).alias("label")), on=["source1_entity_id", "matched_id"], how="left") \
                    .with_columns(pl.col("label").fill_null(0))
    tp = int(labelled["label"].sum()); fp = labelled.height - tp; fn = positives.height - tp
    grouped = pairs.group_by("source1_entity_id").agg(pl.col("matched_id"))
    res = ev.macro_f05_from_rows(truth, grouped.iter_rows(), keep_per_entity=True)   # streamed: no dict of every id
    empty = sum(1 for v in truth.values() if not v) / len(truth)
    per = res.per_entity
    affected = [s for s in per if s.n_pred > 0]
    by_country: Dict[str, List[float]] = {}
    for s in per:
        by_country.setdefault(countries[s.entity_id], []).append(s.f05)
    src_rec = {}
    for src in ("S2", "S3"):
        sub = positives.filter(pl.col("src") == src)
        hit = sub.join(pairs, on=["source1_entity_id", "matched_id"], how="inner").height
        src_rec[src] = round(hit / sub.height, 5) if sub.height else 0.0
    buckets = {"1": (1, 1), "2": (2, 2), "3-5": (3, 5), "6-11": (6, 11), ">11": (12, 10**9)}
    prec_by_n = {}
    npred = labelled.group_by("source1_entity_id").agg(pl.len().alias("n"), pl.col("label").sum().alias("tp"))
    for name, (lo, hi) in buckets.items():
        sub = npred.filter((pl.col("n") >= lo) & (pl.col("n") <= hi))
        if sub.height:
            prec_by_n[name] = (int(sub.height), round(float(sub["tp"].sum() / sub["n"].sum()), 5))
    return RuleResult(
        key=key, variant=variant, n_pairs=pairs.height, tp=tp, fp=fp, fn=fn,
        pair_precision=round(tp / pairs.height, 5) if pairs.height else 0.0,
        pair_recall=round(tp / positives.height, 5) if positives.height else 0.0,
        entities_affected=len(affected),
        entities_with_false_merge=sum(1 for s in affected if s.fp > 0),
        entities_with_missed_match=sum(1 for s in per if s.fn > 0),
        singleton_false_positives=sum(1 for s in affected if s.n_true == 0),
        entities_fully_correct=sum(1 for s in per if s.f05 == 1.0),
        macro_f05=round(res.macro_f05, 5), delta_vs_empty=round(res.macro_f05 - empty, 5),
        by_country={c: round(sum(v) / len(v), 5) for c, v in sorted(by_country.items())},
        by_src_recall=src_rec, precision_by_n_pred=prec_by_n,
    )


def union_pairs(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    frames = list(frames)
    return pl.concat(frames).unique() if frames else pl.DataFrame({"source1_entity_id": [], "matched_id": []}, schema={"source1_entity_id": pl.String, "matched_id": pl.String})
