"""Matcher utilities: prior correction, decision rules, entity-level evaluation and false-merge classification."""
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matcher as M  # noqa: E402


def test_prior_correct_undoes_negative_subsampling():
    # with negatives kept at rate 0.1 the model's odds are 10x too high: p = 0.5 becomes odds 0.1 -> 1/11
    assert M.prior_correct(np.array([0.5], dtype=np.float32), 0.1)[0] == pytest.approx(1 / 11, abs=1e-6)
    assert M.prior_correct(np.array([0.5], dtype=np.float32), 1.0)[0] == pytest.approx(0.5, abs=1e-6)


@pytest.fixture
def scores():
    # entity 0: pool 10 (0.9), 11 (0.6); entity 1: pool 10 (0.8), 12 (0.3); entity 2: nothing above 0.2
    return pl.DataFrame({"q_idx": [0, 0, 1, 1, 2], "p_idx": [10, 11, 10, 12, 13], "m": [0.9, 0.6, 0.8, 0.3, 0.1],
                         "entity_id": ["S1-a", "S1-a", "S1-b", "S1-b", "S1-c"], "matched_id": ["S2-10", "S2-11", "S2-10", "S3-12", "S3-13"]})


def test_decision_rules(scores):
    plain = M.decide(scores, "m", M.Rule(0.5))
    assert set(zip(plain["q_idx"].to_list(), plain["p_idx"].to_list())) == {(0, 10), (0, 11), (1, 10)}
    uniq = M.decide(scores, "m", M.Rule(0.5, unique_pool=True))          # pool 10 goes to entity 0 only
    assert set(zip(uniq["q_idx"].to_list(), uniq["p_idx"].to_list())) == {(0, 10), (0, 11)}
    gated = M.decide(scores, "m", M.Rule(0.5, rel_gate=0.8))              # 0.6 < 0.8 * 0.9
    assert set(zip(gated["q_idx"].to_list(), gated["p_idx"].to_list())) == {(0, 10), (1, 10)}
    capped = M.decide(scores, "m", M.Rule(0.2, cap=1))
    assert set(zip(capped["q_idx"].to_list(), capped["p_idx"].to_list())) == {(0, 10), (1, 10)}
    assert "p" in plain.columns and "entity_id" in plain.columns
    assert M.Rule(0.5, True, 0.8, 2).label() == "p>=0.50 +unique +rel0.8 +cap2"


def test_entity_evaluation_and_breakdowns(scores):
    truth = {"S1-a": ["S2-10", "S2-11"], "S1-b": ["S3-12"], "S1-c": [], "S1-d": ["S3-99"]}
    countries = {"S1-a": "US", "S1-b": "India", "S1-c": "US", "S1-d": "India"}
    kept = M.decide(scores, "m", M.Rule(0.5)).select("entity_id", "matched_id")
    r = M.evaluate(kept, truth, countries, lambda m: m[:2])
    # a: perfect (F 1); b: predicted S2-10 (wrong) and missed S3-12 (F 0); c: empty (1); d: nothing predicted (0) -> macro 0.5
    assert r["macro_f05"] == pytest.approx(0.5) and r["n_entities"] == 4 and r["n_singletons"] == 1
    assert r["tp"] == 2 and r["fp"] == 1 and r["fn"] == 2
    assert r["singletons"]["predicted_empty_share"] == 1.0 and r["one_match"]["n"] == 2 and r["multi_match"]["n"] == 1
    assert r["by_country"]["US"]["mean_f05"] == 1.0 and r["by_country"]["India"]["mean_f05"] == 0.0
    assert r["by_source"]["S2"]["entities_with_truth"] == 1 and r["by_source"]["S2"]["macro_f05"] == 1.0
    assert r["by_source"]["S3"]["entities_with_truth"] == 2 and r["by_source"]["S3"]["macro_f05"] == 0.0
    assert r["false_merges"]["entities_with_fp"] == 1 and r["false_merges"]["fully_correct"] == 2
    assert M.macro_only(kept, truth) == pytest.approx(0.5)
    assert M.macro_only(kept.head(0), truth) == pytest.approx(0.25)          # only the singleton scores with an empty prediction


def test_false_merge_kinds(scores):
    positives = pl.DataFrame({"source1_entity_id": ["S1-a", "S1-a", "S1-b", "S1-zz"], "matched_id": ["S2-10", "S2-11", "S3-12", "S3-13"]})
    kept = M.decide(scores, "m", M.Rule(0.05)).select("q_idx", "p_idx", "entity_id", "matched_id", "p")
    fm = M.false_merge_table(kept, positives)
    kinds = dict(zip(fm["matched_id"].to_list(), fm["kind"].to_list()))
    assert kinds == {"S2-10": "stolen from another entity", "S3-13": "stolen from another entity"}     # b's S2-10 belongs to a; c's S3-13 to S1-zz
    assert fm.filter(pl.col("entity_id") == "S1-b")["true_owner"].to_list() == ["S1-a"]
