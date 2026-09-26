"""Deterministic rules: joins, one-to-one refinement, post filters and the entity-level metrics."""
import math
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rules as R  # noqa: E402

COLS = ["entity_id", "country", "name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted", "addr_norm", "addr_sorted",
        "house_number", "street", "street_type", "city_norm", "state", "state_recognised", "postal_code"]


def rec(eid, country, name, addr, house, street, city, state, recognised=True, postal=None, stype="street"):
    toks = addr.split()
    return dict(entity_id=eid, country=country, name_core=name, name_core_sorted=" ".join(sorted(name.split())), name_compact=name.replace(" ", ""),
                name_translit_core_sorted=" ".join(sorted(name.split())), addr_norm=addr, addr_sorted=" ".join(sorted(toks)),
                house_number=house, street=street, street_type=stype, city_norm=city, state=state, state_recognised=recognised, postal_code=postal)


@pytest.fixture
def data():
    s1 = pl.DataFrame([
        rec("S1-1", "US", "acme tools", "12 main street springfield il", "12", "main", "springfield", "il"),
        rec("S1-2", "US", "acme tools", "40 oak avenue dayton oh", "40", "oak", "dayton", "oh"),
        rec("S1-3", "India", "zeta foods", "5 mg road bangalore karnataka", "5", "mg", "bangalore", "karnataka"),
        rec("S1-4", "US", "solo corp", "7 elm street boston ma", "7", "elm", "boston", "ma"),
    ], schema_overrides={"postal_code": pl.String})
    pool = pl.DataFrame([
        rec("S2-a", "US", "acme tools", "12 main st springfield il", "12", "main", "springfield", "il"),
        rec("S2-b", "US", "acme tools", "40 oak ave dayton oh", "40", "oak", "dayton", "oh"),
        rec("S3-c", "US", "acme tools", "12 main street springfield il", "12", "main", "springfield", "il"),
        rec("S3-d", "India", "zeta foods", "5 mg road chennai tamil nadu", "5", "mg", "chennai", "tamil nadu"),   # decoy: state conflict
        rec("S2-f", "India", "zeta foods", "5 mg road bangalore karnataka", "5", "mg", "bangalore", "karnataka"),
        rec("S3-e", "US", "solo corp", "7 elm street cambridge ma", "7", "elm", "cambridge", "ma"),              # decoy for the singleton
    ], schema_overrides={"postal_code": pl.String}).with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
    truth = {"S1-1": ["S2-a", "S3-c"], "S1-2": ["S2-b"], "S1-3": ["S2-f"], "S1-4": []}
    positives = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-3"], "matched_id": ["S2-a", "S3-c", "S2-b", "S2-f"],
                              "src": ["S2", "S3", "S2", "S2"]})
    countries = {"S1-1": "US", "S1-2": "US", "S1-3": "India", "S1-4": "US"}
    return s1, pool, truth, positives, countries


def pairs_set(df):
    return set(zip(df["source1_entity_id"].to_list(), df["matched_id"].to_list()))


def test_rule1_exact_name_and_address(data):
    s1, pool, truth, positives, countries = data
    pairs = R.apply_rule(R.RULE_BY_KEY["R1"], s1, pool)
    assert pairs_set(pairs) == {("S1-1", "S3-c"), ("S1-3", "S2-f")}
    res = R.evaluate_pairs(pairs, truth, positives, countries, key="R1")
    assert (res.tp, res.fp, res.fn) == (2, 0, 2)
    assert res.entities_affected == 2 and res.singleton_false_positives == 0 and res.entities_with_false_merge == 0
    # S1-1: P=1, R=0.5 -> F0.5 = 1.25*0.5/(0.25+0.5); S1-2: 0; S1-3: 1; S1-4 (singleton, empty): 1
    expected = (1.25 * 0.5 / 0.75 + 0.0 + 1.0 + 1.0) / 4
    assert math.isclose(res.macro_f05, expected, abs_tol=1e-5)
    assert res.by_src_recall == {"S2": 0.33333, "S3": 1.0}


def test_name_only_rule_and_one_to_one(data):
    s1, pool, truth, positives, countries = data
    raw = R.apply_rule(R.RULE_BY_KEY["D1"], s1, pool, keep_keys=True)
    assert raw.height == 9
    res = R.evaluate_pairs(raw, truth, positives, countries, key="D1")
    assert (res.tp, res.fp) == (4, 5)
    assert res.singleton_false_positives == 1 and res.entities_with_false_merge == 4
    assert res.precision_by_n_pred == {"1": (1, 0.0), "2": (1, 0.5), "3-5": (2, 0.5)}
    by_key = R.one_to_one_by_key(raw, s1, R.RULE_BY_KEY["D1"].keys)
    by_record = R.one_to_one(raw.select("source1_entity_id", "matched_id"))
    assert pairs_set(by_key) == pairs_set(by_record) == {("S1-3", "S2-f"), ("S1-3", "S3-d"), ("S1-4", "S3-e")}


def test_rule8_strong_address_rejects_state_and_city_conflicts(data):
    s1, pool, truth, positives, countries = data
    pairs = R.apply_rule(R.RULE_BY_KEY["R8"], s1, pool)
    assert pairs_set(pairs) == {("S1-1", "S2-a"), ("S1-1", "S3-c"), ("S1-2", "S2-b"), ("S1-3", "S2-f")}
    res = R.evaluate_pairs(pairs, truth, positives, countries, key="R8")
    assert (res.tp, res.fp, res.fn, res.macro_f05) == (4, 0, 0, 1.0)
    assert res.entities_fully_correct == 4 and res.by_country == {"India": 1.0, "US": 1.0}


def test_rule7_links_decoys_that_share_number_and_street(data):
    s1, pool, truth, positives, countries = data
    pairs = R.apply_rule(R.RULE_BY_KEY["R7"], s1, pool)
    res = R.evaluate_pairs(pairs, truth, positives, countries, key="R7")
    assert (res.tp, res.fp) == (4, 2)
    assert res.singleton_false_positives == 1          # S1-4 <- S3-e
    assert res.entities_with_false_merge == 2           # S1-3 (S3-d) and S1-4 (S3-e)
    assert res.macro_f05 < 1.0


def test_name_union_rule_joins_every_representation(data):
    s1, pool, truth, positives, countries = data
    pool2 = pool.with_columns(pl.when(pl.col("entity_id") == "S2-a").then(pl.lit("tools acme")).otherwise(pl.col("name_core")).alias("name_core"))
    assert pairs_set(R.apply_rule(R.RULE_BY_KEY["R8"], s1, pool2)) == {("S1-1", "S3-c"), ("S1-2", "S2-b"), ("S1-3", "S2-f")}
    assert ("S1-1", "S2-a") in pairs_set(R.apply_rule(R.RULE_BY_KEY["R8u"], s1, pool2))   # sorted representation recovers it


def test_state_compatibility_and_unrecognised_states():
    df = pl.DataFrame({"country": ["India", "India", "India", "US"], "state_1": ["telangana", "telangana", "telangana", "tx"],
                       "state_2": ["andhra pradesh", "kerala", "kerala", "tx"], "state_recognised_1": [True, True, True, True],
                       "state_recognised_2": [True, True, False, True]})
    comp = {"India": {"telangana": {"andhra pradesh"}}}
    assert R.no_state_conflict(df, comp).to_list() == [True, False, True, True]
    assert R.no_state_conflict(df).to_list() == [False, False, True, True]


def test_cap_predictions_drops_oversized_entities():
    pairs = pl.DataFrame({"source1_entity_id": ["A"] * 12 + ["B"] * 2, "matched_id": [f"S2-{i}" for i in range(14)]})
    assert R.cap_predictions(pairs)["source1_entity_id"].unique().to_list() == ["B"]
