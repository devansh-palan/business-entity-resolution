"""Tests for src/evaluation.py, the local re-implementation of the leaderboard metric.

Run from code/business_entity_resolution:   python -m pytest tests -q

The real-data tests at the bottom read student_resource/dataset/train/train_ground_truth.tsv
(about a minute) and are skipped when that file is absent.
"""
from __future__ import annotations

import math
import pathlib
import random
import sys
from collections import Counter

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import evaluation as ev  # noqa: E402

ROOT = HERE.parents[2]
GT_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


def same(actual, expected) -> bool:
    if expected is None:
        return actual is None
    return actual is not None and close(actual, expected)


# --------------------------------------------------------------------------- #
# The seven requested synthetic cases, plus the worked example from the README
# --------------------------------------------------------------------------- #
SYNTHETIC = [
    # name, truth, predicted, precision, recall, F0.5
    ("A", [], [], None, None, 1.0),
    ("B", [], ["wrong"], 0.0, None, 0.0),
    ("C", ["A"], ["A"], 1.0, 1.0, 1.0),
    ("D", ["A"], [], None, 0.0, 0.0),
    ("E", ["A"], ["A", "B"], 0.5, 1.0, 5 / 9),
    ("F", ["A", "B"], ["A"], 1.0, 0.5, 5 / 6),
    ("G", ["A", "B"], ["A", "B", "C"], 2 / 3, 1.0, 5 / 7),
]


@pytest.mark.parametrize("name,truth,pred,p,r,f", SYNTHETIC)
def test_synthetic_examples(name, truth, pred, p, r, f):
    s = ev.score_entity(name, truth, pred)
    assert same(s.precision, p), (name, s)
    assert same(s.recall, r), (name, s)
    assert close(s.f05, f), (name, s)


def test_readme_worked_example():
    truth = ["S2-00047", "S3-00812"]
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    s = ev.score_entity("S1-00001", truth, pred)
    assert close(s.precision, 2 / 3) and close(s.recall, 1.0)
    assert round(s.f05, 3) == 0.714 and close(s.f05, 5 / 7)


def test_completely_wrong_predictions_score_zero():
    assert ev.score_entity("e", ["A"], ["B"]).f05 == 0.0
    assert ev.score_entity("e", ["A", "B"], ["C", "D"]).f05 == 0.0
    s = ev.score_entity("e", ["A", "B"], ["C", "D", "E"])
    assert s.precision == 0.0 and s.recall == 0.0 and s.f05 == 0.0


def test_partially_correct_with_multiple_predictions():
    s = ev.score_entity("e", ["A", "B", "C"], ["A", "D"])
    assert close(s.precision, 0.5) and close(s.recall, 1 / 3) and close(s.f05, 5 / 11)
    s = ev.score_entity("e", ["A", "B", "C", "D"], ["A", "B", "X", "Y", "Z"])
    assert close(s.precision, 0.4) and close(s.recall, 0.5) and close(s.f05, 5 / 12)
    assert s.tp == 2 and s.fp == 3 and s.fn == 2


def test_order_irrelevant_and_repeats_collapse_in_memory():
    a = ev.score_entity("e", ["A", "B", "C"], ["C", "A"])
    b = ev.score_entity("e", ["C", "B", "A"], ["A", "C", "A"])
    assert a.f05 == b.f05 and a.tp == b.tp == 2 and b.n_pred == 2


def test_f_beta_is_the_statement_formula():
    assert close(ev.f_beta(2 / 3, 1.0), 5 / 7)
    assert close(ev.f_beta(1.0, 0.5), 5 / 6)
    assert ev.f_beta(0.0, 0.0) == 0.0
    # F_beta with beta=1 degenerates to the harmonic mean
    assert close(ev.f_beta(0.5, 1.0, beta=1.0), 2 / 3)


def test_formula_equals_count_form():
    rng = random.Random(0)
    for _ in range(500):
        n_true = rng.randint(1, 6)
        truth = [f"S2-{i}" for i in range(n_true)]
        pred = rng.sample([f"S2-{i}" for i in range(10)], rng.randint(1, 8))
        s = ev.score_entity("e", truth, pred)
        expected = 1.25 * s.tp / (1.25 * s.tp + 0.25 * s.fn + s.fp)
        assert close(s.f05, expected)


def test_bare_string_is_rejected():
    with pytest.raises(TypeError):
        ev.score_entity("e", "S2-1", ["S2-1"])


# --------------------------------------------------------------------------- #
# Macro average: denominator, singletons, missing / extra rows, subsets
# --------------------------------------------------------------------------- #
TRUTH = {"e1": [], "e2": ["S2-1"], "e3": ["S2-2", "S3-3"]}


def test_macro_average_over_all_entities():
    preds = {"e1": [], "e2": ["S2-1"], "e3": ["S2-2"]}
    res = ev.macro_f05(TRUTH, preds)
    assert close(res.macro_f05, (1.0 + 1.0 + 5 / 6) / 3)
    assert res.n_entities == 3 and res.n_singletons == 1
    assert close(res.singleton_f05, 1.0) and close(res.non_singleton_f05, (1.0 + 5 / 6) / 2)
    assert res.by_n_true == {0: (1, 1.0), 1: (1, 1.0), 2: (1, 5 / 6)}
    assert close(res.pooled_precision, 1.0) and close(res.pooled_recall, 2 / 3)
    assert res.n_pred_nonempty == 2


def test_singletons_count_fully():
    preds = {"e1": ["S2-9"], "e2": ["S2-1"], "e3": ["S2-2", "S3-3"]}
    res = ev.macro_f05(TRUTH, preds)
    assert close(res.macro_f05, (0.0 + 1.0 + 1.0) / 3) and res.singleton_f05 == 0.0


def test_missing_prediction_rows():
    preds = {"e1": [], "e2": ["S2-1"]}
    with pytest.raises(ev.SubmissionFormatError):
        ev.macro_f05(TRUTH, preds)
    res = ev.macro_f05(TRUTH, preds, strict=False)  # e3 scored as empty -> 0
    assert close(res.macro_f05, 2 / 3)


def test_extra_prediction_rows():
    preds = {"e1": [], "e2": ["S2-1"], "e3": ["S2-2", "S3-3"], "e4": ["S2-7"]}
    with pytest.raises(ev.SubmissionFormatError):
        ev.macro_f05(TRUTH, preds)
    assert close(ev.macro_f05(TRUTH, preds, strict=False).macro_f05, 1.0)


def test_subset_scoring_like_a_fold_or_the_public_leaderboard():
    preds = {"e1": ["S2-9"], "e2": ["S2-1"], "e3": ["S2-2"]}
    assert close(ev.macro_f05(TRUTH, preds, entity_ids=["e2", "e3"]).macro_f05, (1.0 + 5 / 6) / 2)
    assert close(ev.macro_f05(TRUTH, preds, entity_ids=["e1"]).macro_f05, 0.0)
    # extra prediction rows are fine when a subset is scored
    assert close(ev.macro_f05({"e2": ["S2-1"]}, preds, entity_ids=["e2"]).macro_f05, 1.0)
    with pytest.raises(KeyError):
        ev.macro_f05(TRUTH, preds, entity_ids=["nope"])


def test_per_entity_records_are_optional():
    res = ev.macro_f05(TRUTH, {"e1": [], "e2": [], "e3": ["S3-3", "S2-2", "S2-5"]}, keep_per_entity=True)
    by_id = {s.entity_id: s for s in res.per_entity}
    assert by_id["e2"].f05 == 0.0 and close(by_id["e3"].f05, 5 / 7)
    assert ev.macro_f05(TRUTH, {"e1": [], "e2": [], "e3": []}).per_entity is None


# --------------------------------------------------------------------------- #
# File format: mirror utils/validate_submission.py
# --------------------------------------------------------------------------- #
def _write(path: pathlib.Path, text: str, newline: str = "\n") -> str:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", newline))
    return str(path)


HEADER = "source1_entity_id\tmatched_entity_ids\n"


def test_read_valid_file_lf_and_crlf(tmp_path):
    body = HEADER + "S1-1\tS2-1,S3-2\nS1-2\t\nS1-3\tS3-9\n"
    for newline in ("\n", "\r\n"):
        m = ev.read_id_list_file(_write(tmp_path / "p.tsv", body, newline))
        assert m == {"S1-1": ["S2-1", "S3-2"], "S1-2": [], "S1-3": ["S3-9"]}


def test_blank_lines_and_header_case_are_tolerated(tmp_path):
    body = " Source1_Entity_ID \tMATCHED_ENTITY_IDS\nS1-1\tS2-1\n\n   \nS1-2\t\n"
    assert ev.read_id_list_file(_write(tmp_path / "p.tsv", body)) == {"S1-1": ["S2-1"], "S1-2": []}


@pytest.mark.parametrize(
    "body",
    [
        HEADER + "S1-1\tS2-1, S3-2\n",         # whitespace after the comma
        HEADER + "S1-1\tS2-1,S2-1\n",          # repeated id
        HEADER + "S1-1\tS1-7\n",               # self match
        HEADER + "S1-1\tX-7\n",                # unknown prefix
        HEADER + "S1-1\tS2-1,\n",              # empty element
        HEADER + "S1-1\tS2-1\nS1-1\tS3-2\n",   # duplicate row
        HEADER + " S1-1\tS2-1\n",              # whitespace around the entity id
        HEADER + "S1-1\n",                     # row without a tab
        "source1_entity_id,matched_entity_ids\nS1-1,S2-1\n",  # a CSV
        "entity_id\tmatched_entity_ids\nS1-1\tS2-1\n",        # wrong header
    ],
)
def test_strict_reading_rejects_what_the_validator_rejects(tmp_path, body):
    with pytest.raises(ev.SubmissionFormatError):
        ev.read_id_list_file(_write(tmp_path / "bad.tsv", body))


def test_lenient_reading_repairs(tmp_path):
    body = HEADER + "S1-1\tS2-1, S3-2 ,S2-1,\nS1-1\tS3-9\n S1-2 \t\n"
    m = ev.read_id_list_file(_write(tmp_path / "p.tsv", body), strict=False)
    assert m == {"S1-1": ["S2-1", "S3-2"], "S1-2": []}


def test_empty_file_and_wrong_header_still_fail_leniently(tmp_path):
    with pytest.raises(ev.SubmissionFormatError):
        ev.read_id_list_file(_write(tmp_path / "e.tsv", ""), strict=False)
    with pytest.raises(ev.SubmissionFormatError):
        ev.read_id_list_file(_write(tmp_path / "h.tsv", "a\tb\nS1-1\tS2-1\n"), strict=False)


def test_writer_roundtrip_uses_lf_only(tmp_path):
    mapping = {"S1-1": ["S2-1", "S3-2"], "S1-2": [], "S1-3": ["S3-9"]}
    path = tmp_path / "out.tsv"
    ev.write_id_list_file(str(path), mapping)
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.startswith(b"source1_entity_id\tmatched_entity_ids\n")
    assert ev.read_id_list_file(str(path)) == mapping
    # entity_order emits every required entity, absent ones as empty rows
    ev.write_id_list_file(str(path), {"S1-1": ["S2-1"]}, entity_order=["S1-1", "S1-2"])
    assert ev.read_id_list_file(str(path)) == {"S1-1": ["S2-1"], "S1-2": []}
    ev.write_id_list_file(str(path), {"S1-1": ["S2-1"]}, header=ev.CANDIDATE_HEADER)
    assert ev.read_id_list_file(str(path), ev.CANDIDATE_HEADER) == {"S1-1": ["S2-1"]}


@pytest.mark.parametrize("bad", [["S2-1", "S2-1"], [" S2-1"], ["S1-4"], ["S2-1,S3-2"]])
def test_writer_validates(tmp_path, bad):
    with pytest.raises(ev.SubmissionFormatError):
        ev.write_id_list_file(str(tmp_path / "x.tsv"), {"S1-1": bad})


def test_score_files_and_cli(tmp_path, capsys):
    truth = _write(tmp_path / "truth.tsv", HEADER + "S1-1\tS2-1,S3-2\nS1-2\t\nS1-3\tS3-9\n")
    pred = _write(tmp_path / "pred.tsv", HEADER + "S1-1\tS2-1\nS1-2\t\nS1-3\tS2-5\n")
    res = ev.score_files(truth, pred)
    assert close(res.macro_f05, (5 / 6 + 1.0 + 0.0) / 3)
    ids = _write(tmp_path / "ids.txt", "S1-1\nS1-3\n")
    assert close(ev.score_files(truth, pred, ids).macro_f05, (5 / 6 + 0.0) / 2)
    assert ev.main(["--truth", truth, "--pred", pred, "--by-k"]) == 0
    out = capsys.readouterr().out
    assert "macro F0.5" in out and f"{(5 / 6 + 1.0) / 3:.6f}" in out
    bad = _write(tmp_path / "bad.tsv", HEADER + "S1-1\tS2-1\n")  # rows for S1-2, S1-3 missing
    assert ev.main(["--truth", truth, "--pred", bad]) == 2
    assert ev.main(["--truth", truth, "--pred", bad, "--lenient"]) == 0


# --------------------------------------------------------------------------- #
# Real ground truth: the scorer against closed-form expectations
# --------------------------------------------------------------------------- #
needs_gt = pytest.mark.skipif(not GT_PATH.exists(), reason="train_ground_truth.tsv not available")


@pytest.fixture(scope="module")
def gt():
    return ev.read_id_list_file(str(GT_PATH))


@needs_gt
def test_real_gt_shape_and_self_score(gt):
    assert len(gt) == 2_206_821
    ks = Counter(len(v) for v in gt.values())
    assert ks[0] == 123_247 and max(ks) == 11
    assert sum(k * c for k, c in ks.items()) == 7_638_365
    res = ev.macro_f05(gt, gt)
    assert res.macro_f05 == 1.0 and res.n_singletons == 123_247


@needs_gt
def test_real_gt_oracles_match_closed_form(gt):
    n = len(gt)
    ks = Counter(len(v) for v in gt.values())

    def expect(fn):
        # closed form: the score only depends on how many true matches an entity has
        return math.fsum(c * fn(k) for k, c in ks.items()) / n

    def near(a, b):
        # both sides are float arithmetic over 2.2M terms; the leaderboard shows far fewer digits
        return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)

    # 1. predict empty everywhere: score = singleton share
    res = ev.macro_f05(gt, {e: [] for e in gt}).macro_f05
    assert near(res, ks[0] / n) and math.isclose(res, 0.05585, abs_tol=5e-6)

    # 2. drop one true match from every entity with >= 2 matches
    pred = {e: v[1:] if len(v) >= 2 else v for e, v in gt.items()}
    res = ev.macro_f05(gt, pred).macro_f05
    assert near(res, expect(lambda k: 1.0 if k <= 1 else ev.f_beta(1.0, (k - 1) / k)))
    assert near(res, 0.9250413041568473)  # independent numpy computation from the EDA

    # 3. add one false match to every entity (singletons drop to 0)
    pred = {e: list(v) + ["S2-FAKE"] for e, v in gt.items()}
    res = ev.macro_f05(gt, pred).macro_f05
    assert near(res, expect(lambda k: 0.0 if k == 0 else ev.f_beta(k / (k + 1), 1.0)))
    assert near(res, 0.7515433478284971)

    # 4. exactly one correct match per non-singleton, empty for singletons
    pred = {e: v[:1] for e, v in gt.items()}
    res = ev.macro_f05(gt, pred).macro_f05
    assert near(res, expect(lambda k: 1.0 if k == 0 else ev.f_beta(1.0, 1 / k)))
    assert near(res, 0.6963880336487924)

    # 5. perfect on everything except one false match on every singleton
    pred = {e: (["S3-FAKE"] if not v else v) for e, v in gt.items()}
    res = ev.macro_f05(gt, pred).macro_f05
    assert near(res, 1.0 - ks[0] / n)


def test_macro_f05_from_rows_matches_dict_form():
    import random
    rng = random.Random(7)
    truth = {}
    pred = {}
    for i in range(2000):
        e = f"S1-{i}"
        k = rng.choice([0, 0, 1, 2, 3, 5])
        truth[e] = [f"S2-{i}-{j}" for j in range(k)]
        if rng.random() < 0.7:
            p = [m for m in truth[e] if rng.random() < 0.8] + [f"S3-x{i}-{j}" for j in range(rng.choice([0, 0, 1, 2]))]
            if p:
                pred[e] = p
    a = ev.macro_f05(truth, pred, strict=False, keep_per_entity=True)
    b = ev.macro_f05_from_rows(truth, [(e, p) for e, p in pred.items()] + [("unknown", ["S2-z"])], keep_per_entity=True)
    assert a.macro_f05 == b.macro_f05 and a.n_entities == b.n_entities and a.n_singletons == b.n_singletons
    assert a.by_n_true == b.by_n_true and a.pooled_precision == b.pooled_precision and a.pooled_recall == b.pooled_recall
    assert sorted(s.entity_id for s in a.per_entity) == sorted(s.entity_id for s in b.per_entity)
    assert {s.entity_id: s.f05 for s in a.per_entity} == {s.entity_id: s.f05 for s in b.per_entity}
    with pytest.raises(ValueError):
        ev.macro_f05_from_rows(truth, [("S1-1", ["a"]), ("S1-1", ["b"])])
