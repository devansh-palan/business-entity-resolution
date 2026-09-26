"""Tests for src/validation.py on a small synthetic training set (no real data needed)."""
from __future__ import annotations

import math
import os
import pathlib
import sys

import polars as pl
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import evaluation as ev  # noqa: E402
import validation as va  # noqa: E402


def _tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


@pytest.fixture()
def train_dir(tmp_path):
    d = tmp_path / "train"; d.mkdir()
    h = ["entity_id", "business_name", "business_address", "country"]
    s1 = [
        ["S1-1", "Summit Inc", "10 Main Street, Austin, TX", "US"],
        ["S1-2", "Summit LLC", "55 Oak Road, Dallas, TX", "US"],          # same core name as S1-1
        ["S1-3", "Blue River Corp", "7 Lake Drive, Reno, NV", "US"],
        ["S1-4", "Green Hills", "7 Lake Drive, Reno, NV", "US"],          # same address as S1-3
        ["S1-5", "Shree Traders Private Limited", "Plot 4, Sector 9, Pune, Maharashtra", "India"],
        ["S1-6", "Traders Shree Pvt Ltd", "Flat 2, Andheri, Mumbai, Maharashtra", "India"],  # same core name as S1-5 (order swapped)
        ["S1-7", "Lone Star Bakery", "1 Elm St, Waco, TX", "US"],
        ["S1-8", "Om Foods", "Gali 3, Delhi, Delhi", "India"],
        ["S1-9", "Summit Inc", "99 Pine Avenue, Houston, TX", "India"],  # same name as S1-1 but other country
    ]
    s2 = [["S2-11", "SUMMIT INC", "10 MAIN ST, AUSTIN, TX", "US"], ["S2-12", "Summit", "55 OAK RD, DALLAS, TX", "US"],
          ["S2-13", "Blue River", "7 LAKE DR, RENO, NV", "US"], ["S2-14", "श्री ट्रेडर्स", "PLOT 4, PUNE, Maharashtra", "India"],
          ["S2-15", "Orphan Co", "1 Nowhere, Austin, TX", "US"]]
    s3 = [["S3-21", "Summit Incorporated", "10 Main Street, Austin, Texas", "US"], ["S3-22", "Green Hills Ltd", "7 Lake Dr, Reno, Nevada", "US"],
          ["S3-23", "Om Foods", "Gali 3, Delhi, DL", "India"], ["S3-24", "Summit Inc", "99 Pine Ave, Houston", "India"]]
    gt = [["S1-1", "S2-11,S3-21"], ["S1-2", "S2-12"], ["S1-3", "S2-13"], ["S1-4", "S3-22"], ["S1-5", "S2-14"],
          ["S1-6", ""], ["S1-7", ""], ["S1-8", "S3-23"], ["S1-9", "S3-24"]]
    _tsv(d / "train_source1.tsv", h, s1); _tsv(d / "train_source2.tsv", h, s2); _tsv(d / "train_source3.tsv", h, s3)
    _tsv(d / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    return str(d)


def test_ground_truth_frame(train_dir):
    gt = va.ground_truth_frame(os.path.join(train_dir, "train_ground_truth.tsv"))
    row = gt.filter(pl.col("source1_entity_id") == "S1-1").to_dicts()[0]
    assert row["matched"] == ["S2-11", "S3-21"] and row["n_true"] == 2 and row["n_s2"] == 1 and row["n_s3"] == 1
    assert row["k_class"] == "2" and row["pattern"] == "both"
    assert gt.filter(pl.col("source1_entity_id") == "S1-6").to_dicts()[0]["pattern"] == "none"
    assert gt.filter(pl.col("source1_entity_id") == "S1-4").to_dicts()[0]["pattern"] == "S3 only"


def test_components_group_shared_name_and_address(train_dir):
    s1 = va.read_tsv(os.path.join(train_dir, "train_source1.tsv"))
    cl, info = va.components(s1, merge_address=True)
    c = dict(zip(s1["entity_id"].to_list(), cl.tolist()))
    assert c["S1-1"] == c["S1-2"]           # same core name (Summit)
    assert c["S1-5"] == c["S1-6"]           # same core name, tokens reordered, suffix variants
    assert c["S1-3"] == c["S1-4"]           # same address
    assert c["S1-1"] != c["S1-9"]           # same name but another country
    assert len({c["S1-1"], c["S1-3"], c["S1-5"], c["S1-7"], c["S1-8"], c["S1-9"]}) == 6
    assert info["max_cluster"] == 2 and info["merge_address"] is True
    cl2, info2 = va.components(s1, merge_address=False)
    c2 = dict(zip(s1["entity_id"].to_list(), cl2.tolist()))
    assert c2["S1-3"] != c2["S1-4"] and c2["S1-1"] == c2["S1-2"]


def test_giant_component_fallback(tmp_path):
    rows = [[f"S1-{i}", f"Name {i}", "1 Same Street, Town, TX", "US"] for i in range(200)]
    s1 = pl.DataFrame({k: [r[j] for r in rows] for j, k in enumerate(["entity_id", "business_name", "business_address", "country"])})
    cl, info = va.components(s1, merge_address=True, giant_limit=0.5, addr_max_group=None)
    assert info["merge_address"] is False and "fallback" in info and info["max_cluster"] == 1
    # with the default cap the popular address simply does not merge, so no fallback is needed
    cl, info = va.components(s1, merge_address=True, giant_limit=0.5)
    assert info["merge_address"] is True and "fallback" not in info and info["max_cluster"] == 1


def test_assign_folds_keeps_clusters_together_and_balances():
    import numpy as np
    cluster = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8])
    country = np.array(["US"] * 6 + ["India"] * 6)
    fold = va.assign_folds(cluster, country, n_folds=2)
    for c in np.unique(cluster):
        assert len(set(fold[cluster == c])) == 1
    assert sorted(np.bincount(fold[country == "US"]).tolist()) == [3, 3]
    assert sorted(np.bincount(fold[country == "India"]).tolist()) == [3, 3]


def test_build_and_use(train_dir, tmp_path):
    out = str(tmp_path / "val")
    summary = va.build_folds(train_dir, out, n_folds=2)
    assert summary["n_entities"] == 9 and set(summary["folds"]) == {0, 1}
    ent = va.load_entities(out)
    by = dict(zip(ent["entity_id"].to_list(), ent["fold"].to_list()))
    assert by["S1-1"] == by["S1-2"] and by["S1-5"] == by["S1-6"] and by["S1-3"] == by["S1-4"]
    # artifacts
    for f in (0, 1):
        m = ev.read_id_list_file(os.path.join(out, f"fold{f}_ground_truth.tsv"))
        ids = open(os.path.join(out, f"fold{f}_entity_ids.txt"), encoding="utf-8").read().split()
        assert set(m) == set(ids) == {e for e, ff in by.items() if ff == f}
        assert m == va.entity_truth(out, folds=[f])
    pairs = va.load_positive_pairs(out)
    assert pairs.height == 8 and pairs["matched_id"].n_unique() == 8
    assert set(pairs.filter(pl.col("source1_entity_id") == "S1-1")["matched_id"].to_list()) == {"S2-11", "S3-21"}
    # training side / leave-one-country-out selections
    assert set(va.entity_ids(out, exclude_folds=[0])) == {e for e, ff in by.items() if ff != 0}
    assert set(va.entity_ids(out, country="India")) == {"S1-5", "S1-6", "S1-8", "S1-9"}
    # pair labels: known positives 1, everything else 0, fold attached
    cands = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-1", "S1-1", "S1-6", "S1-2"],
                          "candidate_id": ["S2-11", "S2-12", "S3-21", "S2-14", "S2-12"]})
    lab = va.label_pairs(cands, out)
    assert lab["label"].to_list() == [1, 0, 1, 0, 1]
    assert lab["fold"].to_list() == [by["S1-1"], by["S1-1"], by["S1-1"], by["S1-6"], by["S1-2"]]
    # scoring a fold equals the evaluator on that subset, with slices
    fold0 = [e for e, ff in by.items() if ff == 0]
    pred = {e: (ent.filter(pl.col("entity_id") == e)["matched"][0].to_list()[:1]) for e in fold0}   # first true match only
    rep = va.score(pred, out, folds=[0])
    truth0 = va.entity_truth(out, folds=[0])
    assert math.isclose(rep["macro_f05"], ev.macro_f05(truth0, pred).macro_f05, abs_tol=1e-9)
    assert sum(v["n"] for v in rep["by_country"].values()) == len(fold0)
    assert set(rep["by_k_class"]) <= set(va.K_CLASSES) and set(rep["by_pattern"]) <= set(va.PATTERNS)
    # strict scoring of an incomplete prediction fails, lenient treats missing rows as empty
    with pytest.raises(ev.SubmissionFormatError):
        va.score({fold0[0]: []}, out, folds=[0], strict=True)
    assert va.score({}, out, folds=[0])["empty_pred_rate"] == 1.0


def test_cli_build_and_score(train_dir, tmp_path, capsys):
    out = str(tmp_path / "val")
    assert va.main(["build", "--train-dir", train_dir, "--out-dir", out, "--folds", "2"]) == 0
    pred_path = str(tmp_path / "pred.tsv")
    ev.write_id_list_file(pred_path, va.entity_truth(out, folds=[1]))
    assert va.main(["score", "--out-dir", out, "--pred", pred_path, "--fold", "1"]) == 0
    assert "macro F0.5 = 1.00000" in capsys.readouterr().out


def test_module_never_touches_test_files():
    src = open(HERE.parent / "src" / "validation.py", encoding="utf-8").read()
    assert "test_source" not in src and "dataset/test" not in src
