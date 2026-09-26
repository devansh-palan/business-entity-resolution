"""The candidate CLI end to end on a tiny preprocessed split: retrieval slices in child processes, union, both output files."""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import candidates as C  # noqa: E402
import generate_candidates as G  # noqa: E402
from test_features import split  # noqa: E402,F401


def test_generate_writes_candidate_files(split):
    tmp, _ = split
    out = str(tmp / "out" / "candidate_pairs.tsv")
    stats = G.generate(str(tmp), "train", out, n_threads=1, log=lambda *a: None)
    assert stats["file"]["valid"] and stats["file"]["rows"] == 3 and os.path.exists(stats["pairs_parquet"])
    lines = open(out, encoding="utf-8", newline="").read().split("\n")
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids" and len(lines) == 5 and lines[-1] == ""
    pairs = pl.read_parquet(stats["pairs_parquet"])
    assert set(pairs.columns) == {"q_idx", "p_idx"} and pairs.height == stats["final"]["pairs"] == stats["file"]["ids"]
    # Acme Tools (q 0) must reach ACME TOOLS (S2-a, p 0) through the exact-name block; nothing crosses countries
    assert (0, 0) in set(zip(pairs["q_idx"].to_list(), pairs["p_idx"].to_list()))
    ids_row = dict(l.split("\t") for l in lines[1:4])
    assert all(i.startswith("S3-") or i.startswith("S2-") for i in ids_row["S1-2"].split(",") if i)
    assert "S3-c" in ids_row["S1-2"] and "S2-a" not in ids_row["S1-2"]         # the Indian entity never gets the US record
    assert not os.path.exists(str(tmp / "out" / "_retrieval_spill"))
