"""Candidate generation: blocking functions, vectorisers, top-k retrieval, measurement and the output file."""
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import candidates as C  # noqa: E402
import evaluation as ev  # noqa: E402

S1_IDS = ["S1-1", "S1-2", "S1-3"]
POOL_IDS = ["S2-a", "S2-b", "S3-c", "S3-d", "S2-e"]


@pytest.fixture
def data():
    s1 = C.with_positions(pl.DataFrame({
        "entity_id": S1_IDS,
        "country": ["US", "US", "India"],
        "name_core": ["acme tools", "zeta foods", "acme tools"],
        "name_tokens": [["acme", "tools"], ["zeta", "foods"], ["acme", "tools"]],
        "addr_tokens": [["12", "main", "st"], ["40", "oak", "ave"], ["5", "mg", "road"]],
        "city_norm": ["springfield", "dayton", "bangalore"],
    }), "q_idx")
    pool = C.with_positions(pl.DataFrame({
        "entity_id": POOL_IDS,
        "src": ["S2", "S2", "S3", "S3", "S2"],
        "country": ["US", "US", "US", "India", "US"],
        "name_core": ["acme tools", "acme tools", "zeta food", "acme tools", "omega"],
        "name_tokens": [["acme", "tools"], ["acme", "tools"], ["zeta", "food"], ["acme", "tools"], ["omega"]],
        "addr_tokens": [["12", "main", "st"], ["99", "elm", "st"], ["40", "oak", "ave"], ["5", "mg", "road"], ["12", "main", "st"]],
        "city_norm": ["springfield", "dayton", "dayton", "bangalore", "springfield"],
    }), "p_idx")
    positives = pl.DataFrame({"q_idx": [0, 1, 2], "p_idx": [0, 2, 3], "src": ["S2", "S3", "S3"]}, schema_overrides=C.PAIR_SCHEMA)
    return s1, pool, positives


def pairs_set(df):
    return set(zip(df["q_idx"].to_list(), df["p_idx"].to_list()))


def test_key_block_respects_country_and_block_cap(data):
    s1, pool, _ = data
    assert pairs_set(C.key_block(s1, pool, ["name_core"])) == {(0, 0), (0, 1), (2, 3)}
    assert pairs_set(C.key_block(s1, pool, ["name_core"], max_block=1)) == {(2, 3)}
    assert pairs_set(C.key_block(s1, pool, ["name_core", "city_norm"])) == {(0, 0), (2, 3)}
    assert C.key_block(s1, pool, ["name_core"]).schema == pl.Schema(C.PAIR_SCHEMA)


def test_token_block_uses_pool_document_frequency(data):
    s1, pool, _ = data
    # US: 'acme'/'tools' df 2, 'zeta'/'food'/'omega' df 1; India: 'acme'/'tools' df 1
    assert pairs_set(C.token_block(s1, pool, "name_tokens", df_cap=1)) == {(1, 2), (2, 3)}
    assert pairs_set(C.token_block(s1, pool, "name_tokens", df_cap=2)) == {(0, 0), (0, 1), (1, 2), (2, 3)}
    addr = pairs_set(C.token_block(s1, pool, "addr_tokens", df_cap=2, rarest=1))
    assert (1, 2) in addr and (0, 1) not in addr


def test_sorted_neighbourhood_pairs_adjacent_keys(data):
    s1, pool, _ = data
    pairs = pairs_set(C.sorted_neighbourhood(s1, pool, "name_core", window=1))
    assert (1, 2) in pairs          # 'zeta foods' sits next to 'zeta food'
    assert (2, 3) in pairs
    assert all(q < 3 and p < 5 for q, p in pairs)


def test_char_ngram_vectorizer_matches_cosine_of_counts():
    vec = C.CharNgramVectorizer(min_df=1, max_df=1.0).fit(["acme tools", "acme tool", "zeta foods"])
    M = vec.transform(["acme tools", "acme tool", "zeta foods"])
    sims = (M @ M.T).toarray()
    assert np.allclose(np.diag(sims), 1.0, atol=1e-5)
    assert sims[0, 1] > 0.7 and sims[0, 2] < 0.2
    vec2 = C.CharNgramVectorizer(min_df=1, max_df=0.5).fit(["acme tools", "acme tool", "acme tools"])
    assert vec2.n_features < vec.n_features


def test_token_vectorizer_and_topk_retrieval(data):
    s1, pool, _ = data
    vec = C.TokenVectorizer(min_df=1, max_df=1.0).fit(pool["addr_tokens"])
    P = vec.transform(pool["addr_tokens"])
    Q = vec.transform(s1["addr_tokens"])
    res = C.topk_pairs(s1["q_idx"].to_numpy(), Q, pool["p_idx"].to_numpy(), P, k=2, threshold=0.1, n_threads=1)
    best = res.filter(pl.col("rank") == 1)
    assert dict(zip(best["q_idx"].to_list(), best["p_idx"].to_list()))[1] == 2
    assert res.filter(pl.col("q_idx") == 0)["p_idx"].to_list()[0] in (0, 4)
    assert res.group_by("q_idx").agg(pl.len().alias("n"))["n"].max() <= 2
    assert res.schema["q_idx"] == pl.UInt32 and res.schema["p_idx"] == pl.UInt32


def test_retrieve_per_country_source_keeps_sources_apart(data):
    s1, pool, _ = data
    res = C.retrieve_per_country_source(s1, pool, lambda: C.TokenVectorizer(min_df=1, max_df=1.0), lambda df: df["addr_tokens"], k=5, threshold=0.05, n_threads=1)
    assert (0, 3) not in pairs_set(res)          # different country
    assert (0, 0) in pairs_set(res) and (1, 2) in pairs_set(res)


def test_measure_union_coverage_and_output_file(data, tmp_path):
    s1, pool, positives = data
    exact = C.key_block(s1, pool, ["name_core"])
    addr = C.token_block(s1, pool, "addr_tokens", df_cap=2, rarest=1)
    u = C.union([exact, addr])
    country_of = s1.select("q_idx", "country")
    m = C.measure(u, positives, s1.height, pool.height, country_of, src_boundary=2)
    assert m["recall"] == 1.0 and m["true_pairs_missed"] == 0 and m["entity_all_matches_found"] == 1.0
    assert m["max"] >= 2 and m["entities_with_candidates"] == 3
    assert 0 < m["reduction_ratio"] < 1
    assert m["recall_by_country"] == {"India": 1.0, "US": 1.0} and m["pairs_by_source"]["S2"] + m["pairs_by_source"]["S3"] == u.height
    cov = C.coverage(positives, {"exact": exact, "addr": addr})
    row = cov.filter(pl.col("q_idx") == 1).row(0, named=True)
    assert row["exact"] is False and row["addr"] is True
    path = str(tmp_path / "candidate_pairs.tsv")
    n = C.write_candidate_pairs(u, S1_IDS + ["S1-9"], POOL_IDS, path)
    assert n == 4
    back = ev.read_id_list_file(path, expected_header=ev.CANDIDATE_HEADER)
    assert back["S1-9"] == [] and set(back["S1-1"]) == {"S2-a", "S2-b", "S2-e"} and back["S1-2"] == ["S3-c"]
    with open(path, "rb") as f:
        raw = f.read()
    assert raw.startswith(b"source1_entity_id\tcandidate_entity_ids\n") and b"\r" not in raw and b'"' not in raw
    v = C.validate_candidate_file(path, S1_IDS + ["S1-9"], set(POOL_IDS))
    assert v["valid"] and v["rows"] == 4 and v["empty_rows"] == 1 and v["ids"] == u.height
    assert not C.validate_candidate_file(path, S1_IDS, set(POOL_IDS))["valid"]            # row count mismatch
    assert not C.validate_candidate_file(path, S1_IDS + ["S1-9"], {"S2-a"})["valid"]      # id not in pool


def test_cap_per_entity_keeps_best_scores():
    pairs = pl.DataFrame({"q_idx": [0, 0, 0, 1], "p_idx": [1, 2, 3, 4], "score": [0.5, 0.9, 0.7, 0.1]}, schema_overrides=C.PAIR_SCHEMA)
    out = C.cap_per_entity(pairs, "score", 2)
    assert out.filter(pl.col("q_idx") == 0)["p_idx"].to_list() == [2, 3]
    assert out.height == 3
