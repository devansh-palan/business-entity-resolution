"""Candidate generation measured on validation fold 0 (441,365 Source 1 entities against the whole 10,320,219-record pool).

Every strategy is measured (pair recall, entity-level recall, per source, per country, mean / median / p95 / max
candidates per entity, reduction ratio) and its pairs are saved partitioned by a hash of the Source 1 id, so unions
and the per-true-pair coverage table are computed bucket by bucket without ever holding a whole union in memory.
The selected strategies are unioned, every true pair of the fold is checked against the union (coverage table and
anatomy of the misses), and the fold-0 candidate_pairs.tsv is written in the competition format and read back with
the validator-mirroring reader.

Inputs : artifacts/preprocessed/fold0/*.parquet, artifacts/validation/*
Outputs: eda/results/s15_candidates.json, artifacts/candidates/fold0/{strategies/<name>/b??.parquet, coverage.parquet,
         misses.parquet, candidate_pairs.tsv}
Run from the project root:  python -u eda/scripts/s15_candidate_generation.py     (S15_SMOKE=1 for a 3,000-entity dry run)
"""
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import candidates as C  # noqa: E402
import evaluation as ev  # noqa: E402

PRE, VAL = "artifacts/preprocessed/fold0", "artifacts/validation"
OUT_DIR = "artifacts/candidates/fold0"
OUT_JSON = "eda/results/s15_candidates.json"
N_THREADS = 12
BUCKETS = 16
SMOKE = os.environ.get("S15_SMOKE") == "1"
if SMOKE:
    OUT_DIR, OUT_JSON = os.path.join(OUT_DIR, "smoke"), "eda/results/s15_candidates_smoke.json"
STR_DIR = os.path.join(OUT_DIR, "strategies")
os.makedirs(STR_DIR, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
RES = {"strategies": {}, "timing_s": {}, "buckets": BUCKETS}

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
S1_COLS = ["entity_id", "country", "name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted", "name_translit_core",
           "name_core_tokens", "name_acronym", "addr_tokens", "addr_sorted", "addr_components", "house_number", "street", "city_norm",
           "state", "postal_code", "addr_digit_runs"]
DERIVED = {   # derived column -> (base columns, expression)
    "name_first": (["name_core_tokens"], pl.col("name_core_tokens").list.first()),
    "tr_tokens": (["name_translit_core"], pl.col("name_translit_core").str.split(" ")),
    "addr_last": (["addr_components"], pl.col("addr_components").list.last()),
    "numsig": (["addr_digit_runs"], pl.when(pl.col("addr_digit_runs").list.join("").str.len_chars() >= 3).then(pl.col("addr_digit_runs").list.sort().list.join("-")).otherwise(None)),
    "acronym": (["name_acronym"], pl.when(pl.col("name_acronym").str.len_chars() >= 2).then(pl.col("name_acronym")).otherwise(None)),
    "name_addr_tokens": (["name_translit_core", "addr_tokens"], pl.concat_list([pl.col("name_translit_core").str.split(" "), pl.col("addr_tokens")])),
}


def derive(df: pl.DataFrame, names=None) -> pl.DataFrame:
    names = list(DERIVED) if names is None else names
    return df.with_columns(*[DERIVED[n][1].alias(n) for n in names])


folds = pl.read_parquet(f"{VAL}/entity_folds.parquet", columns=["entity_id", "fold", "country"])
fold0 = folds.filter(pl.col("fold") == 0)
s1 = derive(pl.read_parquet(f"{PRE}/train_source1.parquet", columns=S1_COLS).join(fold0.select("entity_id"), on="entity_id", how="inner"))
positives = pl.read_parquet(f"{VAL}/positive_pairs.parquet").filter(pl.col("fold") == 0).select("source1_entity_id", "matched_id", "src")
POOL_FILES = [f"{PRE}/train_source{k}.parquet" for k in (2, 3)]
POOL_KEEP = None
if SMOKE:
    s1 = s1.sample(n=3000, seed=0)
    positives = positives.filter(pl.col("source1_entity_id").is_in(s1["entity_id"].to_list()))
    sample_pool = pl.concat([pl.read_parquet(f, columns=["entity_id"]) for f in POOL_FILES]).sample(fraction=0.02, seed=0)
    POOL_KEEP = set(sample_pool["entity_id"].to_list()) | set(positives["matched_id"].to_list())
s1 = C.with_positions(s1, "q_idx")
S1_IDS = s1["entity_id"].to_list()
country_of = s1.select("q_idx", "country")
pool_ids = pl.concat([pl.read_parquet(f, columns=["entity_id"]) for f in POOL_FILES])
if POOL_KEEP is not None:
    pool_ids = pool_ids.filter(pl.col("entity_id").is_in(list(POOL_KEEP)))
pool_ids = C.with_positions(pool_ids, "p_idx")
POOL_IDS = pool_ids["entity_id"].to_list()
N_POOL_BY_SRC = {s: int(n) for s, n in pool_ids.group_by(pl.col("entity_id").str.slice(0, 2).alias("s")).agg(pl.len().alias("n")).sort("s").iter_rows()}
N_POOL = sum(N_POOL_BY_SRC.values())
N_S2 = N_POOL_BY_SRC["S2"]
positives = (positives.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "q_idx"), on="source1_entity_id", how="inner")
                      .join(pool_ids.rename({"entity_id": "matched_id"}), on="matched_id", how="inner").select("q_idx", "p_idx", "src"))
BUCKET_EXPR = (pl.col("q_idx") % BUCKETS).cast(pl.Int32).alias("bucket")
positives = positives.with_columns(BUCKET_EXPR)
log(f"fold-0 queries {s1.height:,}, positives {positives.height:,}, pool {N_POOL:,} {N_POOL_BY_SRC}{' (SMOKE)' if SMOKE else ''}")


def iter_pool_parts(cols, n_parts=8):
    """Row slices of the pool straight from parquet (p_idx = global row position), for building token indices."""
    if POOL_KEEP is not None:                       # smoke mode: small enough to slice in memory
        for part in C._parts(load_pool(cols), n_parts):
            yield part
        return
    want = list(dict.fromkeys(["entity_id", "country", *cols]))
    base = list(dict.fromkeys([b for c in want for b in (DERIVED[c][0] if c in DERIVED else [c])]))
    offset = 0
    for f in POOL_FILES:
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        size = -(-n // (n_parts // 2))
        for i in range(0, n, size):
            part = pl.scan_parquet(f).slice(i, size).select(base).collect()
            part = part.with_row_index("p_idx", offset=offset + i).with_columns(pl.col("p_idx").cast(pl.UInt32))
            part = derive(part, [c for c in want if c in DERIVED])
            yield part.select(["p_idx", *want])
            del part
        offset += n


def load_pool_slice(cols, country, src):
    """One (country, source) slice of the pool straight from parquet, with global p_idx positions."""
    if POOL_KEEP is not None:
        return load_pool(cols).filter((pl.col("country") == country) & (pl.col("src") == src))
    want = list(dict.fromkeys(["entity_id", "country", *cols]))
    base = list(dict.fromkeys([b for c in want for b in (DERIVED[c][0] if c in DERIVED else [c])]))
    offset = 0
    for k, f in zip((2, 3), POOL_FILES):
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        if f"S{k}" == src:
            part = (pl.scan_parquet(f).with_row_index("p_idx", offset=offset).filter(pl.col("country") == country).select(["p_idx", *base]).collect()
                      .with_columns(pl.col("p_idx").cast(pl.UInt32)))
            return derive(part, [c for c in want if c in DERIVED]).select(["p_idx", *want])
        offset += n
    return pl.DataFrame(schema={"p_idx": pl.UInt32})


def load_pool(cols):
    """The pool with the requested plain or derived columns (only the base columns each derived one needs are read)."""
    want = list(dict.fromkeys(["entity_id", "src", "country", *cols]))
    base = list(dict.fromkeys([b for c in want for b in (DERIVED[c][0] if c in DERIVED else [c])]))
    df = pl.concat([pl.read_parquet(f, columns=base) for f in POOL_FILES])
    if POOL_KEEP is not None:
        df = df.filter(pl.col("entity_id").is_in(list(POOL_KEEP)))
    df = C.with_positions(df, "p_idx")            # same files, same order, same filter: positions match pool_ids
    df = derive(df, [c for c in want if c in DERIVED])
    return df.select(["p_idx", *want])


# --------------------------------------------------------------------------- #
# Saving, measuring, unions and coverage (all bucketed)
# --------------------------------------------------------------------------- #
def save_strategy(name: str, pairs: pl.DataFrame) -> None:
    d = os.path.join(STR_DIR, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    bucket = pairs.select(BUCKET_EXPR)["bucket"]
    for b in range(BUCKETS):                       # one bucket at a time: peak memory is a sixteenth of the table
        pairs.filter(bucket == b).select(C.PAIR_COLS).write_parquet(os.path.join(d, f"b{b:02d}.parquet"))


def strategy_bucket(name: str, b: int) -> pl.DataFrame:
    p = os.path.join(STR_DIR, name, f"b{b:02d}.parquet")
    return pl.read_parquet(p, columns=C.PAIR_COLS) if os.path.exists(p) else C.empty_pairs()


def bucket_union(names, b: int) -> pl.DataFrame:
    return C.union([strategy_bucket(n, b) for n in names])


RESUME = os.environ.get("S15_RESUME", "1") == "1"


def run(name: str, fn, note: str = "", extra=None):
    """Compute a strategy (or reload it from a previous run), measure it, save it partitioned, free it."""
    t0 = time.time()
    d = os.path.join(STR_DIR, name)
    scored_path = os.path.join(STR_DIR, f"{name}.scored.parquet")
    loaded = False
    if RESUME and os.path.isdir(d) and (extra is None or os.path.exists(scored_path)):
        pairs = pl.read_parquet(scored_path) if extra is not None else pl.concat([strategy_bucket(name, b) for b in range(BUCKETS)])
        loaded = True
    else:
        try:
            pairs = fn()
        except MemoryError as e:                       # one oversized block must not end the whole run
            RES["strategies"][name] = {"error": f"MemoryError: {e}", "note": note}
            log(f"{name:34s} FAILED with MemoryError")
            gc.collect()
            return
        if extra is not None:
            pairs.write_parquet(scored_path)
    RES["timing_s"][name] = round(time.time() - t0, 1) if not loaded else None
    m = C.measure(pairs, positives, len(S1_IDS), N_POOL, country_of, src_boundary=N_S2, assume_unique=("rank" not in pairs.columns))
    m["note"] = note
    if extra:
        m.update(extra(pairs))
    RES["strategies"][name] = m
    if not loaded:
        save_strategy(name, pairs)
    del pairs
    gc.collect()
    log(f"{name:34s} pairs={m['pairs']:>11,} recall={m['recall']:.4f} (S2 {m['recall_by_source'].get('S2', 0):.4f} S3 {m['recall_by_source'].get('S3', 0):.4f} "
        f"| US {m['recall_by_country'].get('US', 0):.4f} IN {m['recall_by_country'].get('India', 0):.4f}) mean={m['mean']:.1f} p50={m['median']:.0f} "
        f"p95={m['p95']:.0f} max={m['max']:,} RR={m['reduction_ratio']:.6f} [{'loaded' if loaded else str(RES['timing_s'][name]) + 's'}]")


def coverage_table(names) -> pl.DataFrame:
    """One row per true pair with a boolean per strategy, built bucket by bucket."""
    out = []
    for b in range(BUCKETS):
        pos_b = positives.filter(pl.col("bucket") == b).select("q_idx", "p_idx", "src")
        for n in names:
            pos_b = pos_b.join(strategy_bucket(n, b).unique().with_columns(pl.lit(True).alias(n)), on=C.PAIR_COLS, how="left") \
                         .with_columns(pl.col(n).fill_null(False))
        out.append(pos_b)
    return pl.concat(out)


def measure_union(names, cov: pl.DataFrame) -> dict:
    """Recall from the coverage table (exact, no union materialised) and volume from per-bucket unions."""
    found = cov.select(pl.any_horizontal([pl.col(n) for n in names]).alias("hit"), "q_idx", "src").join(country_of, on="q_idx")
    vol = np.zeros(len(S1_IDS), dtype=np.int64)
    n_pairs, pos_v = 0, 0
    for b in range(BUCKETS):
        u = bucket_union(names, b)
        n_pairs += u.height
        counts = u.group_by("q_idx").agg(pl.len().alias("n"))
        vol[pos_v: pos_v + counts.height] = counts["n"].to_numpy()
        pos_v += counts.height
        del u
    per_entity = found.group_by("q_idx").agg(pl.col("hit").all().alias("all_found"))
    by = lambda col: {str(r[col]): round(float(r["r"]), 5) for r in found.group_by(col).agg(pl.col("hit").mean().alias("r")).sort(col).to_dicts()}
    return {"pairs": int(n_pairs), "recall": round(float(found["hit"].mean()), 5), "true_pairs_found": int(found["hit"].sum()),
            "true_pairs_missed": int((~found["hit"]).sum()), "entity_all_matches_found": round(float(per_entity["all_found"].mean()), 5),
            "recall_by_source": by("src"), "recall_by_country": by("country"),
            "mean": round(float(vol.mean()), 3), "median": float(np.median(vol)), "p95": float(np.percentile(vol, 95)),
            "p99": float(np.percentile(vol, 99)), "max": int(vol.max()), "entities_with_candidates": int((vol > 0).sum()),
            "reduction_ratio": round(1.0 - n_pairs / (len(S1_IDS) * N_POOL), 6), "members": list(names)}


def rank_recalls(pairs):
    out = {}
    for k in (1, 3, 5, 10, 20):
        out[f"recall@{k}"] = C.measure(pairs.filter(pl.col("rank") <= k), positives, len(S1_IDS), N_POOL, country_of)["recall"]
    return {"by_rank": out}


# --------------------------------------------------------------------------- #
# 1. Exact normalised name (four representations, one pool column at a time)
# --------------------------------------------------------------------------- #
NAME_REPS = ("name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted")
for rep in NAME_REPS:
    pool = load_pool([rep])
    run(f"1_exact_{rep}", lambda rep=rep, pool=pool: C.key_block(s1, pool, [rep]))
    if rep == "name_core":   # 6. country-aware: the same join without the country key shows what the country string buys
        run("6_exact_name_core_without_country", lambda pool=pool: C.key_block(s1.with_columns(pl.lit("*").alias("country")), pool.with_columns(pl.lit("*").alias("country")), ["name_core"]),
            note="cross-country blocks add volume and no true pairs (all 7,638,365 true pairs share the country string)")
    del pool
    gc.collect()

# capped exact-name blocks: the union of the four representations, keys carried by more than 50 pool records dropped
pool = load_pool(list(NAME_REPS))
run("1_exact_name_any_cap50", lambda: C.multi_key_block(s1, pool, [[r] for r in NAME_REPS], max_block=50),
    note="union of the four exact name representations, blocks over 50 pool records dropped")
run("1_exact_name_any_cap200", lambda: C.multi_key_block(s1, pool, [[r] for r in NAME_REPS], max_block=200),
    note="union of the four exact name representations, blocks over 200 pool records dropped")
del pool
gc.collect()

# --------------------------------------------------------------------------- #
# 2. Name token blocking (transliterated core tokens; pool document frequency within the country)
# --------------------------------------------------------------------------- #
index = C.TokenIndex(None, "tr_tokens", min_len=2, parts=iter_pool_parts(["tr_tokens"]))
gc.collect()
for cap in (200, 500, 1000):
    run(f"2_name_token_df{cap}_rarest2", lambda cap=cap: index.block(s1, df_cap=cap, rarest=2), note=f"two rarest name tokens of the query with pool df <= {cap}")
del index
gc.collect()

# --------------------------------------------------------------------------- #
# 3. Character n-gram retrieval on names (TF-IDF 3-grams, n-grams in > 1% of names dropped, top 20 per source)
# --------------------------------------------------------------------------- #
run("3_name_ngram_top20", lambda: C.retrieve_per_country_source(s1, None, lambda: C.CharNgramVectorizer(max_df=0.01), lambda d: d["name_translit_core"].to_list(),
                                                                 k=20, threshold=0.2, n_threads=N_THREADS, log=log,
                                                                 slice_loader=lambda c, src: load_pool_slice(["name_translit_core"], c, src)),
    note="cosine top-20 per (country, source); 3-grams with df > 1% dropped", extra=rank_recalls)
gc.collect()

# --------------------------------------------------------------------------- #
# 4. Address token blocking
# --------------------------------------------------------------------------- #
index = C.TokenIndex(None, "addr_tokens", min_len=2, parts=iter_pool_parts(["addr_tokens"]))
gc.collect()
for cap in (100, 200, 500):
    run(f"4_addr_token_df{cap}_rarest3", lambda cap=cap: index.block(s1, df_cap=cap, rarest=3), note=f"three rarest address tokens of the query with pool df <= {cap}")
del index
gc.collect()

# --------------------------------------------------------------------------- #
# 5. City / state / postal blocking (with a name or number component; alone these blocks are whole cities)
# --------------------------------------------------------------------------- #
pool = load_pool(["name_core_sorted", "name_first", "house_number", "street", "city_norm", "state", "postal_code", "addr_last"])
run("5_state_name_sorted", lambda: C.key_block(s1, pool, ["state", "name_core_sorted"]))
run("5_city_name_first", lambda: C.key_block(s1, pool, ["city_norm", "name_first"], max_block=2000), note="blocks over 2,000 pool records dropped")
run("5_city_name_first_cap100", lambda: C.key_block(s1, pool, ["city_norm", "name_first"], max_block=100), note="blocks over 100 pool records dropped")
run("5_city_house_number", lambda: C.key_block(s1, pool, ["city_norm", "house_number"], max_block=2000))
run("5_state_house_street", lambda: C.key_block(s1, pool, ["state", "house_number", "street"]))
run("5_postal_name_first", lambda: C.key_block(s1, pool, ["postal_code", "name_first"]), note="postal codes exist in 0.007% of records")
run("5u_rawlast_name_sorted", lambda: C.key_block(s1, pool, ["addr_last", "name_core_sorted"]),
    note="unseen-country simulation: the raw last address component replaces the learned state")
del pool
gc.collect()

# --------------------------------------------------------------------------- #
# 8. Approximate name retrieval: sorted neighbourhoods and acronym blocks (the n-gram retrieval of 3 belongs here too)
# --------------------------------------------------------------------------- #
pool = load_pool(["name_core_sorted", "name_compact", "acronym", "city_norm"])
run("8_sn_name_core_sorted_w3", lambda: C.sorted_neighbourhood(s1, pool, "name_core_sorted", window=3))
run("8_sn_name_compact_w3", lambda: C.sorted_neighbourhood(s1, pool, "name_compact", window=3))
run("8_acronym_city", lambda: C.key_block(s1, pool, ["acronym", "city_norm"], max_block=500), note="acronym of the core tokens (>= 2 tokens) + city")
del pool
gc.collect()

# --------------------------------------------------------------------------- #
# 9. Approximate address retrieval: number signature, sorted neighbourhood, token TF-IDF top-20, name + address retrieval
# --------------------------------------------------------------------------- #
pool = load_pool(["addr_sorted", "numsig"])
run("9_numsig", lambda: C.key_block(s1, pool, ["numsig"], max_block=500), note="sorted set of all digit runs of the address (at least 3 digits in total); blocks over 500 dropped")
run("9_sn_addr_sorted_w3", lambda: C.sorted_neighbourhood(s1, pool, "addr_sorted", window=3))
del pool
gc.collect()
run("9_addr_token_top20", lambda: C.retrieve_per_country_source(s1, None, lambda: C.TokenVectorizer(max_df=0.01), lambda d: d["addr_tokens"],
                                                                 k=20, threshold=0.2, n_threads=N_THREADS, log=log,
                                                                 slice_loader=lambda c, src: load_pool_slice(["addr_tokens"], c, src)),
    note="cosine top-20 per (country, source) on address tokens; tokens in > 1% of addresses dropped", extra=rank_recalls)
gc.collect()
run("9_name_addr_token_top20", lambda: C.retrieve_per_country_source(s1, None, lambda: C.TokenVectorizer(max_df=0.01), lambda d: d["name_addr_tokens"],
                                                                      k=20, threshold=0.2, n_threads=N_THREADS, log=log,
                                                                      slice_loader=lambda c, src: load_pool_slice(["name_addr_tokens"], c, src)),
    note="cosine top-20 per (country, source) on name + address tokens", extra=rank_recalls)
gc.collect()

# --------------------------------------------------------------------------- #
# Coverage of every true pair, unions, misses
# --------------------------------------------------------------------------- #
ALL = [n for n in RES["strategies"] if "error" not in RES["strategies"][n] and not n.startswith("6_")]
cov = coverage_table(ALL)
cov.write_parquet(os.path.join(OUT_DIR, "coverage.parquet"))
log("coverage table", cov.shape)

BOUNDED = [f"1_exact_{r}" for r in NAME_REPS] + ["5_state_name_sorted", "5_city_name_first", "5_city_house_number", "5_state_house_street", "9_numsig",
                                                  "8_acronym_city", "8_sn_name_core_sorted_w3", "8_sn_name_compact_w3", "9_sn_addr_sorted_w3",
                                                  "3_name_ngram_top20", "9_addr_token_top20", "9_name_addr_token_top20"]
GROUPS = {
    "U_exact_name_any": [f"1_exact_{r}" for r in NAME_REPS],
    "U_exact_keys": [f"1_exact_{r}" for r in NAME_REPS] + ["5_state_name_sorted", "5_city_name_first", "5_city_house_number", "5_state_house_street", "9_numsig", "8_acronym_city"],
    "U_retrieval_only": ["3_name_ngram_top20", "9_addr_token_top20", "9_name_addr_token_top20"],
    "U_bounded": BOUNDED,
    "U_bounded_plus_tokens_small": BOUNDED + ["2_name_token_df200_rarest2", "4_addr_token_df100_rarest3"],
    "U_bounded_plus_tokens_mid": BOUNDED + ["2_name_token_df500_rarest2", "4_addr_token_df200_rarest3"],
    "U_all_selected": BOUNDED + ["2_name_token_df1000_rarest2", "4_addr_token_df500_rarest3"],
}
RES["unions"] = {}
for gname, members in GROUPS.items():
    members = [m for m in members if m in ALL]
    t0 = time.time()
    RES["unions"][gname] = measure_union(members, cov)
    u = RES["unions"][gname]
    log(f"{gname:28s} pairs={u['pairs']:>12,} recall={u['recall']:.4f} (S2 {u['recall_by_source'].get('S2', 0):.4f} S3 {u['recall_by_source'].get('S3', 0):.4f} "
        f"| US {u['recall_by_country'].get('US', 0):.4f} IN {u['recall_by_country'].get('India', 0):.4f}) all-found={u['entity_all_matches_found']:.4f} "
        f"mean={u['mean']:.1f} p50={u['median']:.0f} p95={u['p95']:.0f} p99={u['p99']:.0f} max={u['max']:,} [{time.time() - t0:.0f}s]")

FINAL_ENV = os.environ.get("S15_FINAL")
FINAL = [m for m in (FINAL_ENV.split(",") if FINAL_ENV else GROUPS["U_bounded"]) if m in ALL]
RES["unions"]["U_final"] = measure_union(FINAL, cov)
u = RES["unions"]["U_final"]
log(f"{'U_final':28s} pairs={u['pairs']:>12,} recall={u['recall']:.4f} (S2 {u['recall_by_source'].get('S2', 0):.4f} S3 {u['recall_by_source'].get('S3', 0):.4f} "
    f"| US {u['recall_by_country'].get('US', 0):.4f} IN {u['recall_by_country'].get('India', 0):.4f}) all-found={u['entity_all_matches_found']:.4f} "
    f"mean={u['mean']:.1f} p50={u['median']:.0f} p95={u['p95']:.0f} p99={u['p99']:.0f} max={u['max']:,}")
in_final = cov.select(pl.any_horizontal([pl.col(n) for n in FINAL]).alias("in_final"))["in_final"]
cov = cov.with_columns(in_final.alias("in_final"))
cov.write_parquet(os.path.join(OUT_DIR, "coverage.parquet"))
k_found = cov.select(pl.sum_horizontal([pl.col(n).cast(pl.Int32) for n in FINAL]).alias("k"))["k"]
RES["coverage"] = {
    "final_members": FINAL, "true_pairs": cov.height, "found": int(in_final.sum()), "missed": int((~in_final).sum()),
    "found_by_exactly_one_strategy": int((k_found == 1).sum()),
    "unique_contribution": {n: int(((k_found == 1) & cov[n]).sum()) for n in FINAL},
    "recall_without": {n: round(float(cov.select(pl.any_horizontal([pl.col(m) for m in FINAL if m != n]).alias("x"))["x"].mean()), 5) for n in FINAL},
}
log("coverage", json.dumps({k: v for k, v in RES["coverage"].items() if k in ("true_pairs", "found", "missed", "found_by_exactly_one_strategy")}))

# ---- anatomy of the misses
cols = ["entity_id", "business_name", "business_address", "name_translit_core", "addr_tokens", "addr_digit_runs", "addr_missing", "name_script"]
miss = cov.filter(~pl.col("in_final")).select("q_idx", "p_idx", "src")
s1_txt = pl.read_parquet(f"{PRE}/train_source1.parquet", columns=cols).join(s1.select("entity_id", "q_idx"), on="entity_id", how="inner").rename({c: f"{c}_1" for c in cols})
pool_txt = pl.concat([pl.read_parquet(f, columns=cols) for f in POOL_FILES])
if POOL_KEEP is not None:
    pool_txt = pool_txt.filter(pl.col("entity_id").is_in(list(POOL_KEEP)))
pool_txt = C.with_positions(pool_txt, "p_idx").rename({c: f"{c}_2" for c in cols})
miss = miss.join(s1_txt, on="q_idx").join(pool_txt, on="p_idx").join(country_of, on="q_idx")
miss = miss.rename({"entity_id_1": "source1_entity_id", "entity_id_2": "matched_id"})
del s1_txt, pool_txt
gc.collect()
n1, n2 = miss["name_translit_core_1"].fill_null("").to_list(), miss["name_translit_core_2"].fill_null("").to_list()
miss = miss.with_columns(
    pl.Series("name_ratio", [fuzz.ratio(a, b) for a, b in zip(n1, n2)], dtype=pl.Float32),
    pl.Series("name_token_jac", [len(set(a.split()) & set(b.split())) / max(len(set(a.split()) | set(b.split())), 1) for a, b in zip(n1, n2)], dtype=pl.Float32),
    pl.col("addr_tokens_1").list.set_intersection(pl.col("addr_tokens_2")).list.len().alias("addr_shared"),
    pl.col("addr_tokens_1").list.set_union(pl.col("addr_tokens_2")).list.len().alias("addr_union"),
    pl.col("addr_digit_runs_1").list.set_intersection(pl.col("addr_digit_runs_2")).list.len().alias("num_shared"),
)
miss = miss.with_columns((pl.col("addr_shared") / pl.col("addr_union").clip(1)).alias("addr_jac"))
miss = miss.with_columns(
    (pl.when(pl.col("addr_missing_1") | pl.col("addr_missing_2")).then(pl.lit("address missing on a side"))
       .when((pl.col("name_ratio") < 50) & (pl.col("addr_jac") < 0.2)).then(pl.lit("different name and different address"))
       .when(pl.col("name_ratio") < 50).then(pl.lit("different name, similar address"))
       .when(pl.col("addr_jac") < 0.2).then(pl.lit("similar name, different address"))
       .otherwise(pl.lit("similar name and similar address"))).alias("category"))
RES["misses"] = {
    "n": miss.height,
    "by_category": {r["category"]: int(r["n"]) for r in miss.group_by("category").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts()},
    "by_country": {r["country"]: int(r["n"]) for r in miss.group_by("country").agg(pl.len().alias("n")).to_dicts()},
    "by_source": {r["src"]: int(r["n"]) for r in miss.group_by("src").agg(pl.len().alias("n")).to_dicts()},
    "name_ratio_quartiles": [float(x) for x in np.percentile(miss["name_ratio"].to_numpy(), [25, 50, 75])] if miss.height else [],
    "addr_jaccard_quartiles": [float(x) for x in np.percentile(miss["addr_jac"].to_numpy(), [25, 50, 75])] if miss.height else [],
    "pool_name_script": {r["name_script_2"]: int(r["n"]) for r in miss.group_by("name_script_2").agg(pl.len().alias("n")).sort("n", descending=True).head(6).to_dicts()},
    "examples": miss.sample(n=min(30, miss.height), seed=5).select("source1_entity_id", "business_name_1", "business_address_1", "matched_id", "business_name_2",
                                                                    "business_address_2", "name_ratio", "addr_jac", "category").to_dicts() if miss.height else [],
}
miss.select("source1_entity_id", "matched_id", "q_idx", "p_idx", "src", "country", "name_ratio", "name_token_jac", "addr_jac", "num_shared", "category").write_parquet(os.path.join(OUT_DIR, "misses.parquet"))
log("misses", json.dumps({k: v for k, v in RES["misses"].items() if k != "examples"}))
del miss
gc.collect()

# --------------------------------------------------------------------------- #
# The competition file for the fold: one row per Source 1 entity, ids sorted, comma-separated, empty allowed
# --------------------------------------------------------------------------- #
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
final_pairs = pl.concat([bucket_union(FINAL, b) for b in range(BUCKETS)])
final_pairs.write_parquet(os.path.join(OUT_DIR, "final_pairs.parquet"))
path = os.path.join(OUT_DIR, "candidate_pairs.tsv")
n_rows = C.write_candidate_pairs(final_pairs, S1_IDS, POOL_IDS, path)
del final_pairs
gc.collect()
check = C.validate_candidate_file(path, S1_IDS, set(POOL_IDS))
RES["final"] = {**RES["unions"]["U_final"], "file": {"path": path, "rows": n_rows, "bytes": os.path.getsize(path), **check}}
log("final", json.dumps(RES["final"]["file"]))
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT_JSON)
