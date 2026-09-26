"""Re-run candidate generation with the fixes proposed by the miss analysis and measure old versus new (fold 0).

Fixes computed here as extra strategies (saved under artifacts/candidates/fold0/strategies/x_*):
  * x_name_addr_top30            the name + address retrieval extended from 20 to 30 per source (candidate limit)
  * x_addrless_name_ngram_top5   name 3-gram retrieval against the pool records that have NO address (missing address)
  * x_indic_name_ngram_top5      transliterated-name 3-gram retrieval against the Indic-script pool records (transliteration)
  * x_name_any_house_number, x_name_any_city, x_compact_name_state, x_house_street_city   cheap keys from s16
Existing strategies are also offered to the selection (address retrieval, name n-gram retrieval, acronym + city).
Selection: greedy by recall gained per candidate added, accepting a fix only while it buys at least MIN_GAIN_PER_CANDIDATE
(one recall point per 50 candidates per entity, the implicit threshold of the fold-0 frontier in candidate_generation.md).
Run from the project root (alone):  python -u eda/scripts/s16b_candidate_rerun.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import candidates as C  # noqa: E402

PRE, VAL, OUT_DIR = "artifacts/preprocessed/fold0", "artifacts/validation", "artifacts/candidates/fold0"
STR_DIR = os.path.join(OUT_DIR, "strategies")
OUT_JSON = "eda/results/s16b_candidate_rerun.json"
N_THREADS = 12
BUCKETS = 16
MIN_GAIN_PER_CANDIDATE = 0.0002
OLD_FINAL = ["9_name_addr_token_top20", "1_exact_name_any_cap200", "5_state_house_street", "5_state_name_sorted", "5_city_name_first_cap100",
             "8_sn_name_compact_w3", "9_sn_addr_sorted_w3"]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
RES = {"old_final": OLD_FINAL}

# --------------------------------------------------------------------------- #
# Tables (positions exactly as in s15)
# --------------------------------------------------------------------------- #
S1_COLS = ["entity_id", "country", "name_core", "name_core_sorted", "name_compact", "name_translit_core_sorted", "name_translit_core", "name_core_tokens",
           "addr_tokens", "house_number", "street", "city_norm", "state"]
folds = pl.read_parquet(f"{VAL}/entity_folds.parquet", columns=["entity_id", "fold"]).filter(pl.col("fold") == 0)
s1 = C.with_positions(pl.read_parquet(f"{PRE}/train_source1.parquet", columns=S1_COLS).join(folds.select("entity_id"), on="entity_id", how="inner"), "q_idx")
s1 = s1.with_columns(pl.concat_list([pl.col("name_translit_core").str.split(" "), pl.col("addr_tokens")]).alias("name_addr_tokens"))
S1_IDS = s1["entity_id"].to_list()
country_of = s1.select("q_idx", "country")
POOL_FILES = [f"{PRE}/train_source{k}.parquet" for k in (2, 3)]
pool_ids = C.with_positions(pl.concat([pl.read_parquet(f, columns=["entity_id"]) for f in POOL_FILES]), "p_idx")
POOL_IDS = pool_ids["entity_id"].to_list()
N_POOL = pool_ids.height
N_S2 = pl.scan_parquet(POOL_FILES[0]).select(pl.len()).collect().item()
cov = pl.read_parquet(os.path.join(OUT_DIR, "coverage.parquet"))
positives = cov.select("q_idx", "p_idx", "src")
causes = pl.read_parquet(os.path.join(OUT_DIR, "miss_causes.parquet"), columns=["q_idx", "p_idx", "cause"]) if os.path.exists(os.path.join(OUT_DIR, "miss_causes.parquet")) else None
log(f"queries {s1.height:,}, pool {N_POOL:,}, positives {positives.height:,}")


def pool_slice(cols, country, src):
    offset = 0
    for k, f in zip((2, 3), POOL_FILES):
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        if f"S{k}" == src:
            base = list(dict.fromkeys([c for c in cols if c != "name_addr_tokens"] + (["name_translit_core", "addr_tokens"] if "name_addr_tokens" in cols else [])))
            part = (pl.scan_parquet(f).with_row_index("p_idx", offset=offset).filter(pl.col("country") == country).select(["p_idx", "country", *base]).collect()
                      .with_columns(pl.col("p_idx").cast(pl.UInt32)))
            if "name_addr_tokens" in cols:
                part = part.with_columns(pl.concat_list([pl.col("name_translit_core").str.split(" "), pl.col("addr_tokens")]).alias("name_addr_tokens"))
            return part
        offset += n


def strategy_pairs(name):
    d = os.path.join(STR_DIR, name)
    if os.path.isdir(d):
        return pl.concat([pl.read_parquet(os.path.join(d, f), columns=C.PAIR_COLS) for f in sorted(os.listdir(d)) if f.startswith("b")])
    return pl.read_parquet(os.path.join(STR_DIR, f"{name}.parquet"), columns=C.PAIR_COLS)


def save(name, pairs):
    pairs.select(C.PAIR_COLS).write_parquet(os.path.join(STR_DIR, f"{name}.parquet"))


def measure(pairs, name=""):
    m = C.measure(pairs, positives, s1.height, N_POOL, country_of, src_boundary=N_S2)
    if name:
        log(f"  {name:34s} pairs={m['pairs']:>11,} recall={m['recall']:.4f} (US {m['recall_by_country'].get('US', 0):.4f} IN {m['recall_by_country'].get('India', 0):.4f}) "
            f"mean={m['mean']:.1f} p95={m['p95']:.0f} max={m['max']:,}")
    return m


# --------------------------------------------------------------------------- #
# New strategies
# --------------------------------------------------------------------------- #
NEW = {}
scored_path = os.path.join(STR_DIR, "x_name_addr_top50.scored.parquet")
if not os.path.exists(scored_path):
    res = C.retrieve_per_country_source(s1, None, lambda: C.TokenVectorizer(max_df=0.01), lambda d: d["name_addr_tokens"], k=50, threshold=0.2, n_threads=N_THREADS,
                                        log=log, slice_loader=lambda c, src: pool_slice(["name_addr_tokens"], c, src), spill_dir=os.path.join(STR_DIR, "_spill_top50"))
    res.write_parquet(scored_path)
    del res
    gc.collect()
scored = pl.read_parquet(scored_path)
# uniform extensions of the list, and adaptive ones: ranks 21-50 only for queries whose 20th score is still high (a flat score curve)
scored = scored.with_columns(pl.col("p_idx").cast(pl.Int64).alias("_p")).with_columns(
    pl.when(pl.col("rank") == 20).then(pl.col("score")).otherwise(None).max().over(["q_idx", (pl.col("_p") < N_S2).alias("_src")]).alias("s20")).drop("_p")
for k in (30, 50):
    name = f"x_name_addr_top{k}"
    save(name, scored.filter(pl.col("rank") <= k))
    NEW[name] = f"name + address token retrieval, top {k} per source (was 20)"
for tau in (0.5, 0.6, 0.7):
    name = f"x_name_addr_adaptive50_s20ge{int(tau * 100)}"
    save(name, scored.filter((pl.col("rank") <= 20) | ((pl.col("rank") <= 50) & (pl.col("s20") >= tau))))
    NEW[name] = f"name + address retrieval: ranks 21-50 added only when the 20th score is >= {tau}"
del scored
gc.collect()

if not os.path.exists(os.path.join(STR_DIR, "x_addrless_name_ngram_top5.parquet")):
    def addrless(c, src):
        p = pool_slice(["name_translit_core", "addr_missing"], c, src)
        return p.filter(pl.col("addr_missing"))
    res = C.retrieve_per_country_source(s1, None, lambda: C.CharNgramVectorizer(max_df=0.05), lambda d: d["name_translit_core"].to_list(), k=5, threshold=0.3,
                                        n_threads=N_THREADS, log=log, slice_loader=addrless)
    save("x_addrless_name_ngram_top5", res)
    del res
    gc.collect()
NEW["x_addrless_name_ngram_top5"] = "name 3-gram retrieval against the address-less pool records, top 5 per source"

if not os.path.exists(os.path.join(STR_DIR, "x_domain_name_ngram_top3.parquet")):
    def domain_style(c, src):
        p = pool_slice(["name_translit_core", "name_flags"], c, src)
        return p.filter(pl.col("name_flags").list.contains("domain_style") | pl.col("name_flags").list.contains("url_segment"))
    res = C.retrieve_per_country_source(s1, None, lambda: C.CharNgramVectorizer(max_df=0.05), lambda d: d["name_translit_core"].to_list(), k=3, threshold=0.3,
                                        n_threads=N_THREADS, log=log, slice_loader=domain_style)
    save("x_domain_name_ngram_top3", res)
    del res
    gc.collect()
NEW["x_domain_name_ngram_top3"] = "name 3-gram retrieval against the domain-style pool names (dentalcare.com, nbrothers.com), top 3 per source"

addr_scored = pl.read_parquet(os.path.join(STR_DIR, "9_addr_token_top20.scored.parquet"))
for k in (5, 10):
    save(f"x_addr_token_top{k}", addr_scored.filter(pl.col("rank") <= k))
    NEW[f"x_addr_token_top{k}"] = f"address token retrieval limited to the top {k} per source"
del addr_scored
gc.collect()

if os.environ.get("S16_MAXDF3", "1") == "1":
    if not os.path.exists(os.path.join(STR_DIR, "x_name_addr_top20_maxdf3.scored.parquet")):
        t0 = time.time()
        res = C.retrieve_per_country_source(s1, None, lambda: C.TokenVectorizer(max_df=0.03), lambda d: d["name_addr_tokens"], k=20, threshold=0.2, n_threads=N_THREADS,
                                            log=log, slice_loader=lambda c, src: pool_slice(["name_addr_tokens"], c, src), spill_dir=os.path.join(STR_DIR, "_spill_maxdf3"))
        res.write_parquet(os.path.join(STR_DIR, "x_name_addr_top20_maxdf3.scored.parquet"))
        save("x_name_addr_top20_maxdf3", res)
        RES["maxdf3_seconds"] = round(time.time() - t0, 1)
        del res
        gc.collect()
    NEW["x_name_addr_top20_maxdf3"] = "name + address retrieval, top 20, tokens pruned above 3% df instead of 1% (compute cost only)"

for k in ("x_name_any_house_number", "x_name_any_city", "x_compact_name_state", "x_house_street_city"):
    if os.path.exists(os.path.join(STR_DIR, f"{k}.parquet")):
        NEW[k] = f"cheap key from the miss analysis: {k[2:]}"
EXISTING = {"9_addr_token_top20": "address token retrieval, top 20 per source", "3_name_ngram_top20": "name 3-gram retrieval, top 20 per source",
            "8_acronym_city": "name acronym + city"}

# --------------------------------------------------------------------------- #
# Old set, marginal value of every fix, greedy selection
# --------------------------------------------------------------------------- #
old_pairs = C.union([strategy_pairs(n) for n in OLD_FINAL])
old_keys = np.unique(C._keys_of(old_pairs))
old_m = measure(old_pairs, "OLD final")
RES["old"] = old_m
miss_keys = C._keys_of(positives)[~np.isin(C._keys_of(positives), old_keys, kind="sort")]
cause_of = None
if causes is not None:
    ck = C.pack(causes["q_idx"].to_numpy(), causes["p_idx"].to_numpy())
    cause_of = dict(zip(ck.tolist(), causes["cause"].to_list()))

cands = {}
for name, desc in {**NEW, **EXISTING}.items():
    p = strategy_pairs(name)
    keys = np.unique(C._keys_of(p))
    alone = C.measure(p, positives, s1.height, N_POOL, country_of)
    new_keys = keys[~np.isin(keys, old_keys, kind="sort")]
    recovered = np.isin(miss_keys, keys, kind="sort")
    by_cause = {}
    if cause_of:
        for kk in miss_keys[recovered].tolist():
            by_cause[cause_of.get(kk, "?")] = by_cause.get(cause_of.get(kk, "?"), 0) + 1
    cands[name] = {"description": desc, "pairs": int(p.height), "recall_alone": alone["recall"], "mean_alone": alone["mean"], "max_alone": alone["max"],
                   "new_candidates_per_entity": round(len(new_keys) / s1.height, 2), "misses_recovered": int(recovered.sum()),
                   "recall_gain": round(recovered.sum() / positives.height, 5), "by_cause": dict(sorted(by_cause.items(), key=lambda x: -x[1])),
                   "gain_per_candidate": round(recovered.sum() / positives.height / max(len(new_keys) / s1.height, 1e-9), 6)}
    log(f"  fix {name:30s} +{len(new_keys) / s1.height:6.2f} cand/entity  recovers {int(recovered.sum()):6,} misses (+{recovered.sum() / positives.height * 100:.3f} pts)  "
        f"gain/cand={cands[name]['gain_per_candidate']:.6f}  {by_cause}")
    del p, keys, new_keys
    gc.collect()
RES["fixes"] = cands

chosen, cur_keys, steps = [], old_keys, []
remaining = list(cands)
while remaining:
    best = None
    for name in remaining:
        keys = np.unique(C._keys_of(strategy_pairs(name)))
        added = keys[~np.isin(keys, cur_keys, kind="sort")]
        gain = np.isin(miss_keys, added, kind="sort").sum() / positives.height
        cost = len(added) / s1.height
        gpc = gain / max(cost, 1e-9)
        if best is None or gpc > best[0]:
            best = (gpc, name, gain, cost, added)
    gpc, name, gain, cost, added = best
    if gpc < MIN_GAIN_PER_CANDIDATE or gain <= 0:
        break
    cur_keys = np.union1d(cur_keys, added)
    chosen.append(name)
    remaining.remove(name)
    steps.append({"added": name, "recall_gain": round(float(gain), 5), "candidates_added_per_entity": round(float(cost), 2), "gain_per_candidate": round(float(gpc), 6)})
    log(f"  chosen +{name}: +{gain * 100:.3f} pts for +{cost:.2f} cand/entity")
RES["selection"] = {"threshold_gain_per_candidate": MIN_GAIN_PER_CANDIDATE, "chosen": chosen, "steps": steps, "rejected": remaining}

# --------------------------------------------------------------------------- #
# New final set: measure, write, validate
# --------------------------------------------------------------------------- #
q, p = C.unpack(cur_keys)
new_pairs = C.pairs_frame(q, p, dedupe=False)
new_m = measure(new_pairs, "NEW final")
RES["new"] = new_m
RES["new_final"] = OLD_FINAL + chosen
new_keys_arr = C._keys_of(new_pairs)
still = miss_keys[~np.isin(miss_keys, new_keys_arr, kind="sort")]
RES["remaining_misses"] = {"n": int(len(still)), "by_cause": {}}
if cause_of:
    for kk in still.tolist():
        c = cause_of.get(kk, "?")
        RES["remaining_misses"]["by_cause"][c] = RES["remaining_misses"]["by_cause"].get(c, 0) + 1
path = os.path.join(OUT_DIR, "candidate_pairs.tsv")
old_path = os.path.join(OUT_DIR, "candidate_pairs_v1.tsv")
if os.path.exists(path) and not os.path.exists(old_path):
    os.replace(path, old_path)
new_pairs.write_parquet(os.path.join(OUT_DIR, "final_pairs.parquet"))
n_rows = C.write_candidate_pairs(new_pairs, S1_IDS, POOL_IDS, path)
RES["file"] = {"path": path, "rows": n_rows, "bytes": os.path.getsize(path), **C.validate_candidate_file(path, S1_IDS, set(POOL_IDS))}
RES["comparison"] = {"old_recall": old_m["recall"], "new_recall": new_m["recall"], "old_mean": old_m["mean"], "new_mean": new_m["mean"],
                     "old_p95": old_m["p95"], "new_p95": new_m["p95"], "old_pairs": old_m["pairs"], "new_pairs": new_m["pairs"],
                     "old_by_country": old_m["recall_by_country"], "new_by_country": new_m["recall_by_country"],
                     "old_all_found": old_m["entity_all_matches_found"], "new_all_found": new_m["entity_all_matches_found"]}
log("comparison", json.dumps(RES["comparison"]))
log("file", json.dumps(RES["file"]))
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT_JSON)
