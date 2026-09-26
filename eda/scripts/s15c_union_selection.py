"""Choose the final candidate set from the saved strategies of s15 (fold 0).

Recall of any union is exact and instant from the coverage table (one row per true pair, one flag per strategy);
volumes of shortlisted unions are computed exactly bucket by bucket.  Also measures recall@k of the three retrievals
from their scored outputs, and a greedy forward selection by marginal recall per added candidate.
Run from the project root:  python -u eda/scripts/s15c_union_selection.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import candidates as C  # noqa: E402

OUT_DIR = "artifacts/candidates/fold0"
STR_DIR = os.path.join(OUT_DIR, "strategies")
OUT_JSON = "eda/results/s15c_union_selection.json"
VAL = "artifacts/validation"
BUCKETS = 16
N_S1 = 441365
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
RES = {}

cov = pl.read_parquet(os.path.join(OUT_DIR, "coverage.parquet"))
STRATS = [c for c in cov.columns if c not in ("q_idx", "p_idx", "src", "in_final", "in_union")]
folds = pl.read_parquet(f"{VAL}/entity_folds.parquet", columns=["entity_id", "fold", "country"]).filter(pl.col("fold") == 0)
s1_ids = pl.read_parquet("artifacts/preprocessed/fold0/train_source1.parquet", columns=["entity_id", "country"]).join(folds.select("entity_id"), on="entity_id")
country_of = C.with_positions(s1_ids, "q_idx").select("q_idx", "country")
cov = cov.join(country_of, on="q_idx", how="left")
log(f"coverage {cov.shape}, strategies: {STRATS}")


def strategy_bucket(name, b):
    p = os.path.join(STR_DIR, name, f"b{b:02d}.parquet")
    return pl.read_parquet(p, columns=C.PAIR_COLS) if os.path.exists(p) else C.empty_pairs()


def recall_of(names):
    hit = cov.select(pl.any_horizontal([pl.col(n) for n in names]).alias("hit"), "src", "country", "q_idx")
    per_entity = hit.group_by("q_idx").agg(pl.col("hit").all().alias("all"))
    return {"recall": round(float(hit["hit"].mean()), 5), "entity_all_found": round(float(per_entity["all"].mean()), 5),
            "by_country": {r["country"]: round(float(r["r"]), 5) for r in hit.group_by("country").agg(pl.col("hit").mean().alias("r")).sort("country").to_dicts()},
            "by_source": {r["src"]: round(float(r["r"]), 5) for r in hit.group_by("src").agg(pl.col("hit").mean().alias("r")).sort("src").to_dicts()},
            "missed": int((~hit["hit"]).sum())}


def volume_of(names):
    vol = np.zeros(N_S1, dtype=np.int64)
    n_pairs = 0
    for b in range(BUCKETS):
        u = C.union([strategy_bucket(n, b) for n in names])
        n_pairs += u.height
        if u.height:
            vol += np.bincount(u["q_idx"].to_numpy(), minlength=N_S1)
    return {"pairs": int(n_pairs), "mean": round(float(vol.mean()), 2), "median": float(np.median(vol)), "p95": float(np.percentile(vol, 95)),
            "p99": float(np.percentile(vol, 99)), "max": int(vol.max()), "entities_without_candidates": int((vol == 0).sum())}


# ---- per-strategy volume (from the saved buckets) and recall
single = {}
for n in STRATS:
    single[n] = {**recall_of([n]), **volume_of([n])}
RES["single"] = single
log("single strategies measured")

# ---- recall@k of the retrievals from their scored outputs
RES["retrieval_by_rank"] = {}
for n in ("3_name_ngram_top20", "9_addr_token_top20", "9_name_addr_token_top20"):
    p = os.path.join(STR_DIR, f"{n}.scored.parquet")
    if not os.path.exists(p):
        continue
    sc = pl.read_parquet(p, columns=["q_idx", "p_idx", "rank"])
    pos_keys = C.pack(cov["q_idx"].to_numpy(), cov["p_idx"].to_numpy())
    out = {}
    for k in (1, 3, 5, 10, 15, 20):
        sub = sc.filter(pl.col("rank") <= k)
        out[f"recall@{k}"] = round(float(np.isin(pos_keys, C.pack(sub["q_idx"].to_numpy(), sub["p_idx"].to_numpy()), kind="sort").mean()), 5)
    RES["retrieval_by_rank"][n] = out
    log(n, out)
    del sc

# ---- greedy forward selection: marginal recall per marginal (single-strategy) candidate
remaining = [n for n in STRATS if not n.startswith("5_postal") and not n.startswith("5u_")]
chosen, steps = [], []
cur_recall = 0.0
while remaining:
    best = None
    for n in remaining:
        r = recall_of(chosen + [n])["recall"]
        gain = r - cur_recall
        cost = single[n]["mean"]
        score = gain / max(cost, 0.5)
        if best is None or score > best[0]:
            best = (score, n, r)
    score, n, r = best
    if r - cur_recall < 0.0005:
        break
    chosen.append(n)
    remaining.remove(n)
    steps.append({"added": n, "recall": r, "gain": round(r - cur_recall, 5), "single_mean_candidates": single[n]["mean"]})
    cur_recall = r
RES["greedy"] = {"order": chosen, "steps": steps}
log("greedy order", chosen)
for i in range(1, len(chosen) + 1):
    v = volume_of(chosen[:i])
    steps[i - 1].update({"union_mean": v["mean"], "union_p95": v["p95"], "union_pairs": v["pairs"], "union_max": v["max"]})
    log(f"greedy {i}: +{chosen[i - 1]:28s} recall={steps[i - 1]['recall']:.5f} mean={v['mean']:.1f} p95={v['p95']:.0f} max={v['max']:,}")

# ---- named candidate sets
RETR = ["3_name_ngram_top20", "9_addr_token_top20", "9_name_addr_token_top20"]
CHEAP = ["5_state_name_sorted", "5_state_house_street", "8_sn_name_core_sorted_w3", "8_sn_name_compact_w3", "9_sn_addr_sorted_w3"]
EXACT = ["1_exact_name_core", "1_exact_name_core_sorted", "1_exact_name_compact", "1_exact_name_translit_core_sorted"]
SETS = {
    "retrieval_only": RETR,
    "retrieval_plus_cheap_keys": RETR + CHEAP,
    "retrieval_cheap_exactname": RETR + CHEAP + EXACT,
    "retrieval_cheap_exactname_cityfirst": RETR + CHEAP + EXACT + ["5_city_name_first"],
    "retrieval_cheap_exactname_acronym_cityhouse": RETR + CHEAP + EXACT + ["8_acronym_city", "5_city_house_number"],
    "retrieval_cheap_exactname_addrtok200": RETR + CHEAP + EXACT + ["4_addr_token_df200_rarest3"],
    "retrieval_cheap_exactname_nametok500": RETR + CHEAP + EXACT + ["2_name_token_df500_rarest2"],
    "core7": ["9_name_addr_token_top20", "5_state_house_street", "5_state_name_sorted", "8_sn_name_compact_w3", "8_sn_name_core_sorted_w3", "9_sn_addr_sorted_w3", "8_acronym_city"],
    "core7_exactcap50": ["9_name_addr_token_top20", "5_state_house_street", "5_state_name_sorted", "8_sn_name_compact_w3", "8_sn_name_core_sorted_w3", "9_sn_addr_sorted_w3", "8_acronym_city", "1_exact_name_any_cap50"],
    "core7_exactcap50_addrretr": ["9_name_addr_token_top20", "5_state_house_street", "5_state_name_sorted", "8_sn_name_compact_w3", "8_sn_name_core_sorted_w3", "9_sn_addr_sorted_w3", "8_acronym_city", "1_exact_name_any_cap50", "9_addr_token_top20"],
    "core7_exactcap200_addrretr": ["9_name_addr_token_top20", "5_state_house_street", "5_state_name_sorted", "8_sn_name_compact_w3", "8_sn_name_core_sorted_w3", "9_sn_addr_sorted_w3", "8_acronym_city", "1_exact_name_any_cap200", "9_addr_token_top20"],
    "core7_exactcap50_addrretr_cityfirst100": ["9_name_addr_token_top20", "5_state_house_street", "5_state_name_sorted", "8_sn_name_compact_w3", "8_sn_name_core_sorted_w3", "9_sn_addr_sorted_w3", "8_acronym_city", "1_exact_name_any_cap50", "9_addr_token_top20", "5_city_name_first_cap100"],
    "core7_exactcap50_addrretr_ngram": ["9_name_addr_token_top20", "5_state_house_street", "5_state_name_sorted", "8_sn_name_compact_w3", "8_sn_name_core_sorted_w3", "9_sn_addr_sorted_w3", "8_acronym_city", "1_exact_name_any_cap50", "9_addr_token_top20", "3_name_ngram_top20"],
    "bounded": EXACT + ["5_state_name_sorted", "5_city_name_first", "5_city_house_number", "5_state_house_street", "9_numsig", "8_acronym_city",
                        "8_sn_name_core_sorted_w3", "8_sn_name_compact_w3", "9_sn_addr_sorted_w3"] + RETR,
    "greedy_selection": chosen,
}
RES["sets"] = {}
for name, members in SETS.items():
    members = [m for m in members if m in STRATS]
    RES["sets"][name] = {"members": members, **recall_of(members), **volume_of(members)}
    r = RES["sets"][name]
    log(f"{name:44s} recall={r['recall']:.5f} (US {r['by_country'].get('US', 0):.4f} IN {r['by_country'].get('India', 0):.4f}) all-found={r['entity_all_found']:.4f} "
        f"mean={r['mean']:.1f} p95={r['p95']:.0f} max={r['max']:,} pairs={r['pairs']:,}")

json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
log("saved", OUT_JSON)
