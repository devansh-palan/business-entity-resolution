"""Validation-design diagnostics: numbers behind the seven investigation questions.

Reads the fold artifacts in artifacts/validation plus the normalised train parquet cache.
The only place test files are touched is the final block-size distribution diagnostic; it sets no parameter.
Run from the project root:  python eda/scripts/s9_validation_design.py
"""
import os
import json, sys, time
import numpy as np
import polars as pl

sys.path.insert(0, "code/business_entity_resolution/src")
import validation as va  # noqa: E402

VAL = "artifacts/validation"
PQ = os.path.join(os.environ.get("BER_SCRATCH", "artifacts/_scratch"), "pq")   # parquet cache, built by s0_inventory.py
OUT = "eda/results/s9_validation_design.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
R = {}

ent = va.load_entities(VAL)
pairs = va.load_positive_pairs(VAL)
summary = json.load(open(f"{VAL}/summary.json"))
R["clusters"] = summary["clusters"]
N_FOLDS = summary["n_folds"]

# ---------------------------------------------------------------- Q1: entity integrity
s4 = json.load(open("eda/results/s4_pairs.json", encoding="utf-8"))
R["Q1"] = {
    "pairs_per_entity_mean": round(pairs.height / ent.height, 3),
    "pct_entities_with_2plus_pairs": round(100 * ent.filter(pl.col("n_true") >= 2).height / ent.height, 2),
    "sibling_records_same_core_name_pct": s4["sibling_same_name_core_sorted_pct"],
    "sibling_records_same_sorted_address_pct": s4["sibling_same_addr_sorted_pct"],
}

# ---------------------------------------------------------------- Q2: shared candidates across folds
s1n = pl.read_parquet(f"{PQ}/train_source1_norm.parquet", columns=["entity_id", "country", "name_core_sorted", "addr_sorted"])
s1n = s1n.join(ent.select("entity_id", "fold", "cluster_id"), on="entity_id")
rng = np.random.default_rng(0)
s1n = s1n.with_columns(pl.Series("rfold", rng.integers(0, N_FOLDS, s1n.height)))

def cross_share(key, foldcol):
    d = s1n.filter(pl.col(key) != "")
    g = d.group_by("country", key).agg(pl.col(foldcol).n_unique().alias("nf"))
    affected = d.join(g.filter(pl.col("nf") >= 2).select("country", key), on=["country", key], how="semi").height
    return round(100 * affected / s1n.height, 3)

R["Q2_pct_S1_entities_with_same_name_entity_in_another_fold"] = {"random_entity_folds": cross_share("name_core_sorted", "rfold"),
                                                                  "cluster_folds": cross_share("name_core_sorted", "fold")}
R["Q2_pct_S1_entities_with_same_address_entity_in_another_fold"] = {"random_entity_folds": cross_share("addr_sorted", "rfold"),
                                                                     "cluster_folds": cross_share("addr_sorted", "fold")}
# pool records that are same-name candidates of fold-0 entities, by who owns them
s23n = pl.concat([pl.read_parquet(f"{PQ}/train_source{k}_norm.parquet", columns=["entity_id", "country", "name_core_sorted"]) for k in (2, 3)])
owner = pairs.select(pl.col("matched_id").alias("entity_id"), pl.col("fold").alias("owner_fold"), pl.col("source1_entity_id").alias("owner"))
s23n = s23n.join(owner, on="entity_id", how="left").join(
    s1n.select(pl.col("entity_id").alias("owner"), pl.col("rfold").alias("owner_rfold")), on="owner", how="left")
for foldcol, ownercol, label in (("fold", "owner_fold", "cluster_folds"), ("rfold", "owner_rfold", "random_entity_folds")):
    names0 = s1n.filter(pl.col(foldcol) == 0).filter(pl.col("name_core_sorted") != "").select("country", "name_core_sorted").unique()
    cand = s23n.join(names0, on=["country", "name_core_sorted"], how="semi")
    R[f"Q2_same_name_pool_records_for_fold0_entities[{label}]"] = {
        "records": cand.height,
        "pct_owned_by_fold0_entities": round(100 * cand.filter(pl.col(ownercol) == 0).height / cand.height, 2),
        "pct_owned_by_training_fold_entities": round(100 * cand.filter(pl.col(ownercol).is_not_null() & (pl.col(ownercol) != 0)).height / cand.height, 2),
        "pct_orphans": round(100 * cand.filter(pl.col(ownercol).is_null()).height / cand.height, 2),
    }
log("Q2 done")

# ---------------------------------------------------------------- Q3: duplicates and near duplicates
s3j = json.load(open("eda/results/s3_text_profile.json", encoding="utf-8"))
d = s3j["train_source1"]["duplicates"]
R["Q3_S1_near_duplicates"] = {
    "rows_sharing_core_name_and_sorted_address": d["(name_core_sorted, addr_sorted): rows sharing"],
    "rows_sharing_sorted_address": d["addr_sorted: rows sharing value w/ another row"],
    "rows_sharing_core_name_same_country": d["name_core_sorted: rows sharing value w/ another row (same country)"],
}
rows = []
for k in (2, 3):
    t = pl.read_parquet(f"{PQ}/train_source{k}.parquet").join(owner.select("entity_id", "owner"), on="entity_id", how="left")
    g = t.group_by("business_name", "business_address", "country").agg(
        pl.len().alias("n"), pl.col("owner").drop_nulls().n_unique().alias("n_owners"), pl.col("owner").is_null().sum().alias("n_orphan")).filter(pl.col("n") >= 2)
    rows.append({
        "source": f"S{k}", "duplicate_text_groups": g.height, "records_in_groups": int(g["n"].sum()),
        "groups_single_owner_no_orphan": g.filter((pl.col("n_owners") == 1) & (pl.col("n_orphan") == 0)).height,
        "groups_two_or_more_owners": g.filter(pl.col("n_owners") >= 2).height,
        "groups_owned_plus_orphan": g.filter((pl.col("n_owners") >= 1) & (pl.col("n_orphan") >= 1)).height,
        "groups_all_orphans": g.filter(pl.col("n_owners") == 0).height,
    })
    del t, g
R["Q3_exact_duplicate_S23_text"] = rows
log("Q3 done")

# ---------------------------------------------------------------- Q4-Q6: fold balance
folds = summary["folds"]
def spread(getter):
    vals = [getter(folds[str(f)]) for f in range(N_FOLDS)]
    return {"min": round(min(vals), 3), "max": round(max(vals), 3), "spread": round(max(vals) - min(vals), 3)}
R["Q4_pattern_pct_spread_across_folds"] = {p: spread(lambda s, p=p: s["pattern_pct"][p]) for p in va.PATTERNS}
R["Q5_country_pct_spread_across_folds"] = {c: spread(lambda s, c=c: s["country_pct"][c]) for c in folds["0"]["country_pct"]}
R["Q6_k_class_pct_spread_across_folds"] = {k: spread(lambda s, k=k: s["k_class_pct"][k]) for k in va.K_CLASSES}
R["Q6_mean_n_true_by_fold"] = {f: folds[str(f)]["mean_n_true"] for f in range(N_FOLDS)}
R["fold_sizes"] = {f: folds[str(f)]["entities"] for f in range(N_FOLDS)}
R["fold_positives"] = {f: folds[str(f)]["positives"] for f in range(N_FOLDS)}
# is match count independent of cluster size (name popularity)?
R["Q6_mean_n_true_by_cluster_size_bucket"] = {str(r["b"]): round(r["m"], 3) for r in ent.with_columns(
    pl.when(pl.col("cluster_size") == 1).then(pl.lit("1")).when(pl.col("cluster_size") <= 3).then(pl.lit("2-3"))
      .when(pl.col("cluster_size") <= 9).then(pl.lit("4-9")).otherwise(pl.lit("10+")).alias("b"))
    .group_by("b").agg(pl.col("n_true").mean().alias("m")).sort("b").to_dicts()}

# ---------------------------------------------------------------- Q7: formatting patterns of matched records per fold
f23 = pl.concat([pl.read_parquet(f"{PQ}/train_source{k}_norm.parquet", columns=["entity_id", "business_name", "name_devanagari", "addr_missing"]) for k in (2, 3)])
f23 = f23.with_columns(
    pl.col("business_name").str.contains(r"[\u0900-\u0DFF]").alias("indic_script"),
    pl.col("business_name").str.contains(r"(?i)\.(com|net|org|in|fr|co|io|biz|c0m)\b|^#\S+$").alias("domain_style"),
    (pl.col("business_name").str.contains(r"\p{Lu}") & ~pl.col("business_name").str.contains(r"\p{Ll}")).alias("all_caps"),
    pl.col("business_name").str.contains(r"[\u00C0-\u024F]").alias("latin_accent"),
).drop("business_name")
m = pairs.join(f23, left_on="matched_id", right_on="entity_id")
q7 = m.group_by("fold").agg(
    pl.len().alias("matches"),
    (pl.col("src") == "S2").mean().alias("share_S2"),
    pl.col("indic_script").mean().alias("indic_script"), pl.col("domain_style").mean().alias("domain_style"),
    pl.col("all_caps").mean().alias("all_caps"), pl.col("latin_accent").mean().alias("latin_accent"),
    pl.col("addr_missing").mean().alias("addr_missing")).sort("fold")
R["Q7_matched_record_patterns_by_fold_pct"] = {str(r["fold"]): {k: (round(100 * v, 3) if k != "matches" else v) for k, v in r.items() if k != "fold"} for r in q7.to_dicts()}
del f23, m
log("Q7 done")

# ---------------------------------------------------------------- diagnostic only: same-name block size per S1, train vs test
def block_sizes(split):
    s1 = pl.read_parquet(f"{PQ}/{split}_source1_norm.parquet", columns=["entity_id", "country", "name_core_sorted"])
    pool = pl.concat([pl.read_parquet(f"{PQ}/{split}_source{k}_norm.parquet", columns=["country", "name_core_sorted"]) for k in (2, 3)])
    blk = pool.group_by("country", "name_core_sorted").len().rename({"len": "blk"})
    j = s1.join(blk, on=["country", "name_core_sorted"], how="left").with_columns(pl.col("blk").fill_null(0))
    out = {}
    for (c,), sub in j.group_by("country"):
        b = sub["blk"].to_numpy()
        out[c] = {"S1": len(b), "pool": pool.filter(pl.col("country") == c).height, "mean": round(float(b.mean()), 1),
                  "median": float(np.median(b)), "p90": float(np.percentile(b, 90)), "pct_zero": round(100 * float((b == 0).mean()), 2),
                  "pct_ge_10": round(100 * float((b >= 10).mean()), 2)}
    return out
R["diagnostic_same_name_block_size_per_S1"] = {"train": block_sizes("train"), "test": block_sizes("test")}
log("diagnostic done")

json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
for k, v in R.items():
    print(k, "=>", json.dumps(v, default=str) if not isinstance(v, (int, float, str)) else v)
log("DONE")
