import os
import polars as pl, json, gc
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
pq = lambda k: pl.read_parquet(f"{SCR}/pq/{k}.parquet")
gt = pq("train_ground_truth")
s1 = pq("train_source1").select("entity_id","country")
s2 = pq("train_source2").select("entity_id","country")
s3 = pq("train_source3").select("entity_id","country")
R = {}
# ---- integrity: GT vs S1 file
R["gt_rows"] = gt.height; R["s1_rows"] = s1.height
R["gt_ids_not_in_s1"] = gt.join(s1, left_on="source1_entity_id", right_on="entity_id", how="anti").height
R["s1_ids_not_in_gt"]  = s1.join(gt, left_on="entity_id", right_on="source1_entity_id", how="anti").height
R["gt_dup_s1_rows"] = gt.height - gt["source1_entity_id"].n_unique()
# ---- explode to pair level
g = gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.strip_chars().alias("m"))
g = g.with_columns(pl.when(pl.col("m")=="").then(pl.lit([],dtype=pl.List(pl.String))).otherwise(pl.col("m").str.split(",")).alias("lst"))
g = g.with_columns(pl.col("lst").list.len().alias("n_total"),
                   pl.col("lst").list.n_unique().alias("n_unique"),
                   pl.col("lst").list.eval(pl.element().str.starts_with("S2-")).list.sum().alias("n_s2"),
                   pl.col("lst").list.eval(pl.element().str.starts_with("S3-")).list.sum().alias("n_s3"),
                   pl.col("lst").list.eval(pl.element().str.starts_with("S1-")).list.sum().alias("n_s1_self"))
R["lists_with_intra_dupes"] = int((g["n_total"]!=g["n_unique"]).sum())
R["lists_with_S1_ids"] = int((g["n_s1_self"]>0).sum())
R["lists_with_other_prefix"] = int(((g["n_total"]-g["n_s2"]-g["n_s3"])>0).sum())
R["lists_with_whitespace_tokens"] = int(g.select(pl.col("lst").list.eval(pl.element().str.contains(r"\s")).list.sum().alias("w")).select(pl.col("w")>0).sum().item())
pairs = g.select("source1_entity_id", pl.col("lst").alias("matched_id")).explode("matched_id").drop_nulls("matched_id")
pairs = pairs.with_columns(pl.col("matched_id").str.slice(0,2).alias("src"))
R["n_positive_pairs"] = pairs.height
R["n_positive_pairs_by_src"] = pairs.group_by("src").len().sort("src").to_dict(as_series=False)
# do matched ids exist?
p2 = pairs.filter(pl.col("src")=="S2"); p3 = pairs.filter(pl.col("src")=="S3")
R["matched_S2_ids_missing_from_S2_file"] = p2.join(s2, left_on="matched_id", right_on="entity_id", how="anti").height
R["matched_S3_ids_missing_from_S3_file"] = p3.join(s3, left_on="matched_id", right_on="entity_id", how="anti").height
# does any matched id appear under >1 S1?  (Q22)
mc = pairs.group_by("matched_id").len()
R["matched_ids_unique"] = mc.height
R["matched_ids_under_multiple_S1"] = int((mc["len"]>1).sum())
R["max_S1_per_matched_id"] = int(mc["len"].max())
# coverage of S2/S3 by GT
R["S2_records_total"] = s2.height; R["S2_records_matched"] = int(p2["matched_id"].n_unique())
R["S3_records_total"] = s3.height; R["S3_records_matched"] = int(p3["matched_id"].n_unique())
# ---- entity-level distributions
tot = g.height
def dist(col, cap=10):
    d = g.with_columns(pl.when(pl.col(col)>=cap).then(pl.lit(f"{cap}+")).otherwise(pl.col(col).cast(pl.String)).alias("k")).group_by("k").len().sort("k")
    return {k: {"count": int(c), "pct": round(100*c/tot,3)} for k, c in zip(d["k"], d["len"])}
R["dist_n_total"] = dist("n_total", 12)
R["dist_n_s2"] = dist("n_s2", 8)
R["dist_n_s3"] = dist("n_s3", 8)
R["n_total_stats"] = {"mean": float(g["n_total"].mean()), "max": int(g["n_total"].max()), "p50": float(g["n_total"].median()), "p90": float(g["n_total"].quantile(0.9)), "p99": float(g["n_total"].quantile(0.99))}
R["singleton_pct"] = round(100*int((g["n_total"]==0).sum())/tot,3)
R["one_match_pct"] = round(100*int((g["n_total"]==1).sum())/tot,3)
R["multi_match_pct"] = round(100*int((g["n_total"]>=2).sum())/tot,3)
# source-pair pattern
pat = g.with_columns(pl.when((pl.col("n_s2")==0)&(pl.col("n_s3")==0)).then(pl.lit("none"))
      .when(pl.col("n_s3")==0).then(pl.lit("S2 only")).when(pl.col("n_s2")==0).then(pl.lit("S3 only")).otherwise(pl.lit("both S2 and S3")).alias("pat"))
d = pat.group_by("pat").len().sort("len", descending=True)
R["source_pair_pattern"] = {k: {"count": int(c), "pct": round(100*c/tot,3)} for k,c in zip(d["pat"], d["len"])}
# joint (n_s2, n_s3) crosstab (capped 4)
j = g.with_columns(pl.col("n_s2").clip(0,4).alias("a"), pl.col("n_s3").clip(0,4).alias("b")).group_by("a","b").len().sort("a","b")
R["joint_ns2_ns3_capped4"] = [(int(a),int(b),int(c), round(100*c/tot,3)) for a,b,c in zip(j["a"],j["b"],j["len"])]
# ---- by country (S1 country)
gc_ = g.join(s1, left_on="source1_entity_id", right_on="entity_id")
byc = gc_.group_by("country").agg(pl.len().alias("n"), (pl.col("n_total")==0).mean().alias("singleton_rate"),
      (pl.col("n_total")==1).mean().alias("one_rate"), (pl.col("n_total")>=2).mean().alias("multi_rate"),
      pl.col("n_s2").mean().alias("mean_s2"), pl.col("n_s3").mean().alias("mean_s3"), pl.col("n_total").mean().alias("mean_total"), pl.col("n_total").max().alias("max_total")).sort("country")
R["by_country"] = byc.to_dict(as_series=False)
# ---- country consistency across matched pairs
pc = pairs.join(s1.rename({"entity_id":"source1_entity_id","country":"c1"}), on="source1_entity_id")
c23 = pl.concat([s2, s3]).rename({"entity_id":"matched_id","country":"c2"})
pc = pc.join(c23, on="matched_id", how="left")
R["pairs_country_mismatch"] = int((pc["c1"]!=pc["c2"]).sum())
R["pairs_country_null"] = int(pc["c2"].null_count())
R["pairs_country_crosstab"] = pc.group_by("c1","c2").len().sort("c1","c2").to_dict(as_series=False)
# ---- Oracle scores for trivial strategies (macro F0.5 on train)
# predict-all-empty: score = singleton fraction
R["oracle_all_empty_macroF05"] = R["singleton_pct"]/100
# per-entity F0.5 if we predict exactly one correct match for every non-singleton (and empty for singletons)
def f05(p, r):
    return 0.0 if (p==0 or r==0) else 1.25*p*r/(0.25*p+r)
import numpy as np
n = g["n_total"].to_numpy()
one_correct = np.where(n==0, 1.0, np.array([f05(1.0, 1.0/max(k,1)) for k in n]))
R["oracle_one_correct_match_per_nonsingleton_macroF05"] = float(one_correct.mean())
# if we predict perfectly on non-singletons but wrongly add one false match to every singleton
R["oracle_perfect_but_all_singletons_get_1_FP"] = float(np.where(n==0, 0.0, 1.0).mean())
# perfect on all except we add one extra FP to every entity
allfp = np.where(n==0, 0.0, np.array([f05(k/(k+1), 1.0) for k in n]))
R["oracle_perfect_plus_one_FP_everywhere"] = float(allfp.mean())
# perfect on all except we miss one match from every multi-match entity
miss1 = np.where(n==0, 1.0, np.array([f05(1.0, (k-1)/k) if k>=2 else 1.0 for k in n]))
R["oracle_perfect_but_miss_one_from_each_multi"] = float(miss1.mean())
# expected-value: for an entity with k true matches, adding one FP vs missing one TP
R["f05_table_k_true"] = {k: {"1FP_added": round(f05(k/(k+1),1.0),4), "1TP_missed": round(f05(1.0,(k-1)/k),4) if k>=2 else None, "only_1_of_k_found": round(f05(1.0,1.0/k),4)} for k in range(1,7)}
# save pair table for later
pairs.write_parquet(f"{SCR}/pq/train_pairs_pos.parquet")
g.select("source1_entity_id","n_total","n_s2","n_s3").write_parquet(f"{SCR}/pq/train_gt_counts.parquet")
json.dump(R, open(f"{SCR}/out/s1_gt.json","w"), indent=1, default=str)
for k,v in R.items(): print(k, "=>", v)
