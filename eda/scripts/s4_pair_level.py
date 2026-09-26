import os
import polars as pl, json, gc, time, numpy as np
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
rng = np.random.default_rng(42)
COLS = ["entity_id","country","business_name","business_address","name_norm","name_core","name_core_sorted","addr_norm","addr_sorted","addr_last","addr_first","name_first_tok","name_devanagari","addr_missing"]
pairs = pl.read_parquet(f"{SCR}/pq/train_pairs_pos.parquet")
s1_ids = pl.read_parquet(f"{SCR}/pq/train_source1_norm.parquet", columns=["entity_id","country","name_core_sorted"])
samp_ids = s1_ids.select("entity_id").sample(n=250_000, seed=42)
s1 = pl.scan_parquet(f"{SCR}/pq/train_source1_norm.parquet").select(COLS).filter(pl.col("entity_id").is_in(samp_ids["entity_id"].implode())).collect()
P = pairs.join(samp_ids, left_on="source1_entity_id", right_on="entity_id")
lite = pl.concat([pl.read_parquet(f"{SCR}/pq/train_source{k}_norm.parquet", columns=["entity_id","country","name_core_sorted"]) for k in (2,3)])
# random negatives
negs = []
for ctry in ["US","India"]:
    ids1 = s1.filter(pl.col("country")==ctry)["entity_id"]
    pool = lite.filter(pl.col("country")==ctry)["entity_id"].to_numpy()
    a = np.repeat(ids1.to_numpy(), 3); b = rng.choice(pool, size=len(a))
    negs.append(pl.DataFrame({"source1_entity_id": a, "matched_id": b}))
neg = pl.concat(negs).join(pairs.select("source1_entity_id","matched_id"), on=["source1_entity_id","matched_id"], how="anti").with_columns(pl.col("matched_id").str.slice(0,2).alias("src"))
# hard negatives: same country + same name_core_sorted, not matched
hn = s1.select(pl.col("entity_id").alias("source1_entity_id"), "name_core_sorted", "country").filter(pl.col("name_core_sorted")!="")
hn = hn.join(lite.select(pl.col("entity_id").alias("matched_id"), "name_core_sorted", "country"), on=["name_core_sorted","country"]).select("source1_entity_id","matched_id")
hn = hn.join(pairs.select("source1_entity_id","matched_id"), on=["source1_entity_id","matched_id"], how="anti")
n_hard_total = hn.height
hn = hn.sample(n=min(600_000, hn.height), seed=1).with_columns(pl.col("matched_id").str.slice(0,2).alias("src"))
needed = pl.concat([P["matched_id"], neg["matched_id"], hn["matched_id"]]).unique()
del lite; gc.collect()
s23 = pl.concat([pl.scan_parquet(f"{SCR}/pq/train_source{k}_norm.parquet").select(COLS).filter(pl.col("entity_id").is_in(needed.implode())).collect() for k in (2,3)])
print("loaded s1", s1.height, "s23", s23.height, flush=True)
L = s1.rename({c: f"{c}_1" for c in COLS}); Rt = s23.rename({c: f"{c}_2" for c in COLS})
def attach(pdf):
    return pdf.join(L, left_on="source1_entity_id", right_on="entity_id_1").join(Rt, left_on="matched_id", right_on="entity_id_2")
pos = attach(P).with_columns(pl.lit(1).alias("label"))
neg = attach(neg).with_columns(pl.lit(0).alias("label"))
hard = attach(hn).with_columns(pl.lit(0).alias("label"))
print("pos", pos.height, "rand-neg", neg.height, "hard-neg(sampled)", hard.height, "hard total for 250k S1:", n_hard_total, flush=True)

def feats(d):
    def toks(c): return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element()!="")).list.unique()
    def jac(a,b,name):
        return (toks(a).list.set_intersection(toks(b)).list.len() / pl.max_horizontal(toks(a).list.set_union(toks(b)).list.len(), pl.lit(1))).alias(name)
    d = d.with_columns([
        (pl.col("business_name_1")==pl.col("business_name_2")).alias("name_exact"),
        (pl.col("name_norm_1")==pl.col("name_norm_2")).alias("name_norm_eq"),
        (pl.col("name_core_1")==pl.col("name_core_2")).alias("name_core_eq"),
        (pl.col("name_core_sorted_1")==pl.col("name_core_sorted_2")).alias("name_core_sorted_eq"),
        (pl.col("name_first_tok_1")==pl.col("name_first_tok_2")).alias("name_first_tok_eq"),
        (pl.col("name_core_1").str.slice(0,3)==pl.col("name_core_2").str.slice(0,3)).alias("name_pre3_eq"),
        jac("name_core_1","name_core_2","name_jac"),
        (toks("name_core_1").list.set_intersection(toks("name_core_2")).list.len()>0).alias("name_any_tok"),
        (pl.col("addr_norm_1")==pl.col("addr_norm_2")).alias("addr_norm_eq"),
        (pl.col("addr_sorted_1")==pl.col("addr_sorted_2")).alias("addr_sorted_eq"),
        jac("addr_norm_1","addr_norm_2","addr_jac"),
        (toks("addr_norm_1").list.set_intersection(toks("addr_norm_2")).list.len()>0).alias("addr_any_tok"),
        (pl.col("addr_last_1").str.to_lowercase()==pl.col("addr_last_2").str.to_lowercase()).alias("addr_last_eq"),
        pl.col("addr_missing_2").alias("addr_missing_2"),
        (pl.col("name_devanagari_1") != pl.col("name_devanagari_2")).alias("script_mismatch"),
        pl.col("name_devanagari_2").alias("dev_2"),
        (pl.col("addr_norm_1").str.extract_all(r"\d+").list.set_intersection(pl.col("addr_norm_2").str.extract_all(r"\d+")).list.len()>0).alias("addr_num_overlap"),
        (pl.col("addr_norm_1").str.extract_all(r"\d+").list.len()>0).alias("addr1_has_num"),
    ])
    return d
t0=time.time()
pos, neg, hard = feats(pos), feats(neg), feats(hard)
print("feats", f"{time.time()-t0:.0f}s", flush=True)
F = ["name_exact","name_norm_eq","name_core_eq","name_core_sorted_eq","name_first_tok_eq","name_pre3_eq","name_any_tok","addr_norm_eq","addr_sorted_eq","addr_any_tok","addr_last_eq","addr_missing_2","script_mismatch","dev_2","addr_num_overlap"]
R = {"n_pos": pos.height, "n_rand_neg": neg.height, "n_hard_neg_sampled": hard.height, "n_hard_neg_total_for_250k_S1": n_hard_total,
     "hard_neg_per_S1_entity": n_hard_total/250_000}
def summarize(d, tag):
    out = {}
    for grp, sub in [("all", d)] + [(f"{c}", d.filter(pl.col("country_1")==c)) for c in ["US","India"]] + [(f"src{s}", d.filter(pl.col("src")==s)) for s in ["S2","S3"]]:
        o = {k: round(100*float(sub[k].fill_null(False).mean()),2) for k in F}
        o["name_jac_mean"] = round(float(sub["name_jac"].mean()),3); o["name_jac_p10"] = round(float(sub["name_jac"].quantile(0.1)),3); o["name_jac_p50"] = round(float(sub["name_jac"].median()),3)
        o["addr_jac_mean"] = round(float(sub["addr_jac"].mean()),3); o["addr_jac_p10"] = round(float(sub["addr_jac"].quantile(0.1)),3); o["addr_jac_p50"] = round(float(sub["addr_jac"].median()),3)
        o["name_jac_ge_0.5"] = round(100*float((sub["name_jac"]>=0.5).mean()),2); o["name_jac_ge_0.3"] = round(100*float((sub["name_jac"]>=0.3).mean()),2)
        o["addr_jac_ge_0.5"] = round(100*float((sub["addr_jac"]>=0.5).mean()),2); o["addr_jac_ge_0.3"] = round(100*float((sub["addr_jac"]>=0.3).mean()),2)
        o["name_or_addr_jac_ge_0.3"] = round(100*float(((sub["name_jac"]>=0.3)|(sub["addr_jac"]>=0.3)).mean()),2)
        o["n"] = sub.height
        out[grp] = o
    R[tag] = out
summarize(pos,"positives"); summarize(neg,"random_negatives"); summarize(hard,"hard_negatives_same_name")
R["hard_neg_addr_jac_dist"] = {f"addr_jac<{t}": round(100*float((hard["addr_jac"]<t).mean()),2) for t in [0.1,0.2,0.3,0.5]}
R["pos_addr_jac_dist"] = {f"addr_jac<{t}": round(100*float((pos["addr_jac"]<t).mean()),2) for t in [0.1,0.2,0.3,0.5]}
R["pos_no_name_tok_and_no_addr_tok_pct"] = round(100*float((~pos["name_any_tok"] & ~pos["addr_any_tok"]).mean()),3)
R["pos_no_name_tok_pct"] = round(100*float((~pos["name_any_tok"]).mean()),3)
R["pos_no_name_tok_by_script_mismatch"] = pos.group_by("script_mismatch").agg((~pl.col("name_any_tok")).mean().alias("no_name_tok"), pl.len()).to_dict(as_series=False)
ex = pos.filter(~pl.col("name_any_tok")).select("business_name_1","business_address_1","business_name_2","business_address_2","country_1").head(25)
R["examples_pos_no_name_token_overlap"] = ex.to_dicts()
ex2 = pos.filter(pl.col("name_any_tok") & (pl.col("name_jac")<0.4)).select("business_name_1","business_address_1","business_name_2","business_address_2","country_1").head(20)
R["examples_pos_low_name_jac"] = ex2.to_dicts()
ex3 = hard.filter(pl.col("addr_jac")>=0.5).select("business_name_1","business_address_1","business_name_2","business_address_2","country_1").head(20)
R["examples_hard_neg_same_name_similar_addr"] = ex3.to_dicts()
ex4 = pos.sample(n=30, seed=3).select("src","business_name_1","business_address_1","business_name_2","business_address_2","country_1")
R["examples_random_positives"] = ex4.to_dicts()
sib = P.join(Rt, left_on="matched_id", right_on="entity_id_2").select("source1_entity_id","matched_id","src","name_core_sorted_2","addr_sorted_2","business_name_2")
sib = sib.join(sib, on=["source1_entity_id","src"], suffix="_b").filter(pl.col("matched_id")<pl.col("matched_id_b"))
R["sibling_pairs_same_source"] = sib.height
R["sibling_same_raw_name_pct"] = round(100*float((sib["business_name_2"]==sib["business_name_2_b"]).mean()),2)
R["sibling_same_name_core_sorted_pct"] = round(100*float((sib["name_core_sorted_2"]==sib["name_core_sorted_2_b"]).mean()),2)
R["sibling_same_addr_sorted_pct"] = round(100*float((sib["addr_sorted_2"]==sib["addr_sorted_2_b"]).mean()),2)
ent = pos.group_by("source1_entity_id").agg(pl.col("name_core_sorted_eq").all().alias("all_name_eq"), pl.col("name_any_tok").all().alias("all_name_anytok"),
        ((pl.col("name_jac")>=0.3)|(pl.col("addr_jac")>=0.3)).all().alias("all_jac03"), pl.len().alias("k"))
R["entities_all_matches_name_core_sorted_eq_pct"] = round(100*float(ent["all_name_eq"].mean()),2)
R["entities_all_matches_name_any_tok_pct"] = round(100*float(ent["all_name_anytok"].mean()),2)
R["entities_all_matches_jac03_pct"] = round(100*float(ent["all_jac03"].mean()),2)
pos.select("source1_entity_id","matched_id","src","country_1","label","name_jac","addr_jac",*F).write_parquet(f"{SCR}/pq/sample_pos_feats.parquet")
hard.select("source1_entity_id","matched_id","src","country_1","label","name_jac","addr_jac",*F).write_parquet(f"{SCR}/pq/sample_hard_feats.parquet")
neg.select("source1_entity_id","matched_id","src","country_1","label","name_jac","addr_jac",*F).write_parquet(f"{SCR}/pq/sample_neg_feats.parquet")
json.dump(R, open(f"{SCR}/out/s4_pairs.json","w",encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print("DONE")
