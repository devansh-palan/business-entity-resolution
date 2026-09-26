import os
import polars as pl, json, numpy as np, gc, sys, time
sys.stdout.reconfigure(encoding="utf-8")
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
STOP = ["road","street","avenue","boulevard","drive","lane","court","floor","flat","plot","near","opp","house","block","sector","nagar","colony","unit","suite","apartment","building","city","county","town","village","district","main","cross","north","south","east","west","phase","area","complex","society","marg","rue","route","chemin","allee","place","boite","postale","cedex","door","shop","office","ground","first","second","third","tower","park","garden","layout","stage","extension","road"]
R = {}
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

def keys(lf):
    an = pl.col("addr_norm")
    toks = an.str.extract_all(r"[a-z]{4,}").list.eval(pl.element().filter(~pl.element().is_in(STOP)))
    longest = toks.list.get(toks.list.eval(pl.element().str.len_chars()).list.arg_max(), null_on_oob=True)
    firstnum = an.str.extract(r"(\d+)", 1)
    numsig = an.str.extract_all(r"\d+").list.unique().list.sort().list.join(" ")
    def H(parts, cond):
        return pl.when(cond).then(pl.concat_str([pl.col("country")] + parts, separator="|").hash()).otherwise(None)
    return lf.select(
        "entity_id", "country",
        H([pl.col("name_core_sorted")], pl.col("name_core_sorted") != "").alias("k_name"),
        H([pl.col("name_first_tok")], pl.col("name_first_tok").is_not_null() & (pl.col("name_first_tok") != "")).alias("k_first"),
        H([pl.col("name_core").str.slice(0, 4)], pl.col("name_core").str.len_chars() >= 4).alias("k_pre4"),
        H([pl.col("addr_sorted")], pl.col("addr_sorted") != "").alias("k_addr"),
        H([numsig], an.str.contains(r"\d")).alias("k_numsig"),
        H([firstnum, longest], an.str.contains(r"\d") & longest.is_not_null()).alias("k_num_long"),
        H([longest], longest.is_not_null()).alias("k_long"),
        H([pl.col("name_first_tok"), longest], (pl.col("name_first_tok") != "") & longest.is_not_null()).alias("k_first_long"),
    )
KEYS = ["k_name","k_first","k_pre4","k_addr","k_numsig","k_num_long","k_long","k_first_long"]
s1_ids = pl.read_parquet(f"{SCR}/pq/train_source1_norm.parquet", columns=["entity_id","country","name_core_sorted"])
samp = s1_ids.select("entity_id").sample(n=250_000, seed=42)
pairs = pl.read_parquet(f"{SCR}/pq/train_pairs_pos.parquet").join(samp, left_on="source1_entity_id", right_on="entity_id")
s1k = keys(pl.scan_parquet(f"{SCR}/pq/train_source1_norm.parquet")).collect()
s1ks = s1k.join(samp, on="entity_id")
s23k = pl.concat([keys(pl.scan_parquet(f"{SCR}/pq/train_source{k}_norm.parquet")).collect() for k in (2, 3)])
log("keys built", s1k.height, s23k.height)
# ---- recall per key on sampled positive pairs
pp = pairs.join(s1ks.rename({c: c+"_1" for c in KEYS}).drop("country"), left_on="source1_entity_id", right_on="entity_id")
pp = pp.join(s23k.rename({c: c+"_2" for c in KEYS}).drop("country"), left_on="matched_id", right_on="entity_id")
for k in KEYS:
    pp = pp.with_columns((pl.col(k+"_1").is_not_null() & (pl.col(k+"_1") == pl.col(k+"_2"))).alias("hit_"+k))
rec = {k: round(100*float(pp["hit_"+k].mean()), 2) for k in KEYS}
R["recall_per_key_pct"] = rec
# ---- volume per key: candidates per sampled S1 (group sizes in S2+S3)
vol = {}
for k in KEYS:
    cnt = s23k.group_by(k).len().drop_nulls(k)
    v = s1ks.select("entity_id", k).join(cnt, on=k, how="left").with_columns(pl.col("len").fill_null(0))
    a = v["len"].to_numpy()
    vol[k] = {"mean": round(float(a.mean()), 1), "p50": float(np.median(a)), "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)), "max": int(a.max()),
              "pct_S1_with_zero_cands": round(100*float((a == 0).mean()), 2), "pct_S1_block_gt_5000": round(100*float((a > 5000).mean()), 2)}
R["volume_per_key"] = vol
# ---- union recall & per-entity full recall for key combos
def union(cols):
    e = pl.any_horizontal([pl.col("hit_"+c) for c in cols])
    d = pp.with_columns(e.alias("u"))
    ent = d.group_by("source1_entity_id").agg(pl.col("u").all().alias("all"))
    return {"pair_recall": round(100*float(d["u"].mean()), 2), "entity_all_matches_recalled": round(100*float(ent["all"].mean()), 2),
            "approx_volume_mean(sum of block sizes)": round(sum(vol[c]["mean"] for c in cols), 1)}
COMBOS = {"name": ["k_name"], "name+addr": ["k_name","k_addr"], "name+addr+numsig": ["k_name","k_addr","k_numsig"],
          "name+addr+numsig+num_long": ["k_name","k_addr","k_numsig","k_num_long"],
          "name+pre4+addr+numsig+num_long": ["k_name","k_pre4","k_addr","k_numsig","k_num_long"],
          "name+first+addr+numsig+num_long": ["k_name","k_first","k_addr","k_numsig","k_num_long"],
          "all keys": KEYS}
R["union_recall"] = {n: union(c) for n, c in COMBOS.items()}
log("key recall/volume done")
del s23k; gc.collect()
# ---- rare-token blocking (name tokens, address tokens): recall on sample pairs and volume upper bound
def tok_df(col, minlen):
    lfs = [pl.scan_parquet(f"{SCR}/pq/train_source{k}_norm.parquet").select("country", pl.col(col).str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= minlen) for k in (2, 3)]
    return pl.concat(lfs).group_by("country", "t").len().collect(engine="streaming")
NM = pl.read_parquet(f"{SCR}/pq/train_source1_norm.parquet", columns=["entity_id","country","name_core","addr_norm"])
S23 = pl.concat([pl.scan_parquet(f"{SCR}/pq/train_source{k}_norm.parquet").select("entity_id","name_core","addr_norm").filter(pl.col("entity_id").is_in(pairs["matched_id"].implode())).collect() for k in (2, 3)])
pt = pairs.join(NM.rename({"name_core":"n1","addr_norm":"a1"}), left_on="source1_entity_id", right_on="entity_id").join(S23.rename({"name_core":"n2","addr_norm":"a2"}), left_on="matched_id", right_on="entity_id")
for col, minlen, thrs, tag in [("name_core", 2, [500, 2000, 5000, 20000], "name"), ("addr_norm", 3, [200, 1000, 3000, 10000], "addr")]:
    df = tok_df(col, minlen)
    log(f"{tag} token df table", df.height)
    c1, c2 = ("n1", "n2") if tag == "name" else ("a1", "a2")
    shared = pt.select("source1_entity_id", "matched_id", "country", pl.col(c1).str.split(" ").list.set_intersection(pl.col(c2).str.split(" ")).alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= minlen)
    shared = shared.join(df, on=["country", "t"], how="left").with_columns(pl.col("len").fill_null(0))
    mn = shared.group_by("source1_entity_id", "matched_id").agg(pl.col("len").min().alias("min_df"))
    pr = pt.select("source1_entity_id", "matched_id").join(mn, on=["source1_entity_id", "matched_id"], how="left")
    s1t = NM.join(samp, on="entity_id").select("entity_id", "country", pl.col(col).str.split(" ").alias("t")).explode("t").filter(pl.col("t").str.len_chars() >= minlen).join(df, on=["country", "t"], how="left").with_columns(pl.col("len").fill_null(0))
    out = {}
    for thr in thrs:
        hit = pr.with_columns((pl.col("min_df").is_not_null() & (pl.col("min_df") <= thr)).alias("h"))
        ent = hit.group_by("source1_entity_id").agg(pl.col("h").all().alias("all"))
        volv = s1t.filter(pl.col("len") <= thr).group_by("entity_id").agg(pl.col("len").sum().alias("v")).join(samp, on="entity_id", how="right").with_columns(pl.col("v").fill_null(0))["v"].to_numpy()
        out[f"df<={thr}"] = {"pair_recall": round(100*float(hit["h"].mean()), 2), "entity_all_recalled": round(100*float(ent["all"].mean()), 2), "vol_mean": round(float(volv.mean()), 1), "vol_p50": float(np.median(volv)), "vol_p99": float(np.percentile(volv, 99))}
    R[f"rare_{tag}_token_blocking"] = out
    pr = pr.rename({"min_df": f"min_df_{tag}"})
    pt = pt.join(pr, on=["source1_entity_id", "matched_id"], how="left")
    del df, shared, mn, s1t; gc.collect()
log("token blocking done")
# ---- grand union: keys + rare tokens
pp2 = pp.select("source1_entity_id", "matched_id", *[f"hit_{k}" for k in KEYS]).join(pt.select("source1_entity_id", "matched_id", "min_df_name", "min_df_addr"), on=["source1_entity_id", "matched_id"])
for nm_thr, ad_thr in [(2000, 1000), (5000, 3000), (20000, 10000)]:
    u = pp2.with_columns((pl.col("hit_k_name") | pl.col("hit_k_addr") | pl.col("hit_k_numsig") | pl.col("hit_k_num_long") |
                          (pl.col("min_df_name").is_not_null() & (pl.col("min_df_name") <= nm_thr)) | (pl.col("min_df_addr").is_not_null() & (pl.col("min_df_addr") <= ad_thr))).alias("u"))
    ent = u.group_by("source1_entity_id").agg(pl.col("u").all().alias("all"), pl.len().alias("k"))
    R[f"grand_union name_df<={nm_thr} addr_df<={ad_thr}"] = {"pair_recall": round(100*float(u["u"].mean()), 2), "entity_all_recalled": round(100*float(ent["all"].mean()), 2)}
    if nm_thr == 5000:
        miss = u.filter(~pl.col("u")).join(pt.select("source1_entity_id", "matched_id", "n1", "a1", "n2", "a2"), on=["source1_entity_id", "matched_id"]).head(25)
        R["examples_missed_by_grand_union_5000_3000"] = miss.select("n1", "a1", "n2", "a2").to_dicts()
# ---- categorize positives with no name-token overlap
cat = pt.with_columns(pl.col("n1").str.split(" ").list.set_intersection(pl.col("n2").str.split(" ")).list.len().alias("shared"))
cat = cat.filter(pl.col("shared") == 0).join(pl.concat([pl.scan_parquet(f"{SCR}/pq/train_source{k}_norm.parquet").select("entity_id", "business_name").filter(pl.col("entity_id").is_in(pairs["matched_id"].implode())).collect() for k in (2, 3)]), left_on="matched_id", right_on="entity_id")
bn = pl.col("business_name")
cat = cat.with_columns(pl.when(bn.str.contains(r"[\u0900-\u0DFF]")).then(pl.lit("indic_script")).when(bn.str.contains(r"(?i)\.(com|net|org|in|fr|co|io|biz)\b")).then(pl.lit("domain_style"))
                       .when(bn.str.contains(r"(?i)\b(dba|d/b/a|t/a|fka|aka|trading as|formerly)\b")).then(pl.lit("dba_fka_alias")).otherwise(pl.lit("other_rename_or_typo")).alias("cat"))
c = cat.group_by("cat").len().sort("len", descending=True)
R["no_name_token_overlap_categories"] = {k: (int(v), round(100*v/pt.height, 2)) for k, v in zip(c["cat"], c["len"])}
R["examples_other_rename"] = cat.filter(pl.col("cat") == "other_rename_or_typo").select("n1", "business_name", "a1", "a2").head(20).to_dicts()
log("categories done")
# ---- orphan near-miss analysis: S2/S3 records with no S1 match that share name with an S1 and have a similar address
allpairs = pl.read_parquet(f"{SCR}/pq/train_pairs_pos.parquet")
orph = pl.concat([pl.scan_parquet(f"{SCR}/pq/train_source{k}_norm.parquet").select("entity_id", "country", "name_core_sorted", "addr_norm").collect() for k in (2, 3)]).join(allpairs.select("matched_id"), left_on="entity_id", right_on="matched_id", how="anti")
R["orphans_total"] = orph.height
orph = orph.sample(n=200_000, seed=7)
s1n = pl.read_parquet(f"{SCR}/pq/train_source1_norm.parquet", columns=["entity_id", "country", "name_core_sorted", "addr_norm"])
grp = s1n.group_by("name_core_sorted", "country").len().filter(pl.col("len") <= 200)
j = orph.rename({"entity_id": "oid", "addr_norm": "ao"}).join(s1n.join(grp.select("name_core_sorted", "country"), on=["name_core_sorted", "country"]).rename({"addr_norm": "a1"}), on=["name_core_sorted", "country"])
def toks(c): return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
j = j.with_columns((toks("ao").list.set_intersection(toks("a1")).list.len() / pl.max_horizontal(toks("ao").list.set_union(toks("a1")).list.len(), pl.lit(1))).alias("aj"),
                   (pl.col("ao").str.extract_all(r"\d+").list.set_intersection(pl.col("a1").str.extract_all(r"\d+")).list.len() > 0).alias("num_ov"))
best = j.group_by("oid").agg(pl.col("aj").max().alias("best_aj"), pl.col("num_ov").any().alias("any_num"), pl.len().alias("n_same_name_S1"))
R["orphan_sample"] = {"n": orph.height, "pct_with_same_name_S1": round(100*best.height/orph.height, 2),
                      "pct_with_same_name_S1_and_addr_jac>=0.5": round(100*float((best["best_aj"] >= 0.5).sum())/orph.height, 2),
                      "pct_with_same_name_S1_and_addr_jac>=0.3": round(100*float((best["best_aj"] >= 0.3).sum())/orph.height, 2),
                      "pct_with_same_name_S1_and_shared_number": round(100*float(best["any_num"].sum())/orph.height, 2),
                      "mean_same_name_S1_per_orphan(when>0)": round(float(best["n_same_name_S1"].mean()), 1)}
R["examples_orphan_near_miss"] = j.filter(pl.col("aj") >= 0.5).select("name_core_sorted", "ao", "a1").head(15).to_dicts()
# S1 side: fraction of S1 entities that have >=1 orphan near-miss (same name, addr_jac>=0.5) -- estimated from the orphan sample scaled
s1hit = j.filter(pl.col("aj") >= 0.5).select("entity_id").unique().height
R["S1_entities_with_near_miss_orphan_in_200k_orphan_sample"] = s1hit
R["scaled_estimate_pct_S1_with_near_miss_orphan"] = round(100 * s1hit * (R["orphans_total"] / 200_000) / s1n.height, 2)
log("orphans done")
# ---- Indic script blocks in names / addresses (train S2+S3 India; test S2+S3 India)
BLOCKS = {"devanagari": r"[\u0900-\u097F]", "bengali": r"[\u0980-\u09FF]", "gurmukhi": r"[\u0A00-\u0A7F]", "gujarati": r"[\u0A80-\u0AFF]", "oriya": r"[\u0B00-\u0B7F]", "tamil": r"[\u0B80-\u0BFF]", "telugu": r"[\u0C00-\u0C7F]", "kannada": r"[\u0C80-\u0CFF]", "malayalam": r"[\u0D00-\u0D7F]", "latin_accented": r"[\u00C0-\u024F]"}
for split in ["train", "test"]:
    d = pl.concat([pl.scan_parquet(f"{SCR}/pq/{split}_source{k}.parquet").filter(pl.col("country") == "India").select("business_name", "business_address").collect() for k in (2, 3)])
    R[f"{split} S2+S3 India: pct names by script"] = {b: round(100*float(d.select(pl.col("business_name").str.contains(rx).mean()).item()), 2) for b, rx in BLOCKS.items()}
    R[f"{split} S2+S3 India: pct addresses by script"] = {b: round(100*float(d.select(pl.col("business_address").fill_null("").str.contains(rx).mean()).item()), 2) for b, rx in BLOCKS.items()}
    R[f"{split} S2+S3 India: pct name entirely non-Latin (no a-z)"] = round(100*float(d.select(~pl.col("business_name").str.contains(r"(?i)[a-z]")).mean().item()), 2)
    del d; gc.collect()
json.dump(R, open(f"{SCR}/out/s6_blocking.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
for k, v in R.items():
    if k.startswith("examples"):
        print("\n==", k); [print("   ", " | ".join(str(x) for x in r.values())) for r in v[:15]]
    else: print(k, "=>", v)
log("DONE")
