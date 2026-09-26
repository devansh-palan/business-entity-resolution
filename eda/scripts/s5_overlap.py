import os
import polars as pl, json, sys
sys.stdout.reconfigure(encoding="utf-8")
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
C = ["entity_id","country","business_name","business_address","name_core_sorted","addr_sorted"]
rd = lambda k: pl.read_parquet(f"{SCR}/pq/{k}_norm.parquet", columns=C)
R = {}
for k in ["source1","source2","source3"]:
    tr, te = rd(f"train_{k}"), rd(f"test_{k}")
    R[f"{k}: exact (name,addr,country) rows of TEST also in TRAIN"] = te.join(tr.select("business_name","business_address","country").unique(), on=["business_name","business_address","country"], how="semi").height
    R[f"{k}: TEST rows whose (name_core_sorted,country) appears in TRAIN"] = te.join(tr.select("name_core_sorted","country").unique(), on=["name_core_sorted","country"], how="semi").height
    R[f"{k}: TEST rows whose (name_core_sorted, addr_sorted, country) appears in TRAIN"] = te.join(tr.select("name_core_sorted","addr_sorted","country").unique(), on=["name_core_sorted","addr_sorted","country"], how="semi").height
    R[f"{k}: test rows"] = te.height
# Orphan proxy: fraction of S2/S3 records whose (name_core_sorted,country) exists in S1 of the same split
for split in ["train","test"]:
    s1 = rd(f"{split}_source1").select("name_core_sorted","country").unique()
    for k in ["source2","source3"]:
        d = rd(f"{split}_{k}")
        tot = d.group_by("country").len().rename({"len":"n"})
        hit = d.join(s1, on=["name_core_sorted","country"], how="semi").group_by("country").len().rename({"len":"hit"})
        m = tot.join(hit, on="country").with_columns((pl.col("hit")/pl.col("n")*100).round(2).alias("pct_name_in_S1")).sort("country")
        R[f"{split} {k}: pct of records whose core name exists in S1 (same country)"] = m.to_dicts()
# S1 name-collision stats by country for test (incl France) and train
for split in ["train","test"]:
    s1 = rd(f"{split}_source1")
    g = s1.group_by("name_core_sorted","country").len()
    for (c,), sub in g.group_by("country"):
        n = int(sub["len"].sum())
        R[f"{split} S1 [{c}]: pct of entities sharing core name with >=1 other S1 entity"] = round(100*int(sub.filter(pl.col("len")>=2)["len"].sum())/n, 2)
        R[f"{split} S1 [{c}]: pct in groups >=10"] = round(100*int(sub.filter(pl.col("len")>=10)["len"].sum())/n, 2)
        R[f"{split} S1 [{c}]: distinct core names"] = sub.height
# ratio of S2/S3 to S1 per country
for split in ["train","test"]:
    n1 = rd(f"{split}_source1").group_by("country").len().rename({"len":"s1"})
    n2 = rd(f"{split}_source2").group_by("country").len().rename({"len":"s2"})
    n3 = rd(f"{split}_source3").group_by("country").len().rename({"len":"s3"})
    m = n1.join(n2, on="country").join(n3, on="country").with_columns((pl.col("s2")/pl.col("s1")).round(3).alias("s2_per_s1"), (pl.col("s3")/pl.col("s1")).round(3).alias("s3_per_s1")).sort("country")
    R[f"{split}: records per S1 entity by country"] = m.to_dicts()
json.dump(R, open(f"{SCR}/out/s5_overlap.json","w",encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, "=>", v)
