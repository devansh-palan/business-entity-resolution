import os
import polars as pl, json, numpy as np, gc
SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
pq = lambda k, cols=None: pl.read_parquet(f"{SCR}/pq/{k}.parquet", columns=cols)
R = {}
num = lambda c: pl.col(c).str.extract(r"-(\d+)$",1).cast(pl.Int64)
# --- numeric-part overlap across sources (train)
n1 = pq("train_source1",["entity_id"]).select(num("entity_id").alias("n"), pl.int_range(pl.len()).alias("row"), pl.col("entity_id"))
n2 = pq("train_source2",["entity_id"]).select(num("entity_id").alias("n"), pl.int_range(pl.len()).alias("row"), pl.col("entity_id"))
n3 = pq("train_source3",["entity_id"]).select(num("entity_id").alias("n"), pl.int_range(pl.len()).alias("row"), pl.col("entity_id"))
R["train_numeric_overlap_S1_S2"] = n1.join(n2, on="n").height
R["train_numeric_overlap_S1_S3"] = n1.join(n3, on="n").height
R["train_numeric_overlap_S2_S3"] = n2.join(n3, on="n").height
# expected overlap if uniform random in [1, max]
mx = max(n1["n"].max(), n2["n"].max(), n3["n"].max())
R["train_numeric_max"] = int(mx)
R["expected_random_overlap_S2_S3"] = float(n2.height*n3.height/mx)
# --- test/train ID overlap
t1 = pq("test_source1",["entity_id"]); t2 = pq("test_source2",["entity_id"]); t3 = pq("test_source3",["entity_id"])
R["id_overlap_train_test_S1"] = n1.join(t1, on="entity_id").height
R["id_overlap_train_test_S2"] = n2.join(t2, on="entity_id").height
R["id_overlap_train_test_S3"] = n3.join(t3, on="entity_id").height
# --- does numeric id / row order leak the match?  (train)
pairs = pl.read_parquet(f"{SCR}/pq/train_pairs_pos.parquet")
p = pairs.join(n1.rename({"entity_id":"source1_entity_id","n":"n1","row":"row1"}), on="source1_entity_id")
n23 = pl.concat([n2, n3]).rename({"entity_id":"matched_id","n":"nm","row":"rowm"})
p = p.join(n23, on="matched_id")
for src in ["S2","S3"]:
    q = p.filter(pl.col("src")==src)
    a = q["n1"].to_numpy().astype(float); b = q["nm"].to_numpy().astype(float)
    r1 = q["row1"].to_numpy().astype(float); rm = q["rowm"].to_numpy().astype(float)
    from scipy.stats import spearmanr
    R[f"{src}_spearman_numericID_matched_vs_S1"] = float(spearmanr(a, b).correlation)
    R[f"{src}_spearman_rowpos_matched_vs_S1"] = float(spearmanr(r1, rm).correlation)
    R[f"{src}_pct_matched_with_same_numeric_id"] = float((a==b).mean()*100)
    R[f"{src}_mean_abs_numeric_diff_matched"] = float(np.abs(a-b).mean())
    rng = np.random.default_rng(0); bb = rng.permutation(b)
    R[f"{src}_mean_abs_numeric_diff_random"] = float(np.abs(a-bb).mean())
    # row-position: are matches for an S1 located near each other in S2 file?  (relative position)
    rel1 = r1/ n1.height; relm = rm/ (n2.height if src=="S2" else n3.height)
    R[f"{src}_mean_abs_relpos_diff_matched"] = float(np.abs(rel1-relm).mean())
    R[f"{src}_mean_abs_relpos_diff_random"] = float(np.abs(rel1-rng.permutation(relm)).mean())
# --- are multiple matches of the same S1 adjacent in the S2/S3 file? (row gap between sibling matches)
for src in ["S2","S3"]:
    q = p.filter(pl.col("src")==src).sort("source1_entity_id","rowm")
    q = q.with_columns((pl.col("rowm")-pl.col("rowm").shift(1)).over("source1_entity_id").alias("gap")).drop_nulls("gap")
    R[f"{src}_sibling_row_gap_median"] = float(q["gap"].median()) if q.height else None
    R[f"{src}_sibling_row_gap_pct_le_5"] = float((q["gap"]<=5).mean()*100) if q.height else None
# --- GT row order vs S1 file order
gt = pq("train_ground_truth",["source1_entity_id"]).with_row_index("gt_row")
o = gt.join(n1.rename({"entity_id":"source1_entity_id"}), on="source1_entity_id")
from scipy.stats import spearmanr
R["gt_row_vs_s1_row_spearman"] = float(spearmanr(o["gt_row"].to_numpy(), o["row"].to_numpy()).correlation)
# --- id digit-length vs source, do lengths differ (could leak?) and is ID numeric distribution uniform?
for k, d in [("S1",n1),("S2",n2),("S3",n3)]:
    h = np.histogram(d["n"].to_numpy(), bins=10, range=(0, mx))[0]
    R[f"{k}_numeric_hist10"] = [int(x) for x in h]
json.dump(R, open(f"{SCR}/out/s2_hidden_ids.json","w"), indent=1)
for k,v in R.items(): print(k, "=>", v)
