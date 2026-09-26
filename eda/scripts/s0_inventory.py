import os
import polars as pl, json, os, gc, re, sys, time
DATA = "student_resource/dataset"
SCR  = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
files = {
 "train_source1": f"{DATA}/train/train_source1.tsv",
 "train_source2": f"{DATA}/train/train_source2.tsv",
 "train_source3": f"{DATA}/train/train_source3.tsv",
 "train_ground_truth": f"{DATA}/train/train_ground_truth.tsv",
 "test_source1": f"{DATA}/test/test_source1.tsv",
 "test_source2": f"{DATA}/test/test_source2.tsv",
 "test_source3": f"{DATA}/test/test_source3.tsv",
}
summary = {}
for key, path in files.items():
    t0 = time.time()
    enc_used = "utf8"
    try:
        df = pl.read_csv(path, separator="\t", quote_char=None, has_header=True,
                         infer_schema_length=0, encoding="utf8")
    except Exception as e:
        print(f"[{key}] strict utf8 read failed: {e!r}; retrying utf8-lossy", flush=True)
        enc_used = "utf8-lossy"
        df = pl.read_csv(path, separator="\t", quote_char=None, has_header=True,
                         infer_schema_length=0, encoding="utf8-lossy")
    info = {"path": path, "size_bytes": os.path.getsize(path), "encoding": enc_used,
            "n_rows": df.height, "n_cols": df.width, "columns": df.columns,
            "dtypes": [str(t) for t in df.dtypes], "read_seconds": round(time.time()-t0,1)}
    # null / blank counts
    nulls, blanks = {}, {}
    for c in df.columns:
        nulls[c] = int(df[c].null_count())
        blanks[c] = int(df.select((pl.col(c).is_null() | (pl.col(c).str.strip_chars() == "")).sum()).item())
    info["null_counts"] = nulls
    info["null_or_blank_counts"] = blanks
    # CR check (windows line endings) on last column
    lastc = df.columns[-1]
    info["rows_with_CR_in_last_col"] = int(df.select(pl.col(lastc).str.contains("\r").sum()).item())
    idcol = df.columns[0]
    ids = df[idcol]
    info["id_n_unique"] = int(ids.n_unique())
    info["id_prefix_counts"] = df.select(pl.col(idcol).str.slice(0,3).alias("p")).group_by("p").len().sort("p").to_dict(as_series=False)
    num = df.select(pl.col(idcol).str.extract(r"^S\d-(\d+)$",1).alias("n"))
    info["id_regex_nonmatch"] = int(num["n"].null_count())
    lens = num.select(pl.col("n").str.len_chars().alias("L")).group_by("L").len().sort("L").to_dict(as_series=False)
    info["id_numeric_len_counts"] = lens
    nn = num.select(pl.col("n").cast(pl.Int64))["n"]
    info["id_numeric_min"] = int(nn.min()); info["id_numeric_max"] = int(nn.max())
    info["id_numeric_n_unique"] = int(nn.n_unique())
    if "country" in df.columns:
        info["country_counts"] = df.group_by("country").len().sort("len", descending=True).to_dict(as_series=False)
    if key.endswith("ground_truth"):
        m = df["matched_entity_ids"]
        info["gt_empty_lists"] = int((m.is_null() | (m.str.strip_chars()=="")).sum())
        info["gt_lists_with_space"] = int(m.fill_null("").str.contains(" ").sum())
        info["gt_lists_with_quote"] = int(m.fill_null("").str.contains('"').sum())
    info["head"] = df.head(6).to_dict(as_series=False)
    df.write_parquet(f"{SCR}/pq/{key}.parquet")
    summary[key] = info
    print(f"[{key}] rows={df.height:,} cols={df.width} uniqueIDs={info['id_n_unique']:,} enc={enc_used} {info['read_seconds']}s", flush=True)
    del df, ids, num, nn; gc.collect()
json.dump(summary, open(f"{SCR}/out/s0_inventory.json","w",encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE")
