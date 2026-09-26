"""Strategy 10 probe: would sentence-embedding retrieval recover the true pairs the union misses?

For every missed pair (and a control sample of found pairs) the Source 1 name and the true pool name are embedded
with a multilingual MiniLM (Apache-2.0, 118M parameters, runs on the 4 GB GPU); the cosine of the true pair is
compared with the cosines to a random same-country sample of pool names.  A true match is counted as
"retrievable" when it would rank within the top r of the sample, which scales to top r * (pool / sample) in the full
pool.  The probe also reports embedding throughput, from which the full-scale cost follows.
Run from the project root after s15:  python -u eda/scripts/s15b_embedding_probe.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
PRE, OUT_DIR = "artifacts/preprocessed/fold0", "artifacts/candidates/fold0"
OUT_JSON = "eda/results/s15b_embedding_probe.json"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SAMPLE_POOL = 300_000
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
RES = {"model": MODEL}

import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)
RES["device"] = device
log("model loaded on", device)

cols = ["entity_id", "country", "business_name", "name_translit_core"]
s1 = pl.read_parquet(f"{PRE}/train_source1.parquet", columns=cols)
pool = pl.concat([pl.read_parquet(f"{PRE}/train_source{k}.parquet", columns=cols) for k in (2, 3)])
misses = pl.read_parquet(os.path.join(OUT_DIR, "misses.parquet"))
cov = pl.read_parquet(os.path.join(OUT_DIR, "coverage.parquet"), columns=["q_idx", "p_idx", "in_final"])
folds = pl.read_parquet("artifacts/validation/entity_folds.parquet", columns=["entity_id", "fold"]).filter(pl.col("fold") == 0)
q_ids = s1.join(folds.select("entity_id"), on="entity_id", how="inner").select("entity_id").with_row_index("q_idx").with_columns(pl.col("q_idx").cast(pl.UInt32))
p_ids = pool.select("entity_id").with_row_index("p_idx").with_columns(pl.col("p_idx").cast(pl.UInt32))
found = (cov.filter(pl.col("in_final")).sample(n=min(5000, cov.height), seed=1)
            .join(q_ids.rename({"entity_id": "source1_entity_id"}), on="q_idx").join(p_ids.rename({"entity_id": "matched_id"}), on="p_idx")
            .select("source1_entity_id", "matched_id"))
missed = misses.select("source1_entity_id", "matched_id", "category")
log(f"missed pairs {missed.height:,}, control found pairs {found.height:,}")


def embed(texts, batch=512):
    t0 = time.time()
    e = model.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return e.astype(np.float32), len(texts) / max(time.time() - t0, 1e-6)


results = {}
for label, pairs in (("missed", missed), ("found_control", found)):
    p = pairs.join(s1.rename({c: f"{c}_1" for c in cols}), left_on="source1_entity_id", right_on="entity_id_1") \
             .join(pool.rename({c: f"{c}_2" for c in cols}), left_on="matched_id", right_on="entity_id_2")
    out = {"n": p.height}
    for text_col, tag in (("business_name", "raw name"), ("name_translit_core", "transliterated core")):
        q, rate_q = embed(p[f"{text_col}_1"].fill_null("").to_list())
        t, _ = embed(p[f"{text_col}_2"].fill_null("").to_list())
        true_cos = np.einsum("ij,ij->i", q, t)
        per_country = {}
        for country in sorted(p["country_1"].unique().to_list()):
            idx = np.where(p["country_1"].to_numpy() == country)[0]
            sample = pool.filter(pl.col("country") == country).sample(n=min(SAMPLE_POOL, pool.filter(pl.col("country") == country).height), seed=2)
            S, rate = embed(sample[text_col].fill_null("").to_list())
            n_full = pool.filter(pl.col("country") == country).height
            scale = n_full / sample.height
            ranks = np.zeros(len(idx), dtype=np.int64)
            for a in range(0, len(idx), 2048):
                block = idx[a:a + 2048]
                sims = q[block] @ S.T                                     # (b, sample)
                ranks[a:a + 2048] = (sims > true_cos[block][:, None]).sum(axis=1)   # sample names scoring above the true match
            per_country[country] = {
                "n": int(len(idx)), "embed_rate_names_per_s": round(rate, 0), "sample": int(sample.height), "scale_to_full_pool": round(scale, 1),
                "true_cosine_median": round(float(np.median(true_cos[idx])), 4),
                "retrievable_top20_full_pool": round(float((ranks * scale < 20).mean()), 4),
                "retrievable_top50_full_pool": round(float((ranks * scale < 50).mean()), 4),
                "retrievable_top200_full_pool": round(float((ranks * scale < 200).mean()), 4),
            }
        out[tag] = per_country
        log(label, tag, json.dumps(per_country))
    if label == "missed":
        p = p.with_columns(pl.Series("true_cos", true_cos))
        out["by_category_top50"] = {}
        for cat in p["category"].unique().to_list():
            sub = np.where(p["category"].to_numpy() == cat)[0]
            out["by_category_top50"][cat] = {"n": int(len(sub)), "true_cosine_median": round(float(np.median(true_cos[sub])), 4)}
    results[label] = out
RES["results"] = results
n_names = s1.height + pool.height
rate = results["missed"]["raw name"][next(iter(results["missed"]["raw name"]))]["embed_rate_names_per_s"]
RES["full_scale_cost"] = {"names_to_embed_train": n_names, "minutes_at_measured_rate": round(n_names / max(rate, 1) / 60, 1)}
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT_JSON)
