"""Turn scored candidate pairs into matching_results.tsv.

Reads the pred_XX.parquet parts written by ``features.py --model`` (q_idx, p_idx, prior-corrected probability), applies
the decision rule chosen on validation (global threshold, then pool-side uniqueness: a Source 2/3 record goes to the
Source 1 entity that scores it highest), maps positions back to ids and writes the file in the specification's layout
(one row per Source 1 entity in file order, ids sorted and comma-joined, empty for singletons, LF endings).

Usage (from the project root):
    python code/business_entity_resolution/src/submission.py --pred-dir artifacts/scores/test --pre-dir artifacts/preprocessed/test \
        --prefix test --out output/matching_results.tsv --threshold 0.75 [--no-unique]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates as C  # noqa: E402


def load_predictions(pred_dir: str) -> pl.DataFrame:
    parts = sorted(f for f in os.listdir(pred_dir) if f.startswith("pred_") and f.endswith(".parquet"))
    if not parts:
        raise FileNotFoundError(f"no pred_XX.parquet in {pred_dir}")
    return pl.concat([pl.read_parquet(os.path.join(pred_dir, f)) for f in parts])


def decide(pred: pl.DataFrame, threshold: float, unique_pool: bool = True) -> pl.DataFrame:
    kept = pred.filter(pl.col("p") >= threshold)
    if unique_pool:
        kept = kept.filter(pl.col("p") == pl.col("p").max().over("p_idx")).unique(subset=["p_idx"], keep="first", maintain_order=True)
    return kept.select("q_idx", "p_idx")


def write_matching_results(kept: pl.DataFrame, pre_dir: str, prefix: str, out_path: str, entity_ids: Optional[Sequence[str]] = None) -> dict:
    s1 = pl.read_parquet(os.path.join(pre_dir, f"{prefix}_source1.parquet"), columns=["entity_id"])
    if entity_ids is not None:
        s1 = s1.join(pl.DataFrame({"entity_id": list(entity_ids)}), on="entity_id", how="inner")
    s1_ids = s1["entity_id"].to_list()
    pool_ids = pl.concat([pl.read_parquet(os.path.join(pre_dir, f"{prefix}_source{k}.parquet"), columns=["entity_id"]) for k in (2, 3)])["entity_id"].to_list()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    C.write_candidate_pairs(kept, s1_ids, pool_ids, out_path, id_column="matched_entity_ids")
    counts = np.bincount(kept["q_idx"].to_numpy(), minlength=len(s1_ids)) if kept.height else np.zeros(len(s1_ids), dtype=np.int64)
    n_s2 = pl.scan_parquet(os.path.join(pre_dir, f"{prefix}_source2.parquet")).select(pl.len()).collect().item()
    return {"path": out_path, "entities": len(s1_ids), "pairs": int(kept.height), "entities_with_matches": int((counts > 0).sum()),
            "singleton_share": round(float((counts == 0).mean()), 5), "mean_matches": round(float(counts.mean()), 4), "median_matches": float(np.median(counts)),
            "max_matches": int(counts.max()), "matches_s2": int((kept["p_idx"].to_numpy() < n_s2).sum()), "matches_s3": int((kept["p_idx"].to_numpy() >= n_s2).sum()),
            "bytes": os.path.getsize(out_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--pre-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--no-unique", action="store_true")
    p.add_argument("--entity-ids", default=None)
    p.add_argument("--stats", default=None, help="write the run statistics as json")
    a = p.parse_args(argv)
    t0 = time.time()
    ids = None
    if a.entity_ids:
        with open(a.entity_ids, encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    pred = load_predictions(a.pred_dir)
    print(f"[{time.time() - t0:4.0f}s] {pred.height:,} scored pairs; mean p {pred['p'].mean():.4f}, share >= {a.threshold}: {float((pred['p'] >= a.threshold).mean()):.4f}", flush=True)
    kept = decide(pred, a.threshold, not a.no_unique)
    del pred
    stats = write_matching_results(kept, a.pre_dir, a.prefix, a.out, ids)
    stats.update({"threshold": a.threshold, "unique_pool": not a.no_unique, "seconds": round(time.time() - t0, 1)})
    print(f"[{time.time() - t0:4.0f}s] wrote {a.out}: {stats}", flush=True)
    if a.stats:
        json.dump(stats, open(a.stats, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
