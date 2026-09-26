"""Inference mode of the feature build (scoring with a LightGBM model) and the matching_results.tsv writer."""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features as F  # noqa: E402
import submission as S  # noqa: E402
from test_features import split  # noqa: E402,F401  (fixture: a tiny preprocessed split with pairs and labels)


def test_scoring_mode_and_matching_results(split):
    tmp, remap = split
    meta = F.build(str(tmp), "train", str(tmp / "pairs.parquet"), str(tmp / "feat"), labels_path=str(tmp / "labels.parquet"), n_buckets=1, workers=1, log=lambda *a: None, in_process=True)
    import lightgbm as lgb
    df = pl.read_parquet(tmp / "feat" / "part_00.parquet")
    feats = [f for f in meta["features"] if f != "country_eq"]
    X = df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy().astype(np.float32)
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 1, "min_data_in_bin": 1}, lgb.Dataset(X, df["label"].to_numpy()), num_boost_round=5)
    booster.save_model(str(tmp / "model.txt"))
    model = {"path": str(tmp / "model.txt"), "features": feats, "neg_rate": 0.5}
    out = F.build(str(tmp), "train", str(tmp / "pairs.parquet"), str(tmp / "scores"), n_buckets=1, workers=1, log=lambda *a: None, in_process=True, model=model)
    pred = pl.read_parquet(tmp / "scores" / "pred_00.parquet")
    assert pred.columns == ["q_idx", "p_idx", "p"] and pred.height == df.height and out["buckets"][0]["scored"]
    assert float(pred["p"].min()) >= 0.0 and float(pred["p"].max()) <= 1.0 and not (tmp / "scores" / "part_00.parquet").exists()
    # matching results: threshold 0 keeps everything, uniqueness gives each pool record to one entity
    kept = S.decide(pred, 0.0, unique_pool=True)
    assert kept["p_idx"].n_unique() == kept.height
    stats = S.write_matching_results(kept, str(tmp), "train", str(tmp / "out" / "matching_results.tsv"))
    lines = open(tmp / "out" / "matching_results.tsv", encoding="utf-8", newline="").read().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids" and lines[-1] == "" and len(lines) == 3 + 2
    assert lines[1].startswith("S1-1\t") and "\r" not in lines[1] and stats["entities"] == 3 and stats["pairs"] == kept.height
    row = dict(l.split("\t") for l in lines[1:4])
    assert all(i.startswith(("S2-", "S3-")) for ids in row.values() for i in ids.split(",") if ids)
