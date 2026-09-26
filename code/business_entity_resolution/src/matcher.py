"""Pairwise matching models on the feature matrix, and the entity-level decision layer.

The feature parts of a split (``features.py``) are bucketed by Source 1 entity (q_idx % n_buckets), so entity-level
splits are made by bucket: the training buckets feed a negative-subsampled sample, the validation buckets are scored in
full, streamed part by part.  Pair probabilities become per-entity match lists through a decision rule (threshold,
pool-side uniqueness, relative gate, cap) whose parameters are chosen on one half of the validation entities and
reported on the other half with the official macro F0.5 evaluator (``evaluation.macro_f05_from_rows``).
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

import evaluation as ev

KEY_COLS = ("q_idx", "p_idx", "label")


# --------------------------------------------------------------------------- #
# Parts and matrices
# --------------------------------------------------------------------------- #
def list_parts(feat_dir: str) -> List[Tuple[int, str]]:
    out = []
    for f in sorted(os.listdir(feat_dir)):
        if f.startswith("part_") and f.endswith(".parquet"):
            out.append((int(f[5:7]), os.path.join(feat_dir, f)))
    return out


def to_matrix(df: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    """float32 matrix with NaN for nulls (LightGBM / XGBoost read NaN as missing)."""
    return df.select([pl.col(c).cast(pl.Float32) for c in features]).to_numpy().astype(np.float32, copy=False)


def sample_training(parts: Sequence[str], features: Sequence[str], neg_rate: float, seed: int = 0, log: Callable = print) -> Tuple[np.ndarray, np.ndarray, dict]:
    """All positives plus a uniform ``neg_rate`` share of the negatives of the given parts."""
    xs, ys, n_pos = [], [], 0
    n_neg_total = n_neg_kept = 0
    rng = np.random.default_rng(seed)
    for i, p in enumerate(parts):
        df = pl.read_parquet(p, columns=[*KEY_COLS, *features])
        lab = df["label"].to_numpy()
        keep = (lab == 1) | (rng.random(len(lab)) < neg_rate)
        n_neg_total += int((lab == 0).sum())
        n_neg_kept += int(((lab == 0) & keep).sum())
        n_pos += int((lab == 1).sum())
        sub = df.filter(pl.Series(keep))
        xs.append(to_matrix(sub, features))
        ys.append(sub["label"].to_numpy().astype(np.int8))
        del df, sub
        gc.collect()
        if (i + 1) % 6 == 0:
            log(f"    sampled {i + 1}/{len(parts)} parts: {sum(len(y) for y in ys):,} rows")
    X, y = np.vstack(xs), np.concatenate(ys)
    del xs, ys
    gc.collect()
    info = {"rows": int(len(y)), "positives": int(n_pos), "negatives_kept": int(n_neg_kept), "negatives_total": int(n_neg_total), "neg_rate": neg_rate,
            "effective_neg_rate": n_neg_kept / max(n_neg_total, 1)}
    return X, y, info


def prior_correct(p: np.ndarray, neg_rate: float) -> np.ndarray:
    """Undo the negative subsampling: a model trained with negatives kept at rate r over-estimates the odds by 1/r."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    odds = p / (1 - p) * neg_rate
    return (odds / (1 + odds)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    info: dict = field(default_factory=dict)
    save: Optional[Callable[[str], None]] = None      # writes the fitted model to a path


def fit_logistic(X: np.ndarray, y: np.ndarray, features: Sequence[str], max_rows: int = 1_500_000, seed: int = 0, C: float = 1.0) -> Model:
    """Standardised logistic regression; nulls are filled with 0 (no evidence), the missingness flags carry the rest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(max_rows, len(y)), replace=False) if len(y) > max_rows else np.arange(len(y))
    Xs = np.nan_to_num(X[idx], nan=0.0)
    scaler = StandardScaler().fit(Xs)
    clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
    t0 = time.time()
    clf.fit(scaler.transform(Xs), y[idx])
    coef = clf.coef_[0]
    order = np.argsort(-np.abs(coef))
    info = {"rows": int(len(idx)), "seconds": round(time.time() - t0, 1), "iterations": int(clf.n_iter_[0]),
            "top_coefficients": [(features[i], round(float(coef[i]), 4)) for i in order[:25]]}
    del Xs

    def predict(M: np.ndarray) -> np.ndarray:
        return clf.predict_proba(scaler.transform(np.nan_to_num(M, nan=0.0)))[:, 1].astype(np.float32)

    def save(path: str) -> None:
        import joblib
        joblib.dump({"scaler": scaler, "clf": clf, "features": list(features)}, path)

    return Model("logistic_regression", predict, info, save)


def fit_lightgbm(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, features: Sequence[str], n_threads: int = 12, seed: int = 0,
                 params: Optional[dict] = None, max_rounds: int = 3000, log: Callable = print) -> Model:
    import lightgbm as lgb
    p = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": n_threads, "seed": seed, "verbose": -1, "metric": ["binary_logloss", "auc"]}
    p.update(params or {})
    t0 = time.time()
    dtrain = lgb.Dataset(X, label=y, feature_name=list(features), free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    booster = lgb.train(p, dtrain, num_boost_round=max_rounds, valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)
    info = {"rows": int(len(y)), "seconds": round(time.time() - t0, 1), "best_iteration": int(booster.best_iteration), "params": {k: v for k, v in p.items() if k != "metric"},
            "val_logloss": float(booster.best_score["val"]["binary_logloss"]), "val_auc": float(booster.best_score["val"]["auc"]),
            "top_gain": [(features[i], round(float(gain[i] / gain.sum()), 4)) for i in order[:25]]}
    log(f"    lightgbm: best iteration {booster.best_iteration}, val logloss {info['val_logloss']:.4f}, auc {info['val_auc']:.4f} [{info['seconds']:.0f}s]")
    del dtrain, dval

    def predict(M: np.ndarray) -> np.ndarray:
        return booster.predict(M, num_iteration=booster.best_iteration).astype(np.float32)

    return Model("lightgbm", predict, info, lambda path: booster.save_model(path, num_iteration=booster.best_iteration))


def fit_xgboost(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, features: Sequence[str], n_threads: int = 12, seed: int = 0,
                params: Optional[dict] = None, max_rounds: int = 3000, log: Callable = print) -> Model:
    import xgboost as xgb
    p = {"objective": "binary:logistic", "eta": 0.05, "max_depth": 8, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
         "tree_method": "hist", "nthread": n_threads, "seed": seed, "eval_metric": ["logloss", "auc"]}
    p.update(params or {})
    t0 = time.time()
    dtrain = xgb.QuantileDMatrix(X, label=y, feature_names=list(features), nthread=n_threads)
    dval = xgb.QuantileDMatrix(X_val, label=y_val, feature_names=list(features), ref=dtrain, nthread=n_threads)
    booster = xgb.train(p, dtrain, num_boost_round=max_rounds, evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=200)
    score = booster.get_score(importance_type="total_gain")
    tot = sum(score.values()) or 1.0
    top = sorted(score.items(), key=lambda kv: -kv[1])[:25]
    info = {"rows": int(len(y)), "seconds": round(time.time() - t0, 1), "best_iteration": int(booster.best_iteration), "params": {k: v for k, v in p.items() if k != "eval_metric"},
            "top_gain": [(k, round(v / tot, 4)) for k, v in top]}
    log(f"    xgboost: best iteration {booster.best_iteration} [{info['seconds']:.0f}s]")
    del dtrain, dval

    def predict(M: np.ndarray) -> np.ndarray:
        return booster.predict(xgb.DMatrix(M, feature_names=list(features), nthread=n_threads), iteration_range=(0, booster.best_iteration + 1)).astype(np.float32)

    return Model("xgboost", predict, info, lambda path: booster.save_model(path))


def score_parts(models: Sequence[Model], parts: Sequence[str], features: Sequence[str], neg_rate: float, log: Callable = print) -> pl.DataFrame:
    """Probabilities of every model for every pair of the given parts (prior-corrected for the negative subsampling)."""
    out = []
    for i, p in enumerate(parts):
        df = pl.read_parquet(p, columns=[*KEY_COLS, *features])
        M = to_matrix(df, features)
        cols = {k: df[k] for k in KEY_COLS}
        for m in models:
            cols[m.name] = pl.Series(m.name, prior_correct(m.predict(M), neg_rate))
        out.append(pl.DataFrame(cols))
        del df, M
        gc.collect()
        log(f"    scored part {i + 1}/{len(parts)}")
    return pl.concat(out)


# --------------------------------------------------------------------------- #
# Decision layer: probabilities -> per-entity match lists
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rule:
    threshold: float
    unique_pool: bool = False      # a pool record goes to the entity that scores it highest
    rel_gate: float = 0.0          # keep a pair only if p >= rel_gate * max p of its entity
    cap: int = 0                   # at most this many matches per entity (0 = no cap)

    def label(self) -> str:
        s = f"p>={self.threshold:.2f}"
        if self.unique_pool:
            s += " +unique"
        if self.rel_gate:
            s += f" +rel{self.rel_gate:.1f}"
        if self.cap:
            s += f" +cap{self.cap}"
        return s


def decide(scores: pl.DataFrame, col: str, rule: Rule) -> pl.DataFrame:
    """The pairs kept by a rule: the input columns plus ``p`` (the score used)."""
    df = scores.with_columns(pl.col(col).alias("p"))
    if rule.unique_pool:
        df = df.filter(pl.col("p") == pl.col("p").max().over("p_idx")).unique(subset=["p_idx"], keep="first", maintain_order=True)
    df = df.filter(pl.col("p") >= rule.threshold)
    if rule.rel_gate:
        df = df.filter(pl.col("p") >= rule.rel_gate * pl.col("p").max().over("q_idx"))
    if rule.cap:
        df = df.filter(pl.col("p").rank(method="ordinal", descending=True).over("q_idx") <= rule.cap)
    return df


def rule_grid(thresholds: Iterable[float], uniques: Iterable[bool] = (False, True), gates: Iterable[float] = (0.0,), caps: Iterable[int] = (0,)) -> List[Rule]:
    return [Rule(t, u, g, c) for u in uniques for g in gates for c in caps for t in thresholds]


# --------------------------------------------------------------------------- #
# Entity-level evaluation
# --------------------------------------------------------------------------- #
class IdMaps:
    """q_idx -> Source 1 id, p_idx -> pool id, for the pairs scored."""

    def __init__(self, s1_ids: Sequence[str], pool_ids: pl.Series):
        self.s1 = pl.Series("entity_id", list(s1_ids))
        self.pool = pool_ids

    def attach(self, pairs: pl.DataFrame) -> pl.DataFrame:
        return pairs.with_columns(self.s1.gather(pairs["q_idx"].to_numpy()).alias("entity_id"), self.pool.gather(pairs["p_idx"].to_numpy()).alias("matched_id"))


def evaluate(pairs: pl.DataFrame, truth: Dict[str, List[str]], countries: Dict[str, str], pool_src: Callable[[str], str]) -> dict:
    """Official macro F0.5 of a prediction table (entity_id, matched_id) against ``truth`` (every entity of the split), with
    the breakdowns the stage asks for: singletons, one- and multi-match entities, sources, countries, false merges."""
    grouped = pairs.group_by("entity_id").agg(pl.col("matched_id")) if pairs.height else pl.DataFrame({"entity_id": [], "matched_id": []})
    res = ev.macro_f05_from_rows(truth, grouped.iter_rows(), keep_per_entity=True)
    per = res.per_entity
    tp = sum(s.tp for s in per); fp = sum(s.fp for s in per); fn = sum(s.fn for s in per)
    by_k: Dict[str, List[float]] = {}
    for s in per:
        k = "0 (singleton)" if s.n_true == 0 else "1" if s.n_true == 1 else "2" if s.n_true == 2 else "3-5" if s.n_true <= 5 else "6+"
        by_k.setdefault(k, []).append(s.f05)
    by_country: Dict[str, List[float]] = {}
    for s in per:
        by_country.setdefault(countries.get(s.entity_id, "?"), []).append(s.f05)
    singles = [s for s in per if s.n_true == 0]
    multi = [s for s in per if s.n_true > 1]
    one = [s for s in per if s.n_true == 1]
    # per-source view: truth and predictions restricted to one source, scored over the entities that have a true match there
    by_src = {}
    for src in ("S2", "S3"):
        t_src = {e: [m for m in v if pool_src(m) == src] for e, v in truth.items()}
        t_src = {e: v for e, v in t_src.items() if v}
        p_src = pairs.filter(pl.col("matched_id").str.slice(0, 2) == src) if pairs.height else pairs
        g = p_src.group_by("entity_id").agg(pl.col("matched_id")) if p_src.height else pl.DataFrame({"entity_id": [], "matched_id": []})
        r = ev.macro_f05_from_rows(t_src, g.iter_rows(), keep_per_entity=True)
        tp_s = sum(s.tp for s in r.per_entity); fp_s = sum(s.fp for s in r.per_entity); fn_s = sum(s.fn for s in r.per_entity)
        by_src[src] = {"entities_with_truth": r.n_entities, "macro_f05": round(r.macro_f05, 5), "pair_precision": round(tp_s / max(tp_s + fp_s, 1), 5),
                       "pair_recall": round(tp_s / max(tp_s + fn_s, 1), 5)}
    return {"macro_f05": round(res.macro_f05, 5), "n_entities": res.n_entities, "n_singletons": res.n_singletons,
            "pairs_predicted": int(pairs.height), "tp": tp, "fp": fp, "fn": fn,
            "pair_precision": round(tp / max(tp + fp, 1), 5), "pair_recall": round(tp / max(tp + fn, 1), 5),
            "pair_f05": round(ev.f_beta(tp / max(tp + fp, 1), tp / max(tp + fn, 1)), 5) if tp else 0.0,
            "singletons": {"n": len(singles), "predicted_empty_share": round(sum(1 for s in singles if s.n_pred == 0) / max(len(singles), 1), 5),
                           "false_positive_entities": sum(1 for s in singles if s.n_pred > 0), "mean_f05": round(float(np.mean([s.f05 for s in singles])) if singles else 0.0, 5)},
            "one_match": {"n": len(one), "mean_f05": round(float(np.mean([s.f05 for s in one])) if one else 0.0, 5),
                          "fully_correct_share": round(sum(1 for s in one if s.f05 == 1.0) / max(len(one), 1), 5)},
            "multi_match": {"n": len(multi), "mean_f05": round(float(np.mean([s.f05 for s in multi])) if multi else 0.0, 5),
                            "fully_correct_share": round(sum(1 for s in multi if s.f05 == 1.0) / max(len(multi), 1), 5),
                            "mean_true": round(float(np.mean([s.n_true for s in multi])) if multi else 0.0, 3), "mean_pred": round(float(np.mean([s.n_pred for s in multi])) if multi else 0.0, 3)},
            "by_n_true": {k: {"n": len(v), "mean_f05": round(float(np.mean(v)), 5)} for k, v in sorted(by_k.items())},
            "by_country": {c: {"n": len(v), "mean_f05": round(float(np.mean(v)), 5)} for c, v in sorted(by_country.items())},
            "by_source": by_src,
            "false_merges": {"entities_with_fp": sum(1 for s in per if s.fp > 0), "fp_pairs": fp,
                             "entities_only_fp": sum(1 for s in per if s.fp > 0 and s.tp == 0 and s.n_true > 0),
                             "entities_with_fn": sum(1 for s in per if s.fn > 0), "fully_correct": sum(1 for s in per if s.f05 == 1.0)}}


def macro_only(pairs: pl.DataFrame, truth: Dict[str, List[str]]) -> float:
    """Just the official macro F0.5 of a prediction table (entity_id, matched_id): the sweep's objective."""
    grouped = pairs.group_by("entity_id").agg(pl.col("matched_id")) if pairs.height else pl.DataFrame({"entity_id": [], "matched_id": []})
    return ev.macro_f05_from_rows(truth, grouped.iter_rows()).macro_f05


def false_merge_table(pairs: pl.DataFrame, positives: pl.DataFrame) -> pl.DataFrame:
    """The predicted pairs that are wrong, with the pool record's true owner when it has one anywhere in the training
    truth (``positives``: source1_entity_id, matched_id) -> kind 'stolen from another entity' or 'ownerless'."""
    pos = positives.select(pl.col("source1_entity_id").alias("entity_id"), "matched_id").with_columns(pl.lit(True).alias("is_true"))
    fp = pairs.join(pos, on=["entity_id", "matched_id"], how="left").filter(pl.col("is_true").is_null()).drop("is_true")
    owners = positives.select("matched_id", pl.col("source1_entity_id").alias("true_owner")).unique(subset=["matched_id"])
    return fp.join(owners, on="matched_id", how="left").with_columns(
        pl.when(pl.col("true_owner").is_null()).then(pl.lit("ownerless")).otherwise(pl.lit("stolen from another entity")).alias("kind"))
