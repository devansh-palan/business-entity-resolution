"""First pairwise matching models on the fold-0 feature matrix, evaluated at the entity level with the official evaluator.

Entity split by feature bucket (q_idx % 32): buckets with b % 4 != 3 train (24 buckets), b % 8 == 3 tune the decision
rule (half A: 4 buckets), b % 8 == 7 report (half B: 4 buckets).  Models: logistic regression (interpretable baseline),
LightGBM, XGBoost.  Every model's probabilities are turned into per-entity match lists by the rule that maximises macro
F0.5 on half A, and reported on half B: overall, singletons, one- and multi-match entities, per source, per country,
false merges.  Run from the project root (alone):  python -u eda/scripts/s18_matcher.py     [S18_RESUME=1 skips training]
"""
import gc
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import matcher as M  # noqa: E402
import validation as V  # noqa: E402

FEAT_DIR = os.environ.get("S18_FEAT_DIR", "artifacts/features/fold0")
VAL_DIR, PRE_DIR = "artifacts/validation", "artifacts/preprocessed/fold0"
MODEL_DIR = os.environ.get("S18_MODEL_DIR", "artifacts/models/fold0")
OUT_JSON = os.environ.get("S18_OUT", "eda/results/s18_matcher.json")
ONLY = os.environ.get("S18_MODELS", "logistic_regression,lightgbm,xgboost").split(",")     # subset of the models to fit
NEG_RATE = 0.065
N_THREADS = 12
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
os.makedirs(MODEL_DIR, exist_ok=True)
RES = {"neg_rate": NEG_RATE}

meta = json.load(open(os.path.join(FEAT_DIR, "features_meta.json")))
FEATS = [f for f in meta["features"] if f not in ("country_eq",)]        # constant within the candidate set
parts = M.list_parts(FEAT_DIR)
train_parts = [p for b, p in parts if b % 4 != 3]
tune_parts = [p for b, p in parts if b % 8 == 3]
report_parts = [p for b, p in parts if b % 8 == 7]
log(f"{len(FEATS)} features; parts: train {len(train_parts)}, tune {len(tune_parts)}, report {len(report_parts)}")

# ids, truth, countries
s1 = pl.read_parquet(os.path.join(FEAT_DIR, "_pool_stats", "s1.parquet"), columns=["q_idx", "entity_id", "country"])
S1_IDS = s1["entity_id"].to_list()
COUNTRY = dict(zip(S1_IDS, s1["country"].to_list()))
pool_ids = pl.concat([pl.read_parquet(os.path.join(PRE_DIR, f"train_source{k}.parquet"), columns=["entity_id"]) for k in (2, 3)])["entity_id"]
assert pool_ids[0].startswith("S2-") and pool_ids[-1].startswith("S3-")
ids = M.IdMaps(S1_IDS, pool_ids)
truth_all = V.entity_truth(VAL_DIR, folds=[0])
q_of = {e: i for i, e in enumerate(S1_IDS)}
truth_A = {e: v for e, v in truth_all.items() if q_of[e] % 8 == 3}
truth_B = {e: v for e, v in truth_all.items() if q_of[e] % 8 == 7}
positives = V.load_positive_pairs(VAL_DIR).select("source1_entity_id", "matched_id")
pool_src = lambda m: m[:2]
log(f"entities: fold 0 {len(truth_all):,}, tune half {len(truth_A):,}, report half {len(truth_B):,}; singletons in report half "
    f"{sum(1 for v in truth_B.values() if not v):,}")
RES["split"] = {"train_buckets": [b for b, _ in parts if b % 4 != 3], "tune_buckets": [b for b, _ in parts if b % 8 == 3], "report_buckets": [b for b, _ in parts if b % 8 == 7],
                "entities_tune": len(truth_A), "entities_report": len(truth_B), "singletons_report": sum(1 for v in truth_B.values() if not v)}

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
scores_path = os.path.join(MODEL_DIR, "val_scores.parquet")
if os.environ.get("S18_RESUME") == "1" and os.path.exists(scores_path):
    scores = pl.read_parquet(scores_path)
    RES["models"] = json.load(open(os.path.join(MODEL_DIR, "models_info.json")))
    log(f"resumed: {scores.height:,} scored pairs")
else:
    X, y, info = M.sample_training(train_parts, FEATS, NEG_RATE, log=log)
    RES["train_sample"] = info
    log(f"training sample: {info['rows']:,} rows, {info['positives']:,} positives, negatives kept {info['negatives_kept']:,} of {info['negatives_total']:,}")
    Xv, yv, _ = M.sample_training(tune_parts[:2], FEATS, NEG_RATE, seed=1, log=lambda *a: None)
    log(f"early-stopping sample: {len(yv):,} rows")
    models, minfo = [], {}
    if "logistic_regression" in ONLY:
        m = M.fit_logistic(X, y, FEATS)
        log(f"  logistic regression: {m.info['rows']:,} rows, {m.info['iterations']} iterations [{m.info['seconds']:.0f}s]; top |coef|: " + ", ".join(f"{f} {c:+.2f}" for f, c in m.info["top_coefficients"][:8]))
        models.append(m); minfo[m.name] = m.info
        m.save(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    if "lightgbm" in ONLY:
        m = M.fit_lightgbm(X, y, Xv, yv, FEATS, n_threads=N_THREADS, log=log)
        models.append(m); minfo[m.name] = m.info
        m.save(os.path.join(MODEL_DIR, "lightgbm.txt"))
    if "xgboost" in ONLY:
        m = M.fit_xgboost(X, y, Xv, yv, FEATS, n_threads=N_THREADS, log=log)
        models.append(m); minfo[m.name] = m.info
        m.save(os.path.join(MODEL_DIR, "xgboost.json"))
    del X, y, Xv, yv
    gc.collect()
    RES["models"] = minfo
    json.dump(minfo, open(os.path.join(MODEL_DIR, "models_info.json"), "w"), indent=1)
    log("scoring the validation halves")
    scores = M.score_parts(models, tune_parts + report_parts, FEATS, NEG_RATE, log=log)
    scores.write_parquet(scores_path)
    del models
    gc.collect()
MODELS = [c for c in scores.columns if c not in M.KEY_COLS]
half = (scores["q_idx"].to_numpy() % 8)
scores_A = scores.filter(pl.Series(half == 3))
scores_B = scores.filter(pl.Series(half == 7))
del scores
gc.collect()
log(f"scores: tune half {scores_A.height:,} pairs, report half {scores_B.height:,} pairs; models {MODELS}")


# --------------------------------------------------------------------------- #
# Pair-level metrics (report half), for context only
# --------------------------------------------------------------------------- #
def auc_ap(p: np.ndarray, y: np.ndarray):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


yB = scores_B["label"].to_numpy()
RES["pair_level_report_half"] = {}
for mname in MODELS:
    a, ap = auc_ap(scores_B[mname].to_numpy(), yB)
    RES["pair_level_report_half"][mname] = {"auc": round(a, 5), "average_precision": round(ap, 5)}
    log(f"  pair-level {mname}: AUC {a:.4f}, AP {ap:.4f}")

# --------------------------------------------------------------------------- #
# Decision rules: sweep on half A, report on half B
# --------------------------------------------------------------------------- #
GRID = M.rule_grid(thresholds=[round(t, 2) for t in np.arange(0.25, 0.96, 0.05)], uniques=(False, True), gates=(0.0, 0.5, 0.8), caps=(0,))
RES["sweep"], RES["chosen"], RES["report"], RES["report_on_tune_half"] = {}, {}, {}, {}
pairs_A = ids.attach(scores_A.select("q_idx", "p_idx", *MODELS))
pairs_B = ids.attach(scores_B.select("q_idx", "p_idx", *MODELS))
for mname in MODELS:
    t0 = time.time()
    rows = []
    for rule in GRID:
        kept = M.decide(pairs_A.select("q_idx", "p_idx", "entity_id", "matched_id", mname), mname, rule)
        f = M.macro_only(kept, truth_A)
        rows.append({"rule": rule.label(), "threshold": rule.threshold, "unique_pool": rule.unique_pool, "rel_gate": rule.rel_gate, "macro_f05_tune": round(f, 5), "pairs": int(kept.height)})
    rows.sort(key=lambda r: -r["macro_f05_tune"])
    RES["sweep"][mname] = rows
    best = rows[0]
    rule = M.Rule(best["threshold"], best["unique_pool"], best["rel_gate"], 0)
    RES["chosen"][mname] = {"rule": rule.label(), **best}
    # the best plain threshold and the best threshold + uniqueness, for the story
    RES["chosen"][mname]["best_plain_threshold"] = max((r for r in rows if not r["unique_pool"] and not r["rel_gate"]), key=lambda r: r["macro_f05_tune"])
    RES["chosen"][mname]["best_unique_only"] = max((r for r in rows if r["unique_pool"] and not r["rel_gate"]), key=lambda r: r["macro_f05_tune"])
    log(f"  {mname}: best rule on the tune half {rule.label()} -> {best['macro_f05_tune']:.4f} ({len(GRID)} rules in {time.time() - t0:.0f}s)")
    for tag, pairs_h, truth_h in (("report", pairs_B, truth_B), ("report_on_tune_half", pairs_A, truth_A)):
        kept = M.decide(pairs_h.select("q_idx", "p_idx", "entity_id", "matched_id", mname), mname, rule)
        RES[tag][mname] = M.evaluate(kept.select("entity_id", "matched_id"), truth_h, COUNTRY, pool_src)
    r = RES["report"][mname]
    log(f"  {mname} on the report half: macro F0.5 {r['macro_f05']:.4f} | singletons empty {r['singletons']['predicted_empty_share']:.3f} | one-match {r['one_match']['mean_f05']:.4f} "
        f"| multi-match {r['multi_match']['mean_f05']:.4f} | S2 {r['by_source']['S2']['macro_f05']:.4f} S3 {r['by_source']['S3']['macro_f05']:.4f} | "
        + " ".join(f"{c} {v['mean_f05']:.4f}" for c, v in r["by_country"].items()) + f" | entities with a false merge {r['false_merges']['entities_with_fp']:,}")

# also: the empty prediction and the deterministic automatic tier for reference
RES["reference"] = {"empty_prediction": round(sum(1 for v in truth_B.values() if not v) / len(truth_B), 5)}

# --------------------------------------------------------------------------- #
# False merges of the best model (report half)
# --------------------------------------------------------------------------- #
best_model = max(MODELS, key=lambda m: RES["report"][m]["macro_f05"])
RES["best_model"] = best_model
rule = M.Rule(RES["chosen"][best_model]["threshold"], RES["chosen"][best_model]["unique_pool"], RES["chosen"][best_model]["rel_gate"], 0)
kept = M.decide(pairs_B.select("q_idx", "p_idx", "entity_id", "matched_id", best_model), best_model, rule)
fm = M.false_merge_table(kept.select("q_idx", "p_idx", "entity_id", "matched_id", "p"), positives)
log(f"false merges of {best_model}: {fm.height:,} pairs; kinds {fm['kind'].value_counts().to_dicts()}")
# characterise them with the pair features
FM_COLS = ["name_exact_core", "name_lev_sim", "addr_exact_norm", "addr_tok_jaccard", "house_eq", "house_conflict", "unit_conflict", "street_conflict", "city_eq",
           "cos_name_addr", "n_cand_q", "rank_cos_q", "addr_missing_2", "name_pool_freq_2", "addr_pool_freq_2"]
feat_fp = pl.concat([pl.read_parquet(p, columns=["q_idx", "p_idx", *FM_COLS]) for p in report_parts]).join(fm.select("q_idx", "p_idx", "kind", "p"), on=["q_idx", "p_idx"], how="inner")
prof = {}
for kind, sub in [("all", feat_fp)] + [(k, feat_fp.filter(pl.col("kind") == k)) for k in feat_fp["kind"].unique().to_list()]:
    prof[kind] = {"n": sub.height, "share_exact_name": round(float(sub["name_exact_core"].mean() or 0), 4), "share_exact_address": round(float(sub["addr_exact_norm"].mean() or 0), 4),
                  "share_house_conflict": round(float(sub["house_conflict"].mean() or 0), 4), "share_unit_conflict": round(float(sub["unit_conflict"].mean() or 0), 4),
                  "share_street_conflict": round(float(sub["street_conflict"].mean() or 0), 4), "share_pool_address_missing": round(float(sub["addr_missing_2"].mean() or 0), 4),
                  "mean_p": round(float(sub["p"].mean() or 0), 4), "mean_cos_name_addr": round(float(sub["cos_name_addr"].mean() or 0), 4),
                  "share_same_name_diff_house": round(float(((sub["name_exact_core"] == 1) & (sub["house_conflict"] == 1)).mean() or 0), 4)}
RES["false_merge_profile"] = prof
# examples with the raw strings
ex_ids = fm.sort("p", descending=True).head(400)
s1_txt = pl.read_parquet(os.path.join(PRE_DIR, "train_source1.parquet"), columns=["entity_id", "business_name", "business_address"]).filter(pl.col("entity_id").is_in(ex_ids["entity_id"].implode()))
pool_txt = pl.concat([pl.read_parquet(os.path.join(PRE_DIR, f"train_source{k}.parquet"), columns=["entity_id", "business_name", "business_address"]) for k in (2, 3)]) \
             .filter(pl.col("entity_id").is_in(ex_ids["matched_id"].implode()))
ex = (ex_ids.join(s1_txt, on="entity_id", how="left").join(pool_txt.rename({"entity_id": "matched_id", "business_name": "name_2", "business_address": "address_2"}), on="matched_id", how="left")
        .join(feat_fp.select("q_idx", "p_idx", "name_exact_core", "house_conflict", "unit_conflict"), on=["q_idx", "p_idx"], how="left"))
RES["false_merge_examples"] = {k: ex.filter(pl.col("kind") == k).head(8).select("business_name", "business_address", "name_2", "address_2", "p", "true_owner").to_dicts() for k in ex["kind"].unique().to_list()}
# missed matches of the best model: how many true pairs were never in the candidate set vs scored below the rule
truth_pairs_B = pl.DataFrame({"entity_id": [e for e, v in truth_B.items() for _ in v], "matched_id": [m for v in truth_B.values() for m in v]})
n_true = truth_pairs_B.height
in_cand = truth_pairs_B.join(pairs_B.select("entity_id", "matched_id"), on=["entity_id", "matched_id"], how="inner").height
found = truth_pairs_B.join(kept.select("entity_id", "matched_id"), on=["entity_id", "matched_id"], how="inner").height
RES["missed_matches"] = {"true_pairs": n_true, "in_candidate_set": in_cand, "predicted": found, "missed_by_candidates": n_true - in_cand, "missed_by_model": in_cand - found}
log(f"missed matches: {n_true - in_cand:,} not in the candidate set, {in_cand - found:,} scored below the rule")
json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT_JSON)
