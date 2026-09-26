"""Is a global probability threshold the best decision rule?  Per-entity signals and strategies A-F, scored with the
official entity-level macro F0.5: every strategy's parameters are chosen on the tuning half and reported on the
report half, with singleton false positives tracked throughout.  Uses the LightGBM probabilities of s18.

Run from the project root:  python -u eda/scripts/s20_decision_rules.py
Writes eda/results/s20_decision_rules.json and eda/decision_rules.md.
"""
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

MODEL = "lightgbm"
T_GLOBAL = 0.75
FEAT_DIR = "artifacts/features/fold0"
PAIR_COLS = ["name_lev_sim", "name_jw", "addr_tok_jaccard", "cos_addr", "cos_name_addr", "name_exact_core", "name_exact_sorted", "name_exact_compact", "addr_exact_norm",
             "house_eq", "street_eq", "city_eq", "state_eq", "p_domain_style", "ident_x_name", "country_us"]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)

# --------------------------------------------------------------------------- #
# Pairs of the two halves with probabilities, labels and the pair signals
# --------------------------------------------------------------------------- #
sc = pl.read_parquet("artifacts/models/fold0/val_scores.parquet", columns=["q_idx", "p_idx", "label", MODEL]).rename({MODEL: "p"})
parts = [p for b, p in M.list_parts(FEAT_DIR) if b % 4 == 3]
feats = pl.concat([pl.read_parquet(p, columns=["q_idx", "p_idx", *PAIR_COLS]) for p in parts])
sc = sc.join(feats, on=["q_idx", "p_idx"], how="left")
del feats
s1 = pl.read_parquet(os.path.join(FEAT_DIR, "_pool_stats", "s1.parquet"), columns=["entity_id", "country"])
S1_IDS = s1["entity_id"].to_list()
COUNTRY = dict(zip(S1_IDS, s1["country"].to_list()))
pool_ids = pl.concat([pl.read_parquet(f"artifacts/preprocessed/fold0/train_source{k}.parquet", columns=["entity_id"]) for k in (2, 3)])["entity_id"]
ids = M.IdMaps(S1_IDS, pool_ids)
truth_all = V.entity_truth("artifacts/validation", folds=[0])
q_of = {e: i for i, e in enumerate(S1_IDS)}
sc = ids.attach(sc)
half = sc["q_idx"].to_numpy() % 8
H = {"tune": (sc.filter(pl.Series(half == 3)), {e: v for e, v in truth_all.items() if q_of[e] % 8 == 3}),
     "report": (sc.filter(pl.Series(half == 7)), {e: v for e, v in truth_all.items() if q_of[e] % 8 == 7})}
del sc
pool_src = lambda m: m[:2]
log(f"pairs: tune {H['tune'][0].height:,}, report {H['report'][0].height:,}")

# --------------------------------------------------------------------------- #
# Per-entity signals
# --------------------------------------------------------------------------- #
def entity_table(pairs: pl.DataFrame, truth: dict) -> pl.DataFrame:
    srt = pairs.sort(["q_idx", "p"], descending=[False, True])
    ent = srt.group_by("q_idx", maintain_order=True).agg(
        pl.col("entity_id").first(), pl.len().alias("n_cand"), pl.col("p").first().alias("top1"), pl.col("p").slice(1, 1).first().alias("top2"),
        pl.col("p").slice(0, 3).mean().alias("mean_top3"), pl.col("p").sum().alias("sum_p"),
        (pl.col("p") >= 0.9).sum().alias("n_ge_90"), (pl.col("p") >= 0.75).sum().alias("n_ge_75"), (pl.col("p") >= 0.5).sum().alias("n_ge_50"), (pl.col("p") >= 0.25).sum().alias("n_ge_25"),
        pl.col("name_lev_sim").max().alias("best_name_sim"), pl.col("name_jw").max().alias("best_name_jw"), pl.col("addr_tok_jaccard").max().alias("best_addr_sim"),
        pl.col("cos_addr").max().alias("best_cos_addr"), pl.col("cos_name_addr").max().alias("best_cos_name_addr"),
        pl.max_horizontal(pl.col("name_exact_core").max(), pl.col("name_exact_sorted").max(), pl.col("name_exact_compact").max()).alias("any_exact_name"),
        pl.col("addr_exact_norm").max().alias("any_exact_addr"), pl.col("house_eq").max().alias("any_house_eq"),
        ((pl.col("p_domain_style") == 1) & (pl.col("ident_x_name") >= 0.9)).any().cast(pl.Int8).alias("ident_agreement"),
        pl.col("country_us").first(), pl.col("label").sum().alias("n_true_in_cand"))
    ent = ent.with_columns(pl.col("top2").fill_null(0.0), (pl.col("top1") - pl.col("top2").fill_null(0.0)).alias("gap"))
    n_true = pl.DataFrame({"entity_id": list(truth.keys()), "n_true": [len(v) for v in truth.values()]})
    return ent.join(n_true, on="entity_id", how="left")


ENT = {h: entity_table(H[h][0], H[h][1]) for h in H}
log(f"entity tables: tune {ENT['tune'].height:,}, report {ENT['report'].height:,}")
RES = {"model": MODEL, "global_threshold": T_GLOBAL}


def q(s: pl.Series, ps=(0.1, 0.5, 0.9)):
    return [round(float(s.quantile(x)), 4) for x in ps]


SIG = ["n_cand", "top1", "top2", "gap", "n_ge_90", "n_ge_50", "best_name_sim", "best_addr_sim", "best_cos_name_addr", "any_exact_name", "any_exact_addr", "ident_agreement"]
prof = {}
for name, cond in (("singleton", pl.col("n_true") == 0), ("one match", pl.col("n_true") == 1), ("2-5 matches", (pl.col("n_true") >= 2) & (pl.col("n_true") <= 5)), ("6+ matches", pl.col("n_true") >= 6)):
    sub = ENT["report"].filter(cond)
    prof[name] = {"entities": sub.height, **{s: q(sub[s].cast(pl.Float64)) for s in SIG}}
RES["signal_profile_report"] = prof


def auc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(x)
    return float(roc_auc_score(y[m], x[m])) if 0 < y[m].sum() < m.sum() else float("nan")


rep = ENT["report"]
y_single = (rep["n_true"] == 0).to_numpy()
RES["singleton_detection_auc_report"] = {s: round(max(a, 1 - a), 4) for s in SIG for a in [auc(rep[s].cast(pl.Float64).to_numpy(), y_single)]}
log("singleton-detection AUC of the entity signals: " + ", ".join(f"{k} {v}" for k, v in sorted(RES["singleton_detection_auc_report"].items(), key=lambda kv: -kv[1])))

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def keep_threshold(pairs, t):
    return pairs.filter(pl.col("p") >= t)


def keep_topk(pairs, t, k):
    return pairs.filter((pl.col("p") >= t) & (pl.col("p").rank(method="ordinal", descending=True).over("q_idx") <= k))


def keep_gapcut(pairs, t_top, t_low, g):
    """Sorted by p, keep the run from the top while p >= t_low and the drop from the previous candidate is at most g; the top needs p >= t_top."""
    srt = pairs.sort(["q_idx", "p"], descending=[False, True])
    srt = srt.with_columns((pl.col("p").shift(1).over("q_idx") - pl.col("p")).fill_null(0.0).alias("drop"))
    srt = srt.with_columns(((pl.col("p") >= t_low) & (pl.col("drop") <= g)).alias("ok"))
    srt = srt.with_columns(pl.col("ok").cast(pl.Int8).cum_min().over("q_idx").alias("run"))       # 1 until the first failure
    return srt.filter((pl.col("run") == 1) & (pl.col("p").max().over("q_idx") >= t_top))


def auto_mask(pairs, variant):
    exact_name = (pl.col("name_exact_core") == 1) | (pl.col("name_exact_sorted") == 1) | (pl.col("name_exact_compact") == 1)
    if variant == "name+address":
        return exact_name & (pl.col("addr_exact_norm") == 1)
    if variant == "name+house+street/city":
        return exact_name & (pl.col("house_eq") == 1) & ((pl.col("street_eq") == 1) | (pl.col("city_eq") == 1))
    raise ValueError(variant)


def keep_auto_plus(pairs, variant, t):
    return pairs.filter(auto_mask(pairs, variant) | (pl.col("p") >= t))


def keep_by_entity_threshold(pairs, ent_t: pl.DataFrame):
    """ent_t: q_idx -> threshold for that entity."""
    return pairs.join(ent_t, on="q_idx", how="left").filter(pl.col("p") >= pl.col("t_ent").fill_null(T_GLOBAL))


def score(kept, truth):
    return M.macro_only(kept.select("entity_id", "matched_id"), truth)


def full(kept, truth):
    return M.evaluate(kept.select("entity_id", "matched_id"), truth, COUNTRY, pool_src)


STRAT = {}


def record(name, tune_rows, report_eval, tune_best):
    r = report_eval
    STRAT[name] = {"tune_best": tune_best, "tune_sweep": tune_rows, "report": {"macro_f05": r["macro_f05"], "singleton_score": r["singletons"]["predicted_empty_share"],
                   "singleton_false_positives": r["singletons"]["false_positive_entities"], "one_match_f05": r["one_match"]["mean_f05"], "multi_match_f05": r["multi_match"]["mean_f05"],
                   "false_merge_pairs": r["fp"], "entities_with_false_merge": r["false_merges"]["entities_with_fp"], "pair_recall": r["pair_recall"], "pairs_predicted": r["pairs_predicted"],
                   "mean_predicted": round(r["pairs_predicted"] / r["n_entities"], 3), "by_n_true": r["by_n_true"]}}
    rr = STRAT[name]["report"]
    log(f"  {name:48s} tune {tune_best['macro_f05_tune']:.4f} | report {rr['macro_f05']:.4f} | singleton score {rr['singleton_score']:.3f} ({rr['singleton_false_positives']} FPs) "
        f"| one {rr['one_match_f05']:.4f} multi {rr['multi_match_f05']:.4f} | false merges {rr['false_merge_pairs']:,}")


PA, TA = H["tune"]
PB, TB = H["report"]

# A. global threshold (the incumbent)
rows = [{"t": t, "macro_f05_tune": round(score(keep_threshold(PA, t), TA), 5)} for t in (0.65, 0.7, 0.75, 0.8)]
best = max(rows, key=lambda r: r["macro_f05_tune"])
record("A. global threshold", rows, full(keep_threshold(PB, best["t"]), TB), best)
BASE = STRAT["A. global threshold"]["report"]["macro_f05"]

# B. top-1 only
rows = [{"t": round(float(t), 2), "macro_f05_tune": round(score(keep_topk(PA, round(float(t), 2), 1), TA), 5)} for t in np.arange(0.2, 0.96, 0.05)]
best = max(rows, key=lambda r: r["macro_f05_tune"])
record("B. top-1 only", rows, full(keep_topk(PB, best["t"], 1), TB), best)

# C. threshold + top-K
rows = [{"t": t, "k": k, "macro_f05_tune": round(score(keep_topk(PA, t, k), TA), 5)} for t in (0.5, 0.6, 0.7, 0.75, 0.8) for k in (1, 2, 3, 4, 5, 6, 8, 11)]
best = max(rows, key=lambda r: r["macro_f05_tune"])
record("C. threshold + top-K", rows, full(keep_topk(PB, best["t"], best["k"]), TB), best)

# D. threshold + probability-gap rules
rows = [{"t_top": tt, "t_low": tl, "gap": g, "macro_f05_tune": round(score(keep_gapcut(PA, tt, tl, g), TA), 5)}
        for tt in (0.75, 0.9) for tl in (0.3, 0.5, 0.6, 0.7, 0.75) for g in (0.1, 0.2, 0.3, 0.5, 1.0)]
best = max(rows, key=lambda r: r["macro_f05_tune"])
record("D1. gap cut (stop at the first big drop)", rows, full(keep_gapcut(PB, best["t_top"], best["t_low"], best["gap"]), TB), best)
rows = [{"t": t, "rel_gate": g, "macro_f05_tune": round(score(M.decide(PA, "p", M.Rule(t, False, g)), TA), 5)} for t in (0.5, 0.6, 0.7, 0.75) for g in (0.3, 0.5, 0.7, 0.9)]
best = max(rows, key=lambda r: r["macro_f05_tune"])
record("D2. threshold + relative gate (p >= g * top-1)", rows, full(M.decide(PB, "p", M.Rule(best["t"], False, best["rel_gate"])), TB), best)

# E. high-confidence automatic matches + ML threshold
for variant in ("name+address", "name+house+street/city"):
    m = auto_mask(PA, variant)
    auto = PA.filter(m)
    prec = float(auto["label"].mean()) if auto.height else float("nan")
    below = auto.filter(pl.col("p") < T_GLOBAL)
    rows = [{"t": t, "macro_f05_tune": round(score(keep_auto_plus(PA, variant, t), TA), 5)} for t in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95)]
    best = max(rows, key=lambda r: r["macro_f05_tune"])
    best.update({"auto_pairs_tune": int(auto.height), "auto_precision_tune": round(prec, 5), "auto_pairs_below_global_t": int(below.height),
                 "auto_below_precision": round(float(below["label"].mean()), 5) if below.height else None})
    record(f"E. auto ({variant}) + ML threshold", rows, full(keep_auto_plus(PB, variant, best["t"]), TB), best)

# F. separate logic for singleton-like versus multi-match entities
# F1: hand rule on the top-1 probability: entities whose best candidate is weak (top1 < s) must clear a higher bar t_s
rows = []
for s in (0.8, 0.9, 0.95):
    for ts in (0.85, 0.9, 0.95, 0.99):
        for tm in (0.5, 0.6, 0.7, 0.75):
            et = ENT["tune"].select("q_idx", pl.when(pl.col("top1") < s).then(ts).otherwise(tm).alias("t_ent"))
            rows.append({"s": s, "t_single": ts, "t_multi": tm, "macro_f05_tune": round(score(keep_by_entity_threshold(PA, et), TA), 5)})
best = max(rows, key=lambda r: r["macro_f05_tune"])
et = ENT["report"].select("q_idx", pl.when(pl.col("top1") < best["s"]).then(best["t_single"]).otherwise(best["t_multi"]).alias("t_ent"))
record("F1. two thresholds by the entity's top-1 probability", rows, full(keep_by_entity_threshold(PB, et), TB), best)

# F2: an entity-level singleton model (out-of-fold on the tune half, refit for the report half) gating a higher threshold
import lightgbm as lgb
E_FEATS = ["n_cand", "top1", "top2", "gap", "mean_top3", "sum_p", "n_ge_90", "n_ge_75", "n_ge_50", "n_ge_25", "best_name_sim", "best_name_jw", "best_addr_sim", "best_cos_addr",
           "best_cos_name_addr", "any_exact_name", "any_exact_addr", "any_house_eq", "ident_agreement", "country_us"]
XA = ENT["tune"].select([pl.col(c).cast(pl.Float32) for c in E_FEATS]).to_numpy()
yA = (ENT["tune"]["n_true"] == 0).to_numpy().astype(np.int8)
XB = ENT["report"].select([pl.col(c).cast(pl.Float32) for c in E_FEATS]).to_numpy()
yB = (ENT["report"]["n_true"] == 0).to_numpy().astype(np.int8)
prm = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}
oof = np.zeros(len(yA), dtype=np.float32)
fold = (ENT["tune"]["q_idx"].to_numpy() // 8) % 2
for f in (0, 1):
    b = lgb.train(prm, lgb.Dataset(XA[fold != f], yA[fold != f]), num_boost_round=300)
    oof[fold == f] = b.predict(XA[fold == f])
full_model = lgb.train(prm, lgb.Dataset(XA, yA), num_boost_round=300)
pB = full_model.predict(XB).astype(np.float32)
RES["singleton_model"] = {"features": E_FEATS, "auc_oof_tune": round(auc(oof, yA.astype(bool)), 4), "auc_report": round(auc(pB, yB.astype(bool)), 4),
                          "gain_top": [(E_FEATS[i], round(float(g / full_model.feature_importance("gain").sum()), 3)) for i, g in sorted(enumerate(full_model.feature_importance("gain")), key=lambda x: -x[1])[:8]]}
log(f"singleton model: OOF AUC {RES['singleton_model']['auc_oof_tune']}, report AUC {RES['singleton_model']['auc_report']}")
rows = []
for s in (0.3, 0.5, 0.7, 0.9):
    for ts in (0.85, 0.9, 0.95, 0.99):
        et = pl.DataFrame({"q_idx": ENT["tune"]["q_idx"], "t_ent": np.where(oof >= s, ts, T_GLOBAL).astype(np.float32)})
        rows.append({"p_single": s, "t_single": ts, "macro_f05_tune": round(score(keep_by_entity_threshold(PA, et), TA), 5)})
best = max(rows, key=lambda r: r["macro_f05_tune"])
et = pl.DataFrame({"q_idx": ENT["report"]["q_idx"], "t_ent": np.where(pB >= best["p_single"], best["t_single"], T_GLOBAL).astype(np.float32)})
record("F2. singleton model gates a higher threshold", rows, full(keep_by_entity_threshold(PB, et), TB), best)

# F3: multi-match branch: entities with many strong candidates get a lower threshold
rows = []
for m in (4, 6, 8):
    for tm in (0.4, 0.5, 0.6, 0.7):
        et = ENT["tune"].select("q_idx", pl.when(pl.col("n_ge_50") >= m).then(tm).otherwise(T_GLOBAL).alias("t_ent"))
        rows.append({"min_n_ge_50": m, "t_multi": tm, "macro_f05_tune": round(score(keep_by_entity_threshold(PA, et), TA), 5)})
best = max(rows, key=lambda r: r["macro_f05_tune"])
et = ENT["report"].select("q_idx", pl.when(pl.col("n_ge_50") >= best["min_n_ge_50"]).then(best["t_multi"]).otherwise(T_GLOBAL).alias("t_ent"))
record("F3. lower threshold for entities with many strong candidates", rows, full(keep_by_entity_threshold(PB, et), TB), best)

# the singleton false positives of the global rule, for the analysis
fp_single = ENT["report"].filter((pl.col("n_true") == 0) & (pl.col("top1") >= T_GLOBAL))
RES["singleton_false_positives_global"] = {"n": fp_single.height, "of_singletons": ENT["report"].filter(pl.col("n_true") == 0).height,
                                            "top1_quantiles": q(fp_single["top1"]), "n_ge_75_mean": round(float(fp_single["n_ge_75"].mean()), 3),
                                            "any_exact_name_share": round(float(fp_single["any_exact_name"].mean()), 3), "any_exact_addr_share": round(float(fp_single["any_exact_addr"].mean()), 3),
                                            "best_addr_sim_quantiles": q(fp_single["best_addr_sim"].cast(pl.Float64)),
                                            "share_with_top1_ge_0.99": round(float((fp_single["top1"] >= 0.99).mean()), 3)}
RES["strategies"] = STRAT
RES["verdict"] = {k: round(v["report"]["macro_f05"] - BASE, 5) for k, v in STRAT.items()}
json.dump(RES, open("eda/results/s20_decision_rules.json", "w", encoding="utf-8"), indent=1)
log("saved eda/results/s20_decision_rules.json")

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
L = []
add = L.append
add("# Decision rules beyond the global threshold (LightGBM probabilities, fold 0)\n")
add("Every strategy's parameters are chosen on the tuning half (55,171 entities) and every number below is the official macro F0.5 on the report half "
    "(55,170 entities, 3,150 true singletons), never used for training or tuning. The incumbent is the global threshold of 0.75 with no other rule.\n")
add("## 1. What the per-entity signals look like\n")
add("Deciles (10% / 50% / 90%) of the entity-level signals on the report half, by the true number of matches:\n")
add("| signal | singleton | one match | 2-5 matches | 6+ matches | singleton-detection AUC |")
add("|---|---|---|---|---|---|")
for s in SIG:
    add(f"| {s} | " + " | ".join(" / ".join(f"{x:g}" for x in prof[k][s]) for k in ("singleton", "one match", "2-5 matches", "6+ matches")) + f" | {RES['singleton_detection_auc_report'][s]} |")
add("")
add(f"Entities: " + ", ".join(f"{k} {v['entities']:,}" for k, v in prof.items()) + ".\n")
sf = RES["singleton_false_positives_global"]
add(f"**Singleton false positives of the global rule.** {sf['n']} of the {sf['of_singletons']:,} true singletons have a candidate above 0.75; their top-1 probability deciles are "
    f"{' / '.join(f'{x:g}' for x in sf['top1_quantiles'])} and {100 * sf['share_with_top1_ge_0.99']:.0f}% of them are at 0.99 or above, {100 * sf['any_exact_name_share']:.0f}% share an exact name and "
    f"{100 * sf['any_exact_addr_share']:.0f}% an exact address with the candidate. These are the decoys of `matcher_baseline.md`: the model is certain, and nothing in the entity's "
    "signals (a lone strong candidate, a large gap) separates them from a genuine one-match entity, which looks the same.\n")
sm = RES["singleton_model"]
add(f"An entity-level LightGBM trained to recognise singletons from these signals ({len(sm['features'])} features, out-of-fold on the tuning half) reaches AUC {sm['auc_oof_tune']} "
    f"(report half {sm['auc_report']}); its gain is dominated by " + ", ".join(f"`{f}` {g}" for f, g in sm["gain_top"][:5]) + ".\n")
add("## 2. Strategies, chosen on the tuning half, scored on the report half\n")
add("| strategy | chosen parameters | tune F0.5 | **report F0.5** | vs global | singleton score | singleton FPs | one-match F0.5 | multi-match F0.5 | false merges | pair recall | mean predicted |")
add("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, v in STRAT.items():
    tb = {k: x for k, x in v["tune_best"].items() if k != "macro_f05_tune"}
    r = v["report"]
    add(f"| {name} | {', '.join(f'{k} {x}' for k, x in tb.items())} | {v['tune_best']['macro_f05_tune']:.4f} | **{r['macro_f05']:.4f}** | {RES['verdict'][name]:+.4f} | {r['singleton_score']:.4f} | "
        f"{r['singleton_false_positives']} | {r['one_match_f05']:.4f} | {r['multi_match_f05']:.4f} | {r['false_merge_pairs']:,} | {r['pair_recall']:.4f} | {r['mean_predicted']:.2f} |")
add("")
if os.path.exists("eda/scripts/s20_conclusions.md"):
    add(open("eda/scripts/s20_conclusions.md", encoding="utf-8").read())
open("eda/decision_rules.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
log("wrote eda/decision_rules.md")
