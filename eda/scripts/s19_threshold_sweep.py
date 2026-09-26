"""Threshold sweep on the competition objective (entity-level macro F0.5), 0.01 to 0.99, with the full per-threshold report,
and the test of whether one global threshold is appropriate (segment-specific thresholds chosen on the tuning half,
evaluated on the report half).  Uses the LightGBM probabilities of s18 (prior-corrected).

Run from the project root:  python -u eda/scripts/s19_threshold_sweep.py
Writes eda/results/s19_threshold_sweep.json and eda/threshold_sweep.md.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import evaluation as ev  # noqa: E402
import matcher as M  # noqa: E402
import validation as V  # noqa: E402

MODEL = os.environ.get("S19_MODEL", "lightgbm")
THRESHOLDS = [round(t, 2) for t in np.arange(0.01, 1.00, 0.01)]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)

sc = pl.read_parquet("artifacts/models/fold0/val_scores.parquet", columns=["q_idx", "p_idx", "label", MODEL])
s1 = pl.read_parquet("artifacts/features/fold0/_pool_stats/s1.parquet", columns=["entity_id", "country"])
S1_IDS = s1["entity_id"].to_list()
COUNTRY = dict(zip(S1_IDS, s1["country"].to_list()))
pool_ids = pl.concat([pl.read_parquet(f"artifacts/preprocessed/fold0/train_source{k}.parquet", columns=["entity_id"]) for k in (2, 3)])["entity_id"]
ids = M.IdMaps(S1_IDS, pool_ids)
truth_all = V.entity_truth("artifacts/validation", folds=[0])
q_of = {e: i for i, e in enumerate(S1_IDS)}
half = sc["q_idx"].to_numpy() % 8
HALVES = {"tune": (ids.attach(sc.filter(pl.Series(half == 3)).select("q_idx", "p_idx", MODEL)), {e: v for e, v in truth_all.items() if q_of[e] % 8 == 3}),
          "report": (ids.attach(sc.filter(pl.Series(half == 7)).select("q_idx", "p_idx", MODEL)), {e: v for e, v in truth_all.items() if q_of[e] % 8 == 7})}
del sc
log(f"model {MODEL}: tune half {HALVES['tune'][0].height:,} pairs / {len(HALVES['tune'][1]):,} entities, report half {HALVES['report'][0].height:,} / {len(HALVES['report'][1]):,}")


def per_entity_scores(pairs: pl.DataFrame, truth: dict, t: float, unique: bool = False):
    """Official evaluator per entity for the plain threshold rule (optionally with pool uniqueness)."""
    kept = M.decide(pairs, MODEL, M.Rule(t, unique))
    grouped = kept.group_by("entity_id").agg(pl.col("matched_id"))
    res = ev.macro_f05_from_rows(truth, grouped.iter_rows(), keep_per_entity=True)
    return res


def summarise(res, entities_order):
    """The per-threshold report from the evaluator's per-entity scores."""
    per = {s.entity_id: s for s in res.per_entity}
    n = len(per)
    f = np.array([per[e].f05 for e in entities_order])
    n_pred = np.array([per[e].n_pred for e in entities_order])
    n_true = np.array([per[e].n_true for e in entities_order])
    fp = np.array([per[e].fp for e in entities_order])
    prec = np.array([per[e].precision if per[e].precision is not None else np.nan for e in entities_order])
    rec = np.array([per[e].recall if per[e].recall is not None else np.nan for e in entities_order])
    singles, one, multi = n_true == 0, n_true == 1, n_true > 1
    return {"macro_f05": round(float(res.macro_f05), 5),
            "mean_precision": round(float(np.nanmean(prec)), 5) if np.isfinite(prec).any() else None,          # over entities with a prediction
            "mean_recall": round(float(np.nanmean(rec)), 5) if np.isfinite(rec).any() else None,                # over entities with a true match
            "singleton_score": round(float(f[singles].mean()), 5), "zero_prediction_share": round(float((n_pred == 0).mean()), 5),
            "mean_predicted": round(float(n_pred.mean()), 4), "median_predicted": float(np.median(n_pred)),
            "false_merges": int(fp.sum()), "entities_with_false_merge": int((fp > 0).sum()),
            "one_match_f05": round(float(f[one].mean()), 5), "multi_match_f05": round(float(f[multi].mean()), 5),
            "pooled_precision": round(float(res.pooled_precision), 5) if res.pooled_precision is not None else None,
            "pooled_recall": round(float(res.pooled_recall), 5) if res.pooled_recall is not None else None}, f


RES = {"model": MODEL, "thresholds": THRESHOLDS, "sweep": {}, "per_entity_f05": {}}
F = {}          # half -> matrix [threshold, entity]
ORDER = {h: list(truth.keys()) for h, (_, truth) in HALVES.items()}
for h, (pairs, truth) in HALVES.items():
    rows, mat = [], np.zeros((len(THRESHOLDS), len(truth)), dtype=np.float32)
    for i, t in enumerate(THRESHOLDS):
        res = per_entity_scores(pairs, truth, t)
        row, f = summarise(res, ORDER[h])
        row["threshold"] = t
        rows.append(row)
        mat[i] = f
        if i % 20 == 0 or t in (0.5, 0.75):
            log(f"  {h} t={t:.2f}: macro F0.5 {row['macro_f05']:.4f}, P {row['mean_precision']}, R {row['mean_recall']}, singletons {row['singleton_score']:.3f}, "
                f"zero-pred {100 * row['zero_prediction_share']:.1f}%, mean pred {row['mean_predicted']:.2f}, false merges {row['false_merges']:,}")
    RES["sweep"][h] = rows
    F[h] = mat
best_i = int(np.argmax([r["macro_f05"] for r in RES["sweep"]["tune"]]))
t_best = THRESHOLDS[best_i]
plateau = [t for t, r in zip(THRESHOLDS, RES["sweep"]["tune"]) if r["macro_f05"] >= RES["sweep"]["tune"][best_i]["macro_f05"] - 0.001]
RES["global"] = {"best_threshold_tune": t_best, "macro_f05_tune": RES["sweep"]["tune"][best_i]["macro_f05"], "macro_f05_report": RES["sweep"]["report"][best_i]["macro_f05"],
                 "plateau_within_0.001": [min(plateau), max(plateau)],
                 "best_threshold_report_itself": THRESHOLDS[int(np.argmax([r["macro_f05"] for r in RES["sweep"]["report"]]))],
                 "macro_f05_report_at_its_own_best": max(r["macro_f05"] for r in RES["sweep"]["report"])}
log(f"global threshold: best on the tune half {t_best} -> {RES['global']['macro_f05_tune']:.4f}; on the report half {RES['global']['macro_f05_report']:.4f} "
    f"(the report half's own optimum {RES['global']['best_threshold_report_itself']} gives {RES['global']['macro_f05_report_at_its_own_best']:.4f}); plateau {RES['global']['plateau_within_0.001']}")

# uniqueness on top of the best plain threshold, for the record
for h, (pairs, truth) in HALVES.items():
    row, _ = summarise(per_entity_scores(pairs, truth, t_best, unique=True), ORDER[h])
    RES["global"][f"with_uniqueness_{h}"] = row["macro_f05"]

# --------------------------------------------------------------------------- #
# Is one global threshold appropriate?  Segment-specific thresholds chosen on the tune half, applied on the report half.
# Segments must be known at test time: country, number of candidates, the entity's best probability, and the source of the pair.
# --------------------------------------------------------------------------- #
def entity_segments(pairs: pl.DataFrame, truth: dict, order):
    agg = pairs.group_by("entity_id").agg(pl.len().alias("n_cand"), pl.col(MODEL).max().alias("max_p"))
    d = {e: (n, m) for e, n, m in agg.iter_rows()}
    seg = {"country": np.array([COUNTRY[e] for e in order]),
           "n_cand": np.array([("1-20" if d[e][0] <= 20 else "21-60" if d[e][0] <= 60 else "61-120" if d[e][0] <= 120 else "121+") for e in order]),
           "max_p": np.array([("<0.5" if d[e][1] < 0.5 else "0.5-0.9" if d[e][1] < 0.9 else "0.9-0.99" if d[e][1] < 0.99 else ">=0.99") for e in order]),
           "n_true(oracle)": np.array([("0" if len(truth[e]) == 0 else "1" if len(truth[e]) == 1 else "2-5" if len(truth[e]) <= 5 else "6+") for e in order])}
    return seg


SEG = {h: entity_segments(HALVES[h][0], HALVES[h][1], ORDER[h]) for h in HALVES}
RES["segments"] = {}
for name in SEG["tune"]:
    out = {"per_segment": {}, "combined_report_macro_f05": None}
    total = np.zeros(len(ORDER["report"]))
    labels = sorted(set(SEG["tune"][name]))
    for lab in labels:
        mt, mr = SEG["tune"][name] == lab, SEG["report"][name] == lab
        curve = F["tune"][:, mt].mean(axis=1)
        i = int(np.argmax(curve))
        out["per_segment"][lab] = {"entities_tune": int(mt.sum()), "entities_report": int(mr.sum()), "best_threshold": THRESHOLDS[i], "f05_tune_at_best": round(float(curve[i]), 5),
                                   "f05_tune_at_global": round(float(curve[best_i]), 5), "f05_report_at_best": round(float(F["report"][i, mr].mean()), 5) if mr.any() else None,
                                   "f05_report_at_global": round(float(F["report"][best_i, mr].mean()), 5) if mr.any() else None}
        total[mr] = F["report"][i, mr]
    out["combined_report_macro_f05"] = round(float(total.mean()), 5)
    out["gain_over_global_report"] = round(float(total.mean() - F["report"][best_i].mean()), 5)
    RES["segments"][name] = out
    log(f"  segment thresholds by {name}: " + ", ".join(f"{k} {v['best_threshold']}" for k, v in out["per_segment"].items()) +
        f" -> report macro F0.5 {out['combined_report_macro_f05']:.4f} (global {F['report'][best_i].mean():.4f}, gain {out['gain_over_global_report']:+.4f})")

# source-specific thresholds (a pair-level segment): 2-D grid on the tune half, applied on the report half
grid = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]
best_src, table = None, []
for t2 in grid:
    for t3 in grid:
        pairs, truth = HALVES["tune"]
        kept = pairs.filter(pl.when(pl.col("matched_id").str.slice(0, 2) == "S2").then(pl.col(MODEL) >= t2).otherwise(pl.col(MODEL) >= t3))
        f = M.macro_only(kept, truth)
        table.append({"t_S2": t2, "t_S3": t3, "macro_f05_tune": round(f, 5)})
        if best_src is None or f > best_src[0]:
            best_src = (f, t2, t3)
pairs, truth = HALVES["report"]
kept = pairs.filter(pl.when(pl.col("matched_id").str.slice(0, 2) == "S2").then(pl.col(MODEL) >= best_src[1]).otherwise(pl.col(MODEL) >= best_src[2]))
RES["segments"]["source"] = {"best_t_S2": best_src[1], "best_t_S3": best_src[2], "macro_f05_tune": round(best_src[0], 5), "macro_f05_report": round(M.macro_only(kept, truth), 5),
                             "gain_over_global_report": round(M.macro_only(kept, truth) - RES["global"]["macro_f05_report"], 5), "grid": table}
log(f"  source thresholds: S2 {best_src[1]}, S3 {best_src[2]} -> report {RES['segments']['source']['macro_f05_report']:.4f} (gain {RES['segments']['source']['gain_over_global_report']:+.4f})")
json.dump(RES, open("eda/results/s19_threshold_sweep.json", "w", encoding="utf-8"), indent=1)
log("saved eda/results/s19_threshold_sweep.json")

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
L = []
add = L.append
g = RES["global"]
add(f"# Threshold sweep on the competition objective ({MODEL}, fold 0)\n")
add("For every threshold from 0.01 to 0.99 the pairs with probability at or above it form each Source 1 entity's predicted list, and the official evaluator "
    "scores the list against the entity's true list; every number below is macro-averaged over entities (`evaluation.macro_f05_from_rows`). Pair-level "
    "F0.5 is not used anywhere. The threshold is chosen on the tuning half (55,171 entities) and confirmed on the report half (55,170 entities, never used "
    "for training or selection). Precision is averaged over the entities that received a prediction, recall over the entities that have a true match; "
    "singleton score is the share of true singletons left empty (their official score).\n")
add(f"**Result.** The competition F0.5 peaks at threshold **{g['best_threshold_tune']}** on the tuning half ({g['macro_f05_tune']:.4f}) and gives "
    f"**{g['macro_f05_report']:.4f}** on the report half, whose own optimum ({g['best_threshold_report_itself']}) would give {g['macro_f05_report_at_its_own_best']:.4f}: "
    f"the choice transfers. Every threshold between {g['plateau_within_0.001'][0]} and {g['plateau_within_0.001'][1]} stays within 0.001 of the peak. Adding pool-side "
    f"uniqueness to the best threshold gives {g['with_uniqueness_tune']:.4f} / {g['with_uniqueness_report']:.4f} (tune / report).\n")
for h in ("tune", "report"):
    add(f"## Sweep on the {h} half\n")
    add("| t | macro F0.5 | mean precision | mean recall | singleton score | zero-prediction entities | mean predicted | median predicted | false merges (pairs / entities) | one-match F0.5 | multi-match F0.5 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in RES["sweep"][h]:
        mark = "**" if r["threshold"] == g["best_threshold_tune"] else ""
        add(f"| {mark}{r['threshold']:.2f}{mark} | {mark}{r['macro_f05']:.4f}{mark} | {r['mean_precision']:.4f} | {r['mean_recall']:.4f} | {r['singleton_score']:.4f} | {100 * r['zero_prediction_share']:.2f}% "
            f"| {r['mean_predicted']:.3f} | {r['median_predicted']:.0f} | {r['false_merges']:,} / {r['entities_with_false_merge']:,} | {r['one_match_f05']:.4f} | {r['multi_match_f05']:.4f} |")
    add("")
add("## Is a single global threshold appropriate?\n")
add("Segment-specific thresholds were chosen on the tuning half for every segment an entity can be assigned to at test time, then applied on the report half. "
    "The oracle segmentation by the true number of matches is included only to show the size of the prize that a perfect 'how many matches' predictor would unlock.\n")
add("| segmentation | thresholds chosen (tune half) | report macro F0.5 with segment thresholds | with the global threshold | gain |")
add("|---|---|---|---|---|")
for name, out in RES["segments"].items():
    if name == "source":
        add(f"| source of the pair | S2 {out['best_t_S2']}, S3 {out['best_t_S3']} | {out['macro_f05_report']:.4f} | {g['macro_f05_report']:.4f} | {out['gain_over_global_report']:+.4f} |")
    else:
        add(f"| {name} | " + ", ".join(f"{k} {v['best_threshold']}" for k, v in out["per_segment"].items()) + f" | {out['combined_report_macro_f05']:.4f} | {g['macro_f05_report']:.4f} | {out['gain_over_global_report']:+.4f} |")
add("")
add("Per segment, the tuning-half optimum against the global threshold (report-half F0.5 of each):\n")
add("| segmentation | segment | entities (report) | best t | F0.5 at its t | F0.5 at the global t |")
add("|---|---|---|---|---|---|")
for name, out in RES["segments"].items():
    if name == "source":
        continue
    for k, v in out["per_segment"].items():
        add(f"| {name} | {k} | {v['entities_report']:,} | {v['best_threshold']} | {v['f05_report_at_best']:.4f} | {v['f05_report_at_global']:.4f} |")
add("")
if os.path.exists("eda/scripts/s19_conclusions.md"):
    add(open("eda/scripts/s19_conclusions.md", encoding="utf-8").read())
open("eda/threshold_sweep.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
log("wrote eda/threshold_sweep.md")
