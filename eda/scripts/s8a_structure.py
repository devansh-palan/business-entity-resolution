"""Structure of the optimisation problem, computed from train_ground_truth.tsv with the verified evaluator.

Run from the project root:  python eda/scripts/s8a_structure.py
"""
import json, math, sys, time
from collections import Counter

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, "code/business_entity_resolution/src")
import evaluation as ev  # noqa: E402

GT = "student_resource/dataset/train/train_ground_truth.tsv"
OUT = "eda/results/s8a_structure.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)

R = {}
gt = ev.read_id_list_file(GT)
ids = list(gt.keys())
n = len(ids)
k = np.fromiter((len(gt[e]) for e in ids), dtype=np.int64, count=n)
k2 = np.fromiter((sum(1 for m in gt[e] if m.startswith("S2-")) for e in ids), dtype=np.int64, count=n)
k3 = k - k2
log("loaded", n)

pct = lambda mask: round(100 * float(np.mean(mask)), 3)
dist = lambda arr: {int(v): [int(c), round(100 * c / n, 3)] for v, c in zip(*np.unique(arr, return_counts=True))}
R["n_entities"] = n
R["1_pct_singletons"] = pct(k == 0)
R["2_pct_exactly_1_match"] = pct(k == 1)
R["3_pct_exactly_2_matches"] = pct(k == 2)
R["4_pct_3_plus_matches"] = pct(k >= 3)
R["8_dist_true_match_count"] = dist(k)
R["dist_S2_matches"] = dist(k2)
R["dist_S3_matches"] = dist(k3)
R["mean_k"] = float(k.mean()); R["mean_k2"] = float(k2.mean()); R["mean_k3"] = float(k3.mean())
R["6_pct_entities_with_2plus_S2_matches"] = pct(k2 >= 2)
R["6_pct_entities_with_2plus_S3_matches"] = pct(k3 >= 2)
R["6_max_S2_matches"] = int(k2.max()); R["6_max_S3_matches"] = int(k3.max())

# 7. does any S2/S3 id appear under more than one S1 entity?
cnt = Counter()
for e in ids:
    cnt.update(gt[e])
R["7_matched_ids_total"] = sum(cnt.values())
R["7_matched_ids_distinct"] = len(cnt)
R["7_max_S1_entities_per_matched_id"] = max(cnt.values())
del cnt
log("uniqueness done")

# 5. do S2 and S3 contribute independently?  crosstab of (k2, k3)
ct = np.zeros((6, 7), dtype=np.int64)
np.add.at(ct, (np.minimum(k2, 5), np.minimum(k3, 6)), 1)
R["5_crosstab_rows_k2_0to5_cols_k3_0to6"] = ct.tolist()
R["5_P_k3_given_k2"] = {int(i): (ct[i] / ct[i].sum()).round(4).tolist() for i in range(6)}
R["5_P_k2_given_k3"] = {int(j): (ct[:, j] / ct[:, j].sum()).round(4).tolist() for j in range(7)}
R["5_P_k3"] = (ct.sum(0) / n).round(4).tolist()
R["5_P_k2"] = (ct.sum(1) / n).round(4).tolist()

def indep(table, label):
    chi2, p, dof, _ = chi2_contingency(table)
    v = math.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    R[f"5_independence_{label}"] = {"chi2": round(chi2, 1), "p_value": p, "dof": int(dof), "cramers_v": round(v, 5), "n": int(table.sum())}

indep(ct, "all_entities")
indep(ct[1:, :], "given_k2_ge_1")          # rows k2 = 1..5
indep(ct[:, 1:], "given_k3_ge_1")          # cols k3 = 1..6
indep(ct[1:, 1:], "given_both_ge_1")
R["5_P_k3_eq_0"] = round(float(np.mean(k3 == 0)), 4)
R["5_P_k3_eq_0_given_k2_eq_0"] = round(float(ct[0, 0] / ct[0].sum()), 4)
R["5_P_k3_eq_0_given_k2_ge_1"] = round(float(ct[1:, 0].sum() / ct[1:].sum()), 4)
R["5_singleton_rate_observed"] = round(float(np.mean(k == 0)), 5)
R["5_singleton_rate_if_k2_k3_independent"] = round(float(np.mean(k2 == 0) * np.mean(k3 == 0)), 5)
R["5_corr_k2_k3_all"] = round(float(np.corrcoef(k2, k3)[0, 1]), 5)
nz = k > 0
R["5_corr_k2_k3_non_singletons"] = round(float(np.corrcoef(k2[nz], k3[nz])[0, 1]), 5)
# model: singleton flag with prob s, else (k2, k3) independent draws conditioned on not both zero
nonsing = ct.copy(); nonsing[0, 0] = 0
row = nonsing.sum(1) / nonsing.sum(); col = nonsing.sum(0) / nonsing.sum()
expected = np.outer(row, col) * nonsing.sum()
expected[0, 0] = 0
expected *= nonsing.sum() / expected.sum()
R["5_max_abs_pct_deviation_from_independent_nonsingleton_model"] = round(float(np.max(np.abs(nonsing - expected)) / nonsing.sum() * 100), 4)
log("independence done")

# ceilings with perfect scores, scored by the evaluator on the full ground truth
def score(pred, label):
    res = ev.macro_f05(gt, pred)
    R[label] = round(res.macro_f05, 6)
    log(label, R[label])

for kf in range(1, 7):
    pred = {e: (gt[e][:kf] if len(gt[e]) >= kf else gt[e] + [f"S2-PAD{i}" for i in range(kf - len(gt[e]))]) for e in ids}
    score(pred, f"ceiling_fixed_top{kf}_perfect_ranking")
    del pred
pred = {}
for e in ids:
    m = gt[e]
    a = next((x for x in m if x.startswith("S2-")), None)
    b = next((x for x in m if x.startswith("S3-")), None)
    pred[e] = [x for x in (a, b) if x]
score(pred, "ceiling_top1_per_source_perfect_ranking"); del pred
pred = {e: (gt[e] if gt[e] else ["S2-PAD0"]) for e in ids}
score(pred, "ceiling_never_predict_empty"); del pred
pred = {e: (gt[e][:1]) for e in ids}
score(pred, "ceiling_one_correct_per_nonsingleton_empty_singletons"); del pred
# per-entity value of knowing k: fixed k=round(mean k)=3 vs adaptive = 1.0
json.dump(R, open(OUT, "w"), indent=1, default=str)
for key, v in R.items():
    print(key, "=>", v)
log("DONE")
