"""Decision-rule comparison on the real match-count structure.

Measures two parameters from the data (how often a hard negative with a similar address
belongs to another S1 entity; how often a true match is address-ambiguous with another
S1 entity), then simulates calibrated pair scores of three strengths for a sample of real
S1 entities and scores decision rules A..E with the verified evaluator.

Run from the project root:  python eda/scripts/s8b_decision_sim.py
"""
import os
import json, math, sys, time

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "code/business_entity_resolution/src")
import evaluation as ev  # noqa: E402

SCR = os.environ.get("BER_SCRATCH", "artifacts/_scratch")   # parquet cache of the raw TSVs, built by s0_inventory.py
PQ = f"{SCR}/pq"
GT = "student_resource/dataset/train/train_ground_truth.tsv"
OUT = "eda/results/s8b_decision_sim.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
R = {}
rng = np.random.default_rng(2026)

# ------------------------------------------------------------------ data-derived parameters
pairs = pl.read_parquet(f"{PQ}/train_pairs_pos.parquet")
hard = pl.read_parquet(f"{PQ}/sample_hard_feats.parquet", columns=["source1_entity_id", "matched_id", "addr_jac"])
hard = hard.with_columns(pl.col("matched_id").is_in(pairs["matched_id"].implode()).alias("owned"))
R["hard_negatives_sampled"] = hard.height
R["hard_neg_pct_owned_by_another_S1_overall"] = round(100 * float(hard["owned"].mean()), 2)
for thr in (0.3, 0.5):
    h = hard.filter(pl.col("addr_jac") >= thr)
    R[f"hard_neg_addr_jac>={thr}_n"] = h.height
    R[f"hard_neg_addr_jac>={thr}_pct_owned_by_another_S1"] = round(100 * float(h["owned"].mean()), 2)
log("owned-elsewhere rates done")

C = ["entity_id", "country", "name_core_sorted", "addr_norm"]
s1 = pl.read_parquet(f"{PQ}/train_source1_norm.parquet", columns=C)
grp = s1.group_by("country", "name_core_sorted").len()
shared = grp.filter(pl.col("len").is_between(2, 200)).select("country", "name_core_sorted")
s1g = s1.join(shared, on=["country", "name_core_sorted"])
possample = pairs.sample(n=200_000, seed=5)
s23 = pl.concat([
    pl.scan_parquet(f"{PQ}/train_source{k}_norm.parquet").select(C)
      .filter(pl.col("entity_id").is_in(possample["matched_id"].implode())).collect() for k in (2, 3)])
pp = possample.join(s23.rename({"entity_id": "matched_id", "addr_norm": "a_rec"}), on="matched_id")
j = pp.join(s1g.rename({"entity_id": "other_s1", "addr_norm": "a_other"}), on=["country", "name_core_sorted"])
j = j.filter(pl.col("other_s1") != pl.col("source1_entity_id"))
def toks(c):
    return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
j = j.with_columns((toks("a_rec").list.set_intersection(toks("a_other")).list.len()
                    / pl.max_horizontal(toks("a_rec").list.set_union(toks("a_other")).list.len(), pl.lit(1))).alias("aj"))
best = j.group_by("matched_id").agg(pl.col("aj").max().alias("best"))
R["positives_sampled"] = possample.height
R["pct_positives_whose_record_shares_name_with_another_S1"] = round(100 * best.height / possample.height, 2)
for thr in (0.3, 0.5):
    R[f"pct_positives_address_ambiguous_with_another_S1_addr_jac>={thr}"] = round(100 * float((best["best"] >= thr).sum()) / possample.height, 3)
f_amb = float((best["best"] >= 0.5).sum()) / possample.height
f_owned = R["hard_neg_addr_jac>=0.5_pct_owned_by_another_S1"] / 100
del j, pp, s23, best
log("ambiguity rates done")

# same-name block sizes for a sample of real entities
lite = pl.concat([pl.read_parquet(f"{PQ}/train_source{k}_norm.parquet", columns=["country", "name_core_sorted"]) for k in (2, 3)])
blk = lite.group_by("country", "name_core_sorted").len().rename({"len": "blk"})
s1b = s1.select("entity_id", "country", "name_core_sorted").join(blk, on=["country", "name_core_sorted"], how="left").with_columns(pl.col("blk").fill_null(0))
del lite, blk, s1, s1g
N_ENT = 200_000
samp = s1b.sample(n=N_ENT, seed=11)
ent_ids = samp["entity_id"].to_list()
blk_sizes = samp["blk"].to_numpy()
gt = ev.read_id_list_file(GT)
truth = {e: gt[e] for e in ent_ids}
del gt
k = np.array([len(truth[e]) for e in ent_ids])
log("sample ready", N_ENT, "mean k", k.mean().round(3), "mean block", blk_sizes.mean().round(1))

# ------------------------------------------------------------------ candidate universe
W = 40                      # candidates kept per entity after a ranking stage
P_WEAK_POS = 0.18           # no name-token overlap or missing address (EDA: 14.7% + 4.4%)
P_NEAR_MISS = 0.10          # entity has a same-name near-miss candidate (EDA: about 7% orphans + owned ones)
KIND_CLEAR, KIND_WEAK, KIND_NEG, KIND_NM = 0, 1, 2, 3
cand_ids, kinds, owned_flag, amb_flag, ent_of = [], [], [], [], []
for i, e in enumerate(ent_ids):
    t = truth[e]
    for m in t:
        cand_ids.append(m); kinds.append(KIND_WEAK if rng.random() < P_WEAK_POS else KIND_CLEAR)
        owned_flag.append(False); amb_flag.append(rng.random() < f_amb); ent_of.append(i)
    n_d = int(min(max(blk_sizes[i] - math.ceil(0.5 * len(t)), 0), 27)) + int(rng.poisson(2.0))
    n_d = min(n_d, W - len(t) - 1)
    for d in range(n_d):
        cand_ids.append(f"S2-NEG{i}_{d}"); kinds.append(KIND_NEG); owned_flag.append(False); amb_flag.append(False); ent_of.append(i)
    if rng.random() < P_NEAR_MISS:
        cand_ids.append(f"S3-NM{i}"); kinds.append(KIND_NM); owned_flag.append(rng.random() < f_owned); amb_flag.append(False); ent_of.append(i)
kinds = np.array(kinds); owned_flag = np.array(owned_flag); amb_flag = np.array(amb_flag); ent_of = np.array(ent_of)
label = (kinds <= 1).astype(int)
n_cand = len(kinds)
R["sim_candidates_total"] = n_cand
R["sim_candidates_per_entity"] = round(n_cand / N_ENT, 2)
R["sim_positive_rate"] = round(float(label.mean()), 4)
R["sim_params"] = {"W": W, "p_weak_pos": P_WEAK_POS, "p_near_miss": P_NEAR_MISS, "f_owned(measured)": round(f_owned, 3), "f_amb(measured)": round(f_amb, 4)}
log("candidates built", n_cand)

STRENGTHS = {   # Beta(a, b) raw-score distributions per candidate kind
    "weak":   {KIND_CLEAR: (4, 2),    KIND_WEAK: (2, 3),     KIND_NEG: (1.3, 4),  KIND_NM: (3.5, 2.5)},
    "medium": {KIND_CLEAR: (6, 1.5),  KIND_WEAK: (2.5, 2.5), KIND_NEG: (1, 7),    KIND_NM: (3.5, 2.5)},
    "strong": {KIND_CLEAR: (12, 1.5), KIND_WEAK: (3, 2),     KIND_NEG: (1, 12),   KIND_NM: (3, 3)},
}
calib_mask_ent = (np.arange(N_ENT) % 2 == 0)      # even entities calibrate, odd entities are scored
eval_ent = np.where(~calib_mask_ent)[0]
eval_ids = [ent_ids[i] for i in eval_ent]
truth_eval = {e: truth[e] for e in eval_ids}
is_eval_cand = ~calib_mask_ent[ent_of]
# padded matrices for the evaluation entities: row = entity, columns = candidates sorted by prob
order_by_ent = np.argsort(ent_of, kind="stable")
starts = np.searchsorted(ent_of[order_by_ent], np.arange(N_ENT))
ends = np.searchsorted(ent_of[order_by_ent], np.arange(N_ENT), side="right")

def padded(values, ent_subset):
    M = np.zeros((len(ent_subset), W)); V = np.zeros((len(ent_subset), W), dtype=bool); I = np.empty((len(ent_subset), W), dtype=object)
    for r, i in enumerate(ent_subset):
        idx = order_by_ent[starts[i]:ends[i]]
        v = values[idx]; o = np.argsort(-v, kind="stable")[:W]; idx = idx[o]
        M[r, :len(idx)] = v[o][:len(idx)]; V[r, :len(idx)] = True
        for c, q in enumerate(idx): I[r, c] = cand_ids[q]
    return M, V, I

def to_pred(mask, I, V):
    mask = mask & V
    return {eval_ids[r]: [I[r, c] for c in np.flatnonzero(mask[r])] for r in range(len(eval_ids))}

def score(mask, I, V):
    return ev.macro_f05(truth_eval, to_pred(mask, I, V)).macro_f05

def by_k(mask, I, V):
    res = ev.macro_f05(truth_eval, to_pred(mask, I, V))
    return {int(kk): [int(nn), round(ff, 4)] for kk, (nn, ff) in sorted(res.by_n_true.items())}

ONLY = [a for a in sys.argv[1:] if a in ("weak", "medium", "strong")]
if ONLY:
    OUT = OUT.replace(".json", "_" + "_".join(ONLY) + ".json")

def expected_f_prefix(P, V, jmax=12):
    """E[F0.5] of predicting the top-j candidates, j = 0..jmax, under independent Bernoulli(p_i)."""
    N, Wd = P.shape
    p = np.where(V, P, 0.0)
    pre = [np.ones((N, 1))]
    for j in range(jmax):
        prev, pj = pre[-1], p[:, j:j + 1]
        nxt = np.zeros((N, prev.shape[1] + 1)); nxt[:, :-1] += prev * (1 - pj); nxt[:, 1:] += prev * pj
        pre.append(nxt)
    suf = np.ones((N, 1))
    EF = np.zeros((N, jmax + 1))
    for j in range(Wd - 1, -1, -1):
        pj = p[:, j:j + 1]
        nxt = np.zeros((N, suf.shape[1] + 1)); nxt[:, :-1] += suf * (1 - pj); nxt[:, 1:] += suf * pj
        suf = nxt                                  # distribution of true matches at positions >= j
        if j <= jmax:
            tp = np.arange(j + 1)[:, None]; fn = np.arange(suf.shape[1])[None, :]; fp = j - tp
            with np.errstate(divide="ignore", invalid="ignore"):
                F = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * fn + fp), 0.0)
            if j == 0:
                F[0, 0] = 1.0                      # empty prediction and nothing to find
            EF[:, j] = np.einsum("nt,nf,tf->n", pre[j], suf, F)
    return EF

def rank_mask(jstar):
    return np.arange(W)[None, :] < jstar[:, None]

results = {}
for name, params in STRENGTHS.items():
    if ONLY and name not in ONLY:
        continue
    raw = np.empty(n_cand)
    for kind, (a, b) in params.items():
        m = kinds == kind
        raw[m] = rng.beta(a, b, size=int(m.sum()))
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw[~is_eval_cand], label[~is_eval_cand])
    prob = iso.predict(raw)
    auc = roc_auc_score(label[is_eval_cand], raw[is_eval_cand])
    P, V, I = padded(prob, eval_ent)
    Praw, Vraw, Iraw = padded(raw, eval_ent)
    res = {"pair_auc_on_eval": round(float(auc), 4)}
    # A: global threshold on calibrated probability
    best = (-1, None)
    for t in np.arange(0.05, 0.96, 0.05):
        s = score(P >= t, I, V)
        res[f"A_threshold_{t:.2f}"] = round(s, 4)
        if s > best[0]: best = (s, round(float(t), 2))
    res["A_best"] = {"macro_f05": round(best[0], 4), "t": best[1]}
    res["A_at_0.5"] = res["A_threshold_0.50"]
    # B: ranking with a fixed cut-off
    for kf in range(1, 6):
        res[f"B_top{kf}"] = round(score(rank_mask(np.full(len(eval_ent), kf)), I, V), 4)
    res["B_top_round_sum_p"] = round(score(rank_mask(np.rint(np.where(V, P, 0).sum(1)).astype(int)), I, V), 4)
    # C: per-entity relative threshold
    pmax = np.where(V, P, 0).max(1, keepdims=True)
    bestc = (-1, None)
    for t0 in (0.1, 0.2, 0.3, 0.4, 0.5):
        for alpha in (0.3, 0.5, 0.7, 0.8, 0.9):
            s = score(P >= np.maximum(t0, alpha * pmax), I, V)
            if s > bestc[0]: bestc = (s, (t0, alpha))
    res["C_best"] = {"macro_f05": round(bestc[0], 4), "t0_alpha": bestc[1]}
    # D: expected-F0.5 prefix under independence (calibrated), and with raw uncalibrated scores
    EF = expected_f_prefix(P, V); jstar = EF.argmax(1)
    res["D_expected_f05_calibrated"] = round(score(rank_mask(jstar), I, V), 4)
    res["D_predicted_empty_pct"] = round(100 * float((jstar == 0).mean()), 2)
    res["D_mean_expected_f05"] = round(float(EF.max(1).mean()), 4)
    # per-k breakdown of the main rules, and how often each predicts an empty set
    res["by_k"] = {
        "A_best": by_k(P >= best[1], I, V),
        "C_best": by_k(P >= np.maximum(bestc[1][0], bestc[1][1] * pmax), I, V),
        "B_top3": by_k(rank_mask(np.full(len(eval_ent), 3)), I, V),
        "B_top_round_sum_p": by_k(rank_mask(np.rint(np.where(V, P, 0).sum(1)).astype(int)), I, V),
        "D": by_k(rank_mask(jstar), I, V),
    }
    res["empty_pred_pct"] = {
        "A_best": round(100 * float(((P >= best[1]) & V).sum(1).__eq__(0).mean()), 2),
        "C_best": round(100 * float(((P >= np.maximum(bestc[1][0], bestc[1][1] * pmax)) & V).sum(1).__eq__(0).mean()), 2),
        "D": round(100 * float((jstar == 0).mean()), 2),
        "truth": round(100 * float((k[eval_ent] == 0).mean()), 2),
    }
    # D with the empirical k prior instead of independence: rescale the empty-set expectation by the
    # singleton excess (observed 5.585% vs 1.57% under independence) to see if singletons are the gap
    EF2 = EF.copy(); EF2[:, 0] = np.minimum(1.0, EF[:, 0] * (0.05585 / 0.01573))
    res["D_with_singleton_prior_boost"] = round(score(rank_mask(EF2.argmax(1)), I, V), 4)
    res["D_with_singleton_prior_boost_empty_pct"] = round(100 * float((EF2.argmax(1) == 0).mean()), 2)
    EFr = expected_f_prefix(Praw, Vraw); res["D_with_raw_uncalibrated_scores"] = round(score(rank_mask(EFr.argmax(1)), Iraw, Vraw), 4)
    # D with per-source caps (k2<=5, k3<=6, total<=11) is implied by jmax=12; report D with jmax=11
    # E: D after one-to-one assignment across S1 entities
    for w in (0.5, 0.7, 0.9, 1.0):
        prob_e = prob.copy()
        owned_removed = owned_flag & (rng.random(n_cand) < w)
        amb_lost = amb_flag & (rng.random(n_cand) < (1 - w))
        prob_e[owned_removed | amb_lost] = 0.0
        Pe, Ve, Ie = padded(prob_e, eval_ent)
        Ve = Ve & (Pe > 0)
        EFe = expected_f_prefix(Pe, Ve)
        res[f"E_assignment_then_D_owner_wins_{w:.1f}"] = round(score(rank_mask(EFe.argmax(1)), Ie, Ve), 4)
    # blocking loss: 3% of true matches never reach the candidate set (medium only)
    if name == "medium":
        drop = (label == 1) & (rng.random(n_cand) < 0.03)
        prob_b = prob.copy(); prob_b[drop] = 0.0
        Pb, Vb, Ib = padded(prob_b, eval_ent); Vb = Vb & (Pb > 0)
        EFb = expected_f_prefix(Pb, Vb)
        res["D_with_3pct_blocking_misses"] = round(score(rank_mask(EFb.argmax(1)), Ib, Vb), 4)
        sA = max(score(Pb >= t, Ib, Vb) for t in np.arange(0.05, 0.96, 0.05))
        res["A_best_with_3pct_blocking_misses"] = round(sA, 4)
    results[name] = res
    log(name, {k_: v for k_, v in res.items() if not k_.startswith("A_threshold")})
R["simulation"] = results
json.dump(R, open(OUT, "w"), indent=1, default=str)
log("DONE")
