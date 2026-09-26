"""Deterministic high-precision baseline: rules 1-8 (plus diagnostic rules) scored entity-level on validation fold 0.

Inputs : artifacts/preprocessed/fold0/train_source{1,2,3}.parquet (names and addresses normalised with the
         dictionaries learned without fold 0), artifacts/validation/* (folds, truth, positive pairs, vocabulary).
Outputs: eda/results/s14_rule_baseline.json, eda/rule_baseline.md, artifacts/baseline/fold0/<rule>.parquet
         (every pair each rule links for fold-0 entities, with its label and one-to-one flag).
Run from the project root:  python eda/scripts/s14_rule_baseline.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import addresses as ad  # noqa: E402
import evaluation as ev  # noqa: E402
import rules as R  # noqa: E402
import validation as va  # noqa: E402

PRE, VAL = "artifacts/preprocessed/fold0", "artifacts/validation"
OUT_JSON, OUT_MD, OUT_PAIRS = "eda/results/s14_rule_baseline.json", "eda/rule_baseline.md", "artifacts/baseline/fold0"
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
RES = {}
os.makedirs(OUT_PAIRS, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Identifier evidence for rules 2-5 (strong identifier, phone, e-mail, registration id)
# --------------------------------------------------------------------------- #
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE = r"(\+?\d[\d\s().\-]{6,}\d)"
URL = r"(?i)(https?://|www\.|\.(com|net|org|co\.in|in|co|io|biz)\b)"
CIN = r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b"
GSTIN = r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"
EIN = r"\b\d{2}-\d{7}\b"
LONG_DIGITS = r"\b\d{9,}\b"

ident = {}
for k in (1, 2, 3):
    df = pl.read_parquet(f"{PRE}/train_source{k}.parquet", columns=["entity_id", "business_name", "business_address", "name_flags"])
    row = {"rows": df.height}
    for col in ("business_name", "business_address"):
        c = pl.col(col).fill_null("")
        digits_in_phone = c.str.extract(PHONE, 1).fill_null("").str.count_matches(r"\d")
        stats = df.select(
            (c.str.contains(EMAIL)).sum().alias("email"),
            (digits_in_phone >= 7).sum().alias("phone_like_7plus_digits"),
            (digits_in_phone >= 10).sum().alias("phone_like_10plus_digits"),
            (c.str.contains(URL)).sum().alias("url_or_domain"),
            (c.str.contains(CIN)).sum().alias("cin_like"), (c.str.contains(GSTIN)).sum().alias("gstin_like"),
            (c.str.contains(EIN)).sum().alias("ein_like"), (c.str.contains(LONG_DIGITS)).sum().alias("digit_run_9plus"),
        ).to_dicts()[0]
        row[col] = {kk: int(v) for kk, v in stats.items()}
    row["name_flags"] = {f: int(df.select(pl.col("name_flags").list.contains(f).sum()).item()) for f in ("phone_stripped", "domain_style", "url_segment")}
    ident[f"S{k}"] = row
    del df
RES["identifier_evidence"] = ident
log("identifier evidence", json.dumps(ident))

# --------------------------------------------------------------------------- #
# 2. Tables
# --------------------------------------------------------------------------- #
COLS = ["entity_id", "country", *R.NAME_KEYS, "addr_norm", "addr_sorted", "house_number", "street", "city_norm", "state",
        "state_recognised", "postal_code"]
folds = pl.read_parquet(f"{VAL}/entity_folds.parquet", columns=["entity_id", "fold", "country", "n_true"])
s1_all = pl.read_parquet(f"{PRE}/train_source1.parquet", columns=COLS).join(folds.select("entity_id", "fold"), on="entity_id", how="left")
s1_f0 = s1_all.filter(pl.col("fold") == 0).drop("fold")
s1_all = s1_all.drop("fold")
fold0_ids = set(s1_f0["entity_id"].to_list())
POOL_FILES = [f"{PRE}/train_source{k}.parquet" for k in (2, 3)]


def load_pool(cols):
    """The Source 2/3 pool with only the columns a rule needs (the full 15-column pool is ~3.5 GB)."""
    return pl.concat([pl.read_parquet(f, columns=list(dict.fromkeys(cols))) for f in POOL_FILES])


pool = load_pool(["entity_id", "country", "name_core", "addr_norm", "house_number", "street", "city_norm", "state", "state_recognised", "postal_code"])
truth = va.entity_truth(VAL, folds=[0])
positives = pl.read_parquet(f"{VAL}/positive_pairs.parquet").filter(pl.col("fold") == 0).select("source1_entity_id", "matched_id", "src")
countries = dict(zip(*[folds.filter(pl.col("fold") == 0)[c].to_list() for c in ("entity_id", "country")]))
country_df = folds.filter(pl.col("fold") == 0).select(pl.col("entity_id").alias("source1_entity_id"), "country")
positives_c = positives.join(country_df, on="source1_entity_id", how="left")
compat = ad.AddressVocab.load(f"{VAL}/address_vocab_excl_fold0.json").compatible
n_true_map = dict(zip(*[folds.filter(pl.col("fold") == 0)[c].to_list() for c in ("entity_id", "n_true")]))
F_EMPTY = sum(1 for v in truth.values() if not v) / len(truth)
log(f"tables: S1 all {s1_all.height:,}, fold 0 {s1_f0.height:,}, pool {pool.height:,}, fold-0 positives {positives.height:,}, F(empty)={F_EMPTY:.5f}")

cov = {}
for name, df in (("S1", s1_all), ("pool", pool)):
    cov[name] = {c: round(float(df[c].is_not_null().mean()), 4) for c in ("name_core", "addr_norm", "house_number", "street", "city_norm", "state", "postal_code")}
    cov[name]["state_recognised"] = round(float(df["state_recognised"].mean()), 4)
RES["key_coverage"] = cov
log("coverage", cov)


# --------------------------------------------------------------------------- #
# 3. Rules
# --------------------------------------------------------------------------- #
def per_country_precision(pairs: pl.DataFrame) -> dict:
    lab = (pairs.join(positives.select("source1_entity_id", "matched_id").with_columns(pl.lit(1).alias("label")),
                      on=["source1_entity_id", "matched_id"], how="left").with_columns(pl.col("label").fill_null(0))
                .join(country_df, on="source1_entity_id", how="left"))
    g = lab.group_by("country").agg(pl.len().alias("n"), pl.col("label").mean().alias("precision"))
    return {r["country"]: {"pairs": int(r["n"]), "precision": round(float(r["precision"]), 5)} for r in g.to_dicts()}


def candidate_stats(pairs: pl.DataFrame) -> dict:
    n = pairs.group_by("source1_entity_id").agg(pl.len().alias("n"))
    counts = np.zeros(len(truth), dtype=np.int64)
    counts[: n.height] = n["n"].to_numpy()
    hit = positives_c.join(pairs, on=["source1_entity_id", "matched_id"], how="inner")
    by_c = {c: round(hit.filter(pl.col("country") == c).height / positives_c.filter(pl.col("country") == c).height, 5) for c in ("US", "India")}
    return {"pairs": int(pairs.height), "mean": round(float(counts.mean()), 3), "p50": float(np.percentile(counts, 50)),
            "p95": float(np.percentile(counts, 95)), "max": int(counts.max()), "entities_with_candidates": int((counts > 0).sum()),
            "pair_recall": round(hit.height / positives.height, 5), "recall_by_country": by_c}


rule_results = {}
saved_pairs = {}
del pool
for rule in R.RULES:
    t0 = time.time()
    keys = list(R.NAME_KEYS) + list(rule.keys) if rule.name_union else list(rule.keys)
    pool = load_pool(["entity_id", "country", *keys, *(R.POST_COLUMNS.get(rule.post, ()) if rule.post else ())])
    if rule.post:
        pairs_all = R.apply_rule(rule, s1_all, pool, compat)
        n_s1 = pairs_all.group_by("matched_id").agg(pl.col("source1_entity_id").n_unique().alias("n_s1"))
        pairs_all = pairs_all.join(n_s1, on="matched_id")
        f0 = pairs_all.filter(pl.col("source1_entity_id").is_in(list(fold0_ids)))
        raw = f0.select("source1_entity_id", "matched_id")
        o2o = f0.filter(pl.col("n_s1") == 1).select("source1_entity_id", "matched_id")
        del pairs_all
    else:
        raw_k = R.apply_rule(rule, s1_f0, pool, keep_keys=True)
        raw = raw_k.select("source1_entity_id", "matched_id")
        o2o = R.one_to_one_by_key(raw_k, s1_all, rule.keys)
        del raw_k
    del pool
    variants = {"raw": raw, "1to1": o2o, "1to1_cap11": R.cap_predictions(o2o, 11)}
    out = {"title": rule.title, "keys": list(rule.keys), "post": rule.post, "name_union": rule.name_union, "note": rule.note,
           "seconds": None, "variants": {}}
    for vname, p in variants.items():
        r = R.evaluate_pairs(p, truth, positives, countries, key=rule.key, variant=vname)
        d = r.as_dict()
        d["precision_by_country"] = per_country_precision(p)
        out["variants"][vname] = d
    out["candidate_stats_raw"] = candidate_stats(raw)
    out["seconds"] = round(time.time() - t0, 1)
    rule_results[rule.key] = out
    lab = raw.join(o2o.with_columns(pl.lit(True).alias("one_to_one")), on=["source1_entity_id", "matched_id"], how="left") \
             .with_columns(pl.col("one_to_one").fill_null(False))
    lab = lab.join(positives.select("source1_entity_id", "matched_id").with_columns(pl.lit(1, dtype=pl.Int8).alias("label")),
                   on=["source1_entity_id", "matched_id"], how="left").with_columns(pl.col("label").fill_null(0))
    lab.write_parquet(f"{OUT_PAIRS}/{rule.key}.parquet")
    if out["variants"]["1to1"]["pair_precision"] >= 0.99:      # only candidates for the automatic tier are kept in memory
        saved_pairs[rule.key] = o2o
    del raw, o2o, lab, variants
    v = out["variants"]
    log(f"{rule.key:4s} {rule.title[:60]:60s} raw: pairs={v['raw']['n_pairs']:>9,} P={v['raw']['pair_precision']:.4f} R={v['raw']['pair_recall']:.4f} "
        f"F={v['raw']['macro_f05']:.4f} | 1to1: pairs={v['1to1']['n_pairs']:>9,} P={v['1to1']['pair_precision']:.4f} R={v['1to1']['pair_recall']:.4f} "
        f"F={v['1to1']['macro_f05']:.4f} singFP={v['1to1']['singleton_false_positives']:,} merges={v['1to1']['entities_with_false_merge']:,} ({out['seconds']}s)")
RES["rules"] = rule_results

# --------------------------------------------------------------------------- #
# 4. Classification and the composite baseline
# --------------------------------------------------------------------------- #
A_PRECISION, A_SINGLETON_FP_SHARE = 0.99, 0.01
B_MIN_RECALL, B_MAX_MEAN_CANDIDATES = 0.05, 50.0
classification = {}
for key, out in rule_results.items():
    v1 = out["variants"]["1to1"]
    cand = out["candidate_stats_raw"]
    a_ok = (v1["n_pairs"] > 0 and v1["pair_precision"] >= A_PRECISION and v1["delta_vs_empty"] > 0
            and v1["singleton_false_positives"] <= A_SINGLETON_FP_SHARE * max(v1["entities_affected"], 1))
    b_ok = cand["pair_recall"] >= B_MIN_RECALL and cand["mean"] <= B_MAX_MEAN_CANDIDATES
    classification[key] = {"A_automatic": bool(a_ok), "B_candidate_signal": bool(b_ok), "C_feature": bool(cand["pairs"] > 0),
                           "pair_precision_1to1": v1["pair_precision"], "pair_recall_raw": cand["pair_recall"], "mean_candidates_raw": cand["mean"]}
RES["classification_thresholds"] = {"A": f"one-to-one variant: pair precision >= {A_PRECISION}, singleton false positives <= {A_SINGLETON_FP_SHARE:.0%} of affected entities, macro F0.5 above the empty prediction",
                                    "B": f"raw variant: pair recall >= {B_MIN_RECALL:.0%} with mean candidates per entity <= {B_MAX_MEAN_CANDIDATES:.0f}",
                                    "C": "any rule that links pairs (its precision is the feature's likelihood)"}
RES["classification"] = classification
log("classification", json.dumps(classification))

# greedy forward union of the A rules (one-to-one variants), ordered by marginal macro F0.5
a_keys = [k for k, c in classification.items() if c["A_automatic"]]
chosen, current, steps = [], None, []
remaining = list(a_keys)
while remaining:
    best = None
    for k in remaining:
        cand_pairs = R.union_pairs([saved_pairs[k]] + ([current] if current is not None else []))
        r = R.evaluate_pairs(cand_pairs, truth, positives, countries, key="+".join(chosen + [k]), variant="union")
        if best is None or r.macro_f05 > best[1].macro_f05:
            best = (k, r, cand_pairs)
    prev = steps[-1]["macro_f05"] if steps else F_EMPTY
    if best[1].macro_f05 <= prev + 1e-9:
        break
    chosen.append(best[0]); current = best[2]; remaining.remove(best[0])
    steps.append({"added": best[0], "macro_f05": best[1].macro_f05, "gain": round(best[1].macro_f05 - prev, 5), "pairs": best[1].n_pairs,
                  "pair_precision": best[1].pair_precision, "pair_recall": best[1].pair_recall,
                  "singleton_false_positives": best[1].singleton_false_positives, "entities_with_false_merge": best[1].entities_with_false_merge})
    log(f"greedy union + {best[0]}: F={best[1].macro_f05:.5f} pairs={best[1].n_pairs:,} P={best[1].pair_precision:.4f} R={best[1].pair_recall:.4f}")
RES["greedy_union"] = {"order": chosen, "steps": steps, "rejected": remaining}

if current is not None:
    comp = R.evaluate_pairs(current, truth, positives, countries, key="composite", variant="union").as_dict()
    comp["precision_by_country"] = per_country_precision(current)
    # per-entity effect versus the empty prediction
    pred = {}
    for s, m in zip(current["source1_entity_id"].to_list(), current["matched_id"].to_list()):
        pred.setdefault(s, []).append(m)
    res = ev.macro_f05(truth, pred, strict=False, keep_per_entity=True)
    improved = hurt = unchanged = 0
    k_hist = {}
    for s in res.per_entity:
        f_empty = 1.0 if s.n_true == 0 else 0.0
        if s.f05 > f_empty + 1e-12:
            improved += 1
        elif s.f05 < f_empty - 1e-12:
            hurt += 1
        else:
            unchanged += 1
        kc = "0" if s.n_true == 0 else ("1" if s.n_true == 1 else ("2" if s.n_true == 2 else "3+"))
        k_hist.setdefault(kc, []).append(s.f05)
    comp["entities_improved_vs_empty"] = improved; comp["entities_hurt_vs_empty"] = hurt; comp["entities_unchanged"] = unchanged
    comp["macro_f05_by_k_class"] = {k: round(float(np.mean(v)), 5) for k, v in sorted(k_hist.items())}
    comp["official_report"] = va.score(pred, VAL, folds=[0])
    RES["composite"] = comp
    current.write_parquet(f"{OUT_PAIRS}/composite_A.parquet")
    log("composite", json.dumps({k: comp[k] for k in ("n_pairs", "pair_precision", "pair_recall", "macro_f05", "singleton_false_positives",
                                                    "entities_with_false_merge", "entities_improved_vs_empty", "entities_hurt_vs_empty")}))

# union of B rules as a candidate set: recall and volume
b_keys = [k for k, c in classification.items() if c["B_candidate_signal"]]
if b_keys:
    b_union = R.union_pairs([pl.read_parquet(f"{OUT_PAIRS}/{k}.parquet", columns=["source1_entity_id", "matched_id"]) for k in b_keys])
    RES["B_union_candidates"] = {"rules": b_keys, **candidate_stats(b_union)}
    log("B union", RES["B_union_candidates"])

json.dump(RES, open(OUT_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT_JSON)
