"""Reliability of parsed address components on true pairs versus same-name negatives.

Builds the fold-safe address vocabulary (pairs of fold 0 excluded), parses both sides of 300,000 true
pairs, 200,000 same-name hard negatives and the near-miss negatives, and reports for each component:
presence, agreement, conflict, and the likelihood ratio between matches and non-matches.
Run from the project root:  python eda/scripts/s13_address_reliability.py
"""
import os
import json, re, sys, time
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import addresses as ad  # noqa: E402

PQ = os.path.join(os.environ.get("BER_SCRATCH", "artifacts/_scratch"), "pq")   # parquet cache, built by s0_inventory.py
TRAIN = "student_resource/dataset/train"
OUT = "eda/results/s13_address_reliability.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
R = {}

folds = pl.read_parquet("artifacts/validation/entity_folds.parquet", columns=["entity_id", "fold"])
fold0 = set(folds.filter(pl.col("fold") == 0)["entity_id"].to_list())
t0 = time.time()
vocab = ad.build_vocab(TRAIN, "artifacts/validation/address_vocab_excl_fold0.json", exclude_entity_ids=fold0)
R["vocab_excl_fold0"] = {"seconds": round(time.time() - t0, 1), "states_aliases": {c: len(m) for c, m in vocab.states.items()},
                        "cities": {c: len(s) for c, s in vocab.cities.items()}, "compatible": {c: {k: sorted(v) for k, v in d.items()} for c, d in vocab.compatible.items()}}
R["vocab_examples"] = {"US aliases of tx": sorted(a for a, canon in vocab.states["US"].items() if canon == "tx")[:12],
                       "India aliases of maharashtra": sorted(a for a, canon in vocab.states["India"].items() if canon == "maharashtra")[:12],
                       "India aliases of orissa": sorted(a for a, canon in vocab.states["India"].items() if canon == "orissa")}
ad.build_vocab(TRAIN, "artifacts/address_vocab_all_train.json")
log("vocab built", R["vocab_excl_fold0"])

s1 = pl.read_parquet(f"{PQ}/train_source1.parquet", columns=["entity_id", "business_address", "country"])
s23 = pl.concat([pl.read_parquet(f"{PQ}/train_source{k}.parquet", columns=["entity_id", "business_address"]).with_columns(pl.lit(f"S{k}").alias("src")) for k in (2, 3)])
pairs = pl.read_parquet(f"{PQ}/train_pairs_pos.parquet")
def attach(df):
    return (df.join(s1.rename({"entity_id": "source1_entity_id", "business_address": "a1"}), on="source1_entity_id")
              .join(s23.rename({"entity_id": "matched_id", "business_address": "a2"}), on="matched_id"))
P = attach(pairs.sample(n=300_000, seed=12)).to_dicts()
hard_all = pl.read_parquet(f"{PQ}/sample_hard_feats.parquet", columns=["source1_entity_id", "matched_id", "addr_jac"])
H = attach(hard_all.sample(n=200_000, seed=13)).to_dicts()
NM = attach(hard_all.filter(pl.col("addr_jac") >= 0.5)).to_dicts()
log("samples", len(P), len(H), len(NM))

def digits(x): return re.sub(r"\D", "", x or "")
def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 0.0

def features(r1, r2, vocab):
    f = {}
    both = lambda x, y: x is not None and y is not None
    # house number
    if both(r1.house_number, r2.house_number):
        n1, n2 = r1.house_number, r2.house_number
        d1, d2 = digits(n1), digits(n2)
        f["number_both"] = True
        f["number_exact"] = n1 == n2
        f["number_digits_equal"] = d1 == d2 and d1 != ""
        f["number_edit1"] = (not f["number_digits_equal"]) and Levenshtein.distance(d1, d2) == 1
        f["number_absdiff_le10"] = (not f["number_digits_equal"]) and d1.isdigit() and d2.isdigit() and abs(int(d1[:9]) - int(d2[:9])) <= 10
        f["number_conflict"] = not (f["number_digits_equal"] or f["number_edit1"])
        f["number_in_other_numbers"] = n1 in r2.numbers or n2 in r1.numbers
    else:
        f["number_both"] = False
        f["number_one_side_only"] = (r1.house_number is not None) != (r2.house_number is not None)
    # street
    if both(r1.street, r2.street):
        f["street_both"] = True
        f["street_exact"] = r1.street == r2.street
        f["street_jaccard_ge_0.5"] = jac(r1.street.split(), r2.street.split()) >= 0.5
        f["street_fuzzy_ge_85"] = fuzz.ratio(r1.street, r2.street) >= 85
        f["street_conflict"] = fuzz.ratio(r1.street, r2.street) < 60
    if both(r1.street_type, r2.street_type):
        f["street_type_both"] = True
        f["street_type_equal"] = r1.street_type == r2.street_type
    # city
    if both(r1.city_norm, r2.city_norm):
        f["city_both"] = True
        f["city_equal"] = r1.city_norm == r2.city_norm
        f["city_fuzzy_ge_85"] = fuzz.ratio(r1.city_norm, r2.city_norm) >= 85
        f["city_conflict"] = fuzz.ratio(r1.city_norm, r2.city_norm) < 70
    # state
    if both(r1.state, r2.state):
        f["state_both"] = True
        f["state_equal_or_compatible"] = vocab.states_compatible(r1.state, r2.state, r1.country)
        f["state_conflict"] = r1.state_recognised and r2.state_recognised and not f["state_equal_or_compatible"]
    # unit
    if both(r1.unit, r2.unit):
        f["unit_both"] = True
        f["unit_equal"] = r1.unit == r2.unit
        f["unit_conflict"] = r1.unit != r2.unit
    # landmark
    if r1.landmark and r2.landmark:
        f["landmark_both"] = True
        f["landmark_overlap"] = jac(" ".join(r1.landmark).split(), " ".join(r2.landmark).split()) >= 0.5
    # sets
    f["numbers_set_jaccard_ge_0.5"] = jac(r1.numbers, r2.numbers) >= 0.5 if (r1.numbers or r2.numbers) else False
    f["numbers_set_disjoint"] = bool(r1.numbers) and bool(r2.numbers) and not (set(r1.numbers) & set(r2.numbers))
    f["tokens_jaccard_ge_0.5"] = jac(r1.tokens, r2.tokens) >= 0.5
    f["tokens_jaccard_ge_0.3"] = jac(r1.tokens, r2.tokens) >= 0.3
    f["locality_overlap"] = jac(" ".join(r1.locality).split(), " ".join(r2.locality).split()) >= 0.5 if (r1.locality and r2.locality) else False
    f["postal_both"] = both(r1.postal_code, r2.postal_code)
    f["either_missing"] = r1.missing or r2.missing
    return f

def run(rows, label):
    t0 = time.time()
    A = [ad.parse_address(r["a1"], r["country"], vocab) for r in rows]
    B = [ad.parse_address(r["a2"], r["country"], vocab) for r in rows]
    secs = time.time() - t0
    cnt = Counter(); n = len(rows)
    for r1, r2 in zip(A, B):
        for k, v in features(r1, r2, vocab).items():
            if v: cnt[k] += 1
    pres = Counter()
    for side, reps in (("S1", A), ("match", B)):
        for r in reps:
            if r.missing: pres[f"{side} missing"] += 1; continue
            for k in ("house_number", "street", "street_type", "unit", "city", "state", "postal_code", "po_box", "care_of"):
                if getattr(r, k): pres[f"{side} {k}"] += 1
            if r.landmark: pres[f"{side} landmark"] += 1
            if r.state_recognised: pres[f"{side} state_recognised"] += 1
            if r.locality: pres[f"{side} locality"] += 1
    return {"n": n, "addresses_per_second": int(2 * n / secs), "feature_rate_pct": {k: round(100 * v / n, 2) for k, v in sorted(cnt.items())},
            "presence_pct": {k: round(100 * v / n, 2) for k, v in sorted(pres.items())}}, A, B

pos, PA, PB = run(P, "positives"); R["positives"] = pos; log("positives parsed")
neg, HA, HB = run(H, "hard_negatives"); R["hard_negatives"] = neg; log("hard negatives parsed")
nmr, NA, NB = run(NM, "near_miss"); R["near_miss_negatives"] = nmr
# by country and by source for positives
for grp, sel in (("US", [i for i, r in enumerate(P) if r["country"] == "US"]), ("India", [i for i, r in enumerate(P) if r["country"] == "India"]),
                 ("S2", [i for i, r in enumerate(P) if r["src"] == "S2"]), ("S3", [i for i, r in enumerate(P) if r["src"] == "S3"])):
    cnt = Counter()
    for i in sel:
        for k, v in features(PA[i], PB[i], vocab).items():
            if v: cnt[k] += 1
    R[f"positives_{grp}"] = {"n": len(sel), "feature_rate_pct": {k: round(100 * v / len(sel), 2) for k, v in sorted(cnt.items())}}
# likelihood ratios match vs hard negative
lr = {}
for k in sorted(set(pos["feature_rate_pct"]) | set(neg["feature_rate_pct"])):
    p, q = pos["feature_rate_pct"].get(k, 0.0), neg["feature_rate_pct"].get(k, 0.0)
    lr[k] = {"pos_pct": p, "hardneg_pct": q, "nearmiss_pct": nmr["feature_rate_pct"].get(k, 0.0), "LR_pos_vs_hardneg": round(p / q, 1) if q > 0 else (float("inf") if p > 0 else None)}
R["likelihood_ratios"] = lr
# examples
ex = defaultdict(list)
for r, r1, r2 in zip(P, PA, PB):
    f = features(r1, r2, vocab)
    if f.get("number_conflict") and len(ex["positive with number conflict"]) < 6: ex["positive with number conflict"].append((r["a1"], r["a2"], r1.house_number, r2.house_number))
    if f.get("city_conflict") and len(ex["positive with city conflict"]) < 6: ex["positive with city conflict"].append((r["a1"], r["a2"], r1.city_norm, r2.city_norm))
    if f.get("state_conflict") and len(ex["positive with state conflict"]) < 6: ex["positive with state conflict"].append((r["a1"], r["a2"], r1.state, r2.state))
    if not r1.house_number and not r1.missing and len(ex["S1 without house number"]) < 6: ex["S1 without house number"].append(r["a1"])
    if not r2.city_norm and not r2.missing and len(ex["match without city"]) < 6: ex["match without city"].append(r["a2"])
for r, r1, r2 in zip(H, HA, HB):
    f = features(r1, r2, vocab)
    if f.get("number_digits_equal") and f.get("street_fuzzy_ge_85") and len(ex["hard negative with same number and street"]) < 6:
        ex["hard negative with same number and street"].append((r["a1"], r["a2"]))
R["examples"] = dict(ex)
# parse samples for the report
R["parse_samples"] = [ad.parse_address(a, c, vocab).as_dict() for a, c in [
    ("1795 Westchester Drive, High Point, NC", "US"), ("<NULL>, IL, 17814. TANGLEWOOD COURT, HAZEL CREST", "US"), ("0286 ELMWOOD DRIVE, PO BOX 5961, RADCLIFF, KY", "US"),
    ("Plot No.9904/13157, Erabang, Nimapara, Puri, Orissa", "India"), ("316, 3Rd Floor, Ratnanjali Square, Near Gloriya Restaurant, Prenatirth Derasar Road, Satellite, Ahmedabad, Gujarat", "India"),
    ("D/1200 B, 28TH MAIN, 35TH C, CROSS RD, 4TH T BLOCK, BANGALORE SOUTH, BANGALORE, ಕರ್ನಾಟಕ", "India"), ("175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France"),
    ("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")]]
json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
log("DONE")
