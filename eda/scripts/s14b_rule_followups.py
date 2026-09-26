"""Follow-ups on the deterministic baseline (reads artifacts/baseline/fold0/*.parquet written by s14_rule_baseline.py).

(a) extended greedy union over every rule's one-to-one variant, starting from the automatic-tier composite, and a
    pure F-maximising greedy from scratch: does any lower-precision rule still raise macro F0.5, at what cost?
(b) anatomy of the composite's false merges and singleton false positives: what does the pool record's true owner
    look like relative to the wrongly linked entity?
(c) rule 8 refinements: R7 with the no-conflict predicates, and R8 split by street presence.
Run from the project root:  python eda/scripts/s14b_rule_followups.py
"""
import json
import sys
import time

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import rules as R  # noqa: E402
import validation as va  # noqa: E402

PRE, VAL, BASE = "artifacts/preprocessed/fold0", "artifacts/validation", "artifacts/baseline/fold0"
OUT = "eda/results/s14b_rule_followups.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)
RES = {}

folds = pl.read_parquet(f"{VAL}/entity_folds.parquet", columns=["entity_id", "fold", "country", "n_true"])
truth = va.entity_truth(VAL, folds=[0])
positives_all = pl.read_parquet(f"{VAL}/positive_pairs.parquet")
positives = positives_all.filter(pl.col("fold") == 0).select("source1_entity_id", "matched_id", "src")
countries = dict(zip(*[folds.filter(pl.col("fold") == 0)[c].to_list() for c in ("entity_id", "country")]))
F_EMPTY = sum(1 for v in truth.values() if not v) / len(truth)

def load_rule(key, one_to_one=True):
    df = pl.read_parquet(f"{BASE}/{key}.parquet")
    return (df.filter(pl.col("one_to_one")) if one_to_one else df).select("source1_entity_id", "matched_id")

composite = pl.read_parquet(f"{BASE}/composite_A.parquet")
ALL_KEYS = [r.key for r in R.RULES if r.key != "R6"]

# --------------------------------------------------------------------------- #
# (a) extended greedy
# --------------------------------------------------------------------------- #
def greedy(start_pairs, start_keys, candidates, label):
    current, chosen, steps = start_pairs, list(start_keys), []
    prev = R.evaluate_pairs(current, truth, positives, countries).macro_f05 if current is not None else F_EMPTY
    remaining = [k for k in candidates if k not in chosen]
    while remaining:
        best = None
        for k in remaining:
            cand = R.union_pairs([load_rule(k)] + ([current] if current is not None else []))
            r = R.evaluate_pairs(cand, truth, positives, countries)
            if best is None or r.macro_f05 > best[1].macro_f05:
                best = (k, r, cand)
        if best[1].macro_f05 <= prev + 1e-9:
            break
        k, r, cand = best
        steps.append({"added": k, "macro_f05": r.macro_f05, "gain": round(r.macro_f05 - prev, 5), "pairs": r.n_pairs, "pair_precision": r.pair_precision,
                      "pair_recall": r.pair_recall, "singleton_false_positives": r.singleton_false_positives,
                      "entities_with_false_merge": r.entities_with_false_merge, "by_country": r.by_country})
        log(f"{label} + {k}: F={r.macro_f05:.5f} (+{r.macro_f05 - prev:.5f}) P={r.pair_precision:.4f} R={r.pair_recall:.4f} singFP={r.singleton_false_positives:,} merges={r.entities_with_false_merge:,}")
        prev, current, chosen = r.macro_f05, cand, chosen + [k]
        remaining.remove(k)
    return {"order": chosen, "steps": steps, "final_macro_f05": prev, "rejected": remaining}

RES["greedy_from_composite"] = greedy(composite, ["D8", "R1s", "R8u", "R7"], ALL_KEYS, "composite")
RES["greedy_from_scratch_all_rules"] = greedy(None, [], ALL_KEYS, "scratch")

# --------------------------------------------------------------------------- #
# (b) anatomy of the composite's errors
# --------------------------------------------------------------------------- #
cols = ["entity_id", "country", "business_name", "business_address", "name_core", "house_number", "street", "city_norm", "state"]
s1 = pl.read_parquet(f"{PRE}/train_source1.parquet", columns=cols)
pool = pl.concat([pl.read_parquet(f"{PRE}/train_source{k}.parquet", columns=cols) for k in (2, 3)])
lab = composite.join(positives.select("source1_entity_id", "matched_id").with_columns(pl.lit(1).alias("label")), on=["source1_entity_id", "matched_id"], how="left") \
               .with_columns(pl.col("label").fill_null(0))
fm = lab.filter(pl.col("label") == 0)
owner = positives_all.select(pl.col("matched_id"), pl.col("source1_entity_id").alias("owner_id"))
fm = fm.join(owner, on="matched_id", how="left")
fm = fm.join(s1.rename({c: f"{c}_1" for c in cols}), left_on="source1_entity_id", right_on="entity_id_1", how="left")
fm = fm.join(pool.rename({c: f"{c}_2" for c in cols}), left_on="matched_id", right_on="entity_id_2", how="left")
fm = fm.join(s1.rename({c: f"{c}_o" for c in cols}), left_on="owner_id", right_on="entity_id_o", how="left")
n_true_f0 = dict(zip(*[folds.filter(pl.col("fold") == 0)[c].to_list() for c in ("entity_id", "n_true")]))
fm = fm.with_columns(pl.col("source1_entity_id").replace_strict(n_true_f0, default=None).alias("n_true_1"))
anat = {"false_merge_pairs": fm.height, "singleton_entity_pairs": int((fm["n_true_1"] == 0).sum()),
        "record_has_no_owner_anywhere": int(fm["owner_id"].is_null().sum())}
has_owner = fm.filter(pl.col("owner_id").is_not_null())
same = lambda a: (pl.col(f"{a}_1") == pl.col(f"{a}_o"))
anat["owner_same_name_core"] = int(has_owner.select(same("name_core").sum()).item())
anat["owner_same_name_and_house_number"] = int(has_owner.select((same("name_core") & same("house_number")).sum()).item())
anat["owner_same_name_house_number_street"] = int(has_owner.select((same("name_core") & same("house_number") & same("street")).sum()).item())
anat["owner_same_name_house_number_street_city"] = int(has_owner.select((same("name_core") & same("house_number") & same("street") & same("city_norm")).sum()).item())
anat["owner_same_full_address_text"] = int(has_owner.select((same("name_core") & same("business_address")).sum()).item())
anat["owner_in_fold0"] = int(has_owner["owner_id"].is_in(list(truth)).sum())
anat["by_country"] = {r["country_1"]: int(r["n"]) for r in fm.group_by("country_1").agg(pl.len().alias("n")).to_dicts()}
RES["composite_false_merges"] = anat
ex_cols = ["source1_entity_id", "business_name_1", "business_address_1", "matched_id", "business_name_2", "business_address_2", "owner_id", "business_name_o", "business_address_o", "n_true_1"]
RES["false_merge_examples"] = fm.sample(n=min(12, fm.height), seed=3).select(ex_cols).to_dicts()
RES["singleton_fp_examples"] = fm.filter(pl.col("n_true_1") == 0).sample(n=min(8, int((fm["n_true_1"] == 0).sum())), seed=4).select(ex_cols).to_dicts()
log("false merges", json.dumps(anat))

# --------------------------------------------------------------------------- #
# (c) rule 8 refinements
# --------------------------------------------------------------------------- #
compat = __import__("addresses").AddressVocab.load(f"{VAL}/address_vocab_excl_fold0.json").compatible
def with_sides(pairs):
    a = s1.select(pl.col("entity_id").alias("source1_entity_id"), "country", *[pl.col(c).alias(f"{c}_1") for c in ("street", "city_norm", "state")])
    b = pool.select(pl.col("entity_id").alias("matched_id"), *[pl.col(c).alias(f"{c}_2") for c in ("street", "city_norm", "state")])
    rec1 = pl.read_parquet(f"{PRE}/train_source1.parquet", columns=["entity_id", "state_recognised"]).rename({"entity_id": "source1_entity_id", "state_recognised": "state_recognised_1"})
    rec2 = pl.concat([pl.read_parquet(f"{PRE}/train_source{k}.parquet", columns=["entity_id", "state_recognised"]) for k in (2, 3)]).rename({"entity_id": "matched_id", "state_recognised": "state_recognised_2"})
    return pairs.join(a, on="source1_entity_id").join(b, on="matched_id").join(rec1, on="source1_entity_id").join(rec2, on="matched_id")

refine = {}
r7 = with_sides(load_rule("R7"))
for name, mask in (("R7", None), ("R7 + no state conflict", R.no_state_conflict(r7, compat)), ("R7 + no state conflict + city agrees", R.strong_address(r7, compat))):
    p = (r7.filter(mask) if mask is not None else r7).select("source1_entity_id", "matched_id")
    r = R.evaluate_pairs(p, truth, positives, countries)
    refine[name] = {"pairs": r.n_pairs, "pair_precision": r.pair_precision, "pair_recall": r.pair_recall, "macro_f05": r.macro_f05,
                    "singleton_false_positives": r.singleton_false_positives, "entities_with_false_merge": r.entities_with_false_merge, "by_country": r.by_country}
    log(name, refine[name])
r8 = with_sides(load_rule("R8"))
both = r8["street_1"].is_not_null() & r8["street_2"].is_not_null()
for name, mask in (("R8 street on both sides", both), ("R8 street missing on a side", ~both)):
    p = r8.filter(mask).select("source1_entity_id", "matched_id")
    labp = p.join(positives.select("source1_entity_id", "matched_id").with_columns(pl.lit(1).alias("label")), on=["source1_entity_id", "matched_id"], how="left").with_columns(pl.col("label").fill_null(0))
    refine[name] = {"pairs": p.height, "pair_precision": round(float(labp["label"].mean()), 5)}
    log(name, refine[name])
RES["rule8_refinements"] = refine

json.dump(RES, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("saved", OUT)
