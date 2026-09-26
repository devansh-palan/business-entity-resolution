"""Evaluate names.py representations against the generic normaliser on real pairs.

Measures exact-key recovery of true matches per representation, collision behaviour on same-name hard
negatives and on Source 1, the fold-safe transliteration dictionary (learned without fold 0, tested on
fold 0), rule frequencies with examples, and throughput.  Run from the project root:
    python eda/scripts/s11_names_pipeline_eval.py
"""
import os
import json, sys, time
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
import names as nm  # noqa: E402
from translit import INDIC_RE  # noqa: E402

PQ = os.path.join(os.environ.get("BER_SCRATCH", "artifacts/_scratch"), "pq")   # parquet cache, built by s0_inventory.py
OUT = "eda/results/s11_names_pipeline.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
R = {}

s1 = pl.read_parquet(f"{PQ}/train_source1.parquet", columns=["entity_id", "business_name", "country"])
s23 = pl.concat([pl.read_parquet(f"{PQ}/train_source{k}.parquet", columns=["entity_id", "business_name"]) for k in (2, 3)])
pairs = pl.read_parquet(f"{PQ}/train_pairs_pos.parquet")
folds = pl.read_parquet("artifacts/validation/entity_folds.parquet", columns=["entity_id", "fold"])
n1 = pl.read_parquet(f"{PQ}/train_source1_norm.parquet", columns=["entity_id", "name_core_sorted"])
n23 = pl.concat([pl.read_parquet(f"{PQ}/train_source{k}_norm.parquet", columns=["entity_id", "name_core_sorted"]) for k in (2, 3)])

def attach(df):
    return (df.join(s1.rename({"entity_id": "source1_entity_id", "business_name": "name1"}), on="source1_entity_id")
              .join(s23.rename({"entity_id": "matched_id", "business_name": "name2"}), on="matched_id")
              .join(folds.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
              .join(n1.rename({"entity_id": "source1_entity_id", "name_core_sorted": "gen1"}), on="source1_entity_id")
              .join(n23.rename({"entity_id": "matched_id", "name_core_sorted": "gen2"}), on="matched_id"))

# ------------------------------------------------------------------ transliteration dictionary: learn without fold 0
allp = attach(pairs.select("source1_entity_id", "matched_id"))
indic = allp.filter(pl.col("name2").str.contains(r"[\u0900-\u0DFF]"))
train_ind = indic.filter(pl.col("fold") != 0)
t0 = time.time()
tdict = nm.learn_translit_dict(train_ind["name1"].to_list(), train_ind["name2"].to_list())
R["translit_dict"] = {"learned_from_pairs": train_ind.height, "entries": len(tdict), "seconds": round(time.time() - t0, 1),
                      "examples": list(tdict.items())[:20]}
nm.save_translit_dict(tdict, "artifacts/validation/translit_dict_excl_fold0.json")
full_dict = nm.learn_translit_dict(indic["name1"].to_list(), indic["name2"].to_list())
nm.save_translit_dict(full_dict, "artifacts/translit_dict_all_train.json")
R["translit_dict"]["entries_all_train"] = len(full_dict)
log("dictionary learned", len(tdict))
# evaluate on fold-0 Indic positives
test_ind = indic.filter(pl.col("fold") == 0)
cov = hit = exact_dict = exact_rules = sim_dict = sim_rules = 0
ex_ok, ex_bad = [], []
for n1_, n2_ in zip(test_ind["name1"].to_list(), test_ind["name2"].to_list()):
    a = nm.normalize_name(n1_)
    bd = nm.normalize_name(n2_, tdict)
    br = nm.normalize_name(n2_, None)
    toks = [t for t in bd.tokens if INDIC_RE.search(t)]
    cov += len(toks); hit += sum(1 for t in toks if t in tdict)
    ed = a.translit_core_sorted == bd.translit_core_sorted; er = a.translit_core_sorted == br.translit_core_sorted
    exact_dict += ed; exact_rules += er
    sim_dict += Levenshtein.normalized_similarity(a.translit_core_sorted, bd.translit_core_sorted)
    sim_rules += Levenshtein.normalized_similarity(a.translit_core_sorted, br.translit_core_sorted)
    if ed and len(ex_ok) < 6: ex_ok.append((n1_, n2_, bd.translit_core))
    if not ed and len(ex_bad) < 8: ex_bad.append((n1_, n2_, bd.translit_core, br.translit_core))
n = test_ind.height
R["translit_eval_fold0"] = {"pairs": n, "token_coverage_pct": round(100 * hit / max(cov, 1), 2),
                            "exact_core_match_with_dictionary_pct": round(100 * exact_dict / n, 2), "exact_core_match_rules_only_pct": round(100 * exact_rules / n, 2),
                            "mean_char_similarity_dictionary": round(sim_dict / n, 3), "mean_char_similarity_rules_only": round(sim_rules / n, 3),
                            "examples_matched": ex_ok, "examples_unmatched": ex_bad}
log("translit eval done")

# ------------------------------------------------------------------ representation recovery on positives and hard negatives
ps = attach(pairs.sample(n=400_000, seed=3))
hn = attach(pl.read_parquet(f"{PQ}/sample_hard_feats.parquet", columns=["source1_entity_id", "matched_id"]).sample(n=300_000, seed=4))

def evaluate(df, label):
    t0 = time.time()
    A = nm.normalize_names(df["name1"].to_list(), tdict)
    B = nm.normalize_names(df["name2"].to_list(), tdict)
    secs = time.time() - t0
    n = len(A); c = Counter(); rule = Counter(); ex = {}
    gen_eq = int((df["gen1"] == df["gen2"]).sum())
    for a, b, g1, g2 in zip(A, B, df["gen1"].to_list(), df["gen2"].to_list()):
        eqs = {
            "original": a.original == b.original,
            "clean": a.clean == b.clean,
            "core": a.core == b.core,
            "core_sorted": a.core_sorted == b.core_sorted,
            "translit_core_sorted": a.translit_core_sorted == b.translit_core_sorted,
            "compact": a.compact == b.compact and a.compact != "",
            "any variant": bool(set([a.translit_core] + a.variants) & set([b.translit_core] + b.variants)),
            "acronym (when S2/S3 is an acronym)": b.flags.get("is_acronym", False) and a.acronym == b.core,
            "compact prefix (domain-style)": b.flags.get("domain_style", False) and a.compact != "" and (a.compact.startswith(b.compact) or b.compact.startswith(a.compact)),
        }
        any_key = eqs["translit_core_sorted"] or eqs["compact"] or eqs["any variant"] or eqs["acronym (when S2/S3 is an acronym)"]
        eqs["ANY exact key (translit_core_sorted / compact / variant / acronym)"] = any_key
        eqs["ANY key or compact prefix"] = any_key or eqs["compact prefix (domain-style)"]
        for k, v in eqs.items(): c[k] += v
        for f, v in b.flags.items():
            if v:
                rule[f] += 1
                if len(ex.setdefault(f, [])) < 4: ex[f].append((a.original, b.original, b.translit_core))
        if b.honorific: rule["honorific"] += 1; ex.setdefault("honorific", []); len(ex["honorific"]) < 4 and ex["honorific"].append((a.original, b.original, b.core))
        if b.designations: rule["designations"] += 1
        if b.alias_prefix: rule["alias_prefix"] += 1
        if b.legal_forms != a.legal_forms: rule["legal_forms_differ"] += 1
        if b.script != "latin": rule["non_latin_script"] += 1
    out = {"pairs": n, "seconds_for_2n_names": round(secs, 1), "names_per_second": int(2 * n / secs),
           "generic_name_core_sorted_equal_pct": round(100 * gen_eq / n, 2),
           "equal_pct": {k: round(100 * v / n, 2) for k, v in c.items()},
           "rule_frequency_pct_S23_side": {k: round(100 * v / n, 3) for k, v in rule.most_common()},
           "examples": ex}
    R[label] = out
    log(label, {k: v for k, v in out["equal_pct"].items()})
evaluate(ps, "positives")
evaluate(hn, "hard_negatives")

# ------------------------------------------------------------------ collision cost on Source 1
t0 = time.time()
reps = nm.normalize_names(s1["business_name"].to_list())
R["s1_normalisation_seconds"] = round(time.time() - t0, 1)
ctry = s1["country"].to_list()
def coll(keys):
    c = Counter(zip(ctry, keys)); shared = sum(v for v in c.values() if v >= 2)
    return {"distinct": len(c), "pct_entities_sharing": round(100 * shared / len(keys), 3)}
R["collision_on_S1"] = {
    "original": coll([r.original for r in reps]),
    "clean": coll([r.clean for r in reps]),
    "core": coll([r.core for r in reps]),
    "core_sorted": coll([r.core_sorted for r in reps]),
    "translit_core_sorted": coll([r.translit_core_sorted for r in reps]),
    "compact": coll([r.compact for r in reps]),
    "generic name_core_sorted (normalize.py)": coll(n1.join(s1.select("entity_id"), on="entity_id")["name_core_sorted"].to_list()) if False else coll(s1.join(n1, on="entity_id")["name_core_sorted"].to_list()),
}
R["s1_flag_rates_pct"] = {k: round(100 * v / len(reps), 3) for k, v in Counter(f for r in reps for f, v in r.flags.items() if v).most_common()}
R["s1_honorific_rate_pct"] = round(100 * sum(1 for r in reps if r.honorific) / len(reps), 3)
R["s1_designation_rate_pct"] = round(100 * sum(1 for r in reps if r.designations) / len(reps), 3)
R["s1_empty_core"] = sum(1 for r in reps if not r.core)
log("collisions done")

json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
for k, v in R.items():
    print("\n##", k); print(json.dumps(v, ensure_ascii=False, default=str)[:2500])
log("DONE")
