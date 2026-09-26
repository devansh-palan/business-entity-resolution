"""Render eda/results/s18_matcher.json as eda/matcher_baseline.md.  Run from the project root:  python eda/scripts/s18_report.py"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = json.load(open("eda/results/s18_matcher.json", encoding="utf-8"))
L = []
add = L.append
num = lambda x: f"{x:,}"
f4 = lambda x: f"{x:.4f}"
NAMES = {"logistic_regression": "logistic regression", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
models = list(R["report"].keys())
best = R["best_model"]
sp = R["split"]

add("# First pairwise matchers, scored at the entity level (fold 0)\n")
add(f"Three models were trained on the pairwise feature matrix of `pair_features.md` and turned into per-entity match lists. The split is by Source 1 entity "
    f"(feature bucket): 24 buckets train the models, 4 buckets ({num(sp['entities_tune'])} entities) choose the decision rule, and 4 untouched buckets "
    f"({num(sp['entities_report'])} entities, {num(sp['singletons_report'])} of them singletons) give every number reported below, through the official macro F0.5 "
    "evaluator (`evaluation.macro_f05_from_rows`). Nothing here touches the test set.\n")
ts = R.get("train_sample", {})
if ts:
    add(f"**Training sample.** All {num(ts['positives'])} positives of the training buckets plus a uniform {100 * ts['neg_rate']:.1f}% of their {num(ts['negatives_total'])} "
        f"negatives ({num(ts['negatives_kept'])}), {num(ts['rows'])} rows in all; probabilities are corrected back to the true prior afterwards "
        "(odds divided by 1 / rate), so the thresholds below are on the calibrated scale. A full-data fit needs the 64 GB machine.\n")
add("## 1. Models\n")
add("| model | rows | fit | pair AUC (report half) | average precision | notes |")
add("|---|---|---|---|---|---|")
for m in models:
    mi = R["models"].get(m, {})
    pl_ = R["pair_level_report_half"][m]
    note = ""
    if m == "logistic_regression":
        note = f"{mi.get('iterations', '?')} lbfgs iterations; standardised features, nulls as 0"
    elif m == "lightgbm":
        note = f"{mi.get('best_iteration', '?')} trees, {mi['params']['num_leaves']} leaves, lr {mi['params']['learning_rate']}; early stopping on held-out pairs (val AUC {mi.get('val_auc', 0):.4f})"
    elif m == "xgboost":
        note = f"{mi.get('best_iteration', '?')} trees, depth {mi['params']['max_depth']}, eta {mi['params']['eta']}"
    add(f"| {NAMES.get(m, m)} | {num(mi.get('rows', 0))} | {mi.get('seconds', 0):.0f} s | {pl_['auc']:.4f} | {pl_['average_precision']:.4f} | {note} |")
add("")
lr = R["models"].get("logistic_regression", {})
if lr.get("top_coefficients"):
    add("Logistic regression, largest standardised coefficients (sign = direction of a match): " +
        ", ".join(f"`{f}` {c:+.2f}" for f, c in lr["top_coefficients"][:15]) + ".\n")
for m in ("lightgbm", "xgboost"):
    if R["models"].get(m, {}).get("top_gain"):
        add(f"{NAMES[m]}, share of total gain: " + ", ".join(f"`{f}` {g:.3f}" for f, g in R["models"][m]["top_gain"][:15]) + ".\n")

add("## 2. From pair probabilities to match lists\n")
add("Rules swept on the tuning half: a probability threshold from 0.25 to 0.95, with or without pool-side uniqueness (a Source 2/3 record is given only to the "
    "entity that scores it highest, the one-to-one structure of the truth), and with or without a relative gate (keep a pair only if its probability is at "
    "least 0.5 or 0.8 times the entity's best). The rule with the highest macro F0.5 on the tuning half is then applied, unchanged, to the report half.\n")
add("| model | best rule (tuning half) | macro F0.5 tuning half | best plain threshold | its F0.5 | threshold + uniqueness | its F0.5 |")
add("|---|---|---|---|---|---|---|")
for m in models:
    c = R["chosen"][m]
    bp, bu = c["best_plain_threshold"], c["best_unique_only"]
    add(f"| {NAMES.get(m, m)} | {c['rule']} | {f4(c['macro_f05_tune'])} | {bp['rule']} | {f4(bp['macro_f05_tune'])} | {bu['rule']} | {f4(bu['macro_f05_tune'])} |")
add("")
add(f"An empty prediction scores {f4(R['reference']['empty_prediction'])} on the report half (the singleton share); the deterministic automatic tier of "
    "`rule_baseline.md` scored 0.612 on the whole fold.\n")

add("## 3. Entity-level results on the report half\n")
add("| | " + " | ".join(NAMES.get(m, m) for m in models) + " |")
add("|---|" + "---|" * len(models))
rows = [("**macro F0.5**", lambda r: f"**{f4(r['macro_f05'])}**"), ("macro F0.5 on the tuning half (rule chosen there)", None),
        ("singletons: share predicted empty", lambda r: f4(r["singletons"]["predicted_empty_share"])),
        ("singletons: entities given a false match", lambda r: num(r["singletons"]["false_positive_entities"])),
        ("one-match entities: mean F0.5", lambda r: f4(r["one_match"]["mean_f05"])), ("one-match entities: fully correct", lambda r: f"{100 * r['one_match']['fully_correct_share']:.1f}%"),
        ("multi-match entities: mean F0.5", lambda r: f4(r["multi_match"]["mean_f05"])), ("multi-match entities: fully correct", lambda r: f"{100 * r['multi_match']['fully_correct_share']:.1f}%"),
        ("multi-match entities: true / predicted matches per entity", lambda r: f"{r['multi_match']['mean_true']:.2f} / {r['multi_match']['mean_pred']:.2f}"),
        ("Source 2: macro F0.5 over entities with an S2 match", lambda r: f4(r["by_source"]["S2"]["macro_f05"])), ("Source 2: pair precision / recall", lambda r: f"{f4(r['by_source']['S2']['pair_precision'])} / {f4(r['by_source']['S2']['pair_recall'])}"),
        ("Source 3: macro F0.5 over entities with an S3 match", lambda r: f4(r["by_source"]["S3"]["macro_f05"])), ("Source 3: pair precision / recall", lambda r: f"{f4(r['by_source']['S3']['pair_precision'])} / {f4(r['by_source']['S3']['pair_recall'])}"),
        ("US: macro F0.5", lambda r: f4(r["by_country"].get("US", {}).get("mean_f05", 0))), ("India: macro F0.5", lambda r: f4(r["by_country"].get("India", {}).get("mean_f05", 0))),
        ("entities with a false merge", lambda r: num(r["false_merges"]["entities_with_fp"])), ("entities with a missed match", lambda r: num(r["false_merges"]["entities_with_fn"])),
        ("entities fully correct", lambda r: f"{num(r['false_merges']['fully_correct'])} ({100 * r['false_merges']['fully_correct'] / r['n_entities']:.1f}%)"),
        ("pairs predicted / true pairs", lambda r: f"{num(r['pairs_predicted'])} / {num(r['tp'] + r['fn'])}"),
        ("pair precision / recall / F0.5 (information only)", lambda r: f"{f4(r['pair_precision'])} / {f4(r['pair_recall'])} / {f4(r['pair_f05'])}")]
for label, fn in rows:
    if fn is None:
        add(f"| {label} | " + " | ".join(f4(R["report_on_tune_half"][m]["macro_f05"]) for m in models) + " |")
    else:
        add(f"| {label} | " + " | ".join(fn(R["report"][m]) for m in models) + " |")
add("")
add("Mean F0.5 by number of true matches (report half):\n")
ks = list(R["report"][best]["by_n_true"].keys())
add("| true matches | entities | " + " | ".join(NAMES.get(m, m) for m in models) + " |")
add("|---|---|" + "---|" * len(models))
for k in ks:
    add(f"| {k} | {num(R['report'][best]['by_n_true'][k]['n'])} | " + " | ".join(f4(R["report"][m]["by_n_true"].get(k, {}).get("mean_f05", 0)) for m in models) + " |")
add("")

add(f"## 4. False merges of the best model ({NAMES.get(best, best)})\n")
fm = R["false_merge_profile"]
mm = R["missed_matches"]
add(f"On the report half the rule keeps {num(R['report'][best]['pairs_predicted'])} pairs, {num(R['report'][best]['fp'])} of them wrong, spread over "
    f"{num(R['report'][best]['false_merges']['entities_with_fp'])} entities. Of the {num(mm['true_pairs'])} true pairs, {num(mm['missed_by_candidates'])} were never in the candidate set "
    f"and {num(mm['missed_by_model'])} were scored below the rule.\n")
add("| false merges | pairs | exact name | exact address | house conflict | unit conflict | street conflict | same name, other house | pool address missing | mean p | mean name+address cosine |")
add("|---|---|---|---|---|---|---|---|---|---|---|")
for k, v in fm.items():
    add(f"| {k} | {num(v['n'])} | {100 * v['share_exact_name']:.1f}% | {100 * v['share_exact_address']:.1f}% | {100 * v['share_house_conflict']:.1f}% | {100 * v['share_unit_conflict']:.1f}% "
        f"| {100 * v['share_street_conflict']:.1f}% | {100 * v['share_same_name_diff_house']:.1f}% | {100 * v['share_pool_address_missing']:.1f}% | {v['mean_p']:.3f} | {v['mean_cos_name_addr']:.3f} |")
add("")
add("'Stolen' means the Source 2/3 record belongs to another Source 1 entity in the training truth (a decoy sibling); 'ownerless' means no entity of the "
    "training data claims it.\n")
for kind, exs in R["false_merge_examples"].items():
    add(f"**{kind}**, highest-probability examples:\n")
    for e in exs[:6]:
        add(f"- `{e['business_name']}` | `{e['business_address']}`  ->  `{e['name_2']}` | `{e['address_2']}`  (p = {e['p']:.2f}" + (f", owner {e['true_owner']}" if e.get("true_owner") else "") + ")")
    add("")
if os.path.exists("eda/scripts/s18_conclusions.md"):
    add(open("eda/scripts/s18_conclusions.md", encoding="utf-8").read())
open("eda/matcher_baseline.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:60]))
