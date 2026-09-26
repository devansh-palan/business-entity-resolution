"""Render eda/results/s14_rule_baseline.json as the markdown tables of eda/rule_baseline.md (sections 1-8).

Run from the project root after s14_rule_baseline.py:  python eda/scripts/s14_report.py
The conclusions (section 9) are appended by hand after reading the numbers.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = json.load(open("eda/results/s14_rule_baseline.json", encoding="utf-8"))
L = []
add = L.append
pct = lambda x: f"{100 * x:.2f}%"
num = lambda x: f"{x:,}"

add("# Deterministic high-precision baseline (validation fold 0)\n")
add("Every rule is an exact equality join between Source 1 and the whole Source 2/3 pool (10,320,219 records) on `country` plus the "
    "listed keys of the preprocessed tables (`artifacts/preprocessed/fold0`, built with the transliteration dictionary and address "
    "vocabulary learned without fold 0). Scores are entity-level macro F0.5 over the 441,365 Source 1 entities of fold 0, computed by the "
    "verified evaluator (`src/evaluation.py`); singletons score 1.0 only when predicted empty, so the empty prediction scores "
    f"**{R['composite']['official_report']['true_singleton_rate']:.4f}** and every number below has to beat that.\n")
add("Two refinements are measured for every rule:\n")
add("- **one-to-one (1to1)**: the ground truth maps each Source 2/3 record to at most one Source 1 entity (7,638,365 matched ids, all distinct), "
    "so when a rule links one pool record to several Source 1 entities all but at most one link are false merges; the refinement drops those records "
    "(uniqueness is counted against all 2,206,821 Source 1 entities, exactly as it would be against the test Source 1).")
add("- **cap 11**: no training entity has more than 11 matches (S2 <= 5, S3 <= 6), so an entity linked to 12 or more records is discarded.\n")

# ---- identifiers
add("## 1. Rules 2-5: no identifiers exist in the data\n")
add("The files have four columns (`entity_id`, `business_name`, `business_address`, `country`); the ids are opaque (0.0% of matched pairs share a numeric id, "
    "Spearman correlation 0.0001, see `s2_hidden_ids.json`). Scanning every name and address for identifier-like content:\n")
add("| source | rows | e-mail | phone-like (>=7 digits) | phone-like (>=10 digits) | URL / domain | CIN-like | GSTIN-like | EIN-like | digit run >= 9 | names with phone tail stripped | domain-style names |")
add("|---|---|---|---|---|---|---|---|---|---|---|---|")
for src, d in R["identifier_evidence"].items():
    n, a, f = d["business_name"], d["business_address"], d["name_flags"]
    add(f"| {src} | {num(d['rows'])} | {n['email'] + a['email']} | {num(n['phone_like_7plus_digits'])} name / {num(a['phone_like_7plus_digits'])} addr "
        f"| {num(n['phone_like_10plus_digits'])} name / {num(a['phone_like_10plus_digits'])} addr | {num(n['url_or_domain'])} name / {num(a['url_or_domain'])} addr "
        f"| {n['cin_like'] + a['cin_like']} | {n['gstin_like'] + a['gstin_like']} | {n['ein_like'] + a['ein_like']} | {num(n['digit_run_9plus'] + a['digit_run_9plus'])} "
        f"| {num(f['phone_stripped'])} | {num(f['domain_style'])} |")
add("")

# ---- coverage
add("## 2. Key coverage after preprocessing\n")
add("| table | name_core | addr_norm | house_number | street | city | state | state recognised | postal_code |")
add("|---|---|---|---|---|---|---|---|---|")
for t, c in R["key_coverage"].items():
    add(f"| {t} | {pct(c['name_core'])} | {pct(c['addr_norm'])} | {pct(c['house_number'])} | {pct(c['street'])} | {pct(c['city_norm'])} | {pct(c['state'])} | {pct(c['state_recognised'])} | {pct(c['postal_code'])} |")
add("")

# ---- rule tables
def rule_table(variant: str, title: str):
    add(f"## {title}\n")
    add("| rule | definition | pairs | pair precision | pair recall | macro F0.5 | gain vs empty | false merges (pairs) | missed matches (pairs) | singleton FPs | entities affected | entities with a false merge | entities fully correct |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, out in R["rules"].items():
        v = out["variants"][variant]
        add(f"| {key} | {out['title']} | {num(v['n_pairs'])} | {v['pair_precision']:.4f} | {v['pair_recall']:.4f} | {v['macro_f05']:.4f} | {v['delta_vs_empty']:+.4f} "
            f"| {num(v['fp'])} | {num(v['fn'])} | {num(v['singleton_false_positives'])} | {num(v['entities_affected'])} | {num(v['entities_with_false_merge'])} | {num(v['entities_fully_correct'])} |")
    add("")

rule_table("raw", "3. Rules as written (raw join)")
rule_table("1to1", "4. Rules with the one-to-one refinement")
rule_table("1to1_cap11", "5. One-to-one plus the cap of 11 predictions")

# ---- precision by prediction count and by country
add("## 6. Precision by number of predictions per entity (one-to-one variant)\n")
add("Entities: count of fold-0 entities with that many predictions; precision: share of those predicted pairs that are true.\n")
add("| rule | 1 prediction | 2 | 3-5 | 6-11 | >11 | US precision | India precision |")
add("|---|---|---|---|---|---|---|---|")
for key, out in R["rules"].items():
    v = out["variants"]["1to1"]
    cells = []
    for b in ("1", "2", "3-5", "6-11", ">11"):
        if b in v["precision_by_n_pred"]:
            n, p = v["precision_by_n_pred"][b]
            cells.append(f"{num(n)} ent, {p:.4f}")
        else:
            cells.append("-")
    pc = v["precision_by_country"]
    add(f"| {key} | " + " | ".join(cells) + f" | {pc.get('US', {}).get('precision', float('nan')):.4f} ({num(pc.get('US', {}).get('pairs', 0))}) "
        f"| {pc.get('India', {}).get('precision', float('nan')):.4f} ({num(pc.get('India', {}).get('pairs', 0))}) |")
add("")

# ---- candidate stats
add("## 7. Rules as candidate-generation signals (raw join, fold-0 entities against the whole pool)\n")
add("| rule | pairs | pair recall | recall US | recall India | recall S2 | recall S3 | mean candidates | p50 | p95 | max | entities with >= 1 candidate |")
add("|---|---|---|---|---|---|---|---|---|---|---|---|")
for key, out in R["rules"].items():
    c = out["candidate_stats_raw"]; v = out["variants"]["raw"]
    add(f"| {key} | {num(c['pairs'])} | {c['pair_recall']:.4f} | {c['recall_by_country']['US']:.4f} | {c['recall_by_country']['India']:.4f} "
        f"| {v['by_src_recall']['S2']:.4f} | {v['by_src_recall']['S3']:.4f} | {c['mean']:.2f} | {c['p50']:.0f} | {c['p95']:.0f} | {num(c['max'])} | {num(c['entities_with_candidates'])} |")
add("")
if "B_union_candidates" in R:
    b = R["B_union_candidates"]
    add(f"Union of the B rules ({', '.join(b['rules'])}): {num(b['pairs'])} pairs, pair recall {b['pair_recall']:.4f} "
        f"(US {b['recall_by_country']['US']:.4f}, India {b['recall_by_country']['India']:.4f}), mean {b['mean']:.2f} candidates per entity, "
        f"p95 {b['p95']:.0f}, max {num(b['max'])}, {num(b['entities_with_candidates'])} entities with at least one candidate.\n")

# ---- classification
add("## 8. Classification and the composite baseline\n")
add("Thresholds (fixed before looking at the composite):\n")
for k, v in R["classification_thresholds"].items():
    add(f"- **{k}**: {v}")
add("")
add("| rule | A automatic match | B candidate signal | C feature | 1to1 precision | raw recall | mean candidates |")
add("|---|---|---|---|---|---|---|")
for key, c in R["classification"].items():
    add(f"| {key} | {'yes' if c['A_automatic'] else 'no'} | {'yes' if c['B_candidate_signal'] else 'no'} | {'yes' if c['C_feature'] else 'no'} "
        f"| {c['pair_precision_1to1']:.4f} | {c['pair_recall_raw']:.4f} | {c['mean_candidates_raw']:.2f} |")
add("")
g = R["greedy_union"]
add("Greedy union of the A rules (one-to-one variants), each step adding the rule with the largest macro F0.5 gain:\n")
add("| step | rule added | macro F0.5 | gain | pairs | pair precision | pair recall | singleton FPs | entities with a false merge |")
add("|---|---|---|---|---|---|---|---|---|")
for i, s in enumerate(g["steps"], 1):
    add(f"| {i} | {s['added']} | {s['macro_f05']:.5f} | {s['gain']:+.5f} | {num(s['pairs'])} | {s['pair_precision']:.4f} | {s['pair_recall']:.4f} | {num(s['singleton_false_positives'])} | {num(s['entities_with_false_merge'])} |")
if g["rejected"]:
    add(f"\nA rules that added nothing on top of the others: {', '.join(g['rejected'])}.")
add("")
if "composite" in R:
    c = R["composite"]; o = c["official_report"]
    add(f"**Composite baseline** (union of {' + '.join(g['order'])}): macro F0.5 **{c['macro_f05']:.5f}** on {num(o['n_entities'])} entities "
        f"(empty prediction {o['true_singleton_rate']:.5f}); US {o['by_country']['US']['macro_f05']:.5f}, India {o['by_country']['India']['macro_f05']:.5f}; "
        f"by true match count: " + ", ".join(f"{k}: {v:.4f}" for k, v in c["macro_f05_by_k_class"].items()) + ".")
    add(f"Pairs {num(c['n_pairs'])}, precision {c['pair_precision']:.4f}, recall {c['pair_recall']:.4f}; false merges {num(c['fp'])}, missed matches {num(c['fn'])}, "
        f"singleton false positives {num(c['singleton_false_positives'])}; entities affected {num(c['entities_affected'])}, with a false merge {num(c['entities_with_false_merge'])}, "
        f"fully correct {num(c['entities_fully_correct'])}; versus the empty prediction {num(c['entities_improved_vs_empty'])} entities improved, "
        f"{num(c['entities_hurt_vs_empty'])} hurt, {num(c['entities_unchanged'])} unchanged.\n")

import os
if os.path.exists("eda/scripts/s14_conclusions.md"):          # hand-written sections 9-10, kept next to the generator
    L.append(open("eda/scripts/s14_conclusions.md", encoding="utf-8").read())
open("eda/rule_baseline.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
