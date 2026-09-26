"""Render eda/results/s16_miss_analysis.json and s16b_candidate_rerun.json as eda/candidate_misses.md.

Run from the project root after s16 and s16b:  python eda/scripts/s16_report.py
Hand-written conclusions in eda/scripts/s16_conclusions.md are appended when present.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
A = json.load(open("eda/results/s16_miss_analysis.json", encoding="utf-8"))
B = json.load(open("eda/results/s16b_candidate_rerun.json", encoding="utf-8")) if os.path.exists("eda/results/s16b_candidate_rerun.json") else None
L = []
add = L.append
num = lambda x: f"{x:,}"
pct = lambda x: f"{100 * x:.2f}%"
n = A["n_missed"]
ORDER = ["missing address", "candidate limit too small", "blocking collision", "name variation", "abbreviation", "address variation", "other", "retrieval failure",
         "typo", "transliteration", "missing name", "country variation"]
TITLES = {"x_name_any_city": "exact name (any representation) + city, blocks <= 100", "x_name_any_house_number": "exact name (any representation) + house number",
          "x_compact_name_state": "compact name + state", "x_house_street_city": "house number + street + city, blocks <= 200",
          "x_addrless_name_ngram_top5": "name 3-gram retrieval against the address-less pool records, top 5 per source",
          "x_domain_name_ngram_top3": "name 3-gram retrieval against the domain-style pool names, top 3 per source",
          "x_name_addr_top30": "name + address retrieval, top 30 per source", "x_name_addr_top50": "name + address retrieval, top 50 per source",
          "x_name_addr_adaptive50_s20ge50": "ranks 21-50 of the name + address retrieval when the 20th score >= 0.5",
          "x_name_addr_adaptive50_s20ge60": "ranks 21-50 of the name + address retrieval when the 20th score >= 0.6",
          "x_name_addr_adaptive50_s20ge70": "ranks 21-50 of the name + address retrieval when the 20th score >= 0.7",
          "x_addr_token_top5": "address token retrieval, top 5 per source", "x_addr_token_top10": "address token retrieval, top 10 per source",
          "x_name_addr_top20_maxdf3": "name + address retrieval, top 20, pruning at 3% df instead of 1%",
          "9_addr_token_top20": "address token retrieval, top 20 per source", "3_name_ngram_top20": "name 3-gram retrieval, top 20 per source", "8_acronym_city": "name acronym + city"}

add("# Why the candidate set misses true pairs, and what recovers them (fold 0)\n")
add(f"The final candidate set of `candidate_generation.md` misses {num(n)} of the 1,527,650 true pairs (1.48%). Every missed pair was diagnosed with both "
    "records in hand (`s16_miss_analysis.py`, per-pair table in `artifacts/candidates/fold0/miss_causes.parquet`): the rank of the true record when the name + "
    "address retrieval is extended to 100, its cosine, whether the tokens the two records share were pruned as too common, whether the pair sat inside an "
    "exact-name or city block dropped by a cap, and name- and address-side signals. One primary cause is assigned in this order: country variation, missing name, "
    "missing address, candidate limit too small (rank 21-100), blocking collision (dropped block), retrieval failure (similar name and similar address but rank "
    "> 100), then the name-side causes when the names differ (transliteration, abbreviation, typo, name variation), address variation when the names agree, "
    "other when neither side is similar.\n")

add("## 1. Failure types\n")
add("| cause | pairs | share of misses | share of all true pairs | India | US | median name ratio | median address Jaccard | reading |")
add("|---|---|---|---|---|---|---|---|---|")
READ = {
    "missing address": "the pool record has no address at all, so only its (usually typo'd) name can find it",
    "candidate limit too small": "the true record is in the retrieval's ranks 21-100; the shared tokens are mostly common ones that the 1% pruning removed",
    "blocking collision": "the pair shares an exact name (or city + first token) but the block exceeded the cap; 92% India, 66% Indic-script names",
    "name variation": "the pool name is a different or scrambled name (`Iriarc`, `League Dvee1poebmnt Economic`); the address is only partly similar",
    "abbreviation": "85% are domain-style names (`nbrothers.com` for `North & Brothers`, `CHANDALFUENTESCPA.COM`)",
    "address variation": "similar but not identical names with a different address, so neither keys nor the retrieval reach them",
    "other": "unrelated synthetic names (`Tavozeph`, `Beloxylo`) with a partly similar address",
    "retrieval failure": "similar name and similar address yet outside the top 100: the shared tokens were pruned and the rest are typo'd (`108th` / `108rd`, `Tacoma` / `Tacma`)",
    "typo": "small character edits in the name with a different address", "transliteration": "an Indic name whose transliteration differs from the Latin name",
    "missing name": "", "country variation": "",
}
for c in ORDER:
    k = A["by_cause"].get(c, 0)
    s = A["signals_by_cause"].get(c, {})
    bc = A["by_cause_country"].get(c, {})
    add(f"| {c} | {num(k)} | {pct(k / n)} | {100 * k / 1527650:.3f}% | {num(bc.get('India', 0))} | {num(bc.get('US', 0))} | "
        f"{s.get('name_ratio_median', float('nan')):.0f} | {s.get('addr_jac_median', float('nan')):.2f} | {READ[c]} |")
add("")
r = A["retrieval_rank_of_misses"]
add(f"Rank of the missed true record in the name + address retrieval extended to 100: ranks 21-30 {num(r['rank_21_30'])}, 31-50 {num(r['rank_31_50'])}, "
    f"51-100 {num(r['rank_51_100'])}, beyond 100 {num(r['not_in_top_100'])}; {num(r['cosine_below_0.2_threshold'])} missed pairs have a cosine below the 0.2 threshold.\n")

add("## 2. What recovers each type\n")
add("Recovery counts of the missed pairs by strategies already measured in `candidate_generation.md` (top 4 per cause) and by the cheap keys tried here:\n")
add("| cause | existing strategies that would recover it (pairs) |")
add("|---|---|")
for c in ORDER:
    d = A["recovered_by_existing_strategy"].get(c, {})
    if d:
        add(f"| {c} | " + ", ".join(f"{k} {num(v)}" for k, v in sorted(d.items(), key=lambda x: -x[1])[:4]) + " |")
add("")
add("| cheap key | mean candidates per entity | p95 | recall alone | missed pairs recovered | by cause |")
add("|---|---|---|---|---|---|")
for k, d in A["cheap_keys"].items():
    add(f"| {TITLES.get('x_' + k, k)} | {d['mean']:.1f} | {d['p95']:.0f} | {d['recall_alone']:.4f} | {num(d['misses_recovered'])} | " +
        ", ".join(f"{kk} {num(v)}" for kk, v in d["misses_recovered_by_cause"].items()) + " |")
add("")

if B:
    add("## 3. Fixes measured on the whole fold\n")
    add("Every fix was computed for all 441,365 fold-0 entities and compared with the old final set: new candidates it adds per entity, missed pairs it "
        "recovers, and the recall gained per candidate added (the fold-0 frontier accepted strategies at >= 0.0002, one recall point per 50 candidates).\n")
    add("| fix | pairs | recall alone | +candidates per entity | missed pairs recovered | recall gain (points) | gain per candidate | recovered by cause |")
    add("|---|---|---|---|---|---|---|---|")
    for k, d in sorted(B["fixes"].items(), key=lambda x: -x[1]["gain_per_candidate"]):
        add(f"| {TITLES.get(k, d['description'])} | {num(d['pairs'])} | {d['recall_alone']:.4f} | +{d['new_candidates_per_entity']:.2f} | {num(d['misses_recovered'])} "
            f"| +{100 * d['recall_gain']:.3f} | {d['gain_per_candidate']:.5f} | " + ", ".join(f"{kk} {num(v)}" for kk, v in list(d["by_cause"].items())[:4]) + " |")
    add("")
    sel = B["selection"]
    add(f"Greedy selection (threshold {sel['threshold_gain_per_candidate']} recall per candidate): " + (" -> ".join(
        f"{TITLES.get(s['added'], s['added'])} (+{100 * s['recall_gain']:.3f} points for +{s['candidates_added_per_entity']:.2f} candidates)" for s in sel["steps"]) or "nothing accepted") + ".")
    add(f"Rejected: {', '.join(TITLES.get(x, x) for x in sel['rejected'])}.\n")
    cmp = B["comparison"]
    add("## 4. Old versus new\n")
    add("| | old | new |")
    add("|---|---|---|")
    add(f"| candidate recall | {cmp['old_recall']:.4f} | {cmp['new_recall']:.4f} |")
    add(f"| recall US / India | {cmp['old_by_country'].get('US', 0):.4f} / {cmp['old_by_country'].get('India', 0):.4f} | {cmp['new_by_country'].get('US', 0):.4f} / {cmp['new_by_country'].get('India', 0):.4f} |")
    add(f"| entities with every match present | {cmp['old_all_found']:.4f} | {cmp['new_all_found']:.4f} |")
    add(f"| average candidates per entity | {cmp['old_mean']:.1f} | {cmp['new_mean']:.1f} |")
    add(f"| 95th percentile | {cmp['old_p95']:.0f} | {cmp['new_p95']:.0f} |")
    add(f"| pairs | {num(cmp['old_pairs'])} | {num(cmp['new_pairs'])} |")
    add("")
    rm = B["remaining_misses"]
    add(f"Remaining misses: {num(rm['n'])} (" + ", ".join(f"{k} {num(v)}" for k, v in sorted(rm["by_cause"].items(), key=lambda x: -x[1])) + ").\n")
    f = B["file"]
    add(f"New file `{f['path']}`: {num(f['rows'])} rows, {num(f['ids'])} ids, {f['bytes'] / 1e6:.0f} MB, specification check {'passed' if f['valid'] else 'FAILED'}; "
        "the previous file is kept as `candidate_pairs_v1.tsv`.\n")

add("## 5. Examples per cause\n")
for c in ORDER:
    ex = A["examples"].get(c, [])
    if not ex:
        continue
    add(f"**{c}**\n")
    for e in ex[:4]:
        add(f"- `{e['business_name_1']}` | `{e['business_address_1']}`  ->  `{e['business_name_2']}` | `{e['business_address_2']}`  "
            f"(retrieval rank {e['retr_rank'] if e['retr_rank'] > 0 else '> 100'}, cosine {e['retr_cos']:.2f}, name ratio {e['name_ratio']:.0f}, address Jaccard {e['addr_jac']:.2f})")
    add("")

if os.path.exists("eda/scripts/s16_conclusions.md"):
    add(open("eda/scripts/s16_conclusions.md", encoding="utf-8").read())
open("eda/candidate_misses.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
