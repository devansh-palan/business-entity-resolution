"""Render eda/results/s15_candidates.json (and s15b if present) as eda/candidate_generation.md.

Run from the project root after s15:  python eda/scripts/s15_report.py
Hand-written conclusions live in eda/scripts/s15_conclusions.md and are appended when present.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = json.load(open("eda/results/s15_candidates.json", encoding="utf-8"))
B = json.load(open("eda/results/s15b_embedding_probe.json", encoding="utf-8")) if os.path.exists("eda/results/s15b_embedding_probe.json") else None
L = []
add = L.append
num = lambda x: f"{x:,}"
pct = lambda x: f"{100 * x:.2f}%"
S = R["strategies"]
U = R["unions"]

TITLES = {
    "1_exact_name_core": "1. exact name core", "1_exact_name_core_sorted": "1. exact name core, tokens sorted", "1_exact_name_compact": "1. exact name, spaces removed",
    "1_exact_name_translit_core_sorted": "1. exact transliterated name, sorted", "6_exact_name_core_without_country": "6. exact name core, country key removed",
    "2_name_token_df200_rarest2": "2. name tokens, df <= 200, 2 rarest", "2_name_token_df500_rarest2": "2. name tokens, df <= 500, 2 rarest",
    "2_name_token_df1000_rarest2": "2. name tokens, df <= 1000, 2 rarest", "3_name_ngram_top20": "3/8. name char 3-gram TF-IDF, top 20 per source",
    "4_addr_token_df200_rarest3": "4. address tokens, df <= 200, 3 rarest", "4_addr_token_df500_rarest3": "4. address tokens, df <= 500, 3 rarest",
    "4_addr_token_df1000_rarest3": "4. address tokens, df <= 1000, 3 rarest", "5_state_name_sorted": "5. state + sorted name", "5_city_name_first": "5. city + first name token",
    "5_city_house_number": "5. city + house number", "5_state_house_street": "5. state + house number + street", "5_postal_name_first": "5. postal code + first name token",
    "5u_rawlast_name_sorted": "5/6. raw last component + sorted name (unseen-country simulation)", "8_sn_name_core_sorted_w3": "8. sorted neighbourhood, sorted name, window 3",
    "8_sn_name_compact_w3": "8. sorted neighbourhood, compact name, window 3", "8_acronym_city": "8. name acronym + city", "9_numsig": "9. address number signature",
    "9_sn_addr_sorted_w3": "9. sorted neighbourhood, sorted address tokens, window 3", "9_addr_token_top20": "9. address token TF-IDF, top 20 per source",
    "9_name_addr_token_top20": "9. name + address token TF-IDF, top 20 per source",
}

add("# Candidate generation (validation fold 0)\n")
add("Queries: the 441,365 Source 1 entities of fold 0; pool: all 10,320,219 Source 2 and Source 3 records; truth: the fold's 1,527,650 true pairs. "
    "Every strategy is an operation on the preprocessed tables (`artifacts/preprocessed/fold0`, dictionaries learned without fold 0); every key includes the "
    "country string, nothing about a country is hard-coded. Reduction ratio = 1 - pairs / (queries x pool). Volume statistics count every query entity, "
    "including those with no candidate. Pairs are saved per strategy under `artifacts/candidates/fold0/strategies/`.\n")

add("## 1. Every strategy\n")
add("| strategy | pairs | pair recall | recall S2 | recall S3 | recall US | recall India | entities with all matches found | mean | median | p95 | max | reduction ratio | seconds |")
add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, m in S.items():
    if "error" in m:
        add(f"| {TITLES.get(k, k)} | failed: {m['error']} | | | | | | | | | | | | |")
        continue
    add(f"| {TITLES.get(k, k)} | {num(m['pairs'])} | {m['recall']:.4f} | {m['recall_by_source'].get('S2', 0):.4f} | {m['recall_by_source'].get('S3', 0):.4f} "
        f"| {m['recall_by_country'].get('US', 0):.4f} | {m['recall_by_country'].get('India', 0):.4f} | {m['entity_all_matches_found']:.4f} | {m['mean']:.1f} | {m['median']:.0f} "
        f"| {m['p95']:.0f} | {num(m['max'])} | {m['reduction_ratio']:.6f} | {R['timing_s'].get(k, 0)} |")
add("")
add("Recall of the top-k retrievals by rank (pairs found within the first k neighbours per source):\n")
add("| retrieval | @1 | @3 | @5 | @10 | @20 |")
add("|---|---|---|---|---|---|")
for k in ("3_name_ngram_top20", "9_addr_token_top20", "9_name_addr_token_top20"):
    if k in S and "by_rank" in S[k]:
        r = S[k]["by_rank"]
        add(f"| {TITLES[k]} | {r['recall@1']:.4f} | {r['recall@3']:.4f} | {r['recall@5']:.4f} | {r['recall@10']:.4f} | {r['recall@20']:.4f} |")
add("")

add("## 2. Unions\n")
add("| union | members | pairs | pair recall | recall S2 | recall S3 | recall US | recall India | entities with all matches found | mean | median | p95 | p99 | max | reduction ratio |")
add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, u in U.items():
    add(f"| {k} | {len(u['members'])} | {num(u['pairs'])} | {u['recall']:.4f} | {u['recall_by_source'].get('S2', 0):.4f} | {u['recall_by_source'].get('S3', 0):.4f} "
        f"| {u['recall_by_country'].get('US', 0):.4f} | {u['recall_by_country'].get('India', 0):.4f} | {u['entity_all_matches_found']:.4f} | {u['mean']:.1f} | {u['median']:.0f} "
        f"| {u['p95']:.0f} | {u['p99']:.0f} | {num(u['max'])} | {u['reduction_ratio']:.6f} |")
add("")
add("Members: U_exact_name_any = the four exact name representations; U_exact_keys adds the state/city/postal keys, the number signature and the acronym key; "
    "U_retrieval_only = the three top-20 retrievals; U_bounded = every strategy whose volume per entity is bounded by construction (exact keys, capped blocks, "
    "sorted neighbourhoods, top-20 retrievals); the token unions add the rare-token blocks at the given df cap; U_all_selected is the final set.\n")

SEL = json.load(open("eda/results/s15c_union_selection.json", encoding="utf-8")) if os.path.exists("eda/results/s15c_union_selection.json") else None
if SEL:
    add("## 2b. Choosing the final set\n")
    add("Recall of every union below is exact from the coverage table; volumes are exact per-bucket unions. Retrieval recall by rank (from the scored outputs):\n")
    add("| retrieval | @1 | @3 | @5 | @10 | @15 | @20 |")
    add("|---|---|---|---|---|---|---|")
    for k, r in SEL["retrieval_by_rank"].items():
        add(f"| {TITLES.get(k, k)} | " + " | ".join(f"{r[f'recall@{j}']:.4f}" for j in (1, 3, 5, 10, 15, 20)) + " |")
    add("")
    add("Greedy forward selection (each step adds the strategy with the best recall gain per candidate it costs on its own):\n")
    add("| step | strategy added | union recall | gain | own mean candidates | union mean | union p95 | union max | union pairs |")
    add("|---|---|---|---|---|---|---|---|---|")
    for i, st in enumerate(SEL["greedy"]["steps"], 1):
        add(f"| {i} | {TITLES.get(st['added'], st['added'])} | {st['recall']:.5f} | {st['gain']:+.5f} | {st['single_mean_candidates']:.1f} | {st.get('union_mean', float('nan')):.1f} "
            f"| {st.get('union_p95', float('nan')):.0f} | {num(st.get('union_max', 0))} | {num(st.get('union_pairs', 0))} |")
    add("")
    add("Named candidate sets:\n")
    add("| set | strategies | recall | recall US | recall India | recall S2 | recall S3 | entities with all matches | mean | median | p95 | p99 | max | pairs | entities without candidates |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in SEL["sets"].items():
        add(f"| {k} | {len(r['members'])} | {r['recall']:.5f} | {r['by_country'].get('US', 0):.4f} | {r['by_country'].get('India', 0):.4f} | {r['by_source'].get('S2', 0):.4f} "
            f"| {r['by_source'].get('S3', 0):.4f} | {r['entity_all_found']:.4f} | {r['mean']:.1f} | {r['median']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} | {num(r['max'])} | {num(r['pairs'])} | {num(r['entities_without_candidates'])} |")
    add("")


c = R["coverage"]
add("## 3. Every true pair of the fold against the final set\n")
add(f"Of the {num(c['true_pairs'])} true pairs, **{num(c['found'])} are in the final candidate set and {num(c['missed'])} are not** "
    f"(recall {c['found'] / c['true_pairs']:.5f}); {num(c['found_by_exactly_one_strategy'])} pairs are produced by exactly one strategy. The coverage table "
    "(`artifacts/candidates/fold0/coverage.parquet`) has one row per true pair with a flag per strategy.\n")
add("| strategy | true pairs only this strategy finds | recall of the final set without it |")
add("|---|---|---|")
for k in c["final_members"]:
    add(f"| {TITLES.get(k, k)} | {num(c['unique_contribution'][k])} | {c['recall_without'][k]:.5f} |")
add("")

m = R["misses"]
add("## 4. Anatomy of the misses\n")
add(f"{num(m['n'])} true pairs are missed. By country: " + ", ".join(f"{k} {num(v)}" for k, v in m["by_country"].items()) + "; by source: " +
    ", ".join(f"{k} {num(v)}" for k, v in m["by_source"].items()) + ".\n")
add("| category (name fuzzy ratio on transliterated cores; address token Jaccard) | pairs |")
add("|---|---|")
for k, v in m["by_category"].items():
    add(f"| {k} | {num(v)} |")
add("")
if m["name_ratio_quartiles"]:
    add(f"Name ratio quartiles of the misses: {', '.join(f'{x:.0f}' for x in m['name_ratio_quartiles'])}; address Jaccard quartiles: "
        f"{', '.join(f'{x:.2f}' for x in m['addr_jaccard_quartiles'])}; script of the missed pool name: " + ", ".join(f"{k} {num(v)}" for k, v in m["pool_name_script"].items()) + ".\n")
add("Examples (Source 1 record, then the missed true record):\n")
for e in m["examples"][:16]:
    add(f"- `{e['business_name_1']}` | `{e['business_address_1']}`  ->  `{e['business_name_2']}` | `{e['business_address_2']}`  (name ratio {e['name_ratio']:.0f}, address Jaccard {e['addr_jac']:.2f}; {e['category']})")
add("")

if B:
    add("## 5. Strategy 10: embedding retrieval probe\n")
    add(f"Model `{B['model']}` on {B['device']}. For each missed pair and a control sample of found pairs, the cosine of the true pair is compared with the "
        "cosines to a random same-country sample of pool names; a pair counts as retrievable at k when fewer than k * (pool / sample) sample names score above the true match.\n")
    add("| set | text | country | pairs | median true cosine | retrievable top-20 | top-50 | top-200 | names embedded per second |")
    add("|---|---|---|---|---|---|---|---|---|")
    for label, out in B["results"].items():
        for tag in ("raw name", "transliterated core"):
            for country, d in out[tag].items():
                add(f"| {label} | {tag} | {country} | {num(d['n'])} | {d['true_cosine_median']:.3f} | {pct(d['retrievable_top20_full_pool'])} | {pct(d['retrievable_top50_full_pool'])} "
                    f"| {pct(d['retrievable_top200_full_pool'])} | {d['embed_rate_names_per_s']:,.0f} |")
    add("")
    fc = B["full_scale_cost"]
    add(f"Full-scale cost: {num(fc['names_to_embed_train'])} training names at the measured rate = {fc['minutes_at_measured_rate']} minutes of GPU embedding per split, before the search.\n")

f = R["final"]["file"]
add("## 6. The competition file\n")
add(f"`{f['path']}`: {num(f['rows'])} rows (one per fold-0 Source 1 entity, in the Source 1 file order), {num(f['ids'])} candidate ids, "
    f"{num(f['empty_rows'])} empty rows, {f['bytes'] / 1e6:.0f} MB. Streaming check against the specification (header, LF endings, two columns, "
    f"S2-/S3- ids only, no duplicates within a row, every id in the pool): {'passed' if f['valid'] else 'FAILED: ' + '; '.join(f['problems'])}.\n")


if os.path.exists("eda/scripts/s15_conclusions.md"):
    add(open("eda/scripts/s15_conclusions.md", encoding="utf-8").read())
open("eda/candidate_generation.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
