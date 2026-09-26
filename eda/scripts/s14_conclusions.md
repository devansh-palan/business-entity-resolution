## 9. Follow-ups (`s14b_rule_followups.py`, `eda/results/s14b_rule_followups.json`)

All numbers in this report come from the rerun on the corrected preprocessed tables (a doubled `#` before a house number, injected in
99,123 Source 2 and 99,328 Source 3 records, used to hide the number from the parser and to pollute every address token; fixing it in
`addresses.py` raised the automatic-tier composite from 0.6002 to 0.6120).

### 9a. Does a lower-precision rule still raise macro F0.5 on top of the automatic tier?

Greedy union over every rule's one-to-one variant, starting from the composite of section 8 (fold 0 has 24,684 singletons and 441,365 entities):

| step | rule added | macro F0.5 | gain | pair precision | pair recall | singleton FPs | entities with a false merge |
|---|---|---|---|---|---|---|---|
| composite (D8 + R1s + R8u + R7s) | | 0.61200 | | 0.9950 | 0.4037 | 247 (1.0% of singletons) | 2,787 (0.6% of entities) |
| 1 | D2 name + state | 0.70807 | +0.09607 | 0.9126 | 0.5461 | 6,057 (24.5%) | 74,341 (16.8%) |
| 2 | D7 house number + street + city | 0.76458 | +0.05651 | 0.9160 | 0.6564 | 6,720 (27.2%) | 83,909 (19.0%) |
| 3 | D5 name + city | 0.77945 | +0.01487 | 0.9029 | 0.6829 | 7,985 (32.3%) | 100,302 (22.7%) |
| 4 | D6 exact address only | 0.78791 | +0.00846 | 0.9030 | 0.6938 | 8,041 (32.6%) | 101,279 (22.9%) |
| 5 | D4 name + street | 0.79463 | +0.00672 | 0.9005 | 0.7041 | 8,401 (34.0%) | 105,870 (24.0%) |
| 6 | D1 exact name only | 0.79889 | +0.00426 | 0.8879 | 0.7221 | 8,684 (35.2%) | 112,077 (25.4%) |

The same greedy from scratch ends at the same set; R7, R7s, R8, D3, R1 and R6c add nothing once D8 and D2 are in.

Reading: the metric is precision-weighted per entity, but the automatic tier leaves 60% of true pairs unmatched, and an entity with nothing predicted scores 0. A one-to-one name+state link is right 91% of the time, so adding it to entities that would otherwise score 0 (or gain a second correct match) outweighs the 5,800 singletons it destroys. The best deterministic system therefore scores **0.799** on fold 0, but it gets there by giving up 35% of the singleton score and putting a false merge into one entity in four. Those two costs are exactly what a learned matcher can avoid while keeping the recall, so 0.799 is the bar the ML matcher has to beat clearly, and the 0.612 tier is the part of the answer that can be committed without a model.

### 9b. What the automatic tier gets wrong

Anatomy of the composite's 3,079 false-merge pairs (2,787 entities; 247 of them true singletons):

| finding | count | share |
|---|---|---|
| linked pool record has no true owner in any fold (a decoy) | 2,460 | 79.9% |
| linked pool record belongs to another Source 1 entity | 619 | 20.1% |
| ... of which that owner has the same name core | 516 | 83.4% of owned |
| ... of which that owner also has the same house number | 17 | 2.7% of owned |
| ... of which that owner also has the same street | 0 | 0 |
| India | 2,689 | 87.3% |
| US | 390 | 12.7% |

The decoys (sampled from `false_merge_examples` and `singleton_fp_examples` in the JSON) are copies of the entity's own record with one number changed or added: `301, Plot 72, Gorai Shiv` against `322, PLOT 72, GORAI SHIV`; `1907 C22 N Grand Avenue` against `1907 C24 N Grand Ave`; `3-220/3, Plot No. 340` against `3-220/7, PLOT NO. 340`; `Unit No. F-8, Plot No.4B` against `H.no 483 Unit No. F-9, Plot No.4b`; `Lepakshi Buildings, 1522` against `28 Lepakshi Buildings, 1522`; `S. No. 36/1 And 4` against `S. No. 36/1 And 13`. The house-number rule fires because the number it extracts (plot, door or building number) is the one the decoy kept, while the number that changed is a flat, unit or survey sub-number. The 20% that belong to another entity are same-name entities in the same state whose house number the D8 rule matched without a street check (`23 Perigee Drive, Brookhaven` linked to `23 Koral Dr, Shinnecock Hills`, whose owner sits at `24 Koral Drive`). India carries 87% of the errors because Indian addresses carry several numbers (plot, door, flat, survey) and the decoys perturb any of them; US decoys change the unit or a digit of a single house number, which the parser isolates.

Consequences for the feature stage (adds to the list in `address_analysis.md` section 4): compare the full multiset of numeric tokens (symmetric difference, "a number present on one side only", "same number of numeric tokens"), treat a unit conflict when both sides carry a unit as a strong negative, and give the matcher the street-agreement predicate that D8 lacks.

### 9c. Rule 8 refinements

| variant | pairs | pair precision | pair recall | macro F0.5 | singleton FPs | entities with a false merge |
|---|---|---|---|---|---|---|
| R7 name + house number + street | 404,193 | 0.99807 | 0.2641 | 0.44835 | 65 | 780 |
| R7 + no recognised-state conflict | 404,089 | 0.99807 | 0.2640 | 0.44827 | 65 | 779 |
| R7 + no state conflict + city agrees (equal, fuzzy >= 85 or missing) | 351,440 | 0.99853 | 0.2297 | 0.40566 | 37 | 515 |
| R8 pairs with a street on both sides | 362,595 | 0.99805 | | | | |
| R8 pairs with the street missing on a side | 41,751 | 0.99214 | | | | |

Requiring the city to agree removes 265 false merges but also 52,000 true pairs (cities are substituted by neighbouring places in a fifth of matches, section 1 of the address analysis), so it lowers macro F0.5 by 4 points: the "very strong" conjunction is not the best automatic rule. The state veto removes almost nothing because a same-name, same-house-number pair in another state is rare (104 pairs). Allowing a missing street (R8 versus R7) adds 41,751 pairs at 0.992 precision; that is the D8/R8u trade the composite already makes.

## 10. Classification

Rules 2-5 (strong identifier, phone, e-mail, registration id) cannot be used: the data has no identifier columns, 3 e-mail addresses and 17 registration-like tokens in 12.5 million records, and phone-like strings occur only in 0.3% of Source 2/3 names as injected noise (15,468 and 15,908 names) and never in Source 1 (0 names with a phone tail), so a phone rule has nothing to join to. The 63,250 Source 1 addresses that look phone-like are hyphen- and slash-separated plot numbers (`8-2-293/82/A`); only 1,007 hold 10 or more digits. Rule 6 as written (name + city + postal code) links 6 pairs because postal codes exist in 0.007% of Source 1 and 0.0% of the pool.

**A. Automatic matches** (one-to-one variant, pair precision >= 0.99, singleton false positives <= 1% of affected entities): R1, R1s, R7, R7s, R8, R8u, D3 and D8. The greedy union keeps D8 + R1s + R8u + R7s (the others add nothing) and scores macro F0.5 **0.6120** on fold 0 (US 0.677, India 0.514) at pair precision 0.9950, hurting 247 entities (all singletons) and improving 297,971. This tier is what a deterministic fallback submission should emit; per country its precision is 0.9990 for the US and 0.9871 for India, so an India-only tightening (require the street when the name is transliterated, or require agreement of all numeric tokens) is the one refinement worth testing on the model's features rather than here.

**B. Candidate-generation signals** (raw join, recall >= 5%, mean candidates <= 50): every rule but R1 and R6. Their union reaches pair recall **0.766** (US 0.819, India 0.688) at a mean of 51 candidates per entity (p95 216, max 1,245), almost all of the volume coming from the exact-name join D1 (49.4 candidates per entity for 56.5% recall). Name + state (D2) gives 51.5% recall at 4.9 candidates, the address-only join D7 gives 36% at 2.0 and is the only rule that reaches matches whose name differs. The 23% of true pairs no exact key reaches (name spelled differently and no equal house number + street) is what the candidate-generation stage recovers with token retrieval (see `candidate_generation.md`: 98.5% at 81 candidates per entity).

**C. Features for the matcher**: every rule as a binary feature, and above all the block-size signal behind the one-to-one refinement: knowing how many Source 1 entities share the key raises the precision of the exact-name link from 0.040 to 0.871, of name + state from 0.364 to 0.887 and of the address-only link from 0.625 to 0.976. The per-rule precisions in section 4 are the likelihood each feature carries, and the error anatomy of section 9b names the three features that separate the remaining decoys: numeric-token symmetric difference, unit conflict, and street agreement.
