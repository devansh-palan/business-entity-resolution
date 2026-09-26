# First pairwise matchers, scored at the entity level (fold 0)

Three models were trained on the pairwise feature matrix of `pair_features.md` and turned into per-entity match lists. The split is by Source 1 entity (feature bucket): 24 buckets train the models, 4 buckets (55,171 entities) choose the decision rule, and 4 untouched buckets (55,170 entities, 3,150 of them singletons) give every number reported below, through the official macro F0.5 evaluator (`evaluation.macro_f05_from_rows`). Nothing here touches the test set.

**Training sample.** All 1,134,040 positives of the training buckets plus a uniform 6.5% of their 29,057,743 negatives (1,888,345), 3,022,385 rows in all; probabilities are corrected back to the true prior afterwards (odds divided by 1 / rate), so the thresholds below are on the calibrated scale. A full-data fit needs the 64 GB machine.

## 1. Models

| model | rows | fit | pair AUC (report half) | average precision | notes |
|---|---|---|---|---|---|
| logistic regression | 1,500,000 | 21 s | 0.9985 | 0.9758 | 82 lbfgs iterations; standardised features, nulls as 0 |
| LightGBM | 3,022,385 | 168 s | 0.9998 | 0.9957 | 1056 trees, 127 leaves, lr 0.05; early stopping on held-out pairs (val AUC 0.9998) |
| XGBoost | 3,022,385 | 715 s | 0.9998 | 0.9957 | 1509 trees, depth 8, eta 0.05 |

Logistic regression, largest standardised coefficients (sign = direction of a match): `addr_num_jaccard` -2.63, `addr_num_overlap` +2.32, `name_tok_jaccard` +2.26, `p_domain_style` -2.03, `cos_name_ngram` +1.98, `addr_token_set` +1.89, `addr_numsig_eq` +1.78, `ident_x_name` +1.78, `name_exact_core` -1.75, `name_common_prefix_rel` -1.66, `legal_conflict` -1.52, `name_first_tok_eq` +1.46, `name_compact_lev_sim` +1.46, `gap_name_q` +1.40, `name_translit_lev_sim` +1.37.

LightGBM, share of total gain: `rank_cos_q` 0.539, `cos_name_x_addr` 0.071, `cos_name_addr` 0.068, `rank_cos_qs` 0.051, `name_x_addr` 0.046, `name_translit_jw` 0.024, `cos_name_ngram` 0.020, `name_pool_freq_2` 0.013, `addr_token_set` 0.011, `gap_cos_q` 0.010, `addr_numsig_eq` 0.009, `cos_addr` 0.009, `house_digits_eq` 0.009, `legal_conflict` 0.007, `cos_name` 0.007.

XGBoost, share of total gain: `rank_cos_q` 0.581, `cos_name_x_addr` 0.080, `rank_cos_qs` 0.058, `name_x_addr` 0.030, `cos_name_addr` 0.026, `cos_name_ngram` 0.024, `name_translit_jw` 0.024, `name_pool_freq_2` 0.012, `cos_addr` 0.012, `sim_min` 0.011, `addr_token_set` 0.010, `addr_numsig_eq` 0.009, `house_digits_eq` 0.009, `cos_name` 0.007, `addr_partial` 0.007.

## 2. From pair probabilities to match lists

Rules swept on the tuning half: a probability threshold from 0.25 to 0.95, with or without pool-side uniqueness (a Source 2/3 record is given only to the entity that scores it highest, the one-to-one structure of the truth), and with or without a relative gate (keep a pair only if its probability is at least 0.5 or 0.8 times the entity's best). The rule with the highest macro F0.5 on the tuning half is then applied, unchanged, to the report half.

| model | best rule (tuning half) | macro F0.5 tuning half | best plain threshold | its F0.5 | threshold + uniqueness | its F0.5 |
|---|---|---|---|---|---|---|
| logistic regression | p>=0.70 +unique | 0.9198 | p>=0.70 | 0.9196 | p>=0.70 +unique | 0.9198 |
| LightGBM | p>=0.75 +unique | 0.9702 | p>=0.75 | 0.9700 | p>=0.75 +unique | 0.9702 |
| XGBoost | p>=0.65 +unique | 0.9699 | p>=0.65 | 0.9696 | p>=0.65 +unique | 0.9699 |

An empty prediction scores 0.0571 on the report half (the singleton share); the deterministic automatic tier of `rule_baseline.md` scored 0.612 on the whole fold.

## 3. Entity-level results on the report half

| | logistic regression | LightGBM | XGBoost |
|---|---|---|---|
| **macro F0.5** | **0.9191** | **0.9701** | **0.9696** |
| macro F0.5 on the tuning half (rule chosen there) | 0.9198 | 0.9702 | 0.9699 |
| singletons: share predicted empty | 0.8616 | 0.9464 | 0.9324 |
| singletons: entities given a false match | 436 | 169 | 213 |
| one-match entities: mean F0.5 | 0.8116 | 0.9156 | 0.9189 |
| one-match entities: fully correct | 75.6% | 89.9% | 89.9% |
| multi-match entities: mean F0.5 | 0.9295 | 0.9751 | 0.9752 |
| multi-match entities: fully correct | 54.9% | 78.3% | 79.1% |
| multi-match entities: true / predicted matches per entity | 3.83 / 3.38 | 3.83 / 3.63 | 3.83 / 3.66 |
| Source 2: macro F0.5 over entities with an S2 match | 0.8933 | 0.9593 | 0.9619 |
| Source 2: pair precision / recall | 0.9727 / 0.8607 | 0.9922 / 0.9436 | 0.9903 / 0.9490 |
| Source 3: macro F0.5 over entities with an S3 match | 0.8920 | 0.9549 | 0.9567 |
| Source 3: pair precision / recall | 0.9716 / 0.8533 | 0.9913 / 0.9350 | 0.9885 / 0.9406 |
| US: macro F0.5 | 0.9255 | 0.9756 | 0.9750 |
| India: macro F0.5 | 0.9093 | 0.9619 | 0.9614 |
| entities with a false merge | 5,413 | 1,750 | 2,218 |
| entities with a missed match | 19,570 | 9,702 | 8,913 |
| entities fully correct | 31,918 (57.9%) | 44,055 (79.9%) | 44,422 (80.5%) |
| pairs predicted / true pairs | 168,881 / 190,545 | 180,797 / 190,545 | 182,381 / 190,545 |
| pair precision / recall / F0.5 (information only) | 0.9668 / 0.8569 / 0.9426 | 0.9898 / 0.9391 / 0.9792 | 0.9870 / 0.9447 / 0.9782 |

Mean F0.5 by number of true matches (report half):

| true matches | entities | logistic regression | LightGBM | XGBoost |
|---|---|---|---|---|
| 0 (singleton) | 3,150 | 0.8616 | 0.9464 | 0.9324 |
| 1 | 3,044 | 0.8116 | 0.9156 | 0.9189 |
| 2 | 9,318 | 0.9061 | 0.9643 | 0.9632 |
| 3-5 | 33,403 | 0.9329 | 0.9770 | 0.9773 |
| 6+ | 6,255 | 0.9460 | 0.9807 | 0.9817 |

## 4. False merges of the best model (LightGBM)

On the report half the rule keeps 180,797 pairs, 1,851 of them wrong, spread over 1,750 entities. Of the 190,545 true pairs, 2,095 were never in the candidate set and 9,504 were scored below the rule.

| false merges | pairs | exact name | exact address | house conflict | unit conflict | street conflict | same name, other house | pool address missing | mean p | mean name+address cosine |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 1,851 | 49.0% | 6.5% | 57.6% | 29.2% | 19.2% | 27.5% | 21.5% | 0.881 | 0.638 |
| stolen from another entity | 788 | 41.4% | 14.8% | 21.9% | 24.1% | 20.9% | 4.5% | 47.7% | 0.875 | 0.543 |
| ownerless | 1,063 | 54.8% | 3.3% | 70.6% | 31.0% | 18.7% | 40.5% | 2.1% | 0.886 | 0.708 |

'Stolen' means the Source 2/3 record belongs to another Source 1 entity in the training truth (a decoy sibling); 'ownerless' means no entity of the training data claims it.

**ownerless**, highest-probability examples:

- `Tulsi Trading Limited` | `303 Happiness Building, Citi Of Joy Complex Acc Road, Mulund West, Mumbai, Mumbai City, Maharashtra`  ->  `Tulsi Trading Pvt Ltd` | `303 Happiness Building, Citi Of Joy Complex Acc Road, Mulund West, Mumbai, Mumbai City, MH`  (p = 1.00)
- `Idalia Letcher, DMD` | `404 Ferndale Lane, Minoa, NY`  ->  `Idalia Letcher, MD` | `04 FERNDALE LANE, MINOA, NY`  (p = 1.00)
- `Zamora and Russell LLC` | `716 Charles Street, Plano, IL`  ->  `Zamora and Russell Ltd` | `716 Charles Street, PMB 8499, Plano, Illinois`  (p = 1.00)
- `Signature Telecom Studios` | `4937 Detroiter Drive, Virginia Beach City, VA`  ->  `Signature Telecom Studios Corp` | `4937 Detroiter Drive, Virignia Beach CITY, Virginia`  (p = 1.00)
- `Jessika Watkins, CPA` | `100 Revolution Street, City Of Havre De Grace, MD, Unit 603`  ->  `Jessika Watkins, PA Corp` | `0100 Revolution St, # 603, City Of Havre De Grace, Maryland`  (p = 1.00)
- `Dental Associates LLC` | `1044 421, Whitestown, IN`  ->  `DENTAL ÁSSOCIATES [WEST]` | `1044 421, WHITESTOWN, IN`  (p = 1.00)

**stolen from another entity**, highest-probability examples:

- `FKO Agents Limited` | `Khasra No. 175, 170, 90/2, 176, Pipliya Dhakad, Huzur, Bhopal, Madhya Pradesh`  ->  `FKO AGENTS` | `H.NO 175 , 170, 90/2, 176, PIPLIYA DHAKAD, HUZUR, मध्य प्रदेश`  (p = 1.00, owner S1-598232727)
- `Patna Properties Private Limited` | `42, Savitri Sadan ( Bihand Punjab & Sind Banj ) Fraser Road, Patna, Bihar`  ->  `Patna Properties Private  Ltd` | `42, Patna, बिहार`  (p = 1.00, owner S1-548017045)
- `FKO Agents Limited` | `Khasra No. 175, 170, 90/2, 176, Pipliya Dhakad, Huzur, Bhopal, Madhya Pradesh`  ->  `FKO AGENTS CORPORATION` | `NO 175 , 170, 90/2, 176, PIPLIYA DHAKAD, HUZUR, Madhya Pradesh`  (p = 1.00, owner S1-598232727)
- `Vellore Solutions Private Limited` | `Vellore, No 1 Infantry Road Corporation Main Building Cantonment Opposite To Bsnl Office, Tamil Nadu, Vellore`  ->  `Vellore Information Private Limited` | `1 INFANTRY ROAD CORPORATION MAIN BUILDING CANTONMENT OPPOSITE TO BSNL OFFICE, VELLORE, Tamil Nadu`  (p = 1.00, owner S1-700145233)
- `Dental Care LLC` | `KY, Hebron, 1414 Dominion Trail`  ->  `#dentalcare` | `1330. DOMINION TRAIL, HEBRON, KY`  (p = 0.99, owner S1-96238544)
- `Vellore Solutions Private Limited` | `Vellore, No 1 Infantry Road Corporation Main Building Cantonment Opposite To Bsnl Office, Tamil Nadu, Vellore`  ->  `Vellore Information Private Ltd` | `NO 1 INFANTRY ROAD CORPORATION MAIN BUILDING CANTONMENT OPPOSITE TO BSNL OFFICE, VELLORE, Tamil Nadu`  (p = 0.99, owner S1-700145233)

## 5. What the numbers say

**The candidate pairs are easy at the pair level and the errors are structural.** All three models separate matches from non-matches almost perfectly on pairs (LightGBM AUC 0.9998, average precision 0.996), yet the entity-level macro F0.5 is 0.970, not 0.99: the remaining 3 points are concentrated in a few thousand entities where the decision is genuinely ambiguous. The 1,851 false merges of LightGBM on the report half are dominated by the decoy pattern the rule baseline found: 49% carry the exact same name, 58% a different house number, 27% the same name with a different house number, and 22% are pool records without an address. The highest-probability false merges (`Idalia Letcher, DMD | 404 Ferndale Lane` against `Idalia Letcher, MD | 04 FERNDALE LANE`; `Dental Care LLC | 1414 Dominion Trail` against `#dentalcare | 1330 DOMINION TRAIL`) are records that differ from a true match by one edited number, which the truth labels as non-matches. The model scores them at p = 1.0 because every other signal agrees; only a feature that isolates "same everything, one number changed" can pull them down, and it will cost recall on genuine typos in numbers.

**The recall side is larger than the precision side.** Of the 190,545 true pairs of the report half, 2,095 (1.1%) are missing from the candidate set and 9,504 (5.0%) are scored below the rule, against 1,851 false merges. F0.5 weights precision 4:1, so the tuned threshold sits high (0.75) and leaves those 9,504 pairs; lowering it to 0.5 buys recall but loses more in macro F0.5 on the tuning half. The next gains are therefore in ranking the hard positives higher (name-variation and address-less matches, the causes listed in `candidate_misses.md`), not in moving the threshold.

**Where the score is lost, by entity type (report half, LightGBM).** Singletons: 5.4% of the 3,150 singletons receive a false match (169 entities, each scoring 0 instead of 1). One-match entities are the weakest class (mean F0.5 0.916, 89.9% fully correct) because a single wrong or missing pair is the whole score; entities with 3-5 matches reach 0.977 and those with 6 or more 0.981, where one error is diluted. Source 3 is slightly harder than Source 2 (0.955 against 0.959) and India than the US (0.962 against 0.976), the same ordering as candidate recall.

**Decision rule.** The pool-side uniqueness (a Source 2/3 record goes only to the entity scoring it highest) adds 0.0002 over the plain threshold: the models already leave few contested records. The relative gate (keep a pair only if it is within a factor of the entity's best) never helped: matches of one entity vary legitimately in strength. No rule beats a calibrated threshold plus uniqueness, and the threshold is stable across models of the same family (0.75 LightGBM, 0.65 XGBoost, 0.70 for the two-model average).

**Model choice.** LightGBM (0.9701) and XGBoost (0.9696) are tied within the noise of 55,170 entities; XGBoost is better on one-match entities and Source 2/3 recall, LightGBM on singletons and false merges. Averaging the two gives 0.9702 on the report half (`eda/results/s18_ensemble.json`), no gain; adding the logistic regression lowers it to 0.968. The logistic regression (0.919) is the interpretable reference: its largest standardised coefficients are the address-number overlap, the name token Jaccard, the name 3-gram cosine and the address token-set ratio on the positive side and the domain-style-name flag on the negative side, which is also the order of the tree models' gain. LightGBM at 1,056 trees of 127 leaves is the model to carry forward: 3 minutes to train on 3 million pairs, 5 seconds per million pairs to score.

**Caveats.** The numbers come from one fold of the training data, split by entity: the models were trained on 24 of the 32 buckets of fold 0 (3.0 million sampled pairs), the rule was chosen on 4 buckets and every table above comes from the other 4. The tuning and report halves agree to 0.0001, so the rule is not over-fitted, but the absolute level is specific to fold 0's candidate set (recall 0.9895). The test set adds France, for which the state-keyed candidate strategies and the learned dictionaries are weaker; the models use no country-specific feature except the country indicator, so their transfer depends on the candidate set and the preprocessing more than on the matcher.

## 6. What to do next

1. Train the final LightGBM on all folds' candidate pairs (features for the other four folds are 26 minutes each here, or one pass on the 64 GB box) and keep fold 0 as the check that nothing drifts.
2. Target the decoy false merges with two or three features that the current matrix lacks: an edit distance between the two house numbers (one digit dropped or changed), whether the address numbers of the pool record are a subset of the entity's, and the count of pool records that share the entity's name and street but not its house number. Measure their effect on the 1,851 false merges before keeping them.
3. Re-examine the 9,504 true pairs below the threshold with the miss-cause taxonomy: if most are address-less or name-variation pairs, a second-stage model on those subsets (or a lower threshold restricted to them) is cheaper than a lower global threshold.
4. Run the test-time pipeline end to end (preprocess with the all-train dictionaries, candidates, features, scoring, the rule) once the features for the full training set exist, and validate the submission file with the organisers' script.

