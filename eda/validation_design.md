# Leakage-safe validation design

Artifacts: `artifacts/validation/` (built by `code/business_entity_resolution/src/validation.py build`).
Diagnostics: `eda/scripts/s9_validation_design.py`, numbers in `eda/results/s9_validation_design.json`.
Nothing in the validation module reads the test files; the single test-set diagnostic at the end sets no parameter.

## Investigation

**1. Must a Source 1 entity stay in one fold? Yes.** The score is computed per S1 entity, an entity has 3.46 pairs on average and 89.0% have two or more. Its sibling records are correlated: two S2 records of the same entity share the core name 30.2% of the time and the sorted address 26.2%. A pair-level random split would put an entity's S1 text and near-copies of its records on both sides. Folds are therefore assigned to entities, never to pairs.

**2. Can an S2/S3 record appear in several candidate sets? Yes, and it is the main leak path.** In the ground truth each record belongs to at most one entity, but as a candidate it appears for every same-name entity: 82.4% of same-name hard negatives are the true matches of another S1 entity. Under random entity-level folds 47.2% of entities have a same-name entity in another fold, and 48.2% of the same-name candidate records of a validation entity are owned by training entities, so the model would have trained on those records as positives. Grouping entities into connected components over shared normalised core name (and shared addresses, capped as below) brings the first figure to 0.0% and the second to 3.6%; the residue comes from records whose corrupted name happens to equal another entity's name, which cannot be removed from the S1 side.

**3. Duplicates and near-duplicates.** Source 1 is deduplicated in practice: 14 rows share both core name and sorted address, 6.8% share a sorted address, 49.5% share a core name. Exact duplicate S2/S3 texts exist (25,060 groups in S2, 18,381 in S3) but every group has a single owner and no orphans, so they are siblings of one entity, carry no label noise, and stay in one fold automatically. Address merging had to be capped: merging on every shared address chained popular addresses and large name groups into components of 263,844 and 241,115 entities; capping at addresses shared by at most 3 entities leaves a largest component of 6,531 (0.3%). The cap 5 still gave 198,209.

| Address merge cap | Largest component |
|---|---|
| none | 263,844 |
| 10 | 263,569 |
| 5 | 198,209 |
| 3 (chosen) | 6,531 |
| 2 | 2,105 |
| name only | 557 |

After the cap, 3.1% of entities still have a same-address entity in another fold (addresses shared by more than 3 entities), against 6.2% under random folds.

**4. Stratify source-pair patterns? Not needed, but verified.** Spread across the five folds: none 0.08 points, S2 only 0.07, S3 only 0.16, both 0.25.

**5. Stratify country? Yes.** Components are per country and assigned greedily per country, so every fold is 59.979% US and 40.021% India to three decimals. Country stays in the entity table so leave-one-country-out runs (train US, validate India and vice versa) estimate the drop expected on France.

**6. Preserve singleton and multi-match shares? Yes, verified.** Singleton share 5.55% to 5.63% across folds, exactly-one 5.32% to 5.46%, two 16.92% to 17.08%, three-plus 71.89% to 72.21%; mean matches 3.457 to 3.466. Match count is independent of name popularity (mean 3.45 to 3.47 across cluster-size buckets), so grouping by name does not distort it.

**7. Preserve source formatting patterns? Automatic.** The S2/S3 pool is never split and every fold is scored against all of it, exactly like the test set. Among each fold's matched records: Indic-script names 7.17% to 7.26%, domain-style names 5.57% to 5.64%, all-caps 11.48% to 11.52%, accented Latin 6.65% to 6.68%, missing address 4.40% to 4.44%, S2 share 48.3% to 48.4%.

## The split

- Unit: connected components of S1 entities over (country, normalised core name) and (country, normalised sorted address shared by at most 3 entities). 1,276,462 components, 1,072,762 of size one, largest 6,531.
- Assignment: per country, components largest first, each to the least-loaded fold, ties broken by a seeded shuffle. Deterministic and reproducible (seed 0).
- Five folds of 441,363 to 441,365 entities and 1.526M to 1.530M positives each. Fold 0 is the development fold; the other four give confidence intervals and the final cross-validated estimate.
- Candidate universe for any fold: the full S2/S3 pool, 10.3M records.
- Leave-one-country-out: selections by country from the same table.

## Artifacts

| File | Content |
|---|---|
| entity_folds.parquet | one row per S1 entity: fold, cluster_id, cluster_size, n_true, n_s2, n_s3, k_class, pattern, matched ids |
| positive_pairs.parquet | 7,638,365 true pairs with source and fold |
| fold{f}_ground_truth.tsv | official ground-truth format restricted to fold f |
| fold{f}_entity_ids.txt | S1 ids of fold f, usable as `evaluation.py --ids` |
| summary.json | per-fold statistics |

Functions in `validation.py`: `entity_ids`, `entity_truth`, `label_pairs` (attaches label and fold to any candidate table; absent pairs are genuine negatives because the ground truth is complete), `score` (macro F0.5 with slices by country, match-count class, source pattern, and the empty-prediction rate against the true singleton rate), `format_report`. Command line: `validation.py build` and `validation.py score --fold f`.

## Known gap between validation and the hidden test set

The test pool is denser in distractors per S1 entity than train (5.76 against 4.68 S2/S3 records per S1 entity, an estimated 40% orphans against 26%), and the France portion has larger same-name blocks. Diagnostic on same-name block sizes per S1 entity (no parameter derived): train US mean 54, median 4, p90 150, 27.7% in blocks of ten or more; test US 33, 3, 88, 23.1%; train India 24, 4, 67, 38.6%; test India 26, 5, 71, 38.7%; test France 53, 5, 119, 34.8%. India is the closest proxy for France's block density, so India slices and the leave-one-country-out run are the numbers to watch, and operating points should be chosen conservatively rather than at the validation optimum.
