# Business Entity Resolution Challenge: exploratory analysis (no model trained)

Scope: every file under `student_resource/dataset` was inspected in full with Polars. Pair-level statistics use an entity-level sample of 250,000 training S1 entities with all of their 865,446 true matches, 750,000 random same-country negatives, and 600,000 of the 10.2 million same-name hard negatives. Scripts are in `eda/scripts`, raw numbers in `eda/results`.

Environment used: Python 3.12.10, polars 1.34, pandas 2.3.1, numpy 1.26.4, scikit-learn 1.7.2, scipy 1.11.4. rapidfuzz is not installed. 16 CPU threads, 16.8 GB RAM with about 4 GB free during the session.

## A. Dataset inventory

All seven files are tab-separated, strict UTF-8, LF line endings, no quoting, no ragged lines, no embedded tabs. Row counts equal `wc -l` minus the header.

| File | Size | Rows | Cols | Country split | Missing values |
|---|---|---|---|---|---|
| train/train_source1.tsv | 210 MB | 2,206,821 | 4 | US 1,323,633; India 883,188 | none |
| train/train_source2.tsv | 489 MB | 5,034,616 | 4 | US 3,016,817; India 2,017,799 | address 168,967 (3.36%) |
| train/train_source3.tsv | 504 MB | 5,285,603 | 4 | US 3,170,056; India 2,115,547 | address 175,916 (3.33%) |
| train/train_ground_truth.tsv | 127 MB | 2,206,821 | 2 | one row per S1 entity | empty list 123,247 (5.585%) |
| test/test_source1.tsv | 175 MB | 1,732,544 | 4 | India 809,986; US 663,106; France 259,452 | none |
| test/test_source2.tsv | 509 MB | 4,887,273 | 4 | India 2,312,565; US 1,871,330; France 703,378 | address 129,408 (2.65%) |
| test/test_source3.tsv | 506 MB | 5,082,316 | 4 | India 2,405,000; US 1,945,701; France 731,615 | address 136,098 (2.68%) |

Columns and types. `entity_id` is a string key: prefix S1-, S2- or S3- plus a 1 to 9 digit number. IDs are unique within every file, the prefix always matches the file, and no ID appears in both train and test. `business_name` is free text and never null. `business_address` is free text, null only in S2 and S3. `country` is a categorical string with two values in train and three in test.

Ratio of S2 and S3 records per S1 entity:

| Split | Country | S2 per S1 | S3 per S1 |
|---|---|---|---|
| train | US | 2.28 | 2.40 |
| train | India | 2.29 | 2.40 |
| test | US | 2.82 | 2.93 |
| test | India | 2.86 | 2.97 |
| test | France | 2.71 | 2.82 |

Test has 23% more S2/S3 records per S1 entity than train. Either test entities have more duplicates or, more likely, test has more distractor records with no S1 entity.

## B. Entity-level target definition

For each S1 entity e, the target is a set M(e) of S2 and S3 IDs. Verified structural facts:

- Every S1 entity has exactly one ground-truth row, and every ground-truth row refers to an existing S1 entity. No duplicate rows, no duplicate IDs inside lists, no S1 IDs inside lists, no whitespace.
- Every matched ID exists in the S2 or S3 file.
- Every S2/S3 record appears in at most one list. 7,638,365 matched IDs, all distinct. The relation is one-to-many from S1 and one-to-one from S2/S3. This is a hard constraint a solver can exploit.
- Country never disagrees between an S1 entity and its matches, across all 7.6 million pairs. Country is a safe hard block.
- Set size ranges from 0 to 11. Mean 3.46, median 3, p90 6, p99 8. S2 contributes at most 5, S3 at most 6.

Distribution of |M(e)| on train:

| Matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9+ |
|---|---|---|---|---|---|---|---|---|---|---|
| Share of S1 | 5.59% | 5.40% | 17.00% | 24.06% | 21.94% | 14.59% | 7.47% | 2.90% | 0.85% | 0.22% |

Per-source counts:

| Count | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| S2 matches per S1 | 13.04% | 35.76% | 29.58% | 15.13% | 5.40% | 1.10% | 0 |
| S3 matches per S1 | 12.07% | 32.46% | 30.29% | 16.88% | 6.58% | 1.60% | 0.13% |

Source-pair patterns: both S2 and S3 80.48%, S3 only 7.45%, S2 only 6.48%, neither 5.59%. Zero counts are correlated: given no S2 match, the chance of no S3 match is 43% against a 12% base rate. "Has any duplicates" is an entity-level latent, not two independent draws.

US and India have identical statistics to three decimals: singleton rate 5.58% in both, mean 3.46 in both, max 11 in both. The generator is country-agnostic, so France should follow the same law.

Pool coverage: 73.4% of S2 records and 74.6% of S3 records belong to some S1 entity. The remaining 2,681,854 records, 26.0% of the pool, are orphans that must be rejected for every S1 entity.

## C. Pair-level target definition

A pair (s1, c) with c in S2 or S3 of the same country is positive if c is in M(s1). Counts and imbalance:

| Universe | Pairs | Positive rate |
|---|---|---|
| All same-country cross pairs | 1.2 x 10^13 | 6.5 x 10^-7 |
| Same country and identical core name | 42.7 candidates per S1 | about 4% |
| Top-30 candidates per S1 at 95% recall | 30 per S1 | about 11% |

Hard negatives: on average 40.9 S2/S3 records per S1 entity share its exact core name, same country, and are not matches. Positives per S1 that share the core name: about 1.7. So inside a name block the imbalance is roughly 1 to 24.

## D. Official scoring implications

Macro F0.5 over S1 entities. Per-entity values for an entity with k true matches:

| k | Perfect | Perfect plus 1 false match | Miss 1 of k | Only 1 of k found | Predict empty |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 1 | 1.000 | 0.556 | 0.000 | 1.000 | 0.000 |
| 2 | 1.000 | 0.714 | 0.833 | 0.833 | 0.000 |
| 3 | 1.000 | 0.789 | 0.909 | 0.714 | 0.000 |
| 4 | 1.000 | 0.833 | 0.938 | 0.625 | 0.000 |
| 5 | 1.000 | 0.862 | 0.952 | 0.556 | 0.000 |

Oracle baselines on train, which anchor what a score means:

| Strategy | Macro F0.5 |
|---|---|
| Predict empty for everything | 0.056 |
| Exactly one correct match per non-singleton, empty for singletons | 0.696 |
| Perfect, plus one false match on every entity | 0.752 |
| Perfect, but miss one match on every multi-match entity | 0.925 |
| Perfect, plus one false match on every singleton | 0.944 |

Consequences. A single false match costs 0.14 to 0.44 per entity while a single miss costs 0.05 to 0.17 when k is at least 2. Predicting empty for an entity that has matches scores zero, so recall still matters at the entity level. The optimal decision is per entity: order candidates by probability and pick the prefix that maximises expected F0.5, which depends on how many confident candidates exist, not on one global threshold. Every S1 entity has equal weight regardless of k.

## E. Singleton analysis

- 123,247 train singletons, 5.585%, identical across US and India. Expect about 97,000 in test.
- Singletons are not empty blocks. 65% have at least one S2/S3 record in their country with the identical core name, median 1, mean 40, and 30% have ten or more. For non-singletons the figures are 95%, median 4, 32%.
- 21% of orphan records share an exact core name with some S1 entity. 5.8% of orphans are near misses: same core name and address token Jaccard at least 0.5, usually the same street with a nearby house number, for example "8208 Flicker Pl" against "8212 Flicker Pl". About 7% of S1 entities have at least one such near-miss orphan.
- True matches themselves contain house-number corruptions: 15.7% of positive pairs with an address present share no numeric token at all, for example "6383 Electric Railway" against "6381 ELECTRIC RAILWAY". Number equality is therefore strong but not decisive evidence in either direction.
- Test likely has a higher orphan share than train, about 40% against 26%, so the false-merge pressure on singletons and on everyone else is larger in test than any train-based validation will show.

## F. Multi-match analysis

- 89.0% of S1 entities have two or more matches. To score well an entity needs most of its 3 to 4 matches found and no extras.
- Sibling records of the same entity are corrupted independently. Two S2 records of the same entity share the identical raw name only 4.7% of the time, the same core name 30.2%, the same sorted address 26.2%. Each match must be found on its own evidence, but siblings agree on the truth, so records that are near-duplicates of a confident match are strong secondary candidates.
- Recall at the entity level is much lower than at the pair level. A blocking union with 98.4% pair recall leaves only 95.5% of entities with all matches present.
- 4.4% of true matches have a null address. With 41 same-name distractors per entity on average, a null-address candidate is only resolvable when the name is rare in the pool.
- The one-to-one constraint on the S2/S3 side means competing S1 entities can be resolved globally: assign each candidate to its best S1 entity rather than letting several S1 entities claim it.

## G. Candidate-generation implications

Recall and candidate volume per S1 entity for concrete same-country keys, measured on the 250,000-entity sample against the full 10.3 million record pool:

| Blocking key | Pair recall | Mean candidates | Median | p99 |
|---|---|---|---|---|
| Exact core name, tokens sorted, suffixes removed | 50.5% | 43 | 4 | 690 |
| Name first token | 72.8% | 5,427 | 1,764 | 27,973 |
| Name 4-character prefix | 80.1% | 7,608 | 4,185 | 58,150 |
| Exact sorted address | 12.2% | 0.5 | 0 | 4 |
| All address numbers, as a signature | 66.8% | 2,299 | 216 | 33,449 |
| First house number plus longest address word | 54.6% | 228 | 3 | 6,934 |
| Name first token plus longest address word | 51.2% | 47 | 3 | 588 |
| Rare name tokens, document frequency at most 5,000 | 58.0% | 1,125 | 223 | 6,462 |
| Rare address tokens, document frequency at most 3,000 | 83.8% | 1,808 | 1,469 | 6,832 |
| Rare address tokens, document frequency at most 10,000 | 91.2% | 7,020 | 5,804 | 25,788 |

Unions:

| Union | Pair recall | Entities with all matches present |
|---|---|---|
| Core name only | 50.5% | 14.5% |
| Core name, sorted address, number signature, number plus street word | 87.9% | 72.0% |
| Above plus name 4-character prefix | 95.1% | 87.0% |
| Above keys plus rare name tokens 5,000 and rare address tokens 3,000 | 98.4% | 95.5% |
| Same with thresholds 20,000 and 10,000 | 99.3% | 98.0% |

What the true matches look like, pair level:

| Signal | Positives | Random negatives | Same-name hard negatives |
|---|---|---|---|
| Identical raw name | 4.6% | 0.0% | 8.9% |
| Identical core name | 48.1% | 0.0% | 97.2% |
| Any shared name token | 85.3% | 1.4% | 100% |
| Name token Jaccard at least 0.5 | 74.1% | 0.01% | 100% |
| Any shared address token | 95.6% | 21.0% | 17.0% |
| Address token Jaccard at least 0.3 | 87.3% | 0.02% | 0.41% |
| Shared house or plot number | 79.9% | 2.7% | 1.8% |
| Name or address Jaccard at least 0.3 | 98.4% | 0.19% | 100% |

The 14.7% of positives with no shared name token split into three populations that need dedicated handling: Indic-script names 6.8%, domain-style names such as "valuerealtydigital com" or "#KOKANINDIA" 4.7%, and outright random renames such as "Iriarc" or "Dovakor" 3.2%. Among India S2/S3 names, 17% contain no Latin letters at all: Devanagari 10.3%, Telugu 1.5%, Kannada 1.4%, Tamil 1.3%, Bengali 1.2%, Gujarati 1.2%, Malayalam 0.7%, Oriya 0.3%, Gurmukhi 0.3%. Test India has the same mix.

Implications:

1. Name-only blocking caps recall near 50% exact or 85% token level. The address is the main recall channel and must be blocked on rare tokens and numbers.
2. Same-name hard negatives are resolved by the address in more than 99% of cases, so the matcher's job is mostly address comparison inside name blocks, plus name comparison inside address blocks.
3. Volumes per entity from the high-recall union run to thousands, so a cheap ranking stage such as character n-gram TF-IDF cosine over name plus address must cut to a top-K of roughly 30 to 50 before the learned matcher. That top-K list is what goes into candidate_pairs.tsv.
4. All thresholds are document-frequency based and country-agnostic, so they transfer to France if the frequencies are computed on the test pool itself.
5. Transliteration for Indic scripts can be learned from the roughly 1.1 million training pairs that have one Latin and one Indic name, which keeps the pipeline free of external resources.

## H. Validation-strategy implications

- Split by S1 entity, never by pair. Pairs of one entity share the S1 record and the siblings share the truth. Hold out about 10% of S1 entities and evaluate the entity-level macro F0.5 with singletons scoring 1 or 0.
- Keep the whole same-country S2/S3 pool as the candidate universe for held-out entities. Per-entity block sizes then match test, because block size depends on the pool, not on how many S1 entities exist.
- Test has a higher orphan share, about 40% against 26%, which changes the prior that a plausible candidate is real and weakens any global assignment step. Simulate it by hiding a further slice of validation S1 entities while keeping their S2/S3 records in the pool, and confirm thresholds are stable.
- Run a leave-one-country-out check, train on US and validate on India and vice versa, to estimate the drop expected on France, which is 15% of the test entities.
- Report F0.5 by match count k, by country, and separately for singletons. Tune decisions on the entity metric, not on pair-level AUC.
- France is absent from train, so any per-country lookup table must have a generic fallback path.

## I. Leakage risks

1. IDs and row order carry no signal. Spearman correlation between matched IDs is 0.0001, between row positions 0.001, sibling rows are a million rows apart, S2 and S3 numeric overlap equals chance. Do not use them, and there is nothing to exploit.
2. No record overlap between train and test at the name plus address level, zero rows in all three sources. 43% of test S1 core names occur in train S1 because the synthetic vocabulary is shared, which is harmless.
3. Pair-level random splits leak through shared S1 records, siblings, and same-name clusters. Group folds by S1 entity.
4. Any feature derived from ground truth, such as whether a pool record is matched or cluster sizes from labels, does not exist at test time. Transductive features from the test pool itself, such as token frequencies and block sizes, are legitimate.
5. Hard-coding the country set or state maps only for US and India breaks France. The README explicitly forbids this.
6. External geocoding, registries, or lookup services are disqualifying. Transliteration and abbreviation maps must be learned from the training pairs or be rule-based.
7. Validation on train will be optimistic because of the orphan-share shift.

## J. Recommended next steps

1. Normalisation module, country-agnostic: NFKD and casefold; strip noise prefixes such as "--", "<<", "***"; legal-suffix dictionary covering US, India and France forms including local-script forms; honorific stripping for Mr, Smt, Dr, Sri, Shri, M/s; alias splitting on fka, formerly, dba, t/a; space-less comparison for domain-style names; digit-for-letter canonicalisation; address NULL-token removal; leading-zero, "#" and trailing-dot cleanup on numbers; street-type maps; state and region canonicalisation learned from matched pairs.
2. Candidate generation as a union of channels with per-country document-frequency thresholds, measured against targets of at least 98% pair recall and 95% entity-complete recall, then a TF-IDF ranking stage to a fixed top-K. Measure recall separately for Indic-script, domain-style and renamed positives.
3. Pair features: name similarities including transliterated and space-less variants, address similarities with explicit number comparison, entity-context features such as block size, margin to the best candidate, and the candidate's best alternative S1 entity.
4. A gradient-boosted matcher with entity-grouped cross-validation, followed by per-entity expected-F0.5 subset selection and a one-to-one assignment across S1 entities.
5. Validation exactly as in section H, plus the orphan-share simulation.
6. Generate both output files, keep candidate_pairs.tsv identical to the matcher's input, and run utils/validate_submission.py with --check-ids.
7. Memory budget: process per country with Polars and Parquet, never materialise a full cross join, and expect roughly 50 to 90 million candidate IDs in the final file.

## Supporting detail for the numbered questions

Name and address lengths, characters, median with p5 and p99:

| Source, country | Name | Address |
|---|---|---|
| S1 US | 22, 11 to 41 | 34, 26 to 55 |
| S1 India | 27, 13 to 43 | 76, 47 to 134 |
| S1 France | 19, 12 to 33 | 48, 40 to 78 |
| S2 US | 23, 11 to 46 | 32, 24 to 49 |
| S2 India | 27, 13 to 49 | 69, 36 to 128 |
| S2 France | 20, 11 to 41 | 41, 24 to 70 |
| S3 US | 23, 11 to 49 | 39, 29 to 60 |
| S3 India | 27, 12 to 51 | 60, 23 to 126 |
| S3 France | 20, 10 to 43 | 41, 24 to 70 |

Names have 3 to 4 tokens at the median everywhere. S3 India addresses are shorter than S1 because components are dropped.

Exact duplicates on name plus address plus country: S1 zero in train and test; S2 0.51% train and 0.46% test; S3 0.36% train and 0.32% test.

Name collisions inside S1, same country: 38% of entities share a raw name, 49.5% share a core name in train and 47% in test; 24% sit in groups of ten or more, India 33%, US 18.5%, France 20%; largest group 556. 5.3% of S1 entities share an exact normalised address with another S1 entity, and in test France one address, "12 RUE Lyderic, Lille", is shared by 99 entities. Only 14 train S1 rows and 93 test rows share both core name and sorted address, so S1 is deduplicated in practice, but 21% of entities share name plus state and are only separated by street-level detail.

Hidden identifiers: none. No PIN codes, no ZIP codes, no phone numbers, no emails. Five-digit numbers in 10.8% of US addresses are house numbers.

Common suffixes and abbreviations. US: LLC, Inc, Corp, PC, PLLC, LP, Co, Corporation, Incorporated, Ltd, with punctuated variants such as L.L.C. India: Limited, Ltd, Private, Pvt, LLP, Co, Sons, Brothers, plus local-script forms of Private Limited and honorific prefixes in 7.6% to 8.3% of S2/S3 names. France: SARL, SAS, EURL, SA, SASU, SCI, EI, SNC, Cie, Ets, Groupe, Fils, Développement, Participations, Holding. Address abbreviations: US Rd, St, Ave, Dr, Ln, Ct, Blvd, Cir, Pl, Trl; India Rd, Ngr, Flr, H.No, Door No, Opp, Nr, Sy No, Kh No, and state codes MH, DL, UP, KA, TN, WB, GJ, TG, HR; France R. for Rue, Av., Bd, Allée, Rte, Imp., and the token "bis" in house numbers.

Country formatting. US S1 is "number street type, city, two-letter state" with 3 components in 84% of rows. S2 US is 90% upper case, 48% abbreviated street types, state moved to the front in 6%, PO Box or PMB inserted in 3.3%, literal NULL tokens in 3.9%. S3 US is title case with full state names in 96% and alias markers such as "formerly" in 2.5% of names. India S1 has a median of 5 comma components, landmarks such as Near or Opp in 13.5%, plot, flat or floor numbers in 55%, full state names, and no PIN codes. S2 India has 24% upper case and the state written in a local script in 24% of rows. S3 India uses two-letter state codes in 57% and drops components. France S1 is "number street, city, region" over three regions and about 18 cities, with accents in 16% of names and 28% of addresses. France S2 is 21% upper case, abbreviates Rue to R. and Avenue to Av., replaces the region with the department in about half the rows, drops the region entirely in a quarter, and turns some names into two-letter acronyms. France S3 is title case with mixed accent stripping.

Injected noise rates in S2/S3 names: double spaces 10% to 12%, accents 4% to 7%, digit-for-letter typos 1.2% to 1.9%, an appended generic token such as Service, Center or Partners 5% to 10%, domain-style rewrite 3.7% to 4.8%, single-token names 4% to 8%. In addresses: NULL tokens 3% to 4%, leading zeros 4% to 5%, "#" prefixes 5% to 11%, Door No or H.No prefixes 11% to 12% in India, CITY, CDP or Township suffixes 5% to 9%, trailing dot after the number 1% to 2%.
