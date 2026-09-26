## 6. The cheapest fix per failure type

Every fix was run on the whole fold and its recovered pairs attributed to the cause of the pair (section 3). The table gives, per cause, the fix that recovers the most pairs per candidate added (measured alone against the old set), its cost, whether the selection kept it, and how many pairs of that cause the new thirteen-strategy set recovers in the end. Costs are for the 441,365 fold-0 queries on this machine (12 threads).

| cause | misses | cheapest fix | recovered by that fix alone | cost | kept | recovered by the new set |
|---|---|---|---|---|---|---|
| missing address | 7,286 | name 3-gram retrieval against the address-less pool records only (58k-111k records per country and source, about 3% of the pool), top 5 per source, cosine >= 0.3 | 2,733 (37.5%) | +7.08 candidates per entity, 40 s | yes | 2,733 (37.5%) |
| candidate limit too small | 5,530 | exact name (any representation) + city, blocks <= 100; then the address-token retrieval limited to its top 5; then exact name + house number and compact name + state | 956, 616, 268, 96 | +0.19, +2.42, +0.01, +0.06 candidates; the address retrieval costs 5 min | yes | 1,627 (29.4%) |
| blocking collision | 3,356 | exact name (any representation) + city, blocks <= 100 (the collisions are common-name blocks in India that the cap-200 exact-name key drops; adding the city splits them) | 1,773 (52.8%), plus 286 by name + house number and 237 by compact name + state | +0.19 candidates per entity, 10 s | yes | 1,966 (58.6%) |
| name variation | 2,113 | address-token retrieval top 5 (different names at the same address: only the address side can reach them) | 73 (3.5%); 212 with the top 20 at +22.6 candidates | +2.42 candidates per entity | yes (top 5) | 75 (3.5%) |
| abbreviation | 1,463 | name 3-gram retrieval against the domain-style pool names (`nbrothers.com`), top 3 per source | 400 (27.3%) | +5.21 candidates per entity, 45 s | no: 0.000128 recall per candidate over the new set | 104 (7.1%) |
| address variation | 1,123 | none below the frontier: the domain-style retrieval recovers 153, the name 3-gram retrieval 140 for +29.6 candidates | - | - | no | 8 |
| other | 850 | none: the address-token retrieval top 20 recovers 53 for +22.6 candidates | - | - | no | 16 |
| retrieval failure | 829 | none: the domain-style retrieval recovers 132, the address-token top 20 87, the 3%-pruning retrieval 67 | - | - | no | 39 |
| typo, transliteration | 58 | none needed (1-3 pairs recovered by any fix) | - | - | no | 1 |

Two fixes that looked promising in section 2 did not survive the whole-fold measurement:

* *Extending the name + address retrieval past rank 20.* Ranks 21-30 add 19.7 candidates per entity for 2,441 pairs (0.000081 recall per candidate), ranks 21-50 add 58.5 for 5,128 (0.000057). Making the extension adaptive (ranks 21-50 only when the 20th score is still >= 0.5, 0.6 or 0.7, i.e. a flat score curve) concentrates it to 3.9, 1.5 and 1.1 candidates but the gain shrinks in proportion (781, 299 and 183 pairs: 0.00013 per candidate). The score curve does not tell where the true record hides.
* *Pruning tokens at 3% document frequency instead of 1%.* The retrieval then keeps tokens like `nagar` or `avenue` and recovers 3,935 of the misses (2,235 of the rank 21-100 ones) for +8.3 candidates, the best ratio among the retrieval variants (0.000309). But it competes for the same pairs as the cheap keys and the address retrieval: after those are added it recovers 2,132 pairs for +7.6 candidates (0.000184 per candidate, below the 0.0002 frontier), and it costs 26 minutes of compute against 9 for the 1% run because the matrices are eight times denser. It stays out; it is the first thing to revisit if the matcher shows that the rank 21-100 misses are cheap to score.

## 7. The new final set: old versus new

Greedy forward selection over the old set, accepting a fix while it buys at least 0.0002 recall per candidate added (the implicit frontier of `candidate_generation.md`, one recall point per 50 candidates), with the gains re-measured after every addition:

| step | fix added | recall gain (points) | candidates added per entity | recall per candidate |
|---|---|---|---|---|
| 1 | exact name (any representation) + house number | +0.036 | +0.01 | 0.0259 |
| 2 | exact name (any representation) + city, blocks <= 100 | +0.154 | +0.19 | 0.0083 |
| 3 | compact name + state | +0.007 | +0.06 | 0.0011 |
| 4 | house number + street + city, blocks <= 200 | +0.001 | +0.03 | 0.0004 |
| 5 | name 3-gram retrieval against the address-less pool records, top 5 | +0.179 | +7.08 | 0.00025 |
| 6 | address-token retrieval, top 5 per source | +0.053 | +2.42 | 0.00022 |

What every rejected fix would still add on top of the new thirteen (`eda/results/s16b_marginal_over_new.json`): 3%-pruning retrieval +0.140 points for +7.6 candidates (0.000184 per candidate), domain-style name retrieval +0.066 for +5.2 (0.000128), adaptive ranks 21-50 at >= 0.5 +0.029 for +3.8 (0.000076), ranks 21-30 +0.098 for +18.7 (0.000053), address-token top 10 +0.032 for +5.4 (0.000059), address-token top 20 +0.068 for +20.2 (0.000034), ranks 21-50 +0.213 for +56.5 (0.000038), name 3-gram retrieval top 20 +0.072 for +29.6 (0.000024), acronym + city +0.046 for +29.0 (0.000016). All of them are below the frontier, and even all nine together could add at most 0.76 points (the sum of their gains) for up to 176 more candidates per entity.

| | old (7 strategies) | new (13 strategies) |
|---|---|---|
| candidate recall (pairs) | 0.9852 | **0.9895** |
| recall Source 2 / Source 3 | 0.9859 / 0.9845 | 0.9907 / 0.9884 |
| recall US / India | 0.9883 / 0.9806 | 0.9909 / 0.9875 |
| non-singleton entities with every match present | 95.12% | 96.43% |
| true pairs missed | 22,608 | 16,039 |
| average candidates per entity | 81.4 | **91.2** |
| median | 51 | 62 |
| 95th percentile | 233 | **241** |
| 99th percentile / maximum | 279 / 689 | 288 / 696 |
| pairs | 35,911,531 | 40,233,392 |
| entities without a candidate | 0 | 0 |
| reduction ratio | 0.999992 | 0.999991 |
| file | 469 MB (`candidate_pairs_v1.tsv`) | 524 MB (`candidate_pairs.tsv`) |

The new set recovers 6,569 of the 22,608 misses (29%) for 12% more candidates: 0.43 points of recall for 9.8 candidates per entity, 0.00044 per candidate, twice the frontier rate. The India gap closes from 0.77 to 0.34 points because the collisions were Indian common-name blocks. The 16,039 remaining misses are 1.05% of the true pairs: missing address 4,553 (typo'd names that the 3-gram top 5 does not reach), candidate limit 3,903, name variation 2,038, blocking collision 1,390, abbreviation 1,359, address variation 1,115, other 834, retrieval failure 790, typo 52, transliteration 5. The next cheapest step would be widening the address-less retrieval (top 10, cosine 0.2), which was not measured.

**Cost.** The production run of the thirteen strategies on the fold takes 18 minutes on this machine (12 threads) against 10.5 for the seven: the address-token retrieval adds 5.6 minutes, the address-less retrieval 44 seconds, the four keys and the extra exact-name joins about 1 minute (`eda/results/s16d_cli_log.txt`). The whole-fold measurement of every variant in this section took 50 minutes, dominated by the two extra full retrievals that were rejected (top 50: 10 minutes, 3% pruning: 26 minutes). For the 1,732,544 test entities the retrievals and joins scale with the number of queries (3.9 times more), so the extended set needs about 1 hour 20 minutes end to end here, and the test file will hold about 158 million ids (2.1 GB).

## 8. The competition file and what the matcher inherits

`artifacts/candidates/fold0/candidate_pairs.tsv` was regenerated from the thirteen-strategy union in the specification's format (header `source1_entity_id<TAB>candidate_entity_ids`, one row per Source 1 entity in file order, ids sorted and comma-separated, LF endings): 441,365 rows, 40,233,392 ids, 524 MB, checked line by line by `candidates.validate_candidate_file` (no problems). `src/generate_candidates.py` now runs the thirteen strategies by default (`--v1` gives the seven-strategy set); the address-token top 5 is taken as ranks 1-5 of the top-20 run, exactly as it was measured, so that ties at the fifth rank cannot differ between the measured and the generated file. Run on the fold-0 entities, the CLI reproduced the measured file byte for byte (`cmp` in `eda/results/s16d_cli_log.txt`): 441,365 rows, 40,233,392 ids, 524,266,254 bytes, in 18 minutes on this machine (12 threads) against 10.5 for the seven-strategy set.

The labelled fold-0 set now has 40,233,392 pairs with 1,511,611 positives (3.76%, down from 4.19%), 91 candidates per entity, and for every pair the membership flags of the thirteen strategies, the retrieval ranks and scores of the name + address, address-only and name-only retrievals, and the exact-key agreements as features. The recall ceiling for the matcher on this fold rises from 0.9852 to 0.9895; at the problem-structure estimate of one point of macro F0.5 per three points of blocking miss, the ceiling now costs about 0.35 points against a perfect candidate set instead of 0.5.
