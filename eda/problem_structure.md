# Structure of the optimisation problem and choice of prediction formulation

All statistics below come from `train_ground_truth.tsv` read and scored with the verified evaluator
(`code/business_entity_resolution/src/evaluation.py`). Scripts: `eda/scripts/s8a_structure.py`
(structure, independence tests, ceilings) and `eda/scripts/s8b_decision_sim.py` (decision-rule
simulation). Raw numbers: `eda/results/s8a_structure.json`, `eda/results/s8b_decision_sim*.json`.

## 1. Match-count structure (items 1 to 4 and 8)

2,206,821 Source 1 entities, 7,638,365 true matches, mean 3.46 per entity.

| True matches k | Entities | Share |
|---|---|---|
| 0 (singleton) | 123,247 | 5.585% |
| 1 | 119,157 | 5.399% |
| 2 | 375,212 | 17.002% |
| 3 or more | 1,589,205 | 72.013% |

Full distribution: 3: 24.06%, 4: 21.94%, 5: 14.59%, 6: 7.47%, 7: 2.90%, 8: 0.85%, 9: 0.19%, 10: 0.024%, 11: 0.002%.
Per source: S2 matches 0/1/2/3/4/5 = 13.0 / 35.8 / 29.6 / 15.1 / 5.4 / 1.1%; S3 matches 0/1/2/3/4/5/6 = 12.1 / 32.5 / 30.3 / 16.9 / 6.6 / 1.6 / 0.13%.

## 2. Source structure (items 5, 6, 7)

**Item 6.** Multiple records from one source do match one entity: 51.2% of entities have two or more S2 matches, 55.5% have two or more S3 matches. Maximum 5 from S2 and 6 from S3, never exceeded.

**Item 7.** No S2/S3 record is ever matched to two Source 1 entities: 7,638,365 matched IDs, all distinct. The relation is one-to-many from S1 and one-to-one from S2/S3.

**Item 5.** S2 and S3 counts are independent among entities that have any match, and dependent only through the singleton class.

| Condition | P(k3 = 0..6) |
|---|---|
| k2 = 0 | 0.428, 0.210, 0.198, 0.110, 0.043, 0.011, 0.001 |
| k2 = 1 | 0.075, 0.342, 0.318, 0.178, 0.069, 0.017, 0.001 |
| k2 = 2 | 0.074, 0.342, 0.319, 0.178, 0.069, 0.017, 0.001 |
| k2 = 3 | 0.075, 0.343, 0.317, 0.178, 0.070, 0.017, 0.001 |
| k2 = 4 | 0.075, 0.341, 0.322, 0.175, 0.068, 0.018, 0.001 |
| k2 = 5 | 0.077, 0.342, 0.323, 0.173, 0.069, 0.016, 0.002 |

| Test | Cramér's V | chi-square p | n |
|---|---|---|---|
| All entities | 0.164 | 0 | 2,206,821 |
| Entities with k2 at least 1 | 0.0021 | 0.070 | 1,919,076 |
| Entities with k3 at least 1 | 0.0020 | 0.060 | 1,940,545 |

Observed singleton rate 5.585% against 1.573% if k2 and k3 were independent: a 3.55 times excess. P(k3 = 0 | k2 = 0) = 42.8% against 7.5% otherwise. Correlation of k2 and k3 is 0.12 overall and -0.02 among non-singletons. The generator behaves like: singleton with probability 5.6%, otherwise draw the S2 count and the S3 count independently.

## 3. What perfect ranking can score

Scored on the full ground truth with the evaluator, using the true matches in place of a ranker.

| Rule with perfect scores | Macro F0.5 |
|---|---|
| Always top 1 | 0.6405 |
| Always top 2 | 0.7930 |
| Always top 3 (best fixed cut-off) | 0.8102 |
| Always top 4 | 0.7605 |
| Best S2 plus best S3 | 0.8574 |
| Perfect sets but never empty | 0.9442 |
| One correct match per non-singleton, empty for singletons | 0.6964 |

## 4. Decision-rule simulation on the real match structure

Setup: 200,000 real S1 entities with their real matched IDs. Candidate set per entity: every true match, the entity's same-name records that are not matches (measured block size, capped at 27), a Poisson(2) number of address-block distractors, and with probability 0.10 one near-miss negative. 13.9 candidates per entity, 24.9% positive. Scores are drawn from Beta distributions by candidate type (82% clear positives, 18% weak positives for the no-name-overlap and missing-address cases, ordinary negatives, near-miss negatives) at three strengths, then calibrated with isotonic regression on half of the entities; every rule is scored on the other 100,000 with the evaluator. Two parameters were measured from the data: 0.5% of near-miss negatives belong to another S1 entity, and 0.006% of true matches are address-ambiguous with another S1 entity.

| Decision rule | Weak scorer, AUC 0.901 | Medium, AUC 0.987 | Strong, AUC 0.998 |
|---|---|---|---|
| A. Global threshold on calibrated p, best t | 0.7209 (t 0.45) | 0.9257 (t 0.65) | 0.9700 (t 0.75) |
| A at t = 0.5 | 0.7196 | 0.9208 | 0.9671 |
| B. Top 1 per entity | 0.5786 | 0.6365 | 0.6424 |
| B. Best fixed top-k (k = 3) | 0.6942 | 0.7980 | 0.8138 |
| B. Top round(sum of p) | 0.7008 | 0.8934 | 0.9629 |
| C. Entity-relative threshold, best | 0.7280 | 0.9252 | 0.9694 |
| D. Expected-F0.5 set, calibrated p | 0.7203 | 0.9230 | 0.9693 |
| D with raw uncalibrated scores | 0.7022 | 0.8971 | 0.9476 |
| E. One-to-one assignment then D | 0.7204 | 0.9231 | 0.9694 |
| A / D with 3% of true matches missing from candidates | | 0.9153 / 0.9128 | |

Per-k view at medium strength (rerun with fresh draws: A 0.9254, D 0.9225, D with singleton prior 0.9242):

| k | Entities | A best | C best | D | B top 3 |
|---|---|---|---|---|---|
| 0 | 5,554 | 0.871 | 0.820 | 0.776 | 0.043 |
| 1 | 5,326 | 0.827 | 0.850 | 0.856 | 0.436 |
| 2 | 17,185 | 0.909 | 0.915 | 0.916 | 0.710 |
| 3 | 24,314 | 0.932 | 0.933 | 0.934 | 0.949 |
| 4 | 21,660 | 0.941 | 0.940 | 0.940 | 0.923 |
| 5 | 14,592 | 0.947 | 0.946 | 0.945 | 0.876 |
| 6 or more | 11,369 | 0.951 | 0.950 | 0.948 | 0.81 |

Empty-set prediction rate: truth 5.55%, A 5.70%, C 5.14%, D 4.78%, D with singleton prior 5.76%.
D's own expected score was 0.9256 against 0.9225 realised, so it estimates its score without labels to within a third of a point; at weak strength the gap is 1.1 points.

## 5. Verdict

- **B, per-entity ranking with a cut-off, is ruled out by the data.** k runs from 0 to 11, 72% of entities need three or more matches and 5.6% need none. Even a perfect ranker with the best fixed k scores 0.81, best S2 plus best S3 scores 0.86, and a rank-then-count rule trails the best rule by 1 to 3 points at every strength. Ranking only supplies an ordering inside a set decision.
- **A, independent pair classification, is the right training formulation** and, with a tuned threshold on calibrated probabilities, is within 0.3 points of the best rule at every strength. Its weakness is the threshold: the optimum moves from 0.45 to 0.75 with scorer quality, it must be tuned on the entity metric, and it cannot be tuned on the test set where the scorer will be weaker (France, more distractors).
- **C adds nothing over A** and has two parameters.
- **D, expected-F0.5 set selection, matches A on aggregate, wins on k = 1 to 3 and loses on singletons** because independent Bernoulli candidates under-predict the empty set (4.8% against 5.55% truth). Adding the empirical singleton prior removes most of that gap. It needs calibrated probabilities (2.5 points lost with raw scores), has no threshold to tune, and yields a label-free estimate of the score.
- **E, adding one-to-one assignment, is worth nothing measurable** because address ambiguity between S1 entities is essentially absent: same-name competitors are separated by address in 99.6% of cases and near-miss negatives are orphans that nothing else claims. Keep it only as a consistency guard.

**Recommended formulation: a hybrid of A and D.** Train a calibrated pairwise classifier on (S1, candidate) labels, then make one set decision per S1 entity by maximising expected F0.5 over the ranked candidates, using the empirical entity prior instead of independence: singleton probability 5.6%, S2 and S3 counts independent given that any match exists, at most 5 S2 and 6 S3 matches. Because P(k3 = 0 | k2 = 0) is 43%, the absence of a credible S2 candidate should raise the bar for S3 candidates and vice versa. Validate the set decision against a global-threshold sweep on the entity metric and keep the one-to-one guard.

The decision layer separates good rules by a few tenths of a point but separates them from ranking by 12 to 20 points. The score is dominated by pair-level discrimination (0.72 to 0.93 to 0.97 as AUC goes 0.90 to 0.987 to 0.998), then by calibration, then by candidate recall (about 1 point per 3% of misses). Modelling effort should go, in that order, to same-name near-miss rejection, weak-positive recovery, calibration, and blocking recall.
