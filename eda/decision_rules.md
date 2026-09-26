# Decision rules beyond the global threshold (LightGBM probabilities, fold 0)

Every strategy's parameters are chosen on the tuning half (55,171 entities) and every number below is the official macro F0.5 on the report half (55,170 entities, 3,150 true singletons), never used for training or tuning. The incumbent is the global threshold of 0.75 with no other rule.

## 1. What the per-entity signals look like

Deciles (10% / 50% / 90%) of the entity-level signals on the report half, by the true number of matches:

| signal | singleton | one match | 2-5 matches | 6+ matches | singleton-detection AUC |
|---|---|---|---|---|---|
| n_cand | 57 / 65 / 207 | 57 / 65 / 218 | 54 / 62 / 196 | 52 / 60 / 190 | 0.5836 |
| top1 | 0.0018 / 0.0332 / 0.5419 | 0.913 / 0.9993 / 1 | 0.9987 / 0.9999 / 1 | 0.9997 / 1 / 1 | 0.9982 |
| top2 | 0.0003 / 0.0064 / 0.0845 | 0.0009 / 0.0208 / 0.3438 | 0.9777 / 0.9995 / 1 | 0.9987 / 0.9999 / 1 | 0.9776 |
| gap | 0.0004 / 0.0173 / 0.3891 | 0.4142 / 0.9614 / 0.9979 | 0 / 0.0003 / 0.0184 | 0 / 0 / 0.0008 | 0.7867 |
| n_ge_90 | 0 / 0 / 0 | 1 / 1 / 1 | 2 / 3 / 5 | 5 / 6 / 7 | 0.9933 |
| n_ge_50 | 0 / 0 / 1 | 1 / 1 / 1 | 2 / 3 / 5 | 5 / 6 / 7 | 0.9921 |
| best_name_sim | 0.7273 / 1 / 1 | 0.8462 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0.663 |
| best_addr_sim | 0.4286 / 0.6667 / 0.8 | 0.5 / 0.6667 / 1 | 0.6 / 0.7857 / 1 | 0.6667 / 0.875 / 1 | 0.7656 |
| best_cos_name_addr | 0.5674 / 0.7442 / 0.884 | 0.6485 / 0.8582 / 0.9863 | 0.7803 / 0.9502 / 1 | 0.8646 / 0.9741 / 1 | 0.8929 |
| any_exact_name | 0 / 1 / 1 | 0 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0.6604 |
| any_exact_addr | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0.6202 |
| ident_agreement | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0.5398 |

Entities: singleton 3,150, one match 3,044, 2-5 matches 42,721, 6+ matches 6,255.

**Singleton false positives of the global rule.** 173 of the 3,150 true singletons have a candidate above 0.75; their top-1 probability deciles are 0.7827 / 0.8945 / 0.982 and 5% of them are at 0.99 or above, 92% share an exact name and 7% an exact address with the candidate. These are the decoys of `matcher_baseline.md`: the model is certain, and nothing in the entity's signals (a lone strong candidate, a large gap) separates them from a genuine one-match entity, which looks the same.

An entity-level LightGBM trained to recognise singletons from these signals (20 features, out-of-fold on the tuning half) reaches AUC 0.9982 (report half 0.9983); its gain is dominated by `top1` 0.602, `n_ge_75` 0.2, `n_ge_90` 0.108, `sum_p` 0.014, `n_cand` 0.011.

## 2. Strategies, chosen on the tuning half, scored on the report half

| strategy | chosen parameters | tune F0.5 | **report F0.5** | vs global | singleton score | singleton FPs | one-match F0.5 | multi-match F0.5 | false merges | pair recall | mean predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A. global threshold | t 0.75 | 0.9700 | **0.9698** | +0.0000 | 0.9451 | 173 | 0.9142 | 0.9749 | 1,926 | 0.9392 | 3.28 |
| B. top-1 only | t 0.75 | 0.6874 | **0.6886** | -0.2813 | 0.9451 | 173 | 0.9258 | 0.6573 | 219 | 0.2713 | 0.94 |
| C. threshold + top-K | t 0.75, k 11 | 0.9700 | **0.9698** | +0.0000 | 0.9451 | 173 | 0.9142 | 0.9749 | 1,926 | 0.9392 | 3.28 |
| D1. gap cut (stop at the first big drop) | t_top 0.75, t_low 0.6, gap 0.3 | 0.9701 | **0.9700** | +0.0002 | 0.9451 | 173 | 0.9119 | 0.9752 | 2,449 | 0.9463 | 3.31 |
| D2. threshold + relative gate (p >= g * top-1) | t 0.75, rel_gate 0.3 | 0.9700 | **0.9698** | +0.0000 | 0.9451 | 173 | 0.9142 | 0.9749 | 1,926 | 0.9392 | 3.28 |
| E. auto (name+address) + ML threshold | t 0.75, auto_pairs_tune 10115, auto_precision_tune 0.9996, auto_pairs_below_global_t 0, auto_below_precision None | 0.9700 | **0.9698** | +0.0000 | 0.9451 | 173 | 0.9142 | 0.9749 | 1,926 | 0.9392 | 3.28 |
| E. auto (name+house+street/city) + ML threshold | t 0.75, auto_pairs_tune 65945, auto_precision_tune 0.9963, auto_pairs_below_global_t 227, auto_below_precision 0.14978 | 0.9692 | **0.9693** | -0.0006 | 0.9419 | 183 | 0.9138 | 0.9745 | 2,068 | 0.9394 | 3.28 |
| F1. two thresholds by the entity's top-1 probability | s 0.8, t_single 0.85, t_multi 0.7 | 0.9698 | **0.9700** | +0.0001 | 0.9536 | 146 | 0.9060 | 0.9750 | 2,179 | 0.9437 | 3.30 |
| F2. singleton model gates a higher threshold | p_single 0.7, t_single 0.9 | 0.9702 | **0.9697** | -0.0001 | 0.9476 | 165 | 0.9109 | 0.9748 | 1,918 | 0.9392 | 3.28 |
| F3. lower threshold for entities with many strong candidates | min_n_ge_50 8, t_multi 0.7 | 0.9700 | **0.9698** | +0.0000 | 0.9451 | 173 | 0.9142 | 0.9749 | 1,933 | 0.9394 | 3.28 |

## 3. Verdict: the global threshold stays

**Nothing beats it by more than noise.** On the report half the global threshold gives 0.9698. The gap cut (D1) and the two-threshold rule on the entity's top-1 probability (F1) reach 0.9700, +0.0002, but F1 is *worse* than the global rule on the tuning half where it was chosen (0.9698 against 0.9700), and D1 gains 0.0001 there: both are inside the noise of 55,000 entities, and neither shows the same direction on both halves. The singleton model (F2) wins the tuning half by 0.0002 and loses the report half by 0.0001. Every other strategy either collapses to the global rule (the top-K cap, the relative gate and the automatic name + address matches never change a single decision at the chosen parameters) or hurts: top-1 only loses 28 points because the typical entity has three or four true matches; the looser automatic tier (exact name + house number + street or city) admits 227 tuning-half pairs the model had rejected, of which 15% are true, and costs 0.0005.

**Why the per-entity signals cannot fix the singleton false positives.** The 173 singletons that receive a match (5.5% of 3,150) look exactly like one-match entities: one strong candidate (median top-1 0.89, second candidate near 0), the same name in 92% of cases, a partly matching address (median token Jaccard 0.67). A singleton detector built on the entity signals reaches AUC 0.998 only because the other 95% of singletons have no candidate above 0.5; on the entities that matter, the strong-candidate ones, it has no information the pair model did not already use. Raising the bar for weak-looking entities (F1: 0.85 when the best candidate is below 0.8) removes 27 of the 173 singleton errors but breaks 25 more one-match entities, a wash. The singleton errors are decoy records, and the pair features are where they must be separated (`matcher_baseline.md`, section 6).

**Why the multi-match side does not need special logic.** The gap between consecutive candidates carries no information beyond the probabilities themselves: the model's outputs are bimodal (most true matches above 0.9, most non-matches below 0.1), so a run cut at a large drop keeps the same pairs as the threshold, and a lower threshold for entities with many strong candidates (F3) changes nothing at its best setting. Country and identifier agreement do not enter: the candidates always share the country, and the identifier-like signal (a domain-style pool name matching the entity's name) covers 0.3% of the entities.

**Decision.** The pipeline keeps the global threshold on the prior-corrected LightGBM probability (0.75, anywhere in 0.63-0.80 is equivalent) plus pool-side uniqueness. No entity-level rule is adopted, since none demonstrated an improvement on the held-out half that also held on the tuning half. The 173 singleton false positives and the 1,926 false merges are a modelling problem for the next stage: features that isolate "same name, one number changed" decoys.
