# The pairwise feature matrix (fold 0)

`src/features.py` computed 157 label-free features for every one of the 40,233,392 candidate pairs of the fold-0 set (1,511,611 positives, 3.76%), in 32 buckets of Source 1 entities written to `artifacts/features/fold0/part_XX.parquet` (3.22 GB on disk). Every feature is a function of the two records' preprocessed representations, of unlabelled pool statistics (TF-IDF vocabularies and name / address frequencies fitted on the pool of the same split, no pruning) and of the candidate set itself; the label is joined on afterwards. Build: 26 minutes on this machine (67 s for the pool statistics, 46 s per bucket, every bucket in a fresh process peaking at 5.2 GB).

| group | features | what is in it |
|---|---|---|
| name | 51 | exact matches of the six name representations, acronym / variant / containment matches, token Jaccard-overlap-common, digit overlap and conflict, legal-form agreement, script, Levenshtein distance and similarity, Jaro-Winkler, token-set and partial ratios (rapidfuzz), compact and transliterated similarities, length and token-count differences, common prefix and suffix, name-flag indicators of both records |
| address | 50 | exact normalised / sorted address, token Jaccard-overlap-common, component and number overlaps and conflicts, digit signature, house number (exact, digits, conflict), street (exact, type, conflict, Jaro-Winkler), unit, city (exact, conflict, Jaro-Winkler), state, postal code, country, landmark and locality overlap, length differences, address-flag indicators of both records, character similarity of the whole address |
| missingness | 16 | missing name, address, house number, street, city, state, postal code, unit on each side |
| source | 1 | Source 2 versus Source 3 |
| tfidf | 4 | cosine of the name + address token TF-IDF (the retrieval space), of the address tokens, of the name tokens and of the name character 3-grams |
| context | 4 | log pool frequency (within the country) of each record's name core and normalised address |
| cross | 13 | name x address, name x city, name x postal, name x country, address x country, identifier (domain-style name) x name and x address, exact name x house and x city, name-cosine x address-cosine, min / max / mean of the name and address similarities |
| entity_relative | 18 | number of candidates of the entity (total and per source), rank and gap to the best of the name + address cosine, the name similarity and the address Jaccard, the margin between the entity's best and second-best cosine, how many candidates share the exact name or address |

**Identifiers.** The sources carry no phone, e-mail, website or registration fields, so the identifier block reduces to what the data offers: the pool record's name being a web domain (`p_domain_style`, `p_url_segment`), and its cross terms with the name and address similarities (`ident_x_name`, `ident_x_addr`). Postal codes exist in 0.007% of records, so `postal_eq` and `name_x_postal` are almost always null; `country_eq` is constant (candidates are generated within the country) and `country_us` carries the country instead.

Pairs by country: US 20,307,184 pairs with 906,950 positives (4.47%), India 19,926,208 pairs with 604,661 positives (3.03%).

## Single-feature discriminative power

AUC of each feature alone on a stratified sample of 1,600,000 pairs (half positives), oriented so that 0.5 is chance and 1 is perfect (nulls excluded); the null rate and the mean by label are over all pairs. Top 40:

| feature | group | AUC | null rate | mean (match) | mean (non-match) |
|---|---|---|---|---|---|
| cos_name_addr | tfidf | 0.977 | 0.0% | 0.784 | 0.255 |
| rank_cos_q | entity_relative | 0.974 | 0.0% | 4.387 | 68.401 |
| rank_cos_qs | entity_relative | 0.971 | 0.0% | 2.543 | 34.882 |
| gap_cos_q | entity_relative | 0.963 | 0.0% | 0.146 | 0.651 |
| gap_cos_qs | entity_relative | 0.955 | 0.0% | 0.087 | 0.562 |
| addr_token_set | address | 0.948 | 18.4% | 0.917 | 0.518 |
| name_x_addr | cross | 0.937 | 0.0% | 0.466 | 0.029 |
| addr_tok_overlap | address | 0.937 | 18.4% | 0.777 | 0.249 |
| addr_partial | address | 0.933 | 18.4% | 0.868 | 0.534 |
| cos_addr | tfidf | 0.931 | 0.0% | 0.758 | 0.168 |
| rank_addr_qs | entity_relative | 0.928 | 0.0% | 3.675 | 24.447 |
| rank_addr_q | entity_relative | 0.926 | 0.0% | 7.135 | 47.889 |
| cos_name_x_addr | cross | 0.926 | 0.0% | 0.632 | 0.020 |
| addr_tok_jaccard | address | 0.925 | 0.0% | 0.579 | 0.112 |
| sim_mean | cross | 0.902 | 18.4% | 0.744 | 0.384 |
| gap_addr_qs | entity_relative | 0.899 | 0.0% | 0.147 | 0.569 |
| street_jw | address | 0.892 | 49.0% | 0.955 | 0.479 |
| gap_addr_q | entity_relative | 0.891 | 0.0% | 0.237 | 0.676 |
| addr_tok_common | address | 0.885 | 0.0% | 5.459 | 1.368 |
| street_eq | address | 0.871 | 49.0% | 0.886 | 0.144 |
| street_conflict | address | 0.871 | 49.0% | 0.114 | 0.856 |
| addr_comp_jaccard | address | 0.869 | 0.0% | 0.448 | 0.093 |
| sim_min | cross | 0.866 | 18.4% | 0.584 | 0.190 |
| addr_lev_sim | address | 0.865 | 18.4% | 0.678 | 0.313 |
| addr_comp_overlap | address | 0.863 | 18.4% | 0.617 | 0.191 |
| addr_num_overlap | address | 0.855 | 29.1% | 0.839 | 0.167 |
| unit_eq | address | 0.855 | 94.8% | 0.866 | 0.159 |
| unit_conflict | address | 0.855 | 94.8% | 0.134 | 0.841 |
| house_digits_eq | address | 0.851 | 46.5% | 0.877 | 0.175 |
| landmark_jaccard | address | 0.848 | 93.2% | 0.710 | 0.017 |
| addr_comp_common | address | 0.845 | 0.0% | 2.344 | 0.565 |
| addr_num_jaccard | address | 0.841 | 1.7% | 0.689 | 0.093 |
| street_type_eq | address | 0.835 | 61.6% | 0.998 | 0.328 |
| addr_numsig_eq | address | 0.828 | 29.1% | 0.747 | 0.091 |
| name_x_city | cross | 0.821 | 18.7% | 0.631 | 0.096 |
| addr_num_common | address | 0.819 | 0.0% | 0.998 | 0.150 |
| locality_jaccard | address | 0.815 | 49.6% | 0.482 | 0.042 |
| house_eq | address | 0.811 | 46.5% | 0.784 | 0.164 |
| house_conflict | address | 0.811 | 46.5% | 0.216 | 0.836 |
| sim_max | cross | 0.776 | 18.4% | 0.904 | 0.578 |

53 features have an AUC below 0.55 on their own (country_us, cos_margin_q, addr_exact_norm, legal_2, script_2_nonlatin, script_eq, n_exact_addr_q, addr_pool_freq_1, street_missing_1, p_all_caps, name_exact_raw, house_missing_1, p_addr_reordered, legal_1, q_addr_unit, unit_missing_1, ident_x_addr, q_addr_reordered, name_variant_match, p_addr_indic, name_num_common, q_addr_landmark, p_addr_care_of, p_addr_null_token, ident_x_name...); they stay in the matrix because a tree model can use them in interactions and their cost is nil, but none of them will carry the matcher.

Best feature per group: tfidf `cos_name_addr` (0.977), entity_relative `rank_cos_q` (0.974), address `addr_token_set` (0.948), cross `name_x_addr` (0.937), name `name_common_prefix` (0.760), context `name_pool_freq_2` (0.603), missingness `street_missing_2` (0.591), source `src_s3` (0.500).

## Leakage check

No feature reads the label or the truth table: the pool statistics are fitted on the unlabelled pool, the dictionaries used by the preprocessing were learned without fold 0, and the entity-relative features only look at the candidate set. The label column is added by a join on (q_idx, p_idx) after every feature is computed, and the same code path runs on the test split, where no label exists.

