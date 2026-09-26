# Candidate generation (validation fold 0)

Queries: the 441,365 Source 1 entities of fold 0; pool: all 10,320,219 Source 2 and Source 3 records; truth: the fold's 1,527,650 true pairs. Every strategy is an operation on the preprocessed tables (`artifacts/preprocessed/fold0`, dictionaries learned without fold 0); every key includes the country string, nothing about a country is hard-coded. Reduction ratio = 1 - pairs / (queries x pool). Volume statistics count every query entity, including those with no candidate. Pairs are saved per strategy under `artifacts/candidates/fold0/strategies/`.

## 1. Every strategy

| strategy | pairs | pair recall | recall S2 | recall S3 | recall US | recall India | entities with all matches found | mean | median | p95 | max | reduction ratio | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. exact name core | 21,804,564 | 0.5646 | 0.5594 | 0.5695 | 0.5867 | 0.5316 | 0.1940 | 49.4 | 4 | 213 | 1,240 | 0.999995 | 8.7 |
| 6. exact name core, country key removed | 22,012,251 | 0.5646 | 0.5594 | 0.5695 | 0.5867 | 0.5316 | 0.1940 | 49.9 | 4 | 213 | 1,244 | 0.999995 | 8.6 |
| 1. exact name core, tokens sorted | 22,391,136 | 0.5898 | 0.5853 | 0.5940 | 0.6212 | 0.5429 | 0.2130 | 50.7 | 4 | 221 | 1,272 | 0.999995 | 8.3 |
| 1. exact name, spaces removed | 22,802,606 | 0.5964 | 0.5912 | 0.6012 | 0.6190 | 0.5626 | 0.2279 | 51.7 | 4 | 223 | 1,302 | 0.999995 | 8.5 |
| 1. exact transliterated name, sorted | 29,533,388 | 0.6613 | 0.6780 | 0.6457 | 0.6212 | 0.7213 | 0.2841 | 66.9 | 4 | 282 | 1,272 | 0.999994 | 8.9 |
| 1_exact_name_any_cap50 | 2,011,079 | 0.5168 | 0.5254 | 0.5088 | 0.5393 | 0.4831 | 0.2217 | 4.6 | 3 | 17 | 54 | 1.000000 | 33.7 |
| 1_exact_name_any_cap200 | 12,873,527 | 0.6297 | 0.6383 | 0.6217 | 0.5957 | 0.6806 | 0.2798 | 29.2 | 4 | 181 | 219 | 0.999997 | 38.3 |
| 2. name tokens, df <= 200, 2 rarest | 14,201,790 | 0.3162 | 0.3169 | 0.3157 | 0.3138 | 0.3199 | 0.2415 | 32.2 | 0 | 169 | 396 | 0.999997 | 21.2 |
| 2. name tokens, df <= 500, 2 rarest | 32,764,405 | 0.4060 | 0.4073 | 0.4048 | 0.3866 | 0.4349 | 0.3114 | 74.2 | 0 | 365 | 989 | 0.999993 | 20.8 |
| 2. name tokens, df <= 1000, 2 rarest | 58,363,025 | 0.4488 | 0.4502 | 0.4475 | 0.4194 | 0.4928 | 0.3456 | 132.2 | 0 | 702 | 1,962 | 0.999987 | 23.0 |
| 3/8. name char 3-gram TF-IDF, top 20 per source | 17,432,120 | 0.6101 | 0.6177 | 0.6031 | 0.6627 | 0.5315 | 0.3679 | 39.5 | 40 | 40 | 40 | 0.999996 | None |
| 4_addr_token_df100_rarest3 | 8,470,914 | 0.3975 | 0.4078 | 0.3878 | 0.2653 | 0.5950 | 0.2857 | 19.2 | 0 | 90 | 269 | 0.999998 | 45.1 |
| 4. address tokens, df <= 200, 3 rarest | 21,475,266 | 0.4878 | 0.4990 | 0.4773 | 0.3703 | 0.6633 | 0.3545 | 48.7 | 8 | 191 | 534 | 0.999995 | 46.6 |
| 4. address tokens, df <= 500, 3 rarest | 72,651,016 | 0.6224 | 0.6342 | 0.6113 | 0.5399 | 0.7456 | 0.4626 | 164.6 | 83 | 535 | 1,405 | 0.999984 | 57.5 |
| 5. state + sorted name | 2,228,451 | 0.5383 | 0.5416 | 0.5353 | 0.5900 | 0.4612 | 0.1752 | 5.0 | 3 | 19 | 286 | 1.000000 | 8.4 |
| 5. city + first name token | 18,681,064 | 0.5888 | 0.5454 | 0.6293 | 0.6922 | 0.4342 | 0.2854 | 42.3 | 5 | 269 | 1,724 | 0.999996 | 8.4 |
| 5_city_name_first_cap100 | 4,175,964 | 0.5486 | 0.5152 | 0.5797 | 0.6813 | 0.3502 | 0.2731 | 9.5 | 4 | 47 | 100 | 0.999999 | 8.3 |
| 5. city + house number | 13,349,399 | 0.4588 | 0.4349 | 0.4812 | 0.5792 | 0.2788 | 0.2140 | 30.2 | 3 | 111 | 1,763 | 0.999997 | 6.1 |
| 5. state + house number + street | 1,255,658 | 0.4506 | 0.4503 | 0.4508 | 0.6283 | 0.1848 | 0.2171 | 2.8 | 1 | 8 | 450 | 1.000000 | 5.3 |
| 5. postal code + first name token | 6 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 | 0 | 0 | 1 | 1.000000 | 0.0 |
| 5/6. raw last component + sorted name (unseen-country simulation) | 793,542 | 0.1971 | 0.3799 | 0.0261 | 0.2311 | 0.1462 | 0.0177 | 1.8 | 1 | 6 | 106 | 1.000000 | 8.2 |
| 8. sorted neighbourhood, sorted name, window 3 | 2,648,190 | 0.3694 | 0.3727 | 0.3663 | 0.3910 | 0.3371 | 0.1220 | 6.0 | 6 | 6 | 6 | 0.999999 | 7.2 |
| 8. sorted neighbourhood, compact name, window 3 | 2,648,190 | 0.4021 | 0.4056 | 0.3987 | 0.4194 | 0.3762 | 0.1628 | 6.0 | 6 | 6 | 6 | 0.999999 | 5.6 |
| 8. name acronym + city | 13,912,558 | 0.4979 | 0.4818 | 0.5128 | 0.5304 | 0.4492 | 0.1884 | 31.5 | 3 | 206 | 500 | 0.999997 | 7.9 |
| 9. address number signature | 22,451,967 | 0.3469 | 0.3293 | 0.3634 | 0.3545 | 0.3356 | 0.1877 | 50.9 | 1 | 331 | 500 | 0.999995 | 5.9 |
| 9. sorted neighbourhood, sorted address tokens, window 3 | 2,648,190 | 0.4381 | 0.4498 | 0.4271 | 0.4157 | 0.4716 | 0.1553 | 6.0 | 6 | 6 | 6 | 0.999999 | 5.7 |
| 9. address token TF-IDF, top 20 per source | 17,633,587 | 0.8473 | 0.8629 | 0.8327 | 0.8522 | 0.8399 | 0.6446 | 40.0 | 40 | 40 | 40 | 0.999996 | 287.6 |
| 9. name + address token TF-IDF, top 20 per source | 17,651,339 | 0.9601 | 0.9662 | 0.9544 | 0.9739 | 0.9395 | 0.8872 | 40.0 | 40 | 40 | 40 | 0.999996 | 539.3 |

Recall of the top-k retrievals by rank (pairs found within the first k neighbours per source):

| retrieval | @1 | @3 | @5 | @10 | @20 |
|---|---|---|---|---|---|
| 3/8. name char 3-gram TF-IDF, top 20 per source | 0.2124 | 0.4244 | 0.4947 | 0.5606 | 0.6101 |
| 9. address token TF-IDF, top 20 per source | 0.3616 | 0.6966 | 0.7700 | 0.8176 | 0.8473 |
| 9. name + address token TF-IDF, top 20 per source | 0.4365 | 0.8485 | 0.9221 | 0.9465 | 0.9601 |

## 2. Unions

| union | members | pairs | pair recall | recall S2 | recall S3 | recall US | recall India | entities with all matches found | mean | median | p95 | p99 | max | reduction ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_exact_name_any | 4 | 30,531,428 | 0.6931 | 0.7098 | 0.6774 | 0.6534 | 0.7523 | 0.3251 | 69.2 | 5 | 289 | 849 | 1,334 | 0.999993 |
| U_exact_keys | 10 | 95,583,961 | 0.9441 | 0.9432 | 0.9450 | 0.9534 | 0.9304 | 0.8252 | 216.6 | 79 | 862 | 1357 | 3,334 | 0.999979 |
| U_retrieval_only | 3 | 43,035,242 | 0.9725 | 0.9763 | 0.9689 | 0.9827 | 0.9572 | 0.9201 | 97.5 | 98 | 111 | 115 | 120 | 0.999991 |
| U_bounded | 16 | 134,947,354 | 0.9923 | 0.9930 | 0.9916 | 0.9928 | 0.9915 | 0.9730 | 305.8 | 168 | 939 | 1438 | 3,425 | 0.999970 |
| U_bounded_plus_tokens_small | 18 | 149,629,353 | 0.9926 | 0.9933 | 0.9919 | 0.9929 | 0.9920 | 0.9741 | 339.0 | 234 | 957 | 1465 | 3,477 | 0.999967 |
| U_bounded_plus_tokens_mid | 18 | 175,381,192 | 0.9929 | 0.9936 | 0.9922 | 0.9931 | 0.9925 | 0.9750 | 397.4 | 305 | 1007 | 1539 | 3,477 | 0.999961 |
| U_all_selected | 18 | 239,702,821 | 0.9932 | 0.9939 | 0.9926 | 0.9933 | 0.9930 | 0.9762 | 543.1 | 446 | 1312 | 1859 | 3,945 | 0.999947 |
| U_final | 7 | 35,911,531 | 0.9852 | 0.9859 | 0.9845 | 0.9883 | 0.9806 | 0.9512 | 81.4 | 51 | 233 | 279 | 689 | 0.999992 |

Members: U_exact_name_any = the four exact name representations; U_exact_keys adds the state/city/postal keys, the number signature and the acronym key; U_retrieval_only = the three top-20 retrievals; U_bounded = every strategy whose volume per entity is bounded by construction (exact keys, capped blocks, sorted neighbourhoods, top-20 retrievals); the token unions add the rare-token blocks at the given df cap; U_all_selected is the final set.

## 2b. Choosing the final set

Recall of every union below is exact from the coverage table; volumes are exact per-bucket unions. Retrieval recall by rank (from the scored outputs):

| retrieval | @1 | @3 | @5 | @10 | @15 | @20 |
|---|---|---|---|---|---|---|
| 3/8. name char 3-gram TF-IDF, top 20 per source | 0.2124 | 0.4244 | 0.4947 | 0.5606 | 0.5908 | 0.6101 |
| 9. address token TF-IDF, top 20 per source | 0.3616 | 0.6966 | 0.7700 | 0.8176 | 0.8366 | 0.8473 |
| 9. name + address token TF-IDF, top 20 per source | 0.4365 | 0.8485 | 0.9221 | 0.9465 | 0.9551 | 0.9601 |

Greedy forward selection (each step adds the strategy with the best recall gain per candidate it costs on its own):

| step | strategy added | union recall | gain | own mean candidates | union mean | union p95 | union max | union pairs |
|---|---|---|---|---|---|---|---|---|
| 1 | 5. state + house number + street | 0.45055 | +0.45055 | 2.8 | 2.8 | 8 | 450 | 1,255,658 |
| 2 | 1_exact_name_any_cap50 | 0.72974 | +0.27919 | 4.6 | 6.6 | 22 | 460 | 2,903,167 |
| 3 | 9. sorted neighbourhood, sorted address tokens, window 3 | 0.80918 | +0.07944 | 6.0 | 11.2 | 26 | 466 | 4,936,459 |
| 4 | 5_city_name_first_cap100 | 0.88521 | +0.07603 | 9.5 | 18.8 | 60 | 492 | 8,312,705 |
| 5 | 5. state + sorted name | 0.89945 | +0.01424 | 5.0 | 21.6 | 68 | 629 | 9,528,498 |
| 6 | 8. sorted neighbourhood, compact name, window 3 | 0.91221 | +0.01276 | 6.0 | 25.2 | 72 | 634 | 11,137,109 |
| 7 | 9. name + address token TF-IDF, top 20 per source | 0.97989 | +0.06768 | 40.0 | 59.2 | 106 | 669 | 26,150,980 |
| 8 | 1_exact_name_any_cap200 | 0.98520 | +0.00531 | 29.2 | 81.4 | 233 | 689 | 35,911,531 |
| 9 | 8. name acronym + city | 0.98697 | +0.00177 | 31.5 | 110.5 | 348 | 1,132 | 48,754,770 |

Named candidate sets:

| set | strategies | recall | recall US | recall India | recall S2 | recall S3 | entities with all matches | mean | median | p95 | p99 | max | pairs | entities without candidates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retrieval_only | 3 | 0.97246 | 0.9827 | 0.9572 | 0.9763 | 0.9689 | 0.9201 | 97.5 | 98 | 111 | 115 | 120 | 43,035,242 | 0 |
| retrieval_plus_cheap_keys | 8 | 0.98099 | 0.9890 | 0.9691 | 0.9824 | 0.9797 | 0.9410 | 110.5 | 109 | 134 | 171 | 727 | 48,757,787 | 0 |
| retrieval_cheap_exactname | 12 | 0.98931 | 0.9908 | 0.9872 | 0.9912 | 0.9875 | 0.9635 | 166.3 | 110 | 383 | 925 | 1,454 | 73,409,374 | 0 |
| retrieval_cheap_exactname_cityfirst | 13 | 0.99124 | 0.9919 | 0.9902 | 0.9922 | 0.9904 | 0.9696 | 205.0 | 115 | 730 | 997 | 2,188 | 90,493,528 | 0 |
| retrieval_cheap_exactname_acronym_cityhouse | 14 | 0.99084 | 0.9916 | 0.9897 | 0.9921 | 0.9896 | 0.9684 | 221.9 | 118 | 746 | 1110 | 2,548 | 97,955,588 | 0 |
| retrieval_cheap_exactname_addrtok200 | 13 | 0.98969 | 0.9910 | 0.9877 | 0.9915 | 0.9880 | 0.9648 | 202.7 | 126 | 481 | 953 | 1,602 | 89,457,807 | 0 |
| retrieval_cheap_exactname_nametok500 | 13 | 0.98990 | 0.9909 | 0.9884 | 0.9916 | 0.9883 | 0.9655 | 227.7 | 165 | 559 | 928 | 1,454 | 100,496,668 | 0 |
| core7 | 7 | 0.98108 | 0.9863 | 0.9732 | 0.9828 | 0.9794 | 0.9383 | 84.3 | 55 | 264 | 470 | 984 | 37,196,416 | 0 |
| core7_exactcap50 | 8 | 0.98273 | 0.9874 | 0.9758 | 0.9843 | 0.9813 | 0.9436 | 85.4 | 56 | 265 | 471 | 984 | 37,697,956 | 0 |
| core7_exactcap50_addrretr | 9 | 0.98426 | 0.9892 | 0.9769 | 0.9858 | 0.9828 | 0.9486 | 108.0 | 82 | 284 | 489 | 1,009 | 47,669,565 | 0 |
| core7_exactcap200_addrretr | 9 | 0.98771 | 0.9896 | 0.9849 | 0.9890 | 0.9865 | 0.9587 | 129.6 | 84 | 350 | 553 | 1,149 | 57,193,342 | 0 |
| core7_exactcap50_addrretr_cityfirst100 | 10 | 0.98506 | 0.9897 | 0.9781 | 0.9862 | 0.9840 | 0.9512 | 114.5 | 85 | 295 | 495 | 1,014 | 50,527,049 | 0 |
| core7_exactcap50_addrretr_ngram | 10 | 0.98558 | 0.9902 | 0.9788 | 0.9870 | 0.9843 | 0.9529 | 140.1 | 115 | 317 | 522 | 1,039 | 61,815,450 | 0 |
| bounded | 16 | 0.99228 | 0.9928 | 0.9915 | 0.9930 | 0.9916 | 0.9730 | 305.8 | 168 | 939 | 1438 | 3,425 | 134,947,354 | 0 |
| greedy_selection | 9 | 0.98697 | 0.9884 | 0.9849 | 0.9879 | 0.9861 | 0.9560 | 110.5 | 59 | 348 | 540 | 1,132 | 48,754,770 | 0 |

## 3. Every true pair of the fold against the final set

Of the 1,527,650 true pairs, **1,505,042 are in the final candidate set and 22,608 are not** (recall 0.98520); 95,842 pairs are produced by exactly one strategy. The coverage table (`artifacts/candidates/fold0/coverage.parquet`) has one row per true pair with a flag per strategy.

| strategy | true pairs only this strategy finds | recall of the final set without it |
|---|---|---|
| 9. name + address token TF-IDF, top 20 per source | 75,246 | 0.93594 |
| 1_exact_name_any_cap200 | 11,809 | 0.97747 |
| 5. state + house number + street | 1,944 | 0.98393 |
| 5. state + sorted name | 851 | 0.98464 |
| 5_city_name_first_cap100 | 2,596 | 0.98350 |
| 8. sorted neighbourhood, compact name, window 3 | 2,424 | 0.98361 |
| 9. sorted neighbourhood, sorted address tokens, window 3 | 972 | 0.98456 |

## 4. Anatomy of the misses

22,608 true pairs are missed. By country: US 10,740, India 11,868; by source: S2 10,409, S3 12,199.

| category (name fuzzy ratio on transliterated cores; address token Jaccard) | pairs |
|---|---|
| similar name and similar address | 8,252 |
| address missing on a side | 7,286 |
| similar name, different address | 3,542 |
| different name, similar address | 2,888 |
| different name and different address | 640 |

Name ratio quartiles of the misses: 62, 80, 94; address Jaccard quartiles: 0.00, 0.19, 0.33; script of the missed pool name: latin 19,104, devanagari 1,954, kannada 319, telugu 231, mixed 225, bengali 215.

Examples (Source 1 record, then the missed true record):

- `Pinnacle & Brothers Private Limited` | `G-4, Krishna Regency, B-87, Janpath, Shyam Nagar, Jaipur, Rajasthan`  ->  `Dr pinnaclebrothers.com` | `H.no B3/760 G-4, Shyam Nagar, Jaipur, RJ`  (name ratio 82, address Jaccard 0.31; similar name and similar address)
- `Orelee Jernigan Superior Communities` | `8424 Shears Court, Laurel, MD`  ->  `Jaxnex` | `424 SHEARS CT, PMB 5433, LAUUREL, MD`  (name ratio 19, address Jaccard 0.25; different name, similar address)
- `Modern Construction Private Limited` | `B-Wing, 703, G Block, Floor No.7, One Bkc, Bandra Kurla Complex, Bandra East, Mumbai, Mumbai City, Maharashtra`  ->  `modernconstructionprivate.com` | `MH, B-wing, Mumbai, Mumbai City`  (name ratio 82, address Jaccard 0.18; similar name, different address)
- `Office of Buildings` | `123 Maple Avenue, Versailles, KY`  ->  `Office Office of` | `None`  (name ratio 64, address Jaccard 0.00; address missing on a side)
- `Atlantic Opportunities LLC` | `402 Third Street, St Charles, MO`  ->  `Atlantic Opbortugities LLC | www.atlantico.com` | `None`  (name ratio 91, address Jaccard 0.00; address missing on a side)
- `Clean Rapid Drugs Group` | `775 Melrose Road, Lockesburg, AR`  ->  `Clean Rpiad Drugs Group` | `None`  (name ratio 96, address Jaccard 0.00; address missing on a side)
- `Rev It Up Tavern Inc.` | `301 Old Brook Road, Bedford County, VA`  ->  `Fluxwexvera` | `Old Brook Rd, Bedford County, Virginia`  (name ratio 30, address Jaccard 0.44; different name, similar address)
- `East Global Limited` | `Uttar Pradesh, Hapur, Ghaziabad, B-503, Sanjay Vihar, Avas Vikas Colony, Meerut Road`  ->  `EAST GLOABL  LIMITED` | `None`  (name ratio 91, address Jaccard 0.00; address missing on a side)
- `Ss Exports Pvt Ltd` | `C/O Samadhan Bhonde, Buldana, Maharashtra, 336, At Post Mhasla (Bk), Buldhana`  ->  `एसएस एक्सपोर्ट्स प्रा. लि.` | `DOOR NO D/714 C/O SAMADHAN BHONDE, BULDHANA, BULDANA, महाराष्ट्र`  (name ratio 100, address Jaccard 0.21; similar name and similar address)
- `Aarvin Innovations Private Limited` | `S-3, G/F, Greater Kailash-1, New Delhi, South Delhi, Delhi`  ->  `APINNOVATIONS.COM` | `S-B3/3, SOUTH DELHI, NEW DELHI, Delhi`  (name ratio 77, address Jaccard 0.38; similar name and similar address)
- `Sukriti Infosystems Pvt Ltd` | `B-306, Iiird Floor, Rg City Centre Motia Khan, Pahar Ganj, Delhi, Central Delhi, Delhi`  ->  `Sukriti Pvt Ltd Service` | `H.no 401 B-306, Central Delhi, Delhi, DL`  (name ratio 59, address Jaccard 0.19; similar name, different address)
- `Prime Software Private Limited` | `Door No: 6-4-19-10, Flat No: 302 & 303 Sree Pearl Apartment, 3Rd Cross, Maruthi, Nagar, Anantapur, Andhra Pradesh`  ->  `ప్రైమ్ సాఫ్ట్‌వేర్ ప్రైవేట్ లిమిటెడ్` | `Door No: 6-4-19-1, Anantapururamu Dist, Anantapur, ఆంధ్రప్రదేశ్`  (name ratio 100, address Jaccard 0.15; similar name, different address)
- `Healthcare Prabodh Institute` | `174 C R Avenue, Kolkata, Calcutta, West Bengal`  ->  `hpinstitute.com` | `Kolkata, 1-74 C R Avenue, Calcutta, WB`  (name ratio 56, address Jaccard 0.50; similar name and similar address)
- `Jayanti Clinic Pvt Ltd` | `C/O Dr. Rahul Prasad Polt No.D/1411 Utto Rayon P.O.Matigara Dist.Darjeeling, Siliguri, Darjeeling, West Bengal`  ->  `Jayanti Pvt Ltd Sérvices` | `PLOT 442 C/O DR. RAHUL PRASAD POLT NO.D/1411 UTTO RAYON P.O.MATIGARA DIST.DARJEELING, SILIGURI, West Bengal`  (name ratio 67, address Jaccard 0.21; similar name and similar address)
- `Gonzales Super` | `6536 R Avenue, Holstein, IA`  ->  `Gonzales Center` | `6536b R Avenue, Silver Township, Iowa`  (name ratio 76, address Jaccard 0.22; similar name and similar address)
- `Piedmont Art` | `504 Crawford Avenue, Villa Grove, IL`  ->  `Evonyla - 1407443343` | `504-A CRAWFORD AVE, IL, VLLA GROVE`  (name ratio 42, address Jaccard 0.33; different name, similar address)

## 5. Strategy 10: embedding retrieval probe

Model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on cuda. For each missed pair and a control sample of found pairs, the cosine of the true pair is compared with the cosines to a random same-country sample of pool names; a pair counts as retrievable at k when fewer than k * (pool / sample) sample names score above the true match.

| set | text | country | pairs | median true cosine | retrievable top-20 | top-50 | top-200 | names embedded per second |
|---|---|---|---|---|---|---|---|---|
| missed | raw name | India | 4,399 | 0.693 | 4.73% | 8.34% | 15.73% | 3,043 |
| missed | raw name | US | 6,309 | 0.710 | 3.65% | 8.08% | 15.66% | 3,072 |
| missed | transliterated core | India | 4,399 | 0.620 | 2.30% | 3.16% | 6.75% | 4,363 |
| missed | transliterated core | US | 6,309 | 0.712 | 2.20% | 5.01% | 9.95% | 3,996 |
| found_control | raw name | India | 2,051 | 0.831 | 40.52% | 46.47% | 54.07% | 3,005 |
| found_control | raw name | US | 2,949 | 0.872 | 47.68% | 60.90% | 69.04% | 3,029 |
| found_control | transliterated core | India | 2,051 | 1.000 | 64.31% | 67.43% | 76.50% | 4,303 |
| found_control | transliterated core | US | 2,949 | 1.000 | 59.44% | 69.65% | 75.75% | 3,975 |

Full-scale cost: 12,527,040 training names at the measured rate = 68.6 minutes of GPU embedding per split, before the search.

## 6. The competition file

`artifacts/candidates/fold0\candidate_pairs.tsv`: 441,365 rows (one per fold-0 Source 1 entity, in the Source 1 file order), 35,911,531 candidate ids, 0 empty rows, 469 MB. Streaming check against the specification (header, LF endings, two columns, S2-/S3- ids only, no duplicates within a row, every id in the pool): passed.

## 7. Design decisions and the evidence behind them

**Rerun after a normalisation fix.** The first measurement showed misses like `Dental Care Inc | 7719 Yellow Pine Drive, Houston` against `dentalcare.com | ##7719 YELLOW PINE DR, HOUSTN`: the `#` injected before numbers in 381,737 Source 2 and 543,041 Source 3 records (99,123 and 99,328 of them doubled) survived in the address tokens, so `##7719` never matched `7719` in any token-based strategy, and a doubled `#` also hid the house number from the parser. `addresses.py` now strips it; every table and strategy in this report was recomputed afterwards. The name + address retrieval gained 0.35 points of recall (0.9566 to 0.9601), the final set 0.10 (0.9842 to 0.9852), and the deterministic baseline of `rule_baseline.md` 1.2 points (0.600 to 0.612).

**Strategy verdicts (fold 0, 441,365 queries, 10,320,219-record pool).**

1. *Exact normalised name*: the four representations together reach 69.3% recall at 69 candidates per entity, but the volume sits in common-name blocks (p99 849, max 1,334). Capped at 200 pool records per key the same join keeps 63.0% at 29 per entity and is the second most valuable member of the final set; capped at 50 it keeps 51.7% at 4.6.
2. *Name token blocking*: two rarest transliterated tokens with df <= 200 / 500 / 1000 give 31.6 / 40.6 / 44.9% recall at 32 / 74 / 132 candidates: the tokens that are rare enough to block on are the ones typos change, so it is dominated by the retrievals and by the exact keys and is not in the final set.
3. *Character n-gram retrieval* (TF-IDF 3-grams on transliterated name cores, n-grams in > 1% of names dropped, cosine top 20 per source): 61.0% recall at 40 per entity, 66% US and 53% India; removing it from the bounded union costs 0.09 points, so it is not in the final set. It is the most expensive strategy (20 minutes for 441k queries on 12 threads).
4. *Address token blocking*: three rarest tokens with df <= 100 / 200 / 500 give 39.8 / 48.8 / 62.2% at 19 / 49 / 165 candidates; India benefits most (59% at df 100) because Indian addresses carry rare locality tokens, but the address retrieval covers the same pairs with a bounded volume.
5. *City / state / postal blocking*: postal codes exist in 0.007% of records (6 pairs), so the keys are state + sorted name (53.8% at 5.0 per entity), city + first name token (58.9% at 42; capped at 100 records 54.9% at 9.5), city + house number (45.9% at 30) and state + house number + street (45.1% at 2.8, the most precise block: 63% recall in the US, 18% in India where street names are rarer).
6. *Country-aware blocking without hard-coding*: every key is hashed together with the country string; dropping it from the exact-name join adds 207,687 cross-country pairs and no recall (all 7,638,365 true pairs share the country string). The only country-dependent vocabulary is the learned state table: replacing the parsed state by the raw last address component (what an unseen country falls back to) drops the state + name key from 53.8% to 19.7% recall, so for France the state-keyed block will be weak while the retrieval, the sorted neighbourhoods and the exact-name blocks are vocabulary-free.
7. *Strong identifier blocking*: no identifiers exist (`rule_baseline.md`, section 1).
8. *Approximate name retrieval*: sorted neighbourhoods (window 3, six candidates per entity) on the sorted name give 36.9% and on the compact name 40.2%; the acronym + city key 49.8% at 31.5. The compact-name neighbourhood is in the final set (2,424 true pairs only it finds), the acronym key is not (+0.18 points for +29 candidates).
9. *Approximate address retrieval*: the number signature (all digit runs, at least three digits, blocks over 500 dropped) gives 34.7% at 51; the sorted-address neighbourhood 43.8% at 6; the address-token TF-IDF top 20 per source 84.7% at 40 (recall@5 75%, @10 80%); the **name + address token TF-IDF top 20 per source 96.0% at 40** (recall@1 43%, @3 84%, @5 92%, @10 94%, @20 96%; 97.4% US, 94.0% India). The last one is the workhorse: 75,246 true pairs are found by it alone, and removing it drops the final set from 98.5% to 93.6%.
10. *Embedding retrieval*: not justified (section 5). A multilingual MiniLM ranks the true match in the top 20 of the full pool for 3.7% (US) and 4.7% (India) of the pairs the union misses, and on a control sample of found pairs only 41-48% (raw names) or 59-64% (transliterated cores), against 96% for the token retrieval; it would cost about 70 GPU-minutes per split at 3,000 names per second.

**The final set.** A greedy forward selection by recall gained per candidate added builds the frontier: seven strategies reach 97.99% recall at 59 candidates per entity, the capped exact-name blocks lift it to 98.52% at 81, and everything after costs 30 or more candidates per 0.2 points (acronym + city: 98.70% at 110; the address retrieval: 98.77% at 130; the whole bounded union: 99.23% at 306). The final set is the union of the eight, with the cap-50 block dropped because it is a subset of the cap-200 one:

| member | pairs | recall alone | true pairs only it finds | final recall without it |
|---|---|---|---|---|
| name + address token TF-IDF, top 20 per source | 17,651,339 | 0.9601 | 75,246 | 0.9359 |
| exact name (any representation), blocks <= 200 | 12,873,527 | 0.6297 | 11,809 | 0.9775 |
| state + house number + street | 1,255,658 | 0.4506 | 1,944 | 0.9839 |
| state + sorted name | 2,228,451 | 0.5383 | 851 | 0.9846 |
| city + first name token, blocks <= 100 | 4,175,964 | 0.5486 | 2,596 | 0.9835 |
| sorted neighbourhood on the compact name, window 3 | 2,648,190 | 0.4021 | 2,424 | 0.9836 |
| sorted neighbourhood on the sorted address tokens, window 3 | 2,648,190 | 0.4381 | 972 | 0.9846 |

Union: **35,911,531 pairs, pair recall 0.9852** (Source 2 0.9859, Source 3 0.9845; US 0.9883, India 0.9806), 95.1% of non-singleton entities with every match present, 81.4 candidates per entity on average (median 51, p95 233, p99 279, max 689), no entity without a candidate, reduction ratio 0.999992. Of the 1,527,650 true pairs of the fold, 1,505,042 are in the set and 22,608 are not; the per-pair flags are in `artifacts/candidates/fold0/coverage.parquet` (column `in_final`).

**The misses.** 22,608 pairs (1.48%): 8,252 with a similar name and a similar address that still fell outside every top-20 list and key (`Pinnacle & Brothers Private Limited | G-4, Krishna Regency, B-87, Janpath, Shyam Nagar, Jaipur` against `Dr pinnaclebrothers.com | H.no B3/760 G-4, Shyam Nagar, Jaipur`: the name became a domain, the address lost its landmark tokens); 7,286 where the pool record has no address at all and the name carries a typo (`Clean Rapid Drugs Group` against `Clean Rpiad Drugs Group`), which only a name-only fuzzy retrieval reaches (the n-gram retrieval would add 0.5 points for 40 more candidates per entity); 3,542 similar names with a different address; 2,888 different names with a similar address (`Orelee Jernigan Superior Communities` against `Jaxnex` at the same street); 640 with neither. India holds 11,868 of them and the US 10,740; 19,104 of the missed pool names are Latin, 1,954 Devanagari.

## 8. The competition file and the test-time procedure

`artifacts/candidates/fold0/candidate_pairs.tsv` holds the fold-0 set in the specification's format (header `source1_entity_id<TAB>candidate_entity_ids`, one row per Source 1 entity in the Source 1 file order, ids sorted and comma-separated, empty when none, LF endings, no quoting), 441,365 rows, 35,911,531 ids, 469 MB, checked line by line (`candidates.validate_candidate_file`) and read back with the validator-mirroring reader. This set is what the matching model will score for fold 0, so it is the file the specification asks for: the final candidate list just before inference.

The test file is produced by the same code path: `src/generate_candidates.py --pre-dir <preprocessed test tables> --prefix test --out output/candidate_pairs.tsv` runs the seven strategies on the tables that `preprocess.py` builds with the all-training dictionaries (no labels are used anywhere in candidate generation). Run on the fold-0 entities it reproduced the measured file byte for byte (`cmp` in `eda/results/chain_log.txt`). Cost on this machine for the 441k-entity fold: 25 minutes, of which the name + address retrieval takes 9 and the exact-name blocks 1; the retrieval scales with the number of queries, so the 1,732,544 test entities need about 35 minutes for it and roughly an hour end to end, and the test file will hold about 140 million ids (1.8 GB). The organisers' `validate_submission.py` loads every id into Python sets, so the submission should be validated with `--check-ids` off first and on a machine with more RAM than this one if the full check is wanted.

**Update after the miss analysis (`candidate_misses.md`).** Diagnosing the 22,608 misses led to six additional strategies (exact name + house number, exact name + city with blocks <= 100, compact name + state, house number + street + city with blocks <= 200, a name 3-gram retrieval restricted to the address-less pool records at top 5, and the address-token retrieval limited to its top 5). The union of the thirteen is now the final set: 40,233,392 pairs, recall 0.9895 (US 0.9909, India 0.9875), 91.2 candidates per entity (median 62, p95 241, max 696), and `candidate_pairs.tsv` was regenerated from it; the seven-strategy file above is kept as `candidate_pairs_v1.tsv` and `generate_candidates.py --v1` reproduces it. The numbers in this section and the next describe the seven-strategy set; the extended set is documented in `candidate_misses.md`, sections 3-7.

## 9. What the matcher inherits

The labelled version of the fold-0 set (`artifacts/candidates/fold0/final_pairs.parquet` joined with the fold's truth) has 35.9 million pairs of which 1,505,042 are positives: a 4.2% positive rate, 81 candidates per entity, and the exact-name, state/city and neighbourhood keys plus the retrieval rank and score available as features for every pair. The recall ceiling for the matcher on this fold is 0.9852 pair recall; with the problem-structure estimate of about one point of macro F0.5 per three points of blocking miss, the ceiling costs roughly half a point against a perfect candidate set, which is the price of keeping 81 rather than 306 candidates per entity.

