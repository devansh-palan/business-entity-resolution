# Deterministic high-precision baseline (validation fold 0)

Every rule is an exact equality join between Source 1 and the whole Source 2/3 pool (10,320,219 records) on `country` plus the listed keys of the preprocessed tables (`artifacts/preprocessed/fold0`, built with the transliteration dictionary and address vocabulary learned without fold 0). Scores are entity-level macro F0.5 over the 441,365 Source 1 entities of fold 0, computed by the verified evaluator (`src/evaluation.py`); singletons score 1.0 only when predicted empty, so the empty prediction scores **0.0559** and every number below has to beat that.

Two refinements are measured for every rule:

- **one-to-one (1to1)**: the ground truth maps each Source 2/3 record to at most one Source 1 entity (7,638,365 matched ids, all distinct), so when a rule links one pool record to several Source 1 entities all but at most one link are false merges; the refinement drops those records (uniqueness is counted against all 2,206,821 Source 1 entities, exactly as it would be against the test Source 1).
- **cap 11**: no training entity has more than 11 matches (S2 <= 5, S3 <= 6), so an entity linked to 12 or more records is discarded.

## 1. Rules 2-5: no identifiers exist in the data

The files have four columns (`entity_id`, `business_name`, `business_address`, `country`); the ids are opaque (0.0% of matched pairs share a numeric id, Spearman correlation 0.0001, see `s2_hidden_ids.json`). Scanning every name and address for identifier-like content:

| source | rows | e-mail | phone-like (>=7 digits) | phone-like (>=10 digits) | URL / domain | CIN-like | GSTIN-like | EIN-like | digit run >= 9 | names with phone tail stripped | domain-style names |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 2,206,821 | 1 | 5 name / 63,250 addr | 2 name / 1,007 addr | 0 name / 59 addr | 0 | 0 | 3 | 191 | 0 | 0 |
| S2 | 5,034,616 | 1 | 15,964 name / 177,496 addr | 15,817 name / 16,106 addr | 201,266 name / 110 addr | 3 | 0 | 6 | 16,235 | 15,468 | 191,511 |
| S3 | 5,285,603 | 1 | 16,422 name / 184,701 addr | 16,267 name / 16,104 addr | 210,922 name / 108 addr | 0 | 0 | 5 | 16,630 | 15,908 | 200,822 |

## 2. Key coverage after preprocessing

| table | name_core | addr_norm | house_number | street | city | state | state recognised | postal_code |
|---|---|---|---|---|---|---|---|---|
| S1 | 100.00% | 100.00% | 86.15% | 81.88% | 99.92% | 99.99% | 94.50% | 0.01% |
| pool | 100.00% | 96.66% | 79.18% | 78.05% | 96.35% | 96.64% | 91.32% | 0.00% |

## 3. Rules as written (raw join)

| rule | definition | pairs | pair precision | pair recall | macro F0.5 | gain vs empty | false merges (pairs) | missed matches (pairs) | singleton FPs | entities affected | entities with a false merge | entities fully correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | exact normalised name + exact normalised address | 75,364 | 0.9997 | 0.0493 | 0.1508 | +0.0949 | 25 | 1,452,311 | 3 | 62,779 | 23 | 26,082 |
| R1s | sorted name tokens + sorted address tokens | 108,547 | 0.9997 | 0.0710 | 0.1906 | +0.1347 | 32 | 1,419,135 | 4 | 88,003 | 30 | 26,823 |
| R6 | name + city + postal code | 2 | 0.5000 | 0.0000 | 0.0559 | +0.0000 | 1 | 1,527,649 | 0 | 2 | 1 | 24,684 |
| R6c | name + city + state (postal code unavailable) | 904,182 | 0.6790 | 0.4019 | 0.5578 | +0.5019 | 290,248 | 913,716 | 7,622 | 334,844 | 101,869 | 51,021 |
| R7 | name + house number + street | 404,201 | 0.9981 | 0.2641 | 0.4484 | +0.3924 | 786 | 1,124,235 | 66 | 218,962 | 784 | 47,604 |
| R7s | sorted name + house number + street | 425,383 | 0.9981 | 0.2779 | 0.4630 | +0.4070 | 792 | 1,103,059 | 67 | 225,012 | 790 | 49,934 |
| R8 | very strong: name + house number, street/city agree, no state conflict | 404,539 | 0.9972 | 0.2641 | 0.4557 | +0.3998 | 1,141 | 1,124,252 | 103 | 226,410 | 1,071 | 46,639 |
| R8u | R8 with any name representation (core, sorted, compact, transliterated) | 482,700 | 0.9970 | 0.3150 | 0.5111 | +0.4552 | 1,459 | 1,046,409 | 128 | 250,607 | 1,334 | 56,610 |
| D1 | exact name only | 21,804,564 | 0.0396 | 0.5646 | 0.4539 | +0.3980 | 20,941,999 | 665,085 | 16,130 | 416,457 | 266,550 | 42,562 |
| D2 | name + state | 2,160,287 | 0.3639 | 0.5146 | 0.5778 | +0.5218 | 1,374,180 | 741,543 | 11,926 | 394,434 | 180,543 | 54,304 |
| D3 | name + house number | 512,704 | 0.9792 | 0.3286 | 0.5294 | +0.4735 | 10,684 | 1,025,630 | 446 | 264,447 | 6,696 | 55,052 |
| D4 | name + street | 650,598 | 0.8828 | 0.3760 | 0.5320 | +0.4761 | 76,253 | 953,305 | 5,919 | 291,440 | 71,956 | 51,649 |
| D5 | name + city | 942,821 | 0.6692 | 0.4130 | 0.5718 | +0.5159 | 311,873 | 896,702 | 7,908 | 346,091 | 106,434 | 51,294 |
| D6 | exact address only | 239,120 | 0.5905 | 0.0924 | 0.2166 | +0.1607 | 97,932 | 1,386,462 | 815 | 114,934 | 14,601 | 26,897 |
| D7 | house number + street + city (no name) | 912,811 | 0.6247 | 0.3733 | 0.4879 | +0.4320 | 342,542 | 957,381 | 2,880 | 256,228 | 47,831 | 75,941 |
| D8 | any name representation + house number, no state conflict | 606,903 | 0.9903 | 0.3935 | 0.5926 | +0.5366 | 5,854 | 926,601 | 306 | 286,990 | 3,785 | 71,458 |

## 4. Rules with the one-to-one refinement

| rule | definition | pairs | pair precision | pair recall | macro F0.5 | gain vs empty | false merges (pairs) | missed matches (pairs) | singleton FPs | entities affected | entities with a false merge | entities fully correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | exact normalised name + exact normalised address | 75,358 | 0.9997 | 0.0493 | 0.1508 | +0.0949 | 22 | 1,452,314 | 3 | 62,777 | 22 | 26,081 |
| R1s | sorted name tokens + sorted address tokens | 108,541 | 0.9997 | 0.0710 | 0.1906 | +0.1347 | 29 | 1,419,138 | 4 | 88,001 | 29 | 26,822 |
| R6 | name + city + postal code | 2 | 0.5000 | 0.0000 | 0.0559 | +0.0000 | 1 | 1,527,649 | 0 | 2 | 1 | 24,684 |
| R6c | name + city + state (postal code unavailable) | 651,100 | 0.8853 | 0.3773 | 0.5424 | +0.4865 | 74,668 | 951,218 | 5,924 | 301,648 | 71,522 | 52,568 |
| R7 | name + house number + street | 404,193 | 0.9981 | 0.2641 | 0.4484 | +0.3924 | 782 | 1,124,239 | 65 | 218,956 | 780 | 47,604 |
| R7s | sorted name + house number + street | 425,375 | 0.9981 | 0.2779 | 0.4629 | +0.4070 | 788 | 1,103,063 | 66 | 225,006 | 786 | 49,934 |
| R8 | very strong: name + house number, street/city agree, no state conflict | 404,346 | 0.9974 | 0.2640 | 0.4556 | +0.3997 | 1,034 | 1,124,338 | 97 | 226,337 | 1,019 | 46,646 |
| R8u | R8 with any name representation (core, sorted, compact, transliterated) | 482,391 | 0.9973 | 0.3149 | 0.5111 | +0.4551 | 1,290 | 1,046,549 | 120 | 250,518 | 1,263 | 56,618 |
| D1 | exact name only | 516,291 | 0.8706 | 0.2942 | 0.3980 | +0.3420 | 66,817 | 1,078,176 | 4,162 | 198,853 | 51,609 | 53,989 |
| D2 | name + state | 688,682 | 0.8869 | 0.3998 | 0.5361 | +0.4801 | 77,879 | 916,847 | 5,919 | 283,847 | 72,632 | 59,992 |
| D3 | name + house number | 500,671 | 0.9920 | 0.3251 | 0.5263 | +0.4704 | 3,991 | 1,030,970 | 250 | 260,493 | 3,538 | 55,177 |
| D4 | name + street | 642,056 | 0.8895 | 0.3738 | 0.5305 | +0.4746 | 70,975 | 956,569 | 5,827 | 289,420 | 70,167 | 51,721 |
| D5 | name + city | 665,905 | 0.8841 | 0.3854 | 0.5544 | +0.4985 | 77,145 | 938,890 | 6,006 | 309,216 | 72,700 | 53,013 |
| D6 | exact address only | 133,704 | 0.9810 | 0.0859 | 0.2109 | +0.1550 | 2,542 | 1,396,488 | 127 | 100,103 | 2,038 | 27,534 |
| D7 | house number + street + city (no name) | 511,744 | 0.9755 | 0.3268 | 0.4656 | +0.4096 | 12,532 | 1,028,438 | 792 | 216,341 | 10,618 | 77,574 |
| D8 | any name representation + house number, no state conflict | 601,743 | 0.9950 | 0.3919 | 0.5915 | +0.5356 | 3,015 | 928,922 | 241 | 285,813 | 2,731 | 71,496 |

## 5. One-to-one plus the cap of 11 predictions

| rule | definition | pairs | pair precision | pair recall | macro F0.5 | gain vs empty | false merges (pairs) | missed matches (pairs) | singleton FPs | entities affected | entities with a false merge | entities fully correct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | exact normalised name + exact normalised address | 75,358 | 0.9997 | 0.0493 | 0.1508 | +0.0949 | 22 | 1,452,314 | 3 | 62,777 | 22 | 26,081 |
| R1s | sorted name tokens + sorted address tokens | 108,541 | 0.9997 | 0.0710 | 0.1906 | +0.1347 | 29 | 1,419,138 | 4 | 88,001 | 29 | 26,822 |
| R6 | name + city + postal code | 2 | 0.5000 | 0.0000 | 0.0559 | +0.0000 | 1 | 1,527,649 | 0 | 2 | 1 | 24,684 |
| R6c | name + city + state (postal code unavailable) | 650,556 | 0.8860 | 0.3773 | 0.5424 | +0.4865 | 74,161 | 951,255 | 5,924 | 301,622 | 71,496 | 52,568 |
| R7 | name + house number + street | 404,193 | 0.9981 | 0.2641 | 0.4484 | +0.3924 | 782 | 1,124,239 | 65 | 218,956 | 780 | 47,604 |
| R7s | sorted name + house number + street | 425,375 | 0.9981 | 0.2779 | 0.4629 | +0.4070 | 788 | 1,103,063 | 66 | 225,006 | 786 | 49,934 |
| R8 | very strong: name + house number, street/city agree, no state conflict | 404,346 | 0.9974 | 0.2640 | 0.4556 | +0.3997 | 1,034 | 1,124,338 | 97 | 226,337 | 1,019 | 46,646 |
| R8u | R8 with any name representation (core, sorted, compact, transliterated) | 482,391 | 0.9973 | 0.3149 | 0.5111 | +0.4551 | 1,290 | 1,046,549 | 120 | 250,518 | 1,263 | 56,618 |
| D1 | exact name only | 508,609 | 0.8823 | 0.2938 | 0.3979 | +0.3419 | 59,853 | 1,078,894 | 4,151 | 198,583 | 51,339 | 54,000 |
| D2 | name + state | 687,655 | 0.8880 | 0.3997 | 0.5360 | +0.4801 | 77,006 | 917,001 | 5,917 | 283,791 | 72,576 | 59,994 |
| D3 | name + house number | 500,671 | 0.9920 | 0.3251 | 0.5263 | +0.4704 | 3,991 | 1,030,970 | 250 | 260,493 | 3,538 | 55,177 |
| D4 | name + street | 642,056 | 0.8895 | 0.3738 | 0.5305 | +0.4746 | 70,975 | 956,569 | 5,827 | 289,420 | 70,167 | 51,721 |
| D5 | name + city | 665,024 | 0.8852 | 0.3854 | 0.5544 | +0.4985 | 76,328 | 938,954 | 6,006 | 309,181 | 72,665 | 53,013 |
| D6 | exact address only | 133,673 | 0.9812 | 0.0859 | 0.2109 | +0.1550 | 2,513 | 1,396,490 | 127 | 100,101 | 2,036 | 27,534 |
| D7 | house number + street + city (no name) | 511,330 | 0.9763 | 0.3268 | 0.4656 | +0.4096 | 12,141 | 1,028,461 | 791 | 216,316 | 10,593 | 77,575 |
| D8 | any name representation + house number, no state conflict | 601,743 | 0.9950 | 0.3919 | 0.5915 | +0.5356 | 3,015 | 928,922 | 241 | 285,813 | 2,731 | 71,496 |

## 6. Precision by number of predictions per entity (one-to-one variant)

Entities: count of fold-0 entities with that many predictions; precision: share of those predicted pairs that are true.

| rule | 1 prediction | 2 | 3-5 | 6-11 | >11 | US precision | India precision |
|---|---|---|---|---|---|---|---|
| R1 | 52,162 ent, 0.9997 | 8,889 ent, 0.9997 | 1,723 ent, 0.9996 | 3 ent, 1.0000 | - | 0.9999 (47,170) | 0.9994 (28,188) |
| R1s | 70,959 ent, 0.9998 | 14,019 ent, 0.9997 | 3,020 ent, 0.9998 | 3 ent, 1.0000 | - | 0.9999 (70,781) | 0.9994 (37,760) |
| R6 | 2 ent, 0.5000 | - | - | - | - | 0.5000 (2) | nan (0) |
| R6c | 111,637 ent, 0.8477 | 92,008 ent, 0.8841 | 93,880 ent, 0.8996 | 4,097 ent, 0.8916 | 26 ent, 0.0680 | 0.8728 (481,978) | 0.9210 (169,122) |
| R7 | 103,174 ent, 0.9974 | 67,760 ent, 0.9983 | 47,103 ent, 0.9983 | 919 ent, 0.9984 | - | 0.9998 (336,237) | 0.9893 (67,956) |
| R7s | 101,875 ent, 0.9974 | 70,203 ent, 0.9983 | 51,866 ent, 0.9984 | 1,062 ent, 0.9986 | - | 0.9998 (355,953) | 0.9895 (69,422) |
| R8 | 113,644 ent, 0.9963 | 67,469 ent, 0.9977 | 44,364 ent, 0.9981 | 860 ent, 0.9972 | - | 0.9996 (315,394) | 0.9899 (88,952) |
| R8u | 112,086 ent, 0.9962 | 76,584 ent, 0.9976 | 60,181 ent, 0.9977 | 1,667 ent, 0.9971 | - | 0.9996 (351,740) | 0.9912 (130,651) |
| D1 | 49,208 ent, 0.8446 | 58,946 ent, 0.8933 | 84,513 ent, 0.8959 | 5,916 ent, 0.7906 | 270 ent, 0.0935 | 0.8454 (326,821) | 0.9140 (189,470) |
| D2 | 79,024 ent, 0.8432 | 86,921 ent, 0.8880 | 111,975 ent, 0.8986 | 5,871 ent, 0.8699 | 56 ent, 0.1499 | 0.8672 (473,323) | 0.9303 (215,359) |
| D3 | 115,286 ent, 0.9912 | 81,361 ent, 0.9930 | 62,318 ent, 0.9924 | 1,528 ent, 0.9779 | - | 0.9952 (365,988) | 0.9834 (134,683) |
| D4 | 98,520 ent, 0.8531 | 90,381 ent, 0.8881 | 96,668 ent, 0.8993 | 3,851 ent, 0.9095 | - | 0.8739 (520,530) | 0.9561 (121,526) |
| D5 | 115,170 ent, 0.8507 | 94,368 ent, 0.8850 | 95,406 ent, 0.8981 | 4,237 ent, 0.8745 | 35 ent, 0.0726 | 0.8724 (480,295) | 0.9144 (185,610) |
| D6 | 74,332 ent, 0.9864 | 19,551 ent, 0.9819 | 6,172 ent, 0.9671 | 46 ent, 0.5493 | 2 ent, 0.0645 | 0.9817 (78,412) | 0.9800 (55,292) |
| D7 | 70,531 ent, 0.9717 | 62,052 ent, 0.9780 | 77,903 ent, 0.9786 | 5,830 ent, 0.9614 | 25 ent, 0.0556 | 0.9866 (434,677) | 0.9132 (77,067) |
| D8 | 108,159 ent, 0.9942 | 89,771 ent, 0.9959 | 84,850 ent, 0.9951 | 3,033 ent, 0.9892 | - | 0.9990 (410,069) | 0.9863 (191,674) |

## 7. Rules as candidate-generation signals (raw join, fold-0 entities against the whole pool)

| rule | pairs | pair recall | recall US | recall India | recall S2 | recall S3 | mean candidates | p50 | p95 | max | entities with >= 1 candidate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 75,364 | 0.0493 | 0.0515 | 0.0460 | 0.0758 | 0.0245 | 0.17 | 0 | 1 | 7 | 62,779 |
| R1s | 108,547 | 0.0710 | 0.0773 | 0.0616 | 0.1089 | 0.0356 | 0.25 | 0 | 1 | 7 | 88,003 |
| R6 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 | 0 | 1 | 2 |
| R6c | 904,182 | 0.4019 | 0.4683 | 0.3027 | 0.3783 | 0.4240 | 2.05 | 1 | 5 | 266 | 334,844 |
| R7 | 404,201 | 0.2641 | 0.3673 | 0.1098 | 0.2642 | 0.2639 | 0.92 | 0 | 3 | 9 | 218,962 |
| R7s | 425,383 | 0.2779 | 0.3888 | 0.1122 | 0.2785 | 0.2774 | 0.96 | 1 | 3 | 9 | 225,012 |
| R8 | 404,539 | 0.2641 | 0.3444 | 0.1439 | 0.2499 | 0.2773 | 0.92 | 1 | 3 | 12 | 226,410 |
| R8u | 482,700 | 0.3150 | 0.3841 | 0.2117 | 0.3076 | 0.3220 | 1.09 | 1 | 4 | 13 | 250,607 |
| D1 | 21,804,564 | 0.5646 | 0.5867 | 0.5316 | 0.5594 | 0.5695 | 49.40 | 4 | 213 | 1,240 | 416,457 |
| D2 | 2,160,287 | 0.5146 | 0.5568 | 0.4515 | 0.5170 | 0.5123 | 4.89 | 2 | 18 | 282 | 394,434 |
| D3 | 512,704 | 0.3286 | 0.4015 | 0.2196 | 0.3260 | 0.3311 | 1.16 | 1 | 4 | 13 | 264,447 |
| D4 | 650,598 | 0.3760 | 0.5000 | 0.1905 | 0.3820 | 0.3703 | 1.47 | 1 | 4 | 23 | 291,440 |
| D5 | 942,821 | 0.4130 | 0.4683 | 0.3305 | 0.3818 | 0.4422 | 2.14 | 2 | 5 | 266 | 346,091 |
| D6 | 239,120 | 0.0924 | 0.0901 | 0.0959 | 0.1372 | 0.0505 | 0.54 | 0 | 2 | 641 | 114,934 |
| D7 | 912,811 | 0.3733 | 0.5302 | 0.1387 | 0.3577 | 0.3879 | 2.07 | 1 | 6 | 345 | 256,228 |
| D8 | 606,903 | 0.3935 | 0.4479 | 0.3120 | 0.4008 | 0.3866 | 1.38 | 1 | 4 | 20 | 286,990 |

Union of the B rules (R1s, R6c, R7, R7s, R8, R8u, D1, D2, D3, D4, D5, D6, D7, D8): 22,531,235 pairs, pair recall 0.7664 (US 0.8189, India 0.6880), mean 51.05 candidates per entity, p95 216, max 1,245, 428,745 entities with at least one candidate.

## 8. Classification and the composite baseline

Thresholds (fixed before looking at the composite):

- **A**: one-to-one variant: pair precision >= 0.99, singleton false positives <= 1% of affected entities, macro F0.5 above the empty prediction
- **B**: raw variant: pair recall >= 5% with mean candidates per entity <= 50
- **C**: any rule that links pairs (its precision is the feature's likelihood)

| rule | A automatic match | B candidate signal | C feature | 1to1 precision | raw recall | mean candidates |
|---|---|---|---|---|---|---|
| R1 | yes | no | yes | 0.9997 | 0.0493 | 0.17 |
| R1s | yes | yes | yes | 0.9997 | 0.0710 | 0.25 |
| R6 | no | no | yes | 0.5000 | 0.0000 | 0.00 |
| R6c | no | yes | yes | 0.8853 | 0.4019 | 2.05 |
| R7 | yes | yes | yes | 0.9981 | 0.2641 | 0.92 |
| R7s | yes | yes | yes | 0.9981 | 0.2779 | 0.96 |
| R8 | yes | yes | yes | 0.9974 | 0.2641 | 0.92 |
| R8u | yes | yes | yes | 0.9973 | 0.3150 | 1.09 |
| D1 | no | yes | yes | 0.8706 | 0.5646 | 49.40 |
| D2 | no | yes | yes | 0.8869 | 0.5146 | 4.89 |
| D3 | yes | yes | yes | 0.9920 | 0.3286 | 1.16 |
| D4 | no | yes | yes | 0.8895 | 0.3760 | 1.47 |
| D5 | no | yes | yes | 0.8841 | 0.4130 | 2.14 |
| D6 | no | yes | yes | 0.9810 | 0.0924 | 0.54 |
| D7 | no | yes | yes | 0.9755 | 0.3733 | 2.07 |
| D8 | yes | yes | yes | 0.9950 | 0.3935 | 1.38 |

Greedy union of the A rules (one-to-one variants), each step adding the rule with the largest macro F0.5 gain:

| step | rule added | macro F0.5 | gain | pairs | pair precision | pair recall | singleton FPs | entities with a false merge |
|---|---|---|---|---|---|---|---|---|
| 1 | D8 | 0.59151 | +0.53558 | 601,743 | 0.9950 | 0.3919 | 241 | 2,731 |
| 2 | R1s | 0.61067 | +0.01916 | 618,163 | 0.9951 | 0.4027 | 243 | 2,748 |
| 3 | R8u | 0.61181 | +0.00114 | 619,525 | 0.9950 | 0.4035 | 246 | 2,781 |
| 4 | R7s | 0.61200 | +0.00019 | 619,837 | 0.9950 | 0.4037 | 247 | 2,787 |

A rules that added nothing on top of the others: R1, R7, R8, D3.

**Composite baseline** (union of D8 + R1s + R8u + R7s): macro F0.5 **0.61200** on 441,365 entities (empty prediction 0.05590); US 0.67733, India 0.51408; by true match count: 0: 0.9900, 1: 0.4057, 2: 0.5383, 3+: 0.6156.
Pairs 619,837, precision 0.9950, recall 0.4037; false merges 3,079, missed matches 910,892, singleton false positives 247; entities affected 298,646, with a false merge 2,787, fully correct 72,098; versus the empty prediction 297,971 entities improved, 247 hurt, 143,147 unchanged.

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

