# Why the candidate set misses true pairs, and what recovers them (fold 0)

The final candidate set of `candidate_generation.md` misses 22,608 of the 1,527,650 true pairs (1.48%). Every missed pair was diagnosed with both records in hand (`s16_miss_analysis.py`, per-pair table in `artifacts/candidates/fold0/miss_causes.parquet`): the rank of the true record when the name + address retrieval is extended to 100, its cosine, whether the tokens the two records share were pruned as too common, whether the pair sat inside an exact-name or city block dropped by a cap, and name- and address-side signals. One primary cause is assigned in this order: country variation, missing name, missing address, candidate limit too small (rank 21-100), blocking collision (dropped block), retrieval failure (similar name and similar address but rank > 100), then the name-side causes when the names differ (transliteration, abbreviation, typo, name variation), address variation when the names agree, other when neither side is similar.

## 1. Failure types

| cause | pairs | share of misses | share of all true pairs | India | US | median name ratio | median address Jaccard | reading |
|---|---|---|---|---|---|---|---|---|
| missing address | 7,286 | 32.23% | 0.477% | 2,830 | 4,456 | 82 | 0.00 | the pool record has no address at all, so only its (usually typo'd) name can find it |
| candidate limit too small | 5,530 | 24.46% | 0.362% | 3,116 | 2,414 | 80 | 0.31 | the true record is in the retrieval's ranks 21-100; the shared tokens are mostly common ones that the 1% pruning removed |
| blocking collision | 3,356 | 14.84% | 0.220% | 3,097 | 259 | 100 | 0.23 | the pair shares an exact name (or city + first token) but the block exceeded the cap; 92% India, 66% Indic-script names |
| name variation | 2,113 | 9.35% | 0.138% | 1,014 | 1,099 | 42 | 0.33 | the pool name is a different or scrambled name (`Iriarc`, `League Dvee1poebmnt Economic`); the address is only partly similar |
| abbreviation | 1,463 | 6.47% | 0.096% | 567 | 896 | 62 | 0.27 | 85% are domain-style names (`nbrothers.com` for `North & Brothers`, `CHANDALFUENTESCPA.COM`) |
| address variation | 1,123 | 4.97% | 0.074% | 440 | 683 | 88 | 0.18 | similar but not identical names with a different address, so neither keys nor the retrieval reach them |
| other | 850 | 3.76% | 0.056% | 438 | 412 | 23 | 0.21 | unrelated synthetic names (`Tavozeph`, `Beloxylo`) with a partly similar address |
| retrieval failure | 829 | 3.67% | 0.054% | 361 | 468 | 87 | 0.38 | similar name and similar address yet outside the top 100: the shared tokens were pruned and the rest are typo'd (`108th` / `108rd`, `Tacoma` / `Tacma`) |
| typo | 53 | 0.23% | 0.003% | 5 | 48 | 73 | 0.29 | small character edits in the name with a different address |
| transliteration | 5 | 0.02% | 0.000% | 0 | 5 | 47 | 0.20 | an Indic name whose transliteration differs from the Latin name |
| missing name | 0 | 0.00% | 0.000% | 0 | 0 | nan | nan |  |
| country variation | 0 | 0.00% | 0.000% | 0 | 0 | nan | nan |  |

Rank of the missed true record in the name + address retrieval extended to 100: ranks 21-30 2,286, 31-50 2,719, 51-100 3,332, beyond 100 14,126; 6,246 missed pairs have a cosine below the 0.2 threshold.

## 2. What recovers each type

Recovery counts of the missed pairs by strategies already measured in `candidate_generation.md` (top 4 per cause) and by the cheap keys tried here:

| cause | existing strategies that would recover it (pairs) |
|---|---|
| missing address | 1_exact_name_translit_core_sorted 820, 2_name_token_df1000_rarest2 792, 1_exact_name_core_sorted 778, 1_exact_name_core 753 |
| candidate limit too small | 9_addr_token_top20 1,357, 4_addr_token_df500_rarest3 1,247, 1_exact_name_translit_core_sorted 1,230, 8_acronym_city 988 |
| blocking collision | 1_exact_name_translit_core_sorted 2,266, 8_acronym_city 1,554, 5_city_name_first 701, 5_city_house_number 321 |
| name variation | 9_numsig 279, 9_addr_token_top20 212, 5_city_house_number 193, 4_addr_token_df500_rarest3 168 |
| abbreviation | 9_addr_token_top20 211, 9_numsig 184, 3_name_ngram_top20 178, 4_addr_token_df500_rarest3 131 |
| address variation | 3_name_ngram_top20 140, 9_numsig 123, 2_name_token_df1000_rarest2 81, 8_acronym_city 77 |
| other | 9_numsig 85, 4_addr_token_df500_rarest3 73, 9_addr_token_top20 53, 4_addr_token_df200_rarest3 36 |
| retrieval failure | 9_numsig 123, 3_name_ngram_top20 102, 9_addr_token_top20 87, 5_city_house_number 80 |
| typo | 9_numsig 11, 4_addr_token_df500_rarest3 5, 9_addr_token_top20 2, 3_name_ngram_top20 1 |

| cheap key | mean candidates per entity | p95 | recall alone | missed pairs recovered | by cause |
|---|---|---|---|---|---|
| exact name (any representation) + house number | 1.4 | 4 | 0.3935 | 554 | candidate limit too small 268, blocking collision 286 |
| exact name (any representation) + city, blocks <= 100 | 2.7 | 7 | 0.5110 | 2,729 | candidate limit too small 956, blocking collision 1,773 |
| compact name + state | 5.1 | 19 | 0.5450 | 333 | candidate limit too small 96, blocking collision 237 |
| house number + street + city, blocks <= 200 | 2.0 | 6 | 0.3732 | 20 | retrieval failure 2, candidate limit too small 8, name variation 2, abbreviation 1, blocking collision 7 |

## 3. Fixes measured on the whole fold

Every fix was computed for all 441,365 fold-0 entities and compared with the old final set: new candidates it adds per entity, missed pairs it recovers, and the recall gained per candidate added (the fold-0 frontier accepted strategies at >= 0.0002, one recall point per 50 candidates).

| fix | pairs | recall alone | +candidates per entity | missed pairs recovered | recall gain (points) | gain per candidate | recovered by cause |
|---|---|---|---|---|---|---|---|
| exact name (any representation) + house number | 617,737 | 0.3935 | +0.01 | 554 | +0.036 | 0.02589 | blocking collision 286, candidate limit too small 268 |
| exact name (any representation) + city, blocks <= 100 | 1,186,089 | 0.5110 | +0.19 | 2,729 | +0.179 | 0.00957 | blocking collision 1,773, candidate limit too small 956 |
| compact name + state | 2,270,002 | 0.5450 | +0.06 | 333 | +0.022 | 0.00342 | blocking collision 237, candidate limit too small 96 |
| house number + street + city, blocks <= 200 | 871,435 | 0.3732 | +0.03 | 20 | +0.001 | 0.00049 | candidate limit too small 8, blocking collision 7, name variation 2, retrieval failure 2 |
| name + address retrieval, top 20, pruning at 3% df instead of 1% | 17,650,736 | 0.9688 | +8.34 | 3,935 | +0.258 | 0.00031 | candidate limit too small 2,235, missing address 979, blocking collision 491, name variation 90 |
| name 3-gram retrieval against the address-less pool records, top 5 per source | 4,398,414 | 0.0394 | +7.08 | 2,733 | +0.179 | 0.00025 | missing address 2,733 |
| address token retrieval, top 5 per source | 4,410,414 | 0.7700 | +2.42 | 875 | +0.057 | 0.00024 | candidate limit too small 616, abbreviation 103, name variation 73, retrieval failure 37 |
| name 3-gram retrieval against the domain-style pool names, top 3 per source | 2,636,765 | 0.0389 | +5.21 | 1,157 | +0.076 | 0.00015 | candidate limit too small 402, abbreviation 400, address variation 153, retrieval failure 132 |
| ranks 21-50 of the name + address retrieval when the 20th score >= 0.5 | 19,034,821 | 0.9620 | +3.88 | 781 | +0.051 | 0.00013 | candidate limit too small 578, missing address 187, blocking collision 11, abbreviation 2 |
| ranks 21-50 of the name + address retrieval when the 20th score >= 0.6 | 17,909,519 | 0.9606 | +1.51 | 299 | +0.020 | 0.00013 | candidate limit too small 171, missing address 115, blocking collision 10, other 1 |
| address token retrieval, top 10 per source | 8,819,153 | 0.8176 | +7.84 | 1,393 | +0.091 | 0.00012 | candidate limit too small 952, abbreviation 160, name variation 135, retrieval failure 61 |
| ranks 21-50 of the name + address retrieval when the 20th score >= 0.7 | 17,702,009 | 0.9602 | +1.07 | 183 | +0.012 | 0.00011 | missing address 105, candidate limit too small 65, blocking collision 10, other 1 |
| name + address retrieval, top 30 per source | 26,470,555 | 0.9657 | +19.65 | 2,441 | +0.160 | 0.00008 | candidate limit too small 1,713, missing address 699, blocking collision 17, name variation 4 |
| name acronym + city | 13,912,558 | 0.4979 | +29.10 | 2,696 | +0.176 | 0.00006 | blocking collision 1,554, candidate limit too small 988, address variation 77, retrieval failure 46 |
| address token retrieval, top 20 per source | 17,633,587 | 0.8473 | +22.60 | 1,997 | +0.131 | 0.00006 | candidate limit too small 1,357, name variation 212, abbreviation 211, retrieval failure 87 |
| name + address retrieval, top 50 per source | 44,074,369 | 0.9714 | +58.49 | 5,128 | +0.336 | 0.00006 | candidate limit too small 3,524, missing address 1,571, blocking collision 19, name variation 4 |
| name 3-gram retrieval, top 20 per source | 17,432,120 | 0.6101 | +29.92 | 1,770 | +0.116 | 0.00004 | candidate limit too small 560, missing address 538, abbreviation 178, blocking collision 158 |

Greedy selection (threshold 0.0002 recall per candidate): exact name (any representation) + house number (+0.036 points for +0.01 candidates) -> exact name (any representation) + city, blocks <= 100 (+0.154 points for +0.19 candidates) -> compact name + state (+0.007 points for +0.06 candidates) -> house number + street + city, blocks <= 200 (+0.001 points for +0.03 candidates) -> name 3-gram retrieval against the address-less pool records, top 5 per source (+0.179 points for +7.08 candidates) -> address token retrieval, top 5 per source (+0.053 points for +2.42 candidates).
Rejected: name + address retrieval, top 30 per source, name + address retrieval, top 50 per source, ranks 21-50 of the name + address retrieval when the 20th score >= 0.5, ranks 21-50 of the name + address retrieval when the 20th score >= 0.6, ranks 21-50 of the name + address retrieval when the 20th score >= 0.7, name 3-gram retrieval against the domain-style pool names, top 3 per source, address token retrieval, top 10 per source, name + address retrieval, top 20, pruning at 3% df instead of 1%, address token retrieval, top 20 per source, name 3-gram retrieval, top 20 per source, name acronym + city.

## 4. Old versus new

| | old | new |
|---|---|---|
| candidate recall | 0.9852 | 0.9895 |
| recall US / India | 0.9883 / 0.9806 | 0.9909 / 0.9875 |
| entities with every match present | 0.9512 | 0.9643 |
| average candidates per entity | 81.4 | 91.2 |
| 95th percentile | 233 | 241 |
| pairs | 35,911,531 | 40,233,392 |

Remaining misses: 16,039 (missing address 4,553, candidate limit too small 3,903, name variation 2,038, blocking collision 1,390, abbreviation 1,359, address variation 1,115, other 834, retrieval failure 790, typo 52, transliteration 5).

New file `artifacts/candidates/fold0\candidate_pairs.tsv`: 441,365 rows, 40,233,392 ids, 524 MB, specification check passed; the previous file is kept as `candidate_pairs_v1.tsv`.

## 5. Examples per cause

**missing address**

- `Rapid Bnb Inc` | `1945 Shirley Drive, Burlington, NC`  ->  `Rapid Inc Service` | `None`  (retrieval rank 49, cosine 0.34, name ratio 55, address Jaccard 0.00)
- `Apex Alpine` | `Upper Allen Township, PA, 2427 Bumble Bee Hollow Road`  ->  `Apex [Services]` | `None`  (retrieval rank > 100, cosine 0.24, name ratio 58, address Jaccard 0.00)
- `Bureau of Finance` | `2032 Mallard Lane, Montgomery, AL`  ->  `Bureau of Center Trading` | `None`  (retrieval rank > 100, cosine 0.30, name ratio 63, address Jaccard 0.00)
- `Hargett Charter School Corp` | `1700 Midland Trail, VA, Alleghany County`  ->  `Hargett Charter Sool Corp` | `None`  (retrieval rank 21, cosine 0.41, name ratio 95, address Jaccard 0.00)

**candidate limit too small**

- `Colonial Family Practice Group` | `6905 Darmstadt Road, Evansville, IN`  ->  `Dovakor` | `EVANSVILLE, IN, 6905-C DARMSTADT RD`  (retrieval rank 26, cosine 0.40, name ratio 16, address Jaccard 0.43)
- `Peridora Restaurant Inc` | `WA, Tacoma, 5519 23rd Avenue`  ->  `Peridora Inc Services` | `23TH AVE, TCAOMA, WA`  (retrieval rank 27, cosine 0.33, name ratio 67, address Jaccard 0.12)
- `White Trading Private Limited` | `K.No 69/458/69 Bellary Rd, 10Th Fl, Near Godrej Apt, Bangalore North, Bangalore, Karnataka`  ->  `ವೈಟ್ ಟ್ರೇಡಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `BANGALORE, HN C-692 K.NO 69/458/69 BLLARY RD, ಕರ್ನಾಟಕ`  (retrieval rank 32, cosine 0.44, name ratio 100, address Jaccard 0.29)
- `Direct Agricultural Union` | `9025 Washington Henry Drive, Hanover County, VA`  ->  `Direct Agrcutlurl Union` | `9025-9029 WASHINGTON HENRY DR, MECHANICSVILLE, VA`  (retrieval rank 39, cosine 0.33, name ratio 92, address Jaccard 0.30)

**blocking collision**

- `Shiva Power Limited` | `No.70, Rajanarayan Towers Race Course Road, Coimbatore, Tamil Nadu`  ->  `சிவா பவர் லிமிடெட்` | `#2-70, COIMBATORE, Tamil Nadu`  (retrieval rank > 100, cosine 0.21, name ratio 100, address Jaccard 0.27)
- `All Foods Private Limited` | `S-24, First Floor, Tirupati Plaza, Palanpur, Banas Kantha, Gujarat`  ->  `ઓલ ફૂડ્સ પ્રાઇવેટ લિમિટેડ` | `S-22, FIRST FLOOR, TIRUPATI PLAZA, BANASKANTHA, PALANPUR, Gujarat`  (retrieval rank > 100, cosine 0.38, name ratio 100, address Jaccard 0.55)
- `My It` | `New Delhi, 1214, South West Delhi, 12Th Floor 38, Ansal Tower, Nehru Place, Delhi`  ->  `माय आईटी` | `C-1214, NEW DELHI, SOUTH WEST DELHI, दिल्ली`  (retrieval rank > 100, cosine 0.22, name ratio 100, address Jaccard 0.29)
- `Super Care Private Limited` | `S/F Backside, T-Extn., Part-Iii Jain Colony, H.No.2, Delhi, West Delhi, New Delhi`  ->  `सुपर केयर प्राइवेट लिमिटेड` | `NO 2, Delhi, BINDAPUR, NEW DELHI`  (retrieval rank > 100, cosine 0.15, name ratio 100, address Jaccard 0.31)

**name variation**

- `RM Functional Corporation` | `No. 56, K No 105, Raghavendra Nagar, 2Nd Cr, Bangalore South, Bangalore, Karnataka`  ->  `Iriarc` | `H.NO 56, BANGALORE SOUTH, BANGALORE, Karnataka`  (retrieval rank > 100, cosine 0.16, name ratio 32, address Jaccard 0.42)
- `Economic Development League` | `114 Lodge Drive, Irondequoit, NY`  ->  `League Dvee1poebmnt Economic` | `114- LODGE DRIVE, ROCHESTER, NY`  (retrieval rank > 100, cosine 0.28, name ratio 44, address Jaccard 0.29)
- `1 800 Flowers` | `5735 75, North Vernon, IN`  ->  `Rizanylaflux` | `75, NORTH VERNON, IN`  (retrieval rank > 100, cosine 0.30, name ratio 16, address Jaccard 0.80)
- `Pediatric Partners LLC` | `2566 Hazelwood Street, MN, Maplewood`  ->  `PEDIATRIC LLC SERVICES` | `MN, NULL, 566 HAZELWOOD SAINT, ST. PAUL`  (retrieval rank > 100, cosine 0.32, name ratio 72, address Jaccard 0.22)

**abbreviation**

- `Fuentes, Chandal, CPA` | `256 46th Place, Chicago, IL`  ->  `CHANDALFUENTESCPA.COM` | `256 46ND PL, CHHICAGO, IL`  (retrieval rank > 100, cosine 0.21, name ratio 56, address Jaccard 0.25)
- `Nrg Trust` | `5F, Everest 46/C, Chowringhee Road, Kolkata, Kolkata, Howrah, West Bengal`  ->  `Sri NTRUST.COM` | `5-5F, KOLKATA, CHOWRINGHEE, West Bengal`  (retrieval rank > 100, cosine 0.20, name ratio 67, address Jaccard 0.40)
- `Banyan Plastics Limited` | `Tasgaon, Sangli, Bansode At Post A/P, Maharashtra, C/O Sou Sushil Nivrutti`  ->  `pbanyan.com` | `PLOT 371 C/O SOU SUSHIL NIVRUTTI, SANGLI, TASGAON, Maharashtra`  (retrieval rank > 100, cosine 0.26, name ratio 55, address Jaccard 0.23)
- `North & Brothers Private Limited` | `Unit No. 21, Second Floor, Plot No 8, Eros Metro Mall, Sector-14, Dwarka, New Delhi, South West Delhi, Delhi`  ->  `nbrothers.com` | `NEW DELHI, UNIT NO. 21, Delhi`  (retrieval rank > 100, cosine 0.21, name ratio 67, address Jaccard 0.31)

**address variation**

- `Chiropractic Associates` | `10306 270th Street, Stanwood, WA`  ->  `CHIROPRACTICASSOC.COM` | `270RD ST, PMB 3976, STANWOOD, WA`  (retrieval rank > 100, cosine 0.23, name ratio 85, address Jaccard 0.29)
- `Youth Chiropractic` | `14324 Compton Road, Fairfax County, VA`  ->  `YCHIROPRACTIC.COM` | `14324- COMPTON ROAD, CENTREVILLE, VA`  (retrieval rank > 100, cosine 0.00, name ratio 84, address Jaccard 0.25)
- `Smart Ridge Soulpower Inc` | `7918 Lynette Street, Houston, TX`  ->  `Smart Rcirce Soulpower Inc` | `7918- LYNETTE SAINT, <NULL>, HOUTON, TX`  (retrieval rank > 100, cosine 0.29, name ratio 88, address Jaccard 0.12)
- `P+ Harmony Inc` | `4360 63rd Street, Unit 17, Sacramento, CA`  ->  `Inc P+ Hammony` | `4360- 63RD STREET, SACRAMNTO, CA`  (retrieval rank > 100, cosine 0.00, name ratio 89, address Jaccard 0.22)

**other**

- `Prudent Affiliate Pvt. Ltd.` | `Plot No-143, Khata No-15, Ramachandrapur Back Side Of Police Station, Jatani, Bhubaneswar, Khordha, Orissa`  ->  `Tavozeph` | `PLOT NO-6-143, BHUBANESWAR, KHORDHA, Orissa`  (retrieval rank > 100, cosine 0.17, name ratio 24, address Jaccard 0.27)
- `Career Services Limited` | `#37/1, 3Rd Floor, Adams Corner, Coles Road, Frazer Town, Bangalore, Karnataka`  ->  `CS` | `NO 037/1, BANGALORE, Karnataka`  (retrieval rank > 100, cosine 0.00, name ratio 24, address Jaccard 0.15)
- `Highland Steel Pvt. Ltd.` | `Zone-Ii, Second Floor, M.P Nagar, Bhopal, Madhya Pradesh, 213`  ->  `Pyrakor` | `21, BHOPAL, मध्य प्रदेश`  (retrieval rank > 100, cosine 0.13, name ratio 10, address Jaccard 0.27)
- `CR Starlink, L.L.C.` | `4318 Gifford Place, Washougal, WA`  ->  `Beloxylo` | `4318- GIFFORD PLACE, WASHOUGAAL, WA`  (retrieval rank > 100, cosine 0.07, name ratio 11, address Jaccard 0.29)

**retrieval failure**

- `Alta Olivares Pgim LLC` | `1502 108th Street, Tacoma, WA`  ->  `olivarespgim.com` | `1502 108RD STREET, TACMA, WA`  (retrieval rank > 100, cosine 0.21, name ratio 80, address Jaccard 0.43)
- `Super Infotech Private Limited` | `Flat No.502, Patel Residency, Masab Tank, Hyderabad, Rangareddy, Telangana`  ->  `*** superinfotech.com` | `Telangana, RANGAREDDY, HYDERABAD, 502`  (retrieval rank > 100, cosine 0.27, name ratio 84, address Jaccard 0.40)
- `Guzman Computing` | `10706 227th Avenue, Vancouver, WA`  ->  `6uzman Cssmutig` | `010706 227ST AVE, VANCOVER CITY, WA`  (retrieval rank > 100, cosine 0.14, name ratio 84, address Jaccard 0.10)
- `Painters Local 284` | `35 Maple Avenue, NY, Brookhaven`  ->  `#painterslocal` | `MAPLE AVE, BROOKHAVEN, NY`  (retrieval rank > 100, cosine 0.32, name ratio 84, address Jaccard 0.50)

**typo**

- `Nisa & Co` | `B-31, Okhla Industrial Area, Phase-1, New Delhi, South Delhi, Delhi`  ->  `Mr Ncba & (Co)` | `B-031, SOUTH DELHI, NEW DELHI, Delhi`  (retrieval rank > 100, cosine 0.00, name ratio 75, address Jaccard 0.33)
- `Citadel Inc` | `1524 10th Street, Unit Apartment A4, Anniston, AL`  ->  `Inc Ciaded` | `1524A TENTH STREET, ANNISTON, AL`  (retrieval rank > 100, cosine 0.18, name ratio 77, address Jaccard 0.30)
- `Baify, Inc` | `Kent, 20745 295th Street, WA`  ->  `Baic, Inc` | `20745-B 295ND ST, KENT, WA`  (retrieval rank > 100, cosine 0.12, name ratio 67, address Jaccard 0.25)
- `Nonadyne, LLC` | `554 Gilbert Avenue, Orangetown, NY`  ->  `Nogadrne, LLC` | `554-C GILBERT AVE, PEARL RIVER, NY`  (retrieval rank > 100, cosine 0.14, name ratio 75, address Jaccard 0.22)

**transliteration**

- `O & T Sound LLC` | `2700 Hayden Road, Unit APT 2003, Scottsdale, AZ`  ->  `@ó` | `2700- HAYDEN ROAD, PO BOX 9632, SOCTTSDALE CDP, AZ`  (retrieval rank > 100, cosine 0.07, name ratio 14, address Jaccard 0.18)
- `1 800 Flowers` | `1318 Highland Avenue, Knoxville, TN`  ->  `@1800` | `1318- Highland Avenue, Knoxville, Tennessee`  (retrieval rank > 100, cosine 0.23, name ratio 47, address Jaccard 0.29)
- `1 800 Flowers Inc` | `111 Sunflower Lane, Beebe, AR`  ->  `@1800` | `111- Sunflower Ln, Beebe, Arkansas`  (retrieval rank > 100, cosine 0.23, name ratio 47, address Jaccard 0.12)
- `1 800 Flowers` | `8230 Meadow Lark Road, Arkwright, NY`  ->  `#1800` | `823 Meadow Lark Road, Cassadaga CITY, New York`  (retrieval rank > 100, cosine 0.26, name ratio 47, address Jaccard 0.27)

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

