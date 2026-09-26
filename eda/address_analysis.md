# Business addresses: evidence, structured representation, reliability

Evidence: `eda/scripts/s12_address_analysis.py` on 300,000 true pairs, 200,000 same-name hard negatives, the 1,164 near-miss negatives (same name, address Jaccard at least 0.5) and all Source 1/2/3 addresses (results in `eda/results/s12_addresses.json`). Parser: `code/business_entity_resolution/src/addresses.py`, tests in `tests/test_addresses.py`. Reliability: `eda/scripts/s13_address_reliability.py` (`eda/results/s13_address_reliability.json`). France has no training data and was examined without labels in the test addresses; that look set no parameter. No geocoding or external service is used anywhere.

## 1. What the addresses look like

**Layout by country and source.** US Source 1: `number street type, city, two-letter state` in 84%, a fourth component for a unit in 16%; 86% start with a digit, 86% end with a two-letter code, the remaining 14% are reordered (`Morristown, TN, 1907 Links Boulevard`). US S2: 92% three components (units dropped), all caps 90%, abbreviated types 48%; US S3: full state names in 96%, title case. India Source 1: median 5 to 6 comma components, 27% start with a digit, full state name last; S2 keeps names, S3 uses two-letter codes in 59%, and both put the state in an Indian script in 20% of addresses. France (test, unsupervised): Source 1 is `number street type name, city, region` in 86% with full street types in 94%; S2/S3 abbreviate the type in 36% (`r`, `av`, `all`, `bd`), replace the region by the department (Gironde, Nord, Loire-Atlantique, Pas-de-Calais) or drop it, and average 2.7 components.

**State abbreviations, learned from true pairs.** For every Source 1 state the match writes the same form 45% of the time (US) or 35% (India), the code 45% or 31%, and for India the Indic-script name 21%; a city stands in the last position in 1% to 3%. Learned aliases include `odisha`/`od`/`ଓଡ଼ିଶା` for Orissa, `keralam`/`കേരളം` for Kerala, and Telangana co-occurs with `Andhra Pradesh` in 16% of matches, which is recorded as a compatible pair rather than a conflict. Position of the Source 1 state inside the match: last 66.5%, middle 13.6%, first 6.7%, absent 8.8%, match has no address 4.4%.

**Street and road abbreviations, learned from true pairs.** street/st 16,883, road/rd 16,535, drive/dr 15,178, avenue/ave 12,373, lane/ln 6,365, court/ct 3,910, circle/cir 1,862, place/pl 1,732, boulevard/blvd 1,458, trail/trl 839, parkway/pkwy 421, terrace/ter 370, saint/st 270, fort/ft 226, cove/cv 164, mount/mt 130, square/sq 82, ridge/rdg 52, point/pt 49, alley/aly 29. India contributes the state codes (maharashtra/mh 8,909, karnataka/ka, gujarat/gj, haryana/hr, rajasthan/rj, bihar/br) and one-letter typos of address words (floor/flor, plot/pot, flat/flt, road/rod). `st` is ambiguous: street after a name, saint before one, and the parser resolves it by position. French forms come from the unsupervised look: `r.` for rue, `av.`, `bd`, `all.`, `imp.`, `rte`.

**Postal codes.** None. A five-digit token at the very end appears in 0.004% of US addresses; six-digit tokens in India in 0.0%; France 0.4%, which are house or road numbers. The parser still extracts a trailing 5/6-digit token strictly, and the feature is expected to be empty.

**House and building numbers.** Inside true matches the Source 1 number is exact in 65.9%, equal after stripping leading zeros, `#` or a trailing dot in 3.3% (`001419`, `#1105`, `17814.`), equal as part of a compound in 0.8% (`B3/6` for `6`), one digit changed, dropped or added in 5.1% (`4388` to `5388`, `85` to `84`, `1103` to `110`), two digit edits in 1.0%, absent or different in 7.3%, no number in Source 1 in 12.7% (India descriptive addresses and reordered US ones), match without address 3.8%. Among same-name hard negatives the number is exact in 0.23% and normalised-equal in 0.01%, so equality is close to a signature; but one-digit edits occur in 4.2% of hard negatives and 30.4% of near-miss negatives, and near-miss numbers lie within 10 of the true number 64% of the time (`8208` against `8212`, `3511` against `3515`). A one-digit difference is therefore ambiguous by itself: 5% of true matches and 30% of near misses.

**Apartment and unit numbers.** 24.5% of true pairs have a unit-like component on the Source 1 side (`Unit 4`, `Unit APARTMENT G`, `Flat No. 101`, `3Rd Floor`, `Shop No. 139`); the match keeps a unit component in 14.9% and the same value in 12.8%. Units are dropped in half of the matches, so unit agreement is evidence and unit absence is not.

**Landmarks.** 5.6% of true pairs carry a landmark component on the Source 1 side (`Near Gloriya Restaurant`, `Opp Big Bazar`, `Above Dr Electricwala Hospital`), 13.5% of India Source 1 addresses overall; the match keeps most of its tokens in 75% of those cases.

**City formats.** Source 1 city tokens are all present in the match in 74.4%, partly in 6.4%, absent in 19.3%: the noise drops the city or substitutes a neighbouring place (`Millcreek, UT` to `SALT LAKE CITY, UT`, `North East, NY` to `PINE PLAINS, NY`), and appends `city`, `township`, `cdp`, `county` or `village of` in 2.8%. US Source 1 sometimes carries a county as the city (`Caroline County, VA`). Indian addresses stack sub-district, city and district (`Bangalore South, Bangalore, Karnataka`; `Mumbai, Mumbai City, Maharashtra`).

**Missing components.** Match address null in 3.4% of pairs. Literal `NULL`, `<NULL>`, `None`, `N/A` tokens replace a component in 2.3% of US and 1.2% of India S2/S3 addresses. Component survival in matches (at least half of the component's tokens present): US street component 81%, city 80%, state 53% at token level (code against name, resolved by the alias table), unit 44%; India first component 89%, middle components 75%, city 79%, state 51%.

**Transliteration.** In India S2/S3, 21% of addresses have only their last component (the state) in an Indic script, 3% have another component in script, and none are entirely in script. The parser resolves script states through the alias table and transliterates other Indic tokens for token features.

**Punctuation.** Source 1 uses `-` (16%), `.` (15%), `/` (14%) inside numbers and abbreviations, `(` `)` for `(W)` / `(India)`. S2/S3 add `#` before numbers in 9.3% and `<` `>` around NULL. Hyphens and slashes inside tokens are kept (`1-98/5/2A`, `18/1`, `6905-C`), everything else becomes a space.

**Component reordering.** Between a Source 1 address and its match the comma components are identical in 8.3%, the same set in another order in 3.3%, a subset (drops) in 4.7%, a superset in 1.5%, and otherwise reformatted in 77.8%. Within Source 1 itself, 14% of US addresses are stored reordered (state or city first). Parsing is therefore position-independent for the state, and the street component is found by content, not position.

**Concatenated and split fields.** India S2/S3 addresses contain a comma component of one or two letters in 14.7% against 0.8% in Source 1: the noise breaks a word across a comma (`Polic, E Colony`, `Navrangpu, Ra`, `Gachibow, Li`, `KUKATPALLY MU, NICIPALITY`). Tokens of 15 or more letters occur in 1.3% of both Source 1 and S2/S3 (`Venkatesansalaiperiyar Nagar`, `SUPPLIERSNEAR`), so long tokens are mostly genuine. `C/O`, `S/O`, `D/O` prefixes occur in 2.4% of India addresses on both sides and are parsed into a `care_of` field.

**Source-specific formatting, summary.** S2: upper case, abbreviated street types, state first in 6%, PO Box / PMB inserted in 3.3% (US), `#` prefixes, leading zeros in 5%, NULL tokens. S3: title case, full US state names, Indian state codes, `CITY` / `CDP` / `Township` suffixes in 9%, more dropped components. Source 1: canonical but with 14% reordering in the US and counties as cities.

## 2. Structured representation

`parse_address(raw, country, vocab)` returns an `AddressRep` with the requested fields `house_number`, `street` (with `street_type` separately, canonical: street, road, avenue, marg, rue, ...), `unit`, `city` (raw) and `city_norm` (noise words removed, `st` to saint), `state` (canonical Source 1 spelling when recognised, else the raw admin component with `state_recognised=False`), `postal_code`, `country`, `landmark`, and the extras the data required: `locality` (remaining India components), `po_box`, `care_of`, `numbers`, `digit_runs`, `tokens` (transliterated), `components`, `missing`, `flags`. The original string is kept in `original`.

Vocabularies are learned from the training files by `build_vocab`: states from Source 1 last components (count at least 20), their aliases (codes, full names, Indic scripts, alternative names) from true pairs with at least 2% share and never a city, compatible-state pairs (Telangana / Andhra Pradesh), cities from Source 1 second-to-last components. A fold-safe version excludes fold 0 pairs (`artifacts/validation/address_vocab_excl_fold0.json`); the full version is `artifacts/address_vocab_all_train.json`. For a country absent from training, `extend_from_source1` learns states and cities from that split's Source 1 file without labels, and positional fallbacks apply otherwise.

Parsing order: NULL tokens and country words removed; PO Box and `care of` extracted; state found anywhere by alias lookup; postal code only as a strict trailing token; landmark components by keyword; units by keyword with chained keywords skipped (`Unit APARTMENT G`); the street component chosen by strength (numbered or strong type before building words such as `square`, `nagar`, `sector`); house number from the leading number (US, France, with `bis`/`ter`) or the India keyword forms (`Plot No.`, `H.No`, `Door No`, `Sy No`, `Kh No`, `Flat No`); city by vocabulary, then by position before the state; everything else to `locality`.

## 3. Which components are reliable (measured)

Both sides of 300,000 true pairs, 200,000 same-name hard negatives and 1,164 near-miss negatives were parsed with the fold-safe vocabulary. Presence on the Source 1 side / match side of true pairs: house number 86% / 75%, street 81% / 76%, city 99.9% / 95%, state recognised 95% / 90%, unit 21% / 12%, landmark 3.8% / 2.7%, postal code 0.01% / 0.0%. Parsing runs at about 13,000 addresses per second.

| Signal | True pairs | Same-name negatives | Near-miss negatives | Likelihood ratio |
|---|---|---|---|---|
| house number equal (digits, zeros stripped) | 60.5% | 0.07% | 3.9% | 865 |
| house number in the other side's numbers | 59.0% | 0.11% | 4.9% | 536 |
| house number one digit apart | 5.2% | 1.6% | 31.1% | 3.2 |
| house number within 10 | 1.2% | 1.3% | 51.3% | 0.9 |
| house number conflict (both present, not equal, not one edit) | 7.0% | 75.2% | 52.0% | 0.09 |
| numeric-token set Jaccard at least 0.5 | 70.1% | 0.08% | 6.7% | 876 |
| numeric-token sets disjoint | 11.8% | 88.3% | 70.7% | 0.13 |
| street name exact | 63.4% | 0.41% | 76.1% | 155 |
| street name fuzzy at least 85 | 65.7% | 0.44% | 78.6% | 149 |
| street name conflict (fuzzy below 60) | 4.5% | 80.0% | 4.0% | 0.06 |
| street type equal | 60.7% | 10.5% | 71.4% | 5.8 |
| city equal (normalised) | 74.4% | 1.7% | 86.5% | 44 |
| city conflict (fuzzy below 70) | 14.2% | 94.8% | 9.9% | 0.15 |
| state equal or compatible | 92.1% | 6.9% | 96.8% | 13 |
| state conflict (both recognised, incompatible) | 0.02% | 85.0% | 0.0% | 0.0002 |
| unit equal | 10.1% | 0.08% | 8.3% | 126 |
| unit conflict | 1.5% | 2.0% | 3.6% | 0.7 |
| landmark overlap | 2.7% | 0.0% | 2.5% | very large |
| locality overlap (India components) | 23.3% | 0.11% | 19.6% | 212 |
| all-token Jaccard at least 0.5 | 65.3% | 0.20% | 95.7% | 327 |
| all-token Jaccard at least 0.3 | 84.9% | 0.42% | 99.3% | 202 |
| postal code on both sides | 0.0% | 0.0% | 0.0% | none |

By country, true pairs: US house number equal 71.6%, street exact 83.1%, city equal 80.2%, state agreement 95.3%; India house number equal 44.0%, street exact 34.0%, city equal 65.7%, city conflict 24.9%, state agreement 87.4%, locality overlap and unit agreement (19.9%) carry more weight. By source: S3 has fewer identical tokens (52% at Jaccard 0.5 against 79% for S2) because of full state names and dropped components, but the same component agreement rates. The numbers above were measured before a final tweak of the state-position preference that affects 0.02% of addresses.

**What this says.** The state is a near-perfect veto: a recognised, incompatible state occurs in 0.02% of true pairs and 85% of same-name negatives. The house number is the strongest positive signal (likelihood ratio 865) and the only signal that separates near-miss negatives, on which token similarity fails completely (96% of near misses pass a 0.5 Jaccard). A one-digit difference is ambiguous: it is 6 times more common in near misses than in true pairs, but true pairs do carry digit typos in 5%, so it must be resolved by the rest of the address and the name. City disagreement is weak evidence (true pairs substitute neighbouring places in 14%), unit and landmark agreement are rare but nearly conclusive, and unit absence means nothing because matches drop units in half of the cases.

## 4. Proposed address similarity features (precision preserving)

1. **Number features:** `num_equal` (digits equal after normalisation), `num_in_other_numbers` (compound numbers), `num_edit1`, `num_absdiff` bucketed (0, 1 to 10, 11 to 100, more), `num_conflict` (both present, not equal, not one edit), `num_one_side_only`, `numbers_jaccard` and `numbers_disjoint` over all numeric tokens. `num_conflict` and `numbers_disjoint` are the precision guards; `num_edit1` is a feature, never a match on its own.
2. **Street features:** `street_exact`, `street_fuzzy` (ratio), `street_jaccard`, `street_conflict` (fuzzy below 60 with both present), `street_type_equal`, `street_type_one_missing`. Abbreviations are already canonical, so the ratio measures typos only.
3. **City features:** `city_equal` on the normalised form, `city_fuzzy`, `city_conflict`, `city_one_missing`. Conflict should lower but not veto, given 14% of true pairs.
4. **State features:** `state_agree` (equal or compatible via the learned table), `state_conflict` (both recognised and incompatible; a veto-grade feature), `state_unrecognised` (department against region for France, or a missing component).
5. **Unit and landmark:** `unit_equal`, `unit_conflict`, `unit_one_missing`; `landmark_overlap`; `po_box_equal`.
6. **Locality and whole-address:** `locality_jaccard` (India components), `tokens_jaccard`, `tokens_containment` (smaller side inside larger side, for dropped components), character 3-gram cosine on the cleaned string, `components_count_diff`, `either_missing`.
7. **Conjunctions for the precision-heavy metric:** `strong_address` = num_equal and (street_fuzzy at least 85 or street missing on one side) and not state_conflict; `near_miss_pattern` = same street and city with a number conflict or a one-digit difference, which is the signature of the decoy records; `weak_address` = address missing or only city and state agree. These three let the matcher and the deterministic baseline express "identical location", "same street but a different building", and "no location evidence" directly.
8. **Country handling:** every feature is computed the same way for every country; the vocabulary decides what is recognised, and unrecognised states fall into `state_unrecognised` rather than into conflict, so France degrades gracefully instead of producing false vetoes.
