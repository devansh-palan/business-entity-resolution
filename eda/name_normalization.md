# Business-name normalisation: evidence and design

Evidence: `eda/scripts/s10_name_analysis.py` (results `eda/results/s10_names.json`) on 400,000 true pairs, 300,000 same-name hard negatives and all 2.2M Source 1 names of the training set. Pipeline evaluation: `eda/scripts/s11_names_pipeline_eval.py` (results `eda/results/s11_names_pipeline.json`). Implementation: `code/business_entity_resolution/src/names.py` with `translit.py`; tests in `tests/test_names.py`. The only test-set use is an unsupervised look at French name vocabulary, section 11, which set no parameter.

## Findings by phenomenon

**1. Legal forms.** Raw spellings in the sources include `inc.`, `l.l.c.`, `pvt.`, `ltd.`, `p.c.`, `l.l.p.`, bracketed forms `(limited)`, `[llc]`, `(inc.)`, the accented `límited`, and nine Indic-script forms of Private Limited and LLP. Inside true matches the form is unreliable: same forms 37.5%, neither side has one 27.8%, dropped by S2/S3 22.4%, added 6.4%, different forms 4.9% (mostly `pvt ltd` reduced to `pvt` or `ltd`; `pc` swapped for `llc` or `inc` in 0.1%), repeated forms 1.1% (`PVT PVT LTD`, `Incorporated Incorporated`). Among same-name hard negatives the pattern is nearly the same (different 26%, added 22%, dropped 19%), so the form separates entities no better than chance. Decision: recognise every spelling, canonicalise (`incorporated` to `inc`, `limited` to `ltd`, `private` to `pvt`, `corporation` to `corp`, `company` to `co`, dotted acronyms joined), keep the canonical list as a feature, exclude forms from the core key. Collision cost on Source 1 of excluding forms: 39.9% to 49.1% of entities sharing a key; without exclusion 34% of true matches would miss the exact key.

**2. Abbreviations and substitutions.** Aligning single-token replacements in true matches gives 58% replacements by a generic business word (`india` to `center`, `services` to `center`, `associates` to `services`, `trading` to `center`), 13.8% digit-for-letter, 20.7% typos of edit distance 1 or 2, 6.1% British/American spelling (`center`/`centre`, 86 cases), 1.2% truncations (`services`/`service`). Real abbreviation pairs outside legal forms are essentially absent, so no abbreviation dictionary is built; `centre` is mapped to `center`. Generic-word substitution is left to similarity features weighted by token frequency. Tokens added by S2/S3: `the` (3,457), honorifics `dr` `mr` `sri` `shri` `smt` (about 1,900 each), alias markers `dba` `formerly` `aka` `fka` `nee` `known as`, `center` `services` `service` `partners`, and `com` `www` from domain-style rewrites.

**3. Punctuation.** S2/S3 inject `.` `,` `-` `(` `)` `[` `]` `/` `#` `+` `:`; Source 1 has `,` `.` `&` `'` `(` `)`. Recovery ladder on true matches: raw equal 4.6%, plus case/accents/spaces 14.8%, plus punctuation and `&` 6.3%. Apostrophes appear in 2.1% of Source 1 names and are written joined by the noise (`Luke's Grand Tax` to `lukesgrandtax.com`), so they are dropped without a space. Brackets are removed and their content kept, because `(India)` is real content.

**4. `&` versus `and`.** Source 1 has `&` in 5.1% and the word `and` in 2.6% of names; true matches swap them in only 0.37%. `&` is mapped to `and`; `and` is kept as a token (it is also dropped or added as noise, 3,069 drops, which similarity features absorb).

**5. Token reordering.** 2.4% of true matches have the same core tokens in another order (`Woodfield, Barr & Baker Chase` versus `Woodfield, Chase & Baker Barr`); sorting merges only 4,234 Source 1 name groups (0.3 points of collision). Both `core` and `core_sorted` are produced.

**6. Typos.** Among true matches whose cores still differ after every rule: edit distance 1 in 3.2% of all pairs, 2 in 2.6%, 3 to 4 in 8.0%, 5 to 8 in 11.8%, 9 or more in 16.8%. Digit-for-letter substitutions are a closed set: `o` to `0` 4,019, `l` to `1` 2,875, `s` to `5` 1,663, `b` to `8` 775, `g` to `6` 556, everything else under 20. The fix is applied only inside tokens with at least three letters and at most two digits from that set; ordinals (`2nd`), `24hr` and `b2b` are untouched, and Source 1 is changed in 0.0% of names. Typo-tolerant equality is dangerous as a key: 16.6% of Source 1 core names have another Source 1 name within edit distance 1 in the same country, 36.6% within 2, so typos are left to similarity features.

**7. Transliteration.** 7.2% of true matches pair a Latin Source 1 name with an Indic-script record, 0.5% are mixed script (`Alpha Sun एक्सपोर्ट्स Limited`). Two implementation facts were found the hard way: (a) generic Unicode cleaning treats Indic vowel signs and viramas as marks or punctuation and shreds the words, so only the Latin diacritics block U+0300 to U+036F is stripped and Indic runs are kept in NFC; (b) NFKD decomposes composed Indic vowel signs, so it is applied to non-Indic runs only. A token dictionary learned from training pairs by positional alignment (names keep word order; unequal lengths fall back to similarity under the rule transliteration) has 1,319 entries, covers 98.5% of Indic tokens in the held-out fold and yields an exact core match for 99.94% of held-out Indic pairs. The rule-based fallback alone gives 3.5% exact matches and 0.60 mean character similarity, enough for fuzzy features on unseen tokens. Legal forms in Indic script are recognised through the transliteration (`प्रा. लि.` as `pvt ltd`). Source 1 spells several Indian words two ways at equal frequency (laxmi 4,025 against lakshmi 4,048; jai 4,585 against jay 4,040; shree 7,993 against sree 4,295), and the Indic script has one form, so these are unified in the transliteration representation only.

**8. Unicode.** Source 1 is pure ASCII in training. S2/S3 contain Indic script in 7.2%, accented Latin letters in 5.9% (only `í é ó á ú` and their capitals, injected as noise), and zero-width joiners in 0.4% (inside Telugu and Kannada conjuncts). No non-breaking spaces, full-width forms or NFC problems. S2 is all-caps in 18.9% and all-lowercase in 5.8% of names, S3 3.0% and 6.3%.

**9. Repeated words.** Source 1 has consecutive repeats in 1.4% of names, almost all legitimate initials or forms (`N N Barreto`, `L.L.C.` tokenised, `DDS PC`); S2/S3 in 4.8% (`CLINIC CLINIC`, `Karjat Karjat`). Repeats of tokens with two or more characters are collapsed; single letters are kept; 1.2% of true matches are recovered and Source 1 changes in 0.2%.

**10. Numbers.** 1.6% of Source 1 names contain digits (`72 Hour Massage`, `Studio 38`, `2195 Jefferson Place Partners`, `Ironworkers Local 661`); S2/S3 5.1%, the excess being digit typos and domain names. 92% of Source 1 numbers survive in the true match, so numeric tokens are kept verbatim and exposed as `digits`.

**11. Country patterns.** India: honorific first tokens in 8.0% to 8.8% of S2/S3 names against 0.8% of Source 1, added to 3.1% of true matches; `& Co`, `& Sons`, `Bros` in 2.3%. The honorific is stripped only when at least two core tokens remain, so `Shree & Co` keeps its word. US: comma-separated professional designations (`DDS`, `MD`, `DO`, `DPM`, `LCSW`, `PhD`, `PA`, `DC`, `DMD`, `CPA`, `Esq`, `OD`, 1,600 to 3,400 each) are detected only after a comma, so `Do It Yourself` is safe. France, from the test names without labels: legal forms `SARL` `SAS` `EURL` `SA` `SASU` `SCI` `SNC` `EI` `Cie` appear as suffixes and also as prefixes (`SARL Dupont`, `Ets Martin`), articles occur in 14% of names, `&` in 5.8%, `et` in 0.9%. Forms are matched in any position; articles are kept.

**12. Source patterns.** Alias markers in 2.1% of S3 names, always as `Fake fka Real`: in 5,914 true matches Source 1 equals the segment after the marker and never the segment before, so the name is taken after the marker and the prefix kept as a variant, and a marker needs text on both sides (`Aka Associates` is a name). Domain-style names in 4.3% of S2 and S3 (`Helios.Com`, `CLAIRSPROTT.COM`, `#KOKANINDIA`, `pacificmaterialsdynamics.c0m`, `Indu-Trust | www.indutrust.com`, `... .COM - 6670506828`): the label becomes the name, URL segments become variants, trailing phone numbers are stripped; a space-free `compact` representation makes 55.6% of them exactly comparable and prefix comparison recovers more. Acronym-only names in 0.24% of S3: initials of the Source 1 core tokens match 74.5% of them, so an `acronym` representation is produced but is never a key on its own (95% of Source 1 entities share an acronym). Leading noise punctuation in 0.6%, trailing in 4.0%, double spaces in 11% of S2/S3.

## Rules and their cost

| Rule | Frequency on the S2/S3 side of true matches | Source 1 changed | Collision cost on Source 1 |
|---|---|---|---|
| casefold, Latin accents, spaces | 18.2% all-caps, 5.9% accents | 0% | 38.3% to 38.3% |
| punctuation, brackets, dotted acronyms, `&` to and, apostrophes | 6.3% recovered | none material | to 39.3% |
| legal forms canonical and excluded from core | 27.5% differ | forms kept as feature | to 49.5% |
| digit-for-letter fix | 2.7% | 0.0% | none |
| honorific stripped | 3.0% | 0.1% | none material |
| repeated word collapsed | 2.1% | 0.2% | none material |
| alias marker resolved | 1.5% | 0.0% | none |
| designations after comma | 0.6% | 1.0% (kept as field) | none |
| domain label, URL segment, phone tail | 5.0%, 0.3%, 0.3% | 0% | none |
| token sorting (as a second key) | 2.4% recovered | n/a | 49.5% to 49.8% |
| Indic transliteration, dictionary then rules | 7.2% | 0% | 49.8% to 49.8% |
| Indian spelling unification, transliteration key only | 0.2% | 0.2% | within 0.05 points |
| rejected: stopword removal | would recover little | | measured 49.6%, not applied |
| rejected: fuzzy or phonetic keys | | | 16.6% of names within edit distance 1 of another entity |
| rejected: generic-word removal | 58% of substitutions | would delete real words like `Primary Care Group` | variant without a trailing generic word is produced instead |

## Representations (all kept per name)

`original`, `clean`, `tokens`, `legal_forms`, `honorific`, `designations`, `alias_prefix`, `core_tokens`, `core`, `core_sorted`, `compact`, `acronym`, `translit_tokens`, `translit_core`, `translit_core_sorted`, `variants`, `digits`, `script`, `flags`; character n-grams through `char_ngrams`.

## Results on real pairs

Exact-key recovery of true matches, 400,000 pairs:

| Key | Equal |
|---|---|
| generic key from normalize.py (baseline) | 50.5% |
| original string | 4.6% |
| clean | 28.4% |
| core | 56.5% |
| core_sorted | 59.0% |
| translit_core_sorted | 66.1% |
| compact | 59.7% |
| any of translit_core_sorted, compact, variants, acronym | 71.6% |
| any key or compact prefix for domain-style names | 72.2% |

Same-name hard negatives, 300,000 pairs: 99.3% equal on the sorted key by construction, 96.5% on `compact`, no new merges introduced by any rule. Source 1 collisions: `core_sorted` 49.78% of entities share a key against 49.50% for the generic key; `translit_core_sorted` 49.83%. Held-out transliteration: 99.94% exact core matches with the fold-safe dictionary. Throughput about 30,000 names per second single-threaded, 71 seconds for all Source 1 names.

## What the numbers say about the rest of the pipeline

28% of true matches still have no exact key after every rule: renames (3.2%), generic-word substitutions, multi-character typos and rare-token transliterations. Those are the job of similarity features over `tokens`, `compact` and character n-grams, and of address evidence, not of further normalisation. The dictionary must be learned without the validation fold when scoring that fold (`artifacts/validation/translit_dict_excl_fold0.json`), and from all training pairs for the test run (`artifacts/translit_dict_all_train.json`).
