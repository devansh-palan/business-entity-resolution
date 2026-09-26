"""Evidence for a competition-specific business-name normaliser.

Measures, on the real files, every name phenomenon that a normalisation rule could address:
legal-form variants, abbreviations and substitutions, punctuation, '&' vs 'and', token reordering,
typos, transliteration, Unicode variation, repeated words, numbers, honorifics, alias markers,
domain-style names, acronyms, per-country and per-source formatting.  For each it reports
examples, frequency among true matches, and the collision cost on Source 1 names.

Train data only, except one clearly labelled unsupervised look at French name vocabulary in the
test files (France has no training data).  Run from the project root:
    python eda/scripts/s10_name_analysis.py
"""
import os
import json, re, sys, time, unicodedata
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

sys.stdout.reconfigure(encoding="utf-8")
PQ = os.path.join(os.environ.get("BER_SCRATCH", "artifacts/_scratch"), "pq")   # parquet cache, built by s0_inventory.py
OUT = "eda/results/s10_names.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
R = {}
rng = np.random.default_rng(10)

# ------------------------------------------------------------------ helpers (analysis-only, plain Python)
INDIC = re.compile(r"[\u0900-\u0DFF]")
LATIN = re.compile(r"[A-Za-z]")
def strip_marks(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def basic(s):
    """casefold + NFKD + marks stripped + '&' -> ' and ' + non-alphanumerics -> space."""
    s = strip_marks(s).casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^\w]+", " ", s).replace("_", " ").split())
LEGAL_CANON = {"incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp", "limited": "ltd", "ltd": "ltd",
               "private": "pvt", "pvt": "pvt", "company": "co", "co": "co", "llc": "llc", "llp": "llp", "plc": "plc", "pc": "pc",
               "pllc": "pllc", "lp": "lp", "opc": "opc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
               "snc": "snc", "sa": "sa", "ei": "ei", "cie": "cie", "ets": "ets", "etablissements": "ets"}
HONORIFICS = {"mr", "mrs", "ms", "smt", "shri", "shree", "sri", "dr", "m", "messrs"}
ALIAS = re.compile(r"\b(fka|formerly|dba|d b a|t a|aka|trading as|now known as)\b")
def tokens(s): return basic(s).split()
def forms(toks): return [LEGAL_CANON[t] for t in toks if t in LEGAL_CANON]
def core(toks): return [t for t in toks if t not in LEGAL_CANON]

# ------------------------------------------------------------------ data
s1 = pl.read_parquet(f"{PQ}/train_source1.parquet", columns=["entity_id", "business_name", "country"])
s2 = pl.read_parquet(f"{PQ}/train_source2.parquet", columns=["entity_id", "business_name", "country"])
s3 = pl.read_parquet(f"{PQ}/train_source3.parquet", columns=["entity_id", "business_name", "country"])
pairs = pl.read_parquet(f"{PQ}/train_pairs_pos.parquet")
folds = pl.read_parquet("artifacts/validation/entity_folds.parquet", columns=["entity_id", "fold"])
s23 = pl.concat([s2, s3])
ps = (pairs.sample(n=400_000, seed=3)
      .join(s1.rename({"entity_id": "source1_entity_id", "business_name": "name1"}), on="source1_entity_id")
      .join(s23.select(pl.col("entity_id").alias("matched_id"), pl.col("business_name").alias("name2")), on="matched_id")
      .join(folds.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id"))
hn = (pl.read_parquet(f"{PQ}/sample_hard_feats.parquet", columns=["source1_entity_id", "matched_id", "addr_jac"]).sample(n=300_000, seed=4)
      .join(s1.rename({"entity_id": "source1_entity_id", "business_name": "name1"}), on="source1_entity_id")
      .join(s23.select(pl.col("entity_id").alias("matched_id"), pl.col("business_name").alias("name2")), on="matched_id"))
P = ps.to_dicts(); H = hn.to_dicts()
log("samples", len(P), len(H))

# ------------------------------------------------------------------ 1. legal forms: raw variants per source / country
def last_raw_tokens(df, n=25):
    out = {}
    for (ctry,), sub in df.group_by("country"):
        last = sub["business_name"].str.extract(r"(\S+)\s*$", 1).str.to_lowercase()
        c = last.value_counts().sort("count", descending=True).head(n)
        out[ctry] = [(a, int(b)) for a, b in zip(c[last.name], c["count"])]
    return out
R["1_raw_last_token_top"] = {"S1": last_raw_tokens(s1), "S2": last_raw_tokens(s2), "S3": last_raw_tokens(s3)}
# raw spellings of the legal forms (with punctuation), all sources
raw_forms = Counter()
for df in (s1, s2, s3):
    for name in df["business_name"].sample(n=300_000, seed=1).to_list():
        for m in re.findall(r"(?i)(?:^|\s)([(\[]?(?:l\.?l\.?c|l\.?l\.?p|p\.?l\.?c|p\.?c|inc|incorporated|corp|corporation|ltd|limited|pvt|private|co|company|l\.?p)\.?[)\]]?)(?=\s|$|,)", name):
            raw_forms[m.lower()] += 1
R["1_raw_legal_form_spellings"] = raw_forms.most_common(45)
# legal-form behaviour inside true matches and inside same-name hard negatives
def form_behaviour(rows):
    c = Counter(); ex = defaultdict(list)
    for r in rows:
        f1, f2 = forms(tokens(r["name1"])), forms(tokens(r["name2"]))
        if not f1 and not f2: k = "neither has a legal form"
        elif f1 == f2: k = "same forms"
        elif not f1: k = "form added in S2/S3"
        elif not f2: k = "form dropped in S2/S3"
        elif set(f1) == set(f2): k = "same forms, different order"
        else: k = "different forms"
        c[k] += 1
        if len(ex[k]) < 4 and k not in ("neither has a legal form", "same forms"): ex[k].append((r["name1"], r["name2"]))
    n = len(rows)
    return {k: (v, round(100 * v / n, 2)) for k, v in c.most_common()}, dict(ex)
R["1_legal_form_behaviour_positives"], R["1_legal_form_examples_positives"] = form_behaviour(P)
R["1_legal_form_behaviour_hard_negatives"], _ = form_behaviour(H)
# which form pairs appear in positives with different forms
sub = Counter()
for r in P:
    f1, f2 = forms(tokens(r["name1"])), forms(tokens(r["name2"]))
    if f1 and f2 and set(f1) != set(f2):
        sub[(" ".join(f1), " ".join(f2))] += 1
R["1_legal_form_substitutions_positives_top"] = [(a, b, n) for (a, b), n in sub.most_common(25)]
log("legal forms done")

# ------------------------------------------------------------------ 2. token substitutions (abbreviations, typos), insertions, deletions
subs = Counter(); adds = Counter(); drops = Counter(); n_sub1 = n_ins1 = n_del1 = 0
import difflib
for r in P:
    a, b = core(tokens(r["name1"])), core(tokens(r["name2"]))
    if a == b: continue
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1: subs[(a[i1], b[j1])] += 1
        elif tag == "insert": [adds.__setitem__(t, adds[t] + 1) for t in b[j1:j2]]
        elif tag == "delete": [drops.__setitem__(t, drops[t] + 1) for t in a[i1:i2]]
def classify(x, y):
    if x == y: return "same"
    if re.sub(r"[^a-z]", "", x) == re.sub(r"[^a-z]", "", y): return "punct/digit"
    dl = {"0": "o", "1": "l", "5": "s", "3": "e", "4": "a", "7": "t", "8": "b", "6": "g", "2": "z", "9": "g"}
    if len(x) == len(y) and all(cx == cy or dl.get(cy) == cx or dl.get(cx) == cy or (cy == "1" and cx == "i") or (cx == "1" and cy == "i") for cx, cy in zip(x, y)) and any(c.isdigit() for c in x + y):
        return "digit-for-letter"
    if y and x.startswith(y) or x and y.startswith(x): return "truncation/prefix"
    vow = lambda s: re.sub(r"[aeiou]", "", s)
    if vow(x) == vow(y) or vow(x) == y or vow(y) == x: return "vowel-dropped abbreviation"
    d = Levenshtein.distance(x, y)
    if d <= 2: return f"typo (edit {d})"
    return "other"
R["2_single_token_substitutions_top"] = [(x, y, n, classify(x, y)) for (x, y), n in subs.most_common(80)]
cls = Counter()
for (x, y), n in subs.items(): cls[classify(x, y)] += n
R["2_substitution_classes"] = {k: (v, round(100 * v / max(sum(cls.values()), 1), 2)) for k, v in cls.most_common()}
R["2_tokens_added_in_S23_top"] = adds.most_common(40)
R["2_tokens_dropped_from_S1_top"] = drops.most_common(40)
log("substitutions done")

# ------------------------------------------------------------------ 3-5. punctuation, '&'/'and', reordering, repeated words: recovery ladder on positives
def ladder(rows):
    n = len(rows); c = Counter()
    for r in rows:
        n1, n2 = r["name1"], r["name2"]
        if n1 == n2: c["L0 raw equal"] += 1; continue
        a, b = strip_marks(n1).casefold(), strip_marks(n2).casefold()
        if " ".join(a.split()) == " ".join(b.split()): c["L1 case/accents/spaces"] += 1; continue
        a2, b2 = basic(n1), basic(n2)
        if a2 == b2: c["L2 punctuation and '&'->'and'"] += 1; continue
        ta, tb = tokens(n1), tokens(n2)
        fa, fb = [LEGAL_CANON.get(t, t) for t in ta], [LEGAL_CANON.get(t, t) for t in tb]
        if fa == fb: c["L3 legal forms canonicalised"] += 1; continue
        ca, cb = core(ta), core(tb)
        if ca == cb: c["L4 legal forms removed"] += 1; continue
        if sorted(ca) == sorted(cb): c["L5 token order"] += 1; continue
        ha = ca[1:] if ca and ca[0] in HONORIFICS else ca; hb = cb[1:] if cb and cb[0] in HONORIFICS else cb
        if sorted(ha) == sorted(hb): c["L6 honorific prefix"] += 1; continue
        dd = lambda t: [x for i, x in enumerate(t) if i == 0 or x != t[i - 1]]
        if sorted(dd(ha)) == sorted(dd(hb)): c["L7 repeated word"] += 1; continue
        if "".join(sorted(ha)) == "".join(sorted(hb)) or "".join(ha) == "".join(hb): c["L8 spacing (compact)"] += 1; continue
        c["not equal after all rules"] += 1
    return {k: (v, round(100 * v / n, 2)) for k, v in c.items()}
R["3_recovery_ladder_positives"] = ladder(P)
R["3_recovery_ladder_hard_negatives"] = ladder(H)
# specific measurements
amp = sum(1 for r in P if ("&" in r["name1"]) != ("&" in r["name2"]) and (re.search(r"(?i)\band\b", r["name1"] + r["name2"])))
R["4_positives_amp_vs_and_pct"] = round(100 * amp / len(P), 3)
R["4_S1_names_with_amp_pct"] = round(100 * float(s1["business_name"].str.contains("&").mean()), 2)
R["4_S1_names_with_word_and_pct"] = round(100 * float(s1["business_name"].str.contains(r"(?i)\band\b").mean()), 2)
apos = [(r["name1"], r["name2"]) for r in P if "'" in r["name1"] and "'" not in r["name2"]]
R["3_apostrophe_examples"] = apos[:8]
R["3_apostrophe_S1_pct"] = round(100 * float(s1["business_name"].str.contains("'").mean()), 2)
joined = sum(1 for a, b in apos if basic(a.replace("'", "")) == basic(b)); spaced = sum(1 for a, b in apos if basic(a) == basic(b))
R["3_apostrophe_dropped_as_joined_vs_spaced"] = {"joined (orelees)": joined, "spaced (orelee s)": spaced, "n": len(apos)}
punct = Counter()
for name in s23["business_name"].sample(n=500_000, seed=2).to_list():
    for ch in set(re.findall(r"[^\w\s]", name)): punct[ch] += 1
R["3_punctuation_chars_in_S23_sample_500k"] = punct.most_common(30)
punct1 = Counter()
for name in s1["business_name"].sample(n=500_000, seed=2).to_list():
    for ch in set(re.findall(r"[^\w\s]", name)): punct1[ch] += 1
R["3_punctuation_chars_in_S1_sample_500k"] = punct1.most_common(20)
log("ladder done")

# ------------------------------------------------------------------ 6. typos: edit distance between cores that still differ
ed = Counter(); ex_typo = []
dl_sub = Counter()
for r in P:
    ca, cb = " ".join(sorted(core(tokens(r["name1"])))), " ".join(sorted(core(tokens(r["name2"]))))
    if ca == cb or not ca or not cb: continue
    if INDIC.search(r["name2"]): ed["indic script"] += 1; continue
    d = Levenshtein.distance(ca, cb)
    key = "1" if d == 1 else "2" if d == 2 else "3-4" if d <= 4 else "5-8" if d <= 8 else "9+"
    ed[key] += 1
    if d <= 2 and len(ex_typo) < 12: ex_typo.append((r["name1"], r["name2"]))
    for x, y in zip(core(tokens(r["name1"])), core(tokens(r["name2"]))):
        if len(x) == len(y) and x != y:
            for cx, cy in zip(x, y):
                if cx != cy and (cx.isdigit() != cy.isdigit()): dl_sub[(cx, cy)] += 1
n_diff = sum(ed.values())
R["6_edit_distance_of_unequal_cores"] = {k: (v, round(100 * v / len(P), 2)) for k, v in sorted(ed.items())}
R["6_typo_examples"] = ex_typo
R["6_digit_letter_substitutions"] = [(a, b, n) for (a, b), n in dl_sub.most_common(20)]
# S1 tokens that legitimately mix digits and letters (risk of digit->letter canonicalisation)
mixed = Counter()
for name in s1["business_name"].to_list():
    for t in tokens(name):
        if re.search(r"\d", t) and re.search(r"[a-z]", t): mixed[t] += 1
R["6_S1_legit_mixed_alnum_tokens_top"] = mixed.most_common(25)
R["6_S1_pct_names_with_digit"] = round(100 * float(s1["business_name"].str.contains(r"\d").mean()), 3)
log("typos done")

# ------------------------------------------------------------------ 7. transliteration
ind = [r for r in P if INDIC.search(r["name2"])]
R["7_positives_with_indic_name2_pct"] = round(100 * len(ind) / len(P), 2)
mixed_script = [r for r in P if INDIC.search(r["name2"]) and LATIN.search(r["name2"])]
R["7_positives_mixed_script_name2_pct"] = round(100 * len(mixed_script) / len(P), 3)
R["7_mixed_script_examples"] = [(r["name1"], r["name2"]) for r in mixed_script[:8]]
R["7_indic_examples"] = [(r["name1"], r["name2"]) for r in ind[:10]]
# positional token dictionary learned from folds != 0, evaluated on fold 0
def ind_tokens(s): return [t for t in re.sub(r"[^\w]+", " ", s).split() if INDIC.search(t)]
align = defaultdict(Counter)
same_len = 0
for r in ind:
    if r["fold"] == 0: continue
    lt = core(tokens(r["name1"])); it = [t for t in re.sub(r"[^\w]+", " ", r["name2"]).split()]
    it_core = [t for t in it if INDIC.search(t)]
    if len(lt) + len(forms(tokens(r["name1"]))) == len(it):   # full length match including legal forms
        for x, y in zip(tokens(r["name1"]), it): align[y][x] += 1
        same_len += 1
R["7_indic_pairs_train_folds"] = sum(1 for r in ind if r["fold"] != 0)
R["7_indic_pairs_with_equal_token_count_pct"] = round(100 * same_len / max(1, sum(1 for r in ind if r["fold"] != 0)), 2)
dic = {y: c.most_common(1)[0][0] for y, c in align.items() if sum(c.values()) >= 2 and c.most_common(1)[0][1] / sum(c.values()) >= 0.6}
R["7_dictionary_size"] = len(dic)
R["7_dictionary_examples"] = list(dic.items())[:15]
cov_tok = hit_tok = 0; name_exact = name_total = 0
for r in ind:
    if r["fold"] != 0: continue
    it = [t for t in re.sub(r"[^\w]+", " ", r["name2"]).split()]
    tr = [dic.get(t, t) for t in it]
    cov_tok += len(it); hit_tok += sum(1 for t in it if t in dic)
    name_total += 1
    if sorted(core([LEGAL_CANON.get(t, t) for t in tr])) == sorted(core(tokens(r["name1"]))): name_exact += 1
R["7_heldout_fold0_token_coverage_pct"] = round(100 * hit_tok / max(cov_tok, 1), 2)
R["7_heldout_fold0_name_exact_after_dictionary_pct"] = round(100 * name_exact / max(name_total, 1), 2)
log("transliteration done")

# ------------------------------------------------------------------ 8. unicode variation
def unicode_profile(df, n=400_000):
    c = Counter(); chars = Counter()
    for name in df["business_name"].sample(n=n, seed=5).to_list():
        kinds = set()
        for ch in name:
            o = ord(ch)
            if o < 128: continue
            if 0x0900 <= o <= 0x0DFF: kinds.add("indic")
            elif unicodedata.category(ch).startswith("L"): kinds.add("accented/other letter"); chars[ch] += 1
            elif unicodedata.category(ch) == "Mn": kinds.add("combining mark")
            elif ch in "\u00a0\u2007\u202f": kinds.add("non-breaking space")
            elif unicodedata.category(ch) in ("Cf", "Cc"): kinds.add("format/control")
            elif unicodedata.category(ch).startswith("P") or unicodedata.category(ch).startswith("S"): kinds.add("non-ascii punctuation/symbol"); chars[ch] += 1
            else: kinds.add("other")
        for k in kinds: c[k] += 1
        if name != unicodedata.normalize("NFC", name): c["not NFC"] += 1
        if any(0xFF00 <= ord(ch) <= 0xFFEF for ch in name): c["fullwidth"] += 1
    return {k: (v, round(100 * v / n, 3)) for k, v in c.most_common()}, chars.most_common(25)
R["8_unicode_S1"], R["8_nonascii_chars_S1"] = unicode_profile(s1)
R["8_unicode_S23"], R["8_nonascii_chars_S23"] = unicode_profile(s23)
case = lambda df: {"all_caps_pct": round(100 * float((df["business_name"].str.contains(r"\p{Lu}") & ~df["business_name"].str.contains(r"\p{Ll}")).mean()), 2),
                   "all_lower_pct": round(100 * float((df["business_name"].str.contains(r"\p{Ll}") & ~df["business_name"].str.contains(r"\p{Lu}")).mean()), 2)}
R["8_case"] = {"S1": case(s1), "S2": case(s2), "S3": case(s3)}
log("unicode done")

# ------------------------------------------------------------------ 9. repeated words
rep = lambda df: round(100 * float(df["business_name"].str.to_lowercase().str.contains(r"\b(\w+)\s+\1\b").mean()), 3) if False else None
def repeated_rate(df, n=500_000):
    c = 0; ex = []
    names = df["business_name"].sample(n=n, seed=6).to_list()
    for name in names:
        t = tokens(name)
        if any(t[i] == t[i - 1] for i in range(1, len(t))):
            c += 1
            if len(ex) < 6: ex.append(name)
    return round(100 * c / n, 3), ex
R["9_repeated_word_S1"], R["9_repeated_word_S1_examples"] = repeated_rate(s1)
R["9_repeated_word_S23"], R["9_repeated_word_S23_examples"] = repeated_rate(s23)
log("repeats done")

# ------------------------------------------------------------------ 10. numbers
num_ex = [n for n in s1["business_name"].sample(n=200_000, seed=7).to_list() if re.search(r"\d", n)][:12]
R["10_S1_number_examples"] = num_ex
kept = tot = 0
for r in P:
    d1 = set(re.findall(r"\d+", r["name1"]))
    if d1:
        tot += 1; kept += d1 <= set(re.findall(r"\d+", r["name2"]))
R["10_positives_numbers_in_S1_kept_in_match_pct"] = {"n": tot, "kept_pct": round(100 * kept / max(tot, 1), 2)}
R["10_pct_names_with_digit"] = {"S1": R["6_S1_pct_names_with_digit"], "S2": round(100 * float(s2["business_name"].str.contains(r"\d").mean()), 2), "S3": round(100 * float(s3["business_name"].str.contains(r"\d").mean()), 2)}

# ------------------------------------------------------------------ 11. country-specific: honorifics, '& co', professional suffixes; France vocabulary (test, unsupervised)
def first_tok_rate(df, toks_set, n=600_000):
    names = df["business_name"].sample(n=min(n, df.height), seed=8).to_list()
    return round(100 * sum(1 for x in names if (tokens(x)[:1] or [""])[0] in toks_set) / len(names), 3)
for ctry in ("US", "India"):
    R[f"11_honorific_first_token_pct_{ctry}"] = {k: first_tok_rate(df.filter(pl.col("country") == ctry), HONORIFICS) for k, df in (("S1", s1), ("S2", s2), ("S3", s3))}
hon_added = sum(1 for r in P if (core(tokens(r["name2"]))[:1] or [""])[0] in HONORIFICS and (core(tokens(r["name1"]))[:1] or [""])[0] not in HONORIFICS)
R["11_positives_honorific_added_pct"] = round(100 * hon_added / len(P), 3)
prof = Counter()
for name in s1.filter(pl.col("country") == "US")["business_name"].to_list():
    m = re.search(r"(?i),\s*((?:[a-z]\.?){1,4}|dds|md|cpa|esq|pa|dc|dmd|od|phd|dvm|rn|lcsw|pllc|pc)\.?\s*$", name)
    if m: prof[m.group(1).lower().replace(".", "")] += 1
R["11_US_S1_trailing_professional_designations"] = prof.most_common(15)
R["11_India_S1_and_co_sons_bros_pct"] = round(100 * float(s1.filter(pl.col("country") == "India")["business_name"].str.contains(r"(?i)\b(& co|& sons|bros|brothers|and co|and sons)\b").mean()), 2)
# France vocabulary from the test files: unsupervised, no labels, no parameter fitted
ft = pl.concat([pl.read_parquet(f"{PQ}/test_source{k}.parquet", columns=["business_name", "country"]).filter(pl.col("country") == "France") for k in (1, 2, 3)])
fr_last = Counter(); fr_first = Counter(); fr_art = 0; fr_amp = 0; fr_et = 0
names = ft["business_name"].sample(n=300_000, seed=9).to_list()
for x in names:
    t = tokens(x)
    if t: fr_last[t[-1]] += 1; fr_first[t[0]] += 1
    if any(a in t for a in ("de", "du", "des", "la", "le", "les", "l", "d")): fr_art += 1
    fr_amp += "&" in x; fr_et += " et " in x.lower()
R["11_France_test_unsupervised"] = {"last_tokens_top": fr_last.most_common(30), "first_tokens_top": fr_first.most_common(15),
                                    "pct_with_article": round(100 * fr_art / len(names), 2), "pct_amp": round(100 * fr_amp / len(names), 2), "pct_et": round(100 * fr_et / len(names), 2)}
log("country done")

# ------------------------------------------------------------------ 12. source-specific: alias markers, domain style, acronyms, noise prefixes, parentheses, double spaces
def rate(df, pattern, n=None):
    return round(100 * float(df["business_name"].str.contains(pattern).mean()), 3)
PAT = {"alias marker (fka/formerly/dba/t/a/aka)": r"(?i)\b(fka|formerly|dba|d/b/a|t/a|aka|trading as)\b",
       "domain style (.com etc, #tag, c0m)": r"(?i)\.(com|net|org|in|fr|co|io|biz|c0m)\b|^#\S+$",
       "2-3 letter acronym only": r"^[A-Z]{2,3}$", "leading noise punctuation": r"^[-<*#=~]+\s", "trailing punctuation": r"[^\w.)]$",
       "parenthesised form e.g. (Limited)": r"\((limited|ltd|p|pvt|inc|llc|corp)\)", "double space": r"\s{2,}", "all caps": r"^[^a-z]*$",
       "generic word appended (service/center/partners/group/holdings)": r"(?i)\b(service|center|partners|group|holdings)$"}
R["12_source_pattern_rates_pct"] = {name: {"S1": rate(s1, p), "S2": rate(s2, p), "S3": rate(s3, p)} for name, p in PAT.items()}
# alias markers in positives: which segment matches S1?
before = after = other = 0; alias_ex = []
for r in P:
    m = ALIAS.search(basic(r["name2"]).replace(" and ", " & "))
    if not m: continue
    b = basic(r["name2"]); m = ALIAS.search(b)
    if not m: continue
    seg_b, seg_a = b[:m.start()].strip(), b[m.end():].strip()
    c1 = " ".join(sorted(core(tokens(r["name1"]))))
    if " ".join(sorted(core(seg_a.split()))) == c1: after += 1
    elif " ".join(sorted(core(seg_b.split()))) == c1: before += 1
    else: other += 1
    if len(alias_ex) < 8: alias_ex.append((r["name1"], r["name2"]))
R["12_alias_marker_positives"] = {"S1 equals segment after marker": after, "S1 equals segment before marker": before, "neither (noisy)": other}
R["12_alias_examples"] = alias_ex
dom = [r for r in P if re.search(r"(?i)\.(com|net|org|in|fr|co|io|biz|c0m)\b|^#\S+$|com$", r["name2"].strip())]
rec = 0
for r in dom:
    comp1 = "".join(core(tokens(r["name1"])))
    comp2 = re.sub(r"(?i)\.?(com|net|org|in|fr|co|io|biz|c0m)$", "", re.sub(r"^#", "", r["name2"].strip()))
    comp2 = "".join(core(tokens(comp2)))
    rec += comp1 == comp2
R["12_domain_style_positives"] = {"n": len(dom), "pct_of_positives": round(100 * len(dom) / len(P), 2), "recovered_by_compact_compare_pct": round(100 * rec / max(len(dom), 1), 2),
                                  "examples": [(r["name1"], r["name2"]) for r in dom[:8]]}
acr = [r for r in P if re.fullmatch(r"[A-Z]{2,4}", r["name2"].strip())]
hit = sum(1 for r in acr if "".join(t[0] for t in core(tokens(r["name1"]))) == r["name2"].strip().lower())
hit_full = sum(1 for r in acr if "".join(t[0] for t in tokens(r["name1"])) == r["name2"].strip().lower())
R["12_acronym_positives"] = {"n": len(acr), "pct_of_positives": round(100 * len(acr) / len(P), 3), "initials_of_core_match_pct": round(100 * hit / max(len(acr), 1), 1),
                             "initials_incl_legal_forms_match_pct": round(100 * hit_full / max(len(acr), 1), 1), "examples": [(r["name1"], r["name2"]) for r in acr[:8]]}
log("source patterns done")

# ------------------------------------------------------------------ collision cost of each rule on Source 1 (same country)
names1 = s1["business_name"].to_list(); ctry1 = s1["country"].to_list()
def collisions(keys):
    c = Counter(zip(ctry1, keys)); shared = sum(v for v in c.values() if v >= 2)
    return {"distinct": len(c), "pct_entities_sharing_key": round(100 * shared / len(keys), 3)}
lvl = {}
k_raw = names1; lvl["raw"] = collisions(k_raw)
k_l1 = [" ".join(strip_marks(x).casefold().split()) for x in names1]; lvl["L1 case/accents/spaces"] = collisions(k_l1)
k_l2 = [basic(x) for x in names1]; lvl["L2 punctuation, &->and"] = collisions(k_l2)
tk = [x.split() for x in k_l2]
k_l3 = [" ".join(LEGAL_CANON.get(t, t) for t in t_) for t_ in tk]; lvl["L3 legal forms canonicalised"] = collisions(k_l3)
ck = [core(t_) for t_ in tk]
k_l4 = [" ".join(c_) for c_ in ck]; lvl["L4 legal forms removed"] = collisions(k_l4)
k_l5 = [" ".join(sorted(c_)) for c_ in ck]; lvl["L5 sorted tokens"] = collisions(k_l5)
hk = [c_[1:] if c_ and c_[0] in HONORIFICS else c_ for c_ in ck]
k_l6 = [" ".join(sorted(c_)) for c_ in hk]; lvl["L6 honorific removed"] = collisions(k_l6)
dd = lambda t: [x for i, x in enumerate(t) if i == 0 or x != t[i - 1]]
k_l7 = [" ".join(sorted(dd(c_))) for c_ in hk]; lvl["L7 repeated word collapsed"] = collisions(k_l7)
k_l8 = ["".join(sorted(c_)) for c_ in hk]; lvl["L8 compact (no spaces)"] = collisions(k_l8)
stop = {"the", "of", "and", "de", "du", "des", "la", "le", "les"}
k_l9 = [" ".join(sorted(t for t in c_ if t not in stop)) for c_ in hk]; lvl["L9 + stopwords removed (NOT recommended)"] = collisions(k_l9)
k_l10 = ["".join(t[0] for t in c_) for c_ in hk]; lvl["acronym key (initials)"] = collisions(k_l10)
R["collision_cost_on_S1"] = lvl
# typo-tolerant equality: share of S1 core names with another S1 core name (same country) within edit distance 1 or 2
sample_idx = rng.choice(len(names1), 6000, replace=False)
by_ctry = defaultdict(list)
for k, c in zip(k_l5, ctry1): by_ctry[c].append(k)
near1 = near2 = 0
for i in sample_idx:
    q = k_l5[i]; pool = by_ctry[ctry1[i]]
    cand = [p for p in pool if p and p[0] == q[:1] and abs(len(p) - len(q)) <= 2] if q else []
    if not cand: continue
    d = cdist([q], cand, scorer=Levenshtein.distance, score_cutoff=2)[0]
    d = d[d <= 2]
    n1 = int((d == 1).sum()); n2 = int((d == 2).sum())
    near1 += n1 > 0; near2 += (n1 + n2) > 0
R["typo_tolerant_collision_S1_sample"] = {"sampled": len(sample_idx), "pct_with_other_S1_within_edit1": round(100 * near1 / len(sample_idx), 2), "pct_within_edit2": round(100 * near2 / len(sample_idx), 2)}
log("collisions done")

json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
for k, v in R.items():
    print("\n##", k); print(json.dumps(v, ensure_ascii=False, default=str)[:3000])
log("DONE")
