"""Evidence for a competition-specific address parser.

Measures, on the training files, the vocabulary and noise of business addresses per country and source:
state names/codes/scripts (learned from Source 1 and from true pairs), street-type abbreviations
(learned from true pairs), house numbers inside true matches and inside near-miss negatives, units,
landmarks, city forms, component survival and reordering, Indic script placement, split/concatenated
words, postal codes, punctuation.  France is looked at unsupervised in the test addresses (no labels).
Run from the project root:  python eda/scripts/s12_address_analysis.py
"""
import os
import json, re, sys, time, unicodedata
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from rapidfuzz.distance import Levenshtein

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "code/business_entity_resolution/src")
from names import strip_latin_accents  # noqa: E402

PQ = os.path.join(os.environ.get("BER_SCRATCH", "artifacts/_scratch"), "pq")   # parquet cache, built by s0_inventory.py
OUT = "eda/results/s12_addresses.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
R = {}
INDIC = re.compile(r"[\u0900-\u0DFF]")
WORD = re.compile(r"[^\w\u0900-\u0DFF/]+")

def clean(s):
    return " ".join(WORD.sub(" ", strip_latin_accents(s or "").casefold()).replace("_", " ").split())
def toks(s):
    return clean(s).split()
def comps(s):
    return [c.strip() for c in (s or "").split(",") if c.strip()]
def ccomps(s):
    return [clean(c) for c in comps(s) if clean(c)]

s1 = pl.read_parquet(f"{PQ}/train_source1.parquet", columns=["entity_id", "business_address", "country"])
s2 = pl.read_parquet(f"{PQ}/train_source2.parquet", columns=["entity_id", "business_address", "country"])
s3 = pl.read_parquet(f"{PQ}/train_source3.parquet", columns=["entity_id", "business_address", "country"])
s23 = pl.concat([s2, s3])
pairs = pl.read_parquet(f"{PQ}/train_pairs_pos.parquet")
ps = (pairs.sample(n=300_000, seed=12)
      .join(s1.rename({"entity_id": "source1_entity_id", "business_address": "a1"}), on="source1_entity_id")
      .join(s23.select(pl.col("entity_id").alias("matched_id"), pl.col("business_address").alias("a2")), on="matched_id"))
hard_all = pl.read_parquet(f"{PQ}/sample_hard_feats.parquet", columns=["source1_entity_id", "matched_id", "addr_jac"])
hn = (hard_all.sample(n=200_000, seed=13)
      .join(s1.rename({"entity_id": "source1_entity_id", "business_address": "a1"}), on="source1_entity_id")
      .join(s23.select(pl.col("entity_id").alias("matched_id"), pl.col("business_address").alias("a2")), on="matched_id"))
nm = (hard_all.filter(pl.col("addr_jac") >= 0.5)
      .join(s1.rename({"entity_id": "source1_entity_id", "business_address": "a1"}), on="source1_entity_id")
      .join(s23.select(pl.col("entity_id").alias("matched_id"), pl.col("business_address").alias("a2")), on="matched_id"))
P, H, NM = ps.to_dicts(), hn.to_dicts(), nm.to_dicts()
log("samples", len(P), len(H), len(NM))

# ------------------------------------------------------------------ A. component counts and last/first components per source
def comp_stats(df, label):
    out = {}
    for (c,), sub in df.filter(pl.col("business_address").is_not_null()).group_by("country"):
        n = sub.height
        ncomp = sub["business_address"].str.count_matches(",") + 1
        last = sub["business_address"].str.extract(r",\s*([^,]+)$", 1).str.strip_chars()
        first = sub["business_address"].str.extract(r"^([^,]+)", 1).str.strip_chars()
        vc = last.value_counts().sort("count", descending=True).head(25)
        out[c] = {"n": n, "components_mean": round(float(ncomp.mean()), 2),
                  "components_dist_pct": {str(k): round(100 * v / n, 1) for k, v in zip(*np.unique(np.minimum(ncomp.to_numpy(), 9), return_counts=True))},
                  "last_component_top": [(a, int(b)) for a, b in zip(vc[last.name], vc["count"])],
                  "first_component_starts_with_digit_pct": round(100 * float(first.str.contains(r"^[#\d]").mean()), 2),
                  "last_component_2_capitals_pct": round(100 * float(last.str.contains(r"^[A-Z]{2}$").mean()), 2)}
    R[label] = out
comp_stats(s1, "A_S1"); comp_stats(s2, "A_S2"); comp_stats(s3, "A_S3")
log("A done")

# ------------------------------------------------------------------ B. state vocabulary from Source 1 and the alias map from true pairs
state_alias = defaultdict(Counter)   # (country, S1 last comp) -> Counter of match last comps
for r in P:
    c1, c2 = comps(r["a1"]), comps(r["a2"])
    if c1 and c2:
        state_alias[(r["country"], c1[-1])][c2[-1]] += 1
smap = {}
for (c, s1state), cnt in state_alias.items():
    tot = sum(cnt.values())
    if tot < 300:
        continue
    smap[f"{c}|{s1state}"] = {"pairs": tot, "match_last_component_top": [(k, round(100 * v / tot, 1)) for k, v in cnt.most_common(6)]}
R["B_state_alias_from_pairs"] = dict(sorted(smap.items(), key=lambda kv: -kv[1]["pairs"])[:45])
alias_sets = {k: {a for a, pct in v["match_last_component_top"] if pct >= 2 and a and a.lower() != "none"} | {k.split("|", 1)[1]} for k, v in smap.items()}
pos_state = Counter()
for r in P:
    c1, c2 = comps(r["a1"]), comps(r["a2"])
    if not c1 or not c2: pos_state["no address"] += 1; continue
    key = f"{r['country']}|{c1[-1]}"
    aliases = alias_sets.get(key, {c1[-1]})
    where = [i for i, c in enumerate(c2) if c in aliases]
    if not where: pos_state["absent"] += 1
    elif where[0] == len(c2) - 1: pos_state["last"] += 1
    elif where[0] == 0: pos_state["first"] += 1
    else: pos_state["middle"] += 1
R["B_S1_state_position_in_match_pct"] = {k: round(100 * v / len(P), 2) for k, v in pos_state.most_common()}
log("B done")

# ------------------------------------------------------------------ C. street-type and other abbreviations from true pairs
sub = Counter()
def vowelless(w): return re.sub(r"[aeiou]", "", w)
for r in P:
    A, B = set(toks(r["a1"])), set(toks(r["a2"]))
    onlyA, onlyB = [a for a in A - B if a.isalpha() and len(a) >= 3], [b for b in B - A if b.isalpha() and 2 <= len(b) <= 5]
    for a in onlyA:
        for b in onlyB:
            if len(b) < len(a) and (a.startswith(b) or vowelless(a) == b or vowelless(a).startswith(b) or (b[0] == a[0] and b[-1] == a[-1] and len(b) <= 4 and Levenshtein.distance(vowelless(a), b) <= 1)):
                sub[(r["country"], a, b)] += 1
R["C_abbreviation_pairs_top"] = [(c, a, b, n) for (c, a, b), n in sub.most_common(70)]
log("C done")

# ------------------------------------------------------------------ D. house / plot numbers inside matches and inside near-miss negatives
NUM_KEY = re.compile(r"(?:^|\s)(?:plot|plot no|h\.?\s?no|house no|door no|d\.?\s?no|no|sy no|survey no|kh no|khasra no|gala no|shop no|flat no|office no|unit|site no|s\.?\s?no)[\s.:#-]*([\w/-]*\d[\w/-]*)", re.I)
def primary_number(addr, country):
    if not addr: return None
    c = clean(addr)
    if country == "US":
        m = re.match(r"^#?\s?0*(\d[\w/-]*)", c)
        return m.group(1) if m else None
    m = NUM_KEY.search(c)
    if m: return m.group(1)
    m = re.search(r"(?:^|\s)#?0*(\d[\w/-]*)", c)
    return m.group(1) if m else None
def digits(x): return re.sub(r"\D", "", x or "")
def number_relation(n1, a2):
    if n1 is None: return "S1 has no number"
    if not a2: return "match has no address"
    c2 = clean(a2)
    t2 = set(c2.replace("#", " ").split())
    if n1 in t2: return "exact"
    if any(re.sub(r"^0+", "", t).rstrip(".") == n1 for t in t2): return "equal after stripping leading zeros / trailing dot"
    if any(re.sub(r"^0+", "", t) == n1 for t in re.sub(r"[/-]", " ", c2).split()): return "equal as a part of a compound number"
    d1 = digits(n1)
    cand = [digits(t) for t in t2 if any(ch.isdigit() for ch in t)]
    if d1 and cand:
        best = min(Levenshtein.distance(d1, d) for d in cand)
        if best == 1: return "one digit changed/dropped/added"
        if best == 2: return "two digit edits"
    return "absent or different"
def relation_table(rows, label):
    c = Counter(); ex = defaultdict(list)
    for r in rows:
        n1 = primary_number(r["a1"], r["country"]); k = number_relation(n1, r["a2"]); c[k] += 1
        if len(ex[k]) < 3 and k not in ("exact",): ex[k].append((r["a1"], r["a2"]))
    n = len(rows)
    R[label] = {k: (v, round(100 * v / n, 2)) for k, v in c.most_common()}
    R[label + "_examples"] = dict(ex)
relation_table(P, "D_number_relation_positives")
relation_table([r for r in P if r["country"] == "US"], "D_number_relation_positives_US")
relation_table([r for r in P if r["country"] == "India"], "D_number_relation_positives_India")
relation_table(H, "D_number_relation_hard_negatives")
relation_table(NM, "D_number_relation_near_miss_negatives")
diffs = []
for r in NM:
    if r["country"] != "US": continue
    n1, n2 = primary_number(r["a1"], "US"), primary_number(r["a2"], "US")
    if n1 and n2 and digits(n1) and digits(n2):
        diffs.append(abs(int(digits(n1)[:9]) - int(digits(n2)[:9])))
if diffs:
    d = np.array(diffs)
    R["D_near_miss_US_abs_number_difference"] = {"n": len(d), "pct_equal": round(100 * float((d == 0).mean()), 1), "pct_1_to_10": round(100 * float(((d >= 1) & (d <= 10)).mean()), 1),
                                                  "pct_11_to_100": round(100 * float(((d > 10) & (d <= 100)).mean()), 1), "pct_gt_100": round(100 * float((d > 100).mean()), 1)}
log("D done")

# ------------------------------------------------------------------ E. units, F. cities, G. landmarks, H. component survival, K. reordering
UNIT = re.compile(r"\b(unit|apt|apartment|suite|ste|pmb|po box|p o box|floor|flr|fl|bldg|building|room|rm|shop|office|flat|gala|wing|block|blk|tower|phase)\b|#")
LAND = re.compile(r"\b(near|nr|opp|opposite|behind|beside|next to|adjacent|above|below|in front of|b/h|bh|nearby)\b")
CITYNOISE = re.compile(r"\b(city|township|cdp|village of|town of|city of|county)\b")
unit_c = Counter(); city_c = Counter(); land_c = Counter(); surv = defaultdict(Counter); reorder = Counter(); tiny = Counter()
for r in P:
    a1, a2 = r["a1"], r["a2"]
    c1, c2 = ccomps(a1), ccomps(a2)
    t2 = set(toks(a2))
    u1 = [c for c in c1 if UNIT.search(c)]
    if u1:
        unit_c["S1 has unit"] += 1
        if any(UNIT.search(c) for c in c2): unit_c["match has a unit component"] += 1
        u1n = [t for c in u1 for t in c.split() if any(ch.isdigit() for ch in t) or (len(t) == 1 and t.isalpha())]
        if u1n and all(t in t2 for t in u1n): unit_c["unit value kept"] += 1
    if len(c1) >= 2:
        city = c1[-2]
        ct = city.split()
        if all(t in t2 for t in ct): city_c["S1 city tokens all present in match"] += 1
        elif any(t in t2 for t in ct): city_c["partly present"] += 1
        else: city_c["absent"] += 1
        if any(CITYNOISE.search(c) for c in c2) and not any(CITYNOISE.search(c) for c in c1): city_c["match adds city/township/cdp/county word"] += 1
    l1 = [c for c in c1 if LAND.search(c)]
    if l1:
        land_c["S1 has landmark"] += 1
        if any(LAND.search(c) for c in c2): land_c["match has landmark component"] += 1
        if any(len(set(c.split()) & t2) >= max(1, len(c.split()) // 2) for c in l1): land_c["landmark tokens mostly kept"] += 1
    for i, c in enumerate(c1):
        role = "last(state)" if i == len(c1) - 1 else "second last(city)" if i == len(c1) - 2 else "first" if i == 0 else "middle"
        ct = c.split(); kept = sum(t in t2 for t in ct) / max(len(ct), 1)
        surv[r["country"] + " " + role]["n"] += 1
        surv[r["country"] + " " + role]["kept>=50%"] += kept >= 0.5
        surv[r["country"] + " " + role]["kept=100%"] += kept == 1.0
    if c1 and c2:
        if sorted(c1) == sorted(c2) and c1 != c2: reorder["same components, different order"] += 1
        elif c1 == c2: reorder["identical components"] += 1
        else:
            s1c, s2c = set(c1), set(c2)
            if s1c <= s2c: reorder["S1 components all present, extra in match"] += 1
            elif s2c <= s1c: reorder["match components subset of S1 (drops)"] += 1
            else: reorder["components differ"] += 1
    for c in c2:
        if len(c) <= 2 and c.isalpha(): tiny["match tiny alpha component"] += 1
    for c in c1:
        if len(c) <= 2 and c.isalpha(): tiny["S1 tiny alpha component"] += 1
n = len(P)
R["E_units"] = {k: (v, round(100 * v / n, 2)) for k, v in unit_c.items()}
R["F_city"] = {k: (v, round(100 * v / n, 2)) for k, v in city_c.items()}
R["G_landmarks"] = {k: (v, round(100 * v / n, 2)) for k, v in land_c.items()}
R["H_component_survival_pct"] = {k: {"n": v["n"], "kept>=50%": round(100 * v["kept>=50%"] / v["n"], 1), "kept=100%": round(100 * v["kept=100%"] / v["n"], 1)} for k, v in surv.items()}
R["K_reordering"] = {k: (v, round(100 * v / n, 2)) for k, v in reorder.most_common()}
R["L_tiny_components_per_pair"] = {k: round(v / n, 4) for k, v in tiny.items()}
log("E-L done")

# ------------------------------------------------------------------ I. Indic script placement, J. punctuation, L2. split/concat words, M. postal codes
def indic_place(df, label):
    sub = df.filter((pl.col("country") == "India") & pl.col("business_address").is_not_null()).sample(n=200_000, seed=14)
    c = Counter()
    for a in sub["business_address"].to_list():
        cs = comps(a)
        if not cs: continue
        last_i = bool(INDIC.search(cs[-1])); other_i = any(INDIC.search(x) for x in cs[:-1])
        if last_i and not other_i: c["only last component (state) in Indic script"] += 1
        elif last_i and other_i: c["last and other components"] += 1
        elif other_i: c["only non-last components"] += 1
        else: c["no Indic"] += 1
        if all(INDIC.search(x) for x in cs): c["entire address Indic"] += 1
    R[label] = {k: round(100 * v / sub.height, 2) for k, v in c.most_common()}
indic_place(s2, "I_indic_placement_S2_India"); indic_place(s3, "I_indic_placement_S3_India")
def punct(df, label, n=300_000):
    c = Counter()
    for a in df.filter(pl.col("business_address").is_not_null())["business_address"].sample(n=n, seed=15).to_list():
        for ch in set(re.findall(r"[^\w\s\u0900-\u0DFF]", a)): c[ch] += 1
    R[label] = c.most_common(22)
punct(s1, "J_punct_S1"); punct(s23, "J_punct_S23")
def split_concat(df, label, n=300_000):
    c = Counter(); ex = defaultdict(list)
    rows = df.filter(pl.col("business_address").is_not_null()).sample(n=n, seed=16)
    for a, ctry in zip(rows["business_address"].to_list(), rows["country"].to_list()):
        cs = comps(a)
        if any(len(x) <= 2 and x.isalpha() for x in cs):
            c[f"{ctry}: component of 1-2 letters"] += 1
            if len(ex["tiny"]) < 5: ex["tiny"].append(a)
        if re.search(r"\b[a-z]{15,}\b", clean(a)):
            c[f"{ctry}: token of 15+ letters"] += 1
            if len(ex["long"]) < 5: ex["long"].append(a)
        if re.search(r"(?i)\b[cswd]/o\b", a): c[f"{ctry}: c/o s/o d/o w/o"] += 1
        if re.search(r"(?i)\bnull\b|<null>|\bn/?a\b|\bnone\b", a): c[f"{ctry}: NULL token"] += 1
        if re.search(r"(?i)\b(india|usa|united states)\b", a): c[f"{ctry}: country word inside address"] += 1
        if re.search(r"\b\d{6}\b", a): c[f"{ctry}: 6-digit token"] += 1
        if re.search(r"(?<!^)\b\d{5}\b", a) and not re.match(r"^\s*#?\d{5}\b", a): c[f"{ctry}: 5-digit token not at the start"] += 1
        if re.search(r"\b\d{5}\b\s*$", a): c[f"{ctry}: 5-digit token at the very end"] += 1
    R[label] = {k: round(100 * v / n, 3) for k, v in sorted(c.items())}
    R[label + "_examples"] = dict(ex)
split_concat(s1, "L_split_concat_postal_S1"); split_concat(s23, "L_split_concat_postal_S23")
log("I-M done")

# ------------------------------------------------------------------ N. France, unsupervised, from the test files
ft = pl.concat([pl.read_parquet(f"{PQ}/test_source{k}.parquet", columns=["business_address", "country"]).filter(pl.col("country") == "France").with_columns(pl.lit(f"S{k}").alias("src")) for k in (1, 2, 3)])
fr = {}
for (src,), sub in ft.filter(pl.col("business_address").is_not_null()).group_by("src"):
    rows = sub["business_address"].sample(n=min(150_000, sub.height), seed=17).to_list()
    last = Counter(comps(a)[-1] for a in rows if comps(a)); first_word = Counter(); n = len(rows); feat = Counter()
    for a in rows:
        c = clean(a); t = c.split()
        m = re.match(r"^(\d+)\s*(bis|ter|quater)?\s*([a-z.]+)", c)
        if m: first_word[m.group(3)] += 1
        if re.search(r"\b\d+\s*(bis|ter)\b", c): feat["bis/ter"] += 1
        if re.search(r"\bcedex\b", c): feat["cedex"] += 1
        if re.search(r"\b\d{5}\b", c): feat["5-digit token"] += 1
        if re.match(r"^\d", c): feat["starts with number"] += 1
        if re.search(r"\b(r|av|bd|all|imp|rte|ch|pl|sq|bvd)\b", c): feat["abbreviated street type"] += 1
        if re.search(r"\b(rue|avenue|boulevard|allee|impasse|route|chemin|place|quai|cours|square|voie|passage)\b", c): feat["full street type"] += 1
    fr[src] = {"n": n, "components_mean": round(float(sub["business_address"].str.count_matches(",").mean()) + 1, 2), "last_component_top": last.most_common(18),
               "street_type_after_number_top": first_word.most_common(15), "features_pct": {k: round(100 * v / n, 2) for k, v in feat.items()}}
R["N_France_test_unsupervised"] = fr
log("N done")

json.dump(R, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
for k, v in R.items():
    print("\n##", k); print(json.dumps(v, ensure_ascii=False, default=str)[:2500])
log("DONE")
