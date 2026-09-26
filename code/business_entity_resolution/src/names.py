"""Competition-specific business-name normalisation.

Every rule here was motivated by a measurement on the training data (eda/name_normalization.md,
eda/scripts/s10_name_analysis.py) and nothing is destroyed: each name is turned into several
representations that later stages can pick from.

    original        the raw string
    clean           casefolded, Latin accents stripped, Indic scripts kept intact, dotted acronyms joined,
                    brackets and punctuation removed, '&' -> 'and', apostrophes dropped
    tokens          clean.split() with digit-for-letter fixes (0/1/5/8/6 inside alphabetic tokens)
    legal_forms     canonical legal-form tokens found anywhere in the name (ltd, pvt, inc, corp, co, llc, ...)
    honorific       leading Mr/Smt/Shri/Dr/M/s ... (Indian S2/S3 add these to 8% of names)
    designations    trailing comma-separated professional designations (DDS, MD, PC, ...)
    alias_prefix    the fake former name in 'X fka Y', 'X d.b.a. Y', 'X formerly known as Y' (Y is the name)
    core_tokens     tokens without forms, honorific, designations, alias prefix, and consecutive repeats
    core            ' '.join(core_tokens);  core_sorted: order-insensitive
    compact         core without spaces (domain-style names: 'valuerealtydigital com' vs 'value realty digital')
    acronym         initials of core tokens ('Adroit Advisors Group' -> 'aag'; S3 writes 0.24% of names this way)
    translit_core   core with Indic tokens replaced by Latin: learned dictionary first, rule-based fallback
    variants        alternative cores: alias prefix, domain label, name without an appended generic word
    digits          numeric tokens (kept verbatim; 92% of S1 numbers survive in true matches)
    script, flags   'latin' / 'devanagari' / ... / 'mixed'; all_caps, domain_style, has_alias, ...

Nothing here reads test data; the transliteration dictionary is learned from training pairs only
(``learn_translit_dict``) and must be learned without the validation fold when scoring that fold.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translit import INDIC_RE, script_of, transliterate_indic  # noqa: E402

# --------------------------------------------------------------------------- #
# Vocabularies (all observed in the data; counts in the report)
# --------------------------------------------------------------------------- #
LEGAL_FORM_CANON: Dict[str, str] = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc", "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt", "opc": "opc",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "scp": "scp",
    "ei": "ei", "eirl": "eirl", "sel": "sel", "selarl": "selarl", "cie": "cie", "ets": "ets", "etablissements": "ets",
    "ste": "ste", "societe": "ste", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}
# transliterated Indic legal-form tokens are matched fuzzily against these targets
INDIC_FORM_TARGETS = {"limited": "ltd", "private": "pvt", "llp": "llp", "pvt": "pvt", "ltd": "ltd"}
INDIC_FORM_SHORT = {"pra": "pvt", "li": "ltd", "prai": "pvt", "lim": "ltd"}   # 'प्रा.' 'लि.' abbreviations
HONORIFICS = {"mr", "mrs", "ms", "smt", "shri", "shree", "sri", "dr", "messrs"}
DESIGNATIONS = {"dds", "md", "do", "dpm", "lcsw", "phd", "pa", "dc", "dmd", "cpa", "esq", "od", "dvm", "rn", "mba",
                "cfp", "lpc", "pt", "ot", "np", "lmft", "psyd", "aprn", "dnp", "facs", "jd"}
GENERIC_APPENDED = {"center", "centre", "services", "service", "partners", "group", "holdings"}
SPELLING_VARIANTS = {"centre": "center"}      # the only non-legal abbreviation/spelling pair with real support (86 cases)
# Latin spellings of one Indian word; applied only in the transliteration representation so `core` stays faithful
INDIAN_SPELLINGS = {"shree": "shri", "sree": "shri", "sri": "shri", "laxmi": "lakshmi", "jay": "jai", "jey": "jai",
                    "shiva": "shiv", "shib": "shiv"}
DIGIT_FIX = {"0": "o", "1": "l", "5": "s", "8": "b", "6": "g"}   # 99% of digit-for-letter substitutions
TLDS = "com|net|org|in|fr|co|io|biz|c0m|co\\.in|co\\.uk|info|us|eu"

ALIAS_RE = re.compile(
    r"(?:^|\s)(d\.?\s?b\.?\s?a\.?|doing business as|a\.?\s?k\.?\s?a\.?|f\.?\s?k\.?\s?a\.?|formerly(?:\s+known\s+as)?\s*:?|"
    r"t\s?/\s?a|trading as|now known as|nee)(?=\s|$)")
PHONE_TAIL_RE = re.compile(r"\s*[-–|]\s*\+?\d[\d\s\-]{6,}\s*$")
DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b")
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*?)\.(" + TLDS + r")$")
LATIN_MARKS_RE = re.compile(r"[̀-ͯ]+")
WORD_RE = re.compile(r"[^\wऀ-෿]+")
ORDINAL_RE = re.compile(r"^\d+(st|nd|rd|th)$")


@dataclass
class NameRep:
    original: str
    clean: str
    tokens: List[str]
    legal_forms: List[str]
    honorific: Optional[str]
    designations: List[str]
    alias_prefix: Optional[str]
    core_tokens: List[str]
    core: str
    core_sorted: str
    compact: str
    acronym: str
    translit_tokens: List[str]
    translit_core: str
    translit_core_sorted: str
    variants: List[str]
    digits: List[str]
    script: str
    flags: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Character level helpers
# --------------------------------------------------------------------------- #
_INDIC_SPLIT_RE = re.compile(r"([ऀ-෿]+)")


def strip_latin_accents(text: str) -> str:
    """Drop Latin accents without touching Indic text.

    Non-Indic runs get NFKD and lose the Latin combining-diacritics block.  Indic runs get NFC only:
    NFKD would decompose composed vowel signs (Telugu AI -> E + length mark, Bengali O -> E + AA, ...)
    and the transliteration tables expect the composed characters.
    """
    parts = _INDIC_SPLIT_RE.split(text)
    for i, seg in enumerate(parts):
        if i % 2 == 1:
            parts[i] = unicodedata.normalize("NFC", seg)
        elif seg:
            parts[i] = LATIN_MARKS_RE.sub("", unicodedata.normalize("NFKD", seg))
    return "".join(parts)


def char_ngrams(text: str, n: int = 3) -> List[str]:
    s = text.replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def _fix_digits(tok: str) -> str:
    """'capita1' -> 'capital', 'de1hi' -> 'delhi'; ordinals, '24hr', 'b2b' untouched."""
    if not any(c.isdigit() for c in tok) or ORDINAL_RE.match(tok):
        return tok
    letters = sum(c.isalpha() for c in tok)
    digits = [c for c in tok if c.isdigit()]
    if letters >= 3 and len(digits) <= 2 and all(d in DIGIT_FIX for d in digits):
        return "".join(DIGIT_FIX.get(c, c) for c in tok)
    return tok


def _join_dotted_acronyms(text: str) -> str:
    return DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), text)


def _basic_clean(text: str) -> str:
    """casefold, Latin accents stripped, dotted acronyms joined, '&' -> and, apostrophes dropped, punctuation -> space."""
    t = strip_latin_accents(unicodedata.normalize("NFC", text)).casefold()
    t = "".join(c for c in t if unicodedata.category(c) != "Cf")     # zero-width joiners etc.
    t = _join_dotted_acronyms(t)
    t = t.replace("&", " and ").replace("'", "").replace("’", "")
    t = WORD_RE.sub(" ", t).replace("_", " ")
    return " ".join(t.split())


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def normalize_name(raw: str, translit_dict: Optional[Dict[str, str]] = None) -> NameRep:
    original = raw if raw is not None else ""
    flags: Dict[str, bool] = {}
    text = " ".join(original.split())
    letters = [c for c in text if c.isalpha()]
    flags["all_caps"] = bool(letters) and all(c.isupper() for c in letters if c.isascii())
    flags["mixed_script"] = script_of(text) == "mixed"

    # 1. trailing phone number, '| www.x.com' style URL segments
    variants: List[str] = []
    m = PHONE_TAIL_RE.search(text)
    if m:
        text = text[:m.start()]
        flags["phone_stripped"] = True
    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        keep = []
        for p in parts:
            dm = DOMAIN_RE.match(_fix_digits(p.casefold().replace(" ", "")))
            if dm and keep:
                variants.append(dm.group(1).replace(".", " ").replace("-", " "))
                flags["url_segment"] = True
            else:
                keep.append(p)
        text = " ".join(keep) if keep else parts[0]

    # 2. alias markers: 'Fake Name fka Real Name' -> Real Name is the name, Fake Name a variant
    alias_prefix = None
    low = " ".join(text.casefold().split())
    am = ALIAS_RE.search(low)
    # a marker needs a name on both sides: 'Aka Associates' and 'Nee Corp' are ordinary names
    if am and am.start() > 0 and low[:am.start()].strip() and low[am.end():].strip():
        alias_prefix = low[:am.start()].strip()
        text = low[am.end():].strip()
        flags["has_alias"] = True

    # 3. trailing comma-separated designations / legal forms ('Reed, Edyth N., D.D.S.', 'Monarch, Inc')
    designations: List[str] = []
    segs = [s.strip() for s in text.split(",")]
    while len(segs) > 1:
        toks = _basic_clean(segs[-1]).split()
        if toks and all(t in DESIGNATIONS or t in LEGAL_FORM_CANON for t in toks):
            designations = [t for t in toks if t in DESIGNATIONS] + designations
            text = ", ".join(segs[:-1]) + (" " + " ".join(t for t in toks if t in LEGAL_FORM_CANON) if any(t in LEGAL_FORM_CANON for t in toks) else "")
            segs = [s.strip() for s in text.split(",")]
        else:
            break

    # 4. domain-style names: 'Helios.Com', 'CLAIRSPROTT.COM', '#KOKANINDIA', 'pacificmaterialsdynamics.c0m'
    domain_label = None
    squashed = _fix_digits(text.casefold().replace(" ", ""))
    dm = DOMAIN_RE.match(squashed)
    if dm:
        domain_label = dm.group(1).replace(".", " ").replace("-", " ")
        flags["domain_style"] = True
    elif re.fullmatch(r"#\S+", text.strip()):
        domain_label = text.strip()[1:].casefold()
        flags["domain_style"] = True

    # 5. clean string and tokens
    clean = _basic_clean(domain_label if domain_label else text)
    raw_tokens = clean.split()
    tokens = [_fix_digits(t) for t in raw_tokens]
    if tokens != raw_tokens:
        flags["digit_fixed"] = True
    spelled = [SPELLING_VARIANTS.get(t, t) for t in tokens]
    if spelled != tokens:
        flags["spelling_variant"] = True
    tokens = spelled

    # 6. transliteration of Indic tokens (dictionary, then rules); parallel list to tokens
    translit_tokens: List[str] = []
    for idx, t in enumerate(tokens):
        if INDIC_RE.search(t):
            tr = (translit_dict or {}).get(t)
            if tr is None:
                tr = _basic_clean(transliterate_indic(t)).replace(" ", "") or t
                tr = INDIC_FORM_SHORT.get(tr, tr)
                if tr not in LEGAL_FORM_CANON:
                    # legal forms sit in the last two positions ('... प्राइवेट लिमिटेड'); the rule
                    # transliteration lands within edit distance 3 of the Latin spelling ('praivet')
                    near_end = idx >= len(tokens) - 2
                    for target, canon in INDIC_FORM_TARGETS.items():
                        if len(target) < 5:
                            continue
                        d = Levenshtein.distance(tr, target)
                        if d <= 2 or (near_end and d <= 3 and tr[:1] == target[:1] and len(tr) >= 6):
                            tr = canon
                            break
            translit_tokens.append(INDIAN_SPELLINGS.get(tr, tr))
        else:
            translit_tokens.append(INDIAN_SPELLINGS.get(t, t))

    # 7. legal forms (recognised on the transliterated tokens), honorific, repeats
    legal_forms = [LEGAL_FORM_CANON[t] for t in translit_tokens if t in LEGAL_FORM_CANON]
    keep = [i for i, t in enumerate(translit_tokens) if t not in LEGAL_FORM_CANON]
    honorific = None
    if len(keep) >= 3 and translit_tokens[keep[0]] in HONORIFICS:
        honorific = translit_tokens[keep[0]]
        keep = keep[1:]
    elif len(keep) >= 4 and translit_tokens[keep[0]] == "m" and translit_tokens[keep[1]] == "s":
        honorific = "ms"
        keep = keep[2:]
    core_idx: List[int] = []
    for i in keep:
        if core_idx and len(translit_tokens[i]) >= 2 and translit_tokens[i] == translit_tokens[core_idx[-1]]:
            flags["repeat_removed"] = True
            continue
        core_idx.append(i)
    core_tokens = [tokens[i] for i in core_idx]
    tr_core_tokens = [translit_tokens[i] for i in core_idx]
    if not core_tokens:                                  # a name made only of forms/honorifics: keep everything
        core_tokens, tr_core_tokens = tokens[:], translit_tokens[:]

    core = " ".join(core_tokens)
    translit_core = " ".join(tr_core_tokens)
    if alias_prefix:
        ap = " ".join(t for t in _basic_clean(alias_prefix).split() if t not in LEGAL_FORM_CANON)
        if ap:
            variants.append(ap)
    if len(tr_core_tokens) >= 2 and tr_core_tokens[-1] in GENERIC_APPENDED:
        variants.append(" ".join(tr_core_tokens[:-1]))
    variants = list(dict.fromkeys(v for v in variants if v and v != translit_core))
    flags["is_acronym"] = len(core_tokens) == 1 and 2 <= len(core_tokens[0]) <= 4 and core_tokens[0].isalpha() and flags.get("all_caps", False)

    return NameRep(
        original=original, clean=clean, tokens=tokens, legal_forms=legal_forms, honorific=honorific,
        designations=designations, alias_prefix=alias_prefix, core_tokens=core_tokens, core=core,
        core_sorted=" ".join(sorted(core_tokens)), compact=core.replace(" ", ""),
        acronym="".join(t[0] for t in tr_core_tokens) if len(tr_core_tokens) >= 2 else "",
        translit_tokens=translit_tokens, translit_core=translit_core, translit_core_sorted=" ".join(sorted(tr_core_tokens)),
        variants=variants, digits=[t for t in core_tokens if t.isdigit()], script=script_of(text), flags=flags,
    )


def normalize_names(names: Iterable[Optional[str]], translit_dict: Optional[Dict[str, str]] = None) -> List[NameRep]:
    return [normalize_name(n or "", translit_dict) for n in names]


# --------------------------------------------------------------------------- #
# Transliteration dictionary learned from training pairs
# --------------------------------------------------------------------------- #
def learn_translit_dict(latin_names: Iterable[str], indic_names: Iterable[str], min_count: int = 2, min_purity: float = 0.6) -> Dict[str, str]:
    """Indic token -> Latin token, from aligned true-match pairs (Source 1 Latin name, S2/S3 Indic name).

    Equal token counts align by position (transliterated names keep word order); otherwise each Indic token
    is aligned to the most similar Latin token under the rule-based transliteration.  Latin targets are the
    canonical tokens, so legal forms come out as ltd / pvt / llp.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for lat, ind in zip(latin_names, indic_names):
        if not lat or not ind or not INDIC_RE.search(ind):
            continue
        lt = [LEGAL_FORM_CANON.get(t, t) for t in (_fix_digits(x) for x in _basic_clean(lat).split())]
        it = _basic_clean(ind).split()
        it_idx = [i for i, t in enumerate(it) if INDIC_RE.search(t)]
        if not lt or not it_idx:
            continue
        if len(it) == len(lt):
            pairs = [(it[i], lt[i]) for i in it_idx]
        else:
            rt = {i: _basic_clean(transliterate_indic(it[i])).replace(" ", "") for i in it_idx}
            pairs = []
            for i in it_idx:
                best, score = None, 0.0
                for cand in lt:
                    s = Levenshtein.normalized_similarity(rt[i], cand)
                    if s > score:
                        best, score = cand, s
                if best is not None and score >= 0.45:
                    pairs.append((it[i], best))
        for a, b in pairs:
            counts.setdefault(a, {})
            counts[a][b] = counts[a].get(b, 0) + 1
    out: Dict[str, str] = {}
    for a, c in counts.items():
        total = sum(c.values())
        b, n = max(c.items(), key=lambda kv: kv[1])
        if n >= min_count and n / total >= min_purity:
            out[a] = b
    return out


def save_translit_dict(d: Dict[str, str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=0)


def load_translit_dict(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
