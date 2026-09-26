"""Structured business-address representation, learned vocabularies, no external resources.

Every address is parsed into the fields the task asks for (house_number, street, unit, city, state,
postal_code, country, landmark) plus extras that the data made necessary (locality, po_box, care_of,
numbers, tokens) while the original string is kept.  Evidence for each rule: eda/address_analysis.md.

Vocabularies are learned from the training files by ``build_vocab`` (state names, codes and scripts
from Source 1 and true pairs; city names from Source 1; street-type abbreviations from true pairs plus
generic forms) and saved as JSON.  For a country absent from training the parser still works: states
and cities can be learned from that split's Source 1 file without labels (``extend_from_source1``),
and positional fallbacks apply otherwise.  Nothing here calls any external service.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from names import strip_latin_accents  # noqa: E402
from translit import INDIC_RE, transliterate_indic  # noqa: E402

# --------------------------------------------------------------------------- #
# Vocabularies observed in the data
# --------------------------------------------------------------------------- #
NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bnone\b|\bn\s*/\s*a\b|\bnan\b")
UNIT_KW = {"unit", "apt", "apartment", "suite", "ste", "pmb", "room", "rm", "flat", "floor", "flr", "fl", "shop", "office",
           "gala", "wing", "bldg", "building", "tower", "cabin", "chamber", "basement", "mezzanine", "penthouse", "lot", "space", "spc"}
UNIT_RE = re.compile(r"\b(unit|apt|apartment|suite|ste|pmb|room|rm|flat|floor|flr|fl|shop|office|gala|wing|bldg|building|tower|cabin|chamber|basement|lot|space|spc)\b\.?\s*(?:no\.?|number|#)?\s*[:.-]?\s*([\w/-]+)?")
ORD_FLOOR_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\s+(floor|flr|fl)\b|\b(ground|first|second|third|fourth|fifth|top|basement|mezzanine)\s+(floor|flr|fl)\b")
HASH_UNIT_RE = re.compile(r"(?<!\S)#+\s*([\w/-]*\d[\w/-]*)")
PO_RE = re.compile(r"\b(?:p\.?\s?o\.?\s?box|post box|post office box|pmb)\s*#?\s*([\w-]+)?")
CARE_RE = re.compile(r"\b([cswd])\s*/\s*o\b\.?\s*(.*)")
LANDMARK_RE = re.compile(r"^(near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adjacent|above|below|in front of|front of|b/h|bh|back side of|backside of|nearby)\b\.?\s*(.*)")
CITY_NOISE_RE = re.compile(r"\b(city of|town of|village of|township of|city|town|township|cdp|village|borough|county|municipality|taluka|tehsil|district|dist|distt|suburban)\b")
COUNTRY_WORDS = {"india", "usa", "u s a", "united states", "france", "us"}
INDIA_NUM_KW = ["plot no", "plot", "h no", "hno", "house no", "house", "door no", "d no", "dno", "door", "sy no", "survey no", "survey", "s no", "sno", "kh no",
                "khasra no", "khasra", "gala no", "shop no", "flat no", "office no", "site no", "site", "unit no", "room no", "gat no", "gat", "no", "old no", "new no"]
INDIA_NUM_RE = re.compile(r"(?:^|\s)(?:" + "|".join(re.escape(k) for k in sorted(INDIA_NUM_KW, key=len, reverse=True)) + r")\.?\s*[:#-]?\s*([\w/-]*\d[\w/-]*)")
LEADING_NUM_RE = re.compile(r"^#*\s*(\d[\w/-]*(?:\s+\d/\d)?)\s*(.*)$")
POSTAL_RE = {"US": re.compile(r"(?<!\S)(\d{5})(?:-\d{4})?(?!\S)"), "India": re.compile(r"(?<!\S)(\d{6})(?!\S)"), "France": re.compile(r"(?<!\S)(\d{5})(?!\S)")}
FR_BIS_RE = re.compile(r"^(\d+)\s*(bis|ter|quater)\b")

# canonical street type -> abbreviations (US/India pairs learned from true matches; French forms generic)
STREET_TYPES: Dict[str, List[str]] = {
    "street": ["st", "str", "strt"], "road": ["rd"], "drive": ["dr", "drv"], "avenue": ["ave", "av", "aven", "avn"], "lane": ["ln"],
    "court": ["ct", "crt"], "circle": ["cir", "circ"], "place": ["pl"], "boulevard": ["blvd", "bd", "bvd", "boul"], "trail": ["trl"],
    "parkway": ["pkwy", "pky"], "terrace": ["ter", "terr"], "highway": ["hwy"], "way": ["wy"], "square": ["sq"], "cove": ["cv"],
    "ridge": ["rdg"], "point": ["pt"], "alley": ["aly"], "plaza": ["plz"], "expressway": ["expy"], "freeway": ["fwy"], "turnpike": ["tpke"],
    "crossing": ["xing"], "extension": ["ext"], "heights": ["hts"], "junction": ["jct"], "loop": [], "run": [], "path": [], "pike": [],
    "marg": [], "nagar": ["ngr"], "colony": ["col"], "sector": ["sec"], "phase": ["ph"], "block": ["blk"], "main": [], "cross": ["crs"],
    "layout": [], "gali": [], "chowk": [], "bazar": ["bazaar"], "mandi": [], "enclave": ["encl"], "vihar": [], "puram": [], "peth": [],
    "rue": ["r"], "allee": ["all"], "impasse": ["imp"], "route": ["rte"], "chemin": ["ch", "chem"], "quai": [], "cours": [], "passage": ["pass"],
    "faubourg": ["fbg"], "promenade": ["prom"], "esplanade": ["espl"],
}
ABBR_TO_TYPE: Dict[str, str] = {a: c for c, al in STREET_TYPES.items() for a in al}
ABBR_TO_TYPE.update({c: c for c in STREET_TYPES})
DIRECTIONS = {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
              "north": "north", "south": "south", "east": "east", "west": "west"}
WORD_RE = re.compile(r"[^\wऀ-෿/]+")


def clean_text(text: str) -> str:
    """casefold, Latin accents stripped, Indic kept, NULL tokens removed, punctuation to space except '/' and '-' inside tokens."""
    t = strip_latin_accents(unicodedata.normalize("NFC", text or "")).casefold()
    t = "".join(c for c in t if unicodedata.category(c) != "Cf")
    t = NULL_RE.sub(" ", t)
    t = re.sub(r"(?<=\w)-(?=\w)", "-", t)                  # keep intra-token hyphens
    t = re.sub(r"[^\wऀ-෿/\-#,]+", " ", t).replace("_", " ")
    t = re.sub(r"\s*-\s*", "-", t)                         # ' - ' inside numbers
    t = re.sub(r"(?<!\w)-|-(?!\w)", " ", t)                # dangling hyphens
    return ", ".join(" ".join(c.split()) for c in t.split(",") if c.strip())


def norm_number(tok: str) -> str:
    """'001419' -> '1419', '17814.' -> '17814', '#1105' -> '1105'; compound numbers keep their shape ('18/1', '1-98/5/2a')."""
    t = tok.strip().lstrip("#").rstrip(".").strip()
    t = re.sub(r"^0+(?=\d)", "", t)
    return t


@dataclass
class AddressRep:
    original: str
    country: str
    house_number: Optional[str]
    street: Optional[str]               # street name without number, unit and type
    street_type: Optional[str]          # canonical type: street, road, avenue, marg, rue ...
    unit: Optional[str]
    city: Optional[str]                 # raw city component (cleaned)
    city_norm: Optional[str]            # city without noise words, 'st' -> 'saint'
    state: Optional[str]                # canonical state as Source 1 writes it, or the raw admin component
    state_recognised: bool
    postal_code: Optional[str]
    landmark: List[str]
    locality: List[str]                 # remaining components (India: nagar/colony/sector/village ...)
    po_box: Optional[str]
    care_of: Optional[str]
    numbers: List[str]                  # every numeric token, normalised ('d/1200', '18/1', '1419')
    digit_runs: List[str]               # every run of digits, leading zeros stripped ('1200', '18', '1')
    tokens: List[str]                   # all cleaned tokens (transliterated where Indic), for set similarities
    components: List[str]
    missing: bool
    flags: Dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Learned vocabularies
# --------------------------------------------------------------------------- #
class AddressVocab:
    def __init__(self) -> None:
        self.states: Dict[str, Dict[str, str]] = defaultdict(dict)       # country -> alias -> canonical
        self.compatible: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))  # canonical -> canonicals seen as its match
        self.cities: Dict[str, Set[str]] = defaultdict(set)              # country -> normalised city names
        self.city_tokens: Dict[str, Set[str]] = defaultdict(set)

    # ---- learning
    def add_source1(self, addresses: Iterable[Optional[str]], countries: Iterable[str], min_state: int = 20, min_city: int = 3,
                    last_to_prev_ratio: float = 3.0) -> None:
        """States are values that end an address far more often than they precede the end.

        Source 1 stores 14% of US addresses reordered, so frequent cities also appear as last components;
        a real state is last in 86% of its occurrences while a city is last in about 14%, hence the ratio test.
        """
        last = Counter(); prev = Counter(); anywhere = Counter()
        for a, c in zip(addresses, countries):
            if not a:
                continue
            comps = [x.strip() for x in clean_text(a).split(",") if x.strip()]
            if len(comps) >= 2:
                last[(c, comps[-1])] += 1
                prev[(c, comps[-2])] += 1
                for x in comps[:-2]:
                    anywhere[(c, x)] += 1
        for (c, s), n in last.items():
            others = prev.get((c, s), 0) + anywhere.get((c, s), 0)
            if n >= min_state and not re.search(r"\d", s) and n >= last_to_prev_ratio * max(others, 1):
                self.states[c][s] = s
        for (c, ci), n in prev.items():
            if n >= min_city and not re.search(r"\d", ci) and ci not in self.states[c]:
                self.cities[c].add(_city_norm(ci))
                for t in _city_norm(ci).split():
                    self.city_tokens[c].add(t)

    def add_pairs(self, addr1: Iterable[Optional[str]], addr2: Iterable[Optional[str]], countries: Iterable[str],
                  min_share: float = 0.02, min_count: int = 20, compat_share: float = 0.05) -> None:
        co = defaultdict(Counter)
        for a1, a2, c in zip(addr1, addr2, countries):
            if not a1 or not a2:
                continue
            c1 = [x.strip() for x in clean_text(a1).split(",") if x.strip()]
            c2 = [x.strip() for x in clean_text(a2).split(",") if x.strip()]
            if not c1 or not c2:
                continue
            canon = self.states[c].get(c1[-1])
            if canon:
                co[(c, canon)][c2[-1]] += 1
        for (c, canon), cnt in co.items():
            tot = sum(cnt.values())
            for alias, n in cnt.items():
                if n < min_count or n / tot < min_share or re.search(r"\d", alias):
                    continue
                if alias in self.states[c] and self.states[c][alias] != canon:
                    if n / tot >= compat_share:
                        self.compatible[c][canon].add(self.states[c][alias])
                        self.compatible[c][self.states[c][alias]].add(canon)
                    continue
                # a city that happens to end matches of this state has a small share; a state's own full name
                # has a large one ('texas' is also the city vocabulary's 'Texas City', hence the share test)
                if n / tot < 0.2 and _city_norm(alias) in self.cities[c]:
                    continue
                self.states[c].setdefault(alias, canon)
        # transliterated forms of Indic aliases so that mixed spellings still resolve
        for c in list(self.states):
            for alias, canon in list(self.states[c].items()):
                if INDIC_RE.search(alias):
                    tr = " ".join(transliterate_indic(alias).split())
                    self.states[c].setdefault(tr, canon)

    def extend_from_source1(self, addresses: Iterable[Optional[str]], countries: Iterable[str]) -> None:
        """Unsupervised: states and cities of a split's Source 1 file (for countries absent from training)."""
        self.add_source1(addresses, countries)

    # ---- persistence
    def save(self, path: str) -> None:
        json.dump({"states": {c: d for c, d in self.states.items()},
                   "compatible": {c: {k: sorted(v) for k, v in d.items()} for c, d in self.compatible.items()},
                   "cities": {c: sorted(v) for c, v in self.cities.items()}}, open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "AddressVocab":
        v = cls()
        d = json.load(open(path, encoding="utf-8"))
        for c, m in d["states"].items():
            v.states[c].update(m)
        for c, m in d.get("compatible", {}).items():
            for k, lst in m.items():
                v.compatible[c][k] = set(lst)
        for c, lst in d["cities"].items():
            v.cities[c] = set(lst)
            for ci in lst:
                for t in ci.split():
                    v.city_tokens[c].add(t)
        return v

    # ---- lookup
    def state_of(self, comp: str, country: str) -> Optional[str]:
        m = self.states.get(country, {})
        if comp in m:
            return m[comp]
        if INDIC_RE.search(comp):
            tr = " ".join(transliterate_indic(comp).split())
            if tr in m:
                return m[tr]
        return None

    def states_compatible(self, a: str, b: str, country: str) -> bool:
        return a == b or b in self.compatible.get(country, {}).get(a, set())


def _city_norm(comp: str) -> str:
    t = CITY_NOISE_RE.sub(" ", comp)
    t = re.sub(r"\bst\b", "saint", t)
    t = re.sub(r"\bft\b", "fort", t)
    t = re.sub(r"\bmt\b", "mount", t)
    return " ".join(t.split())


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _split_components(text: str) -> List[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


FLOOR_KW = {"floor", "flr", "fl", "basement", "penthouse", "mezzanine"}
UNIT_FILLER = {"no", "number", "#", "num"}


def _extract_unit(comp: str) -> Tuple[str, Optional[str]]:
    """Return (component without the unit phrase, unit value or keyword).

    'unit apartment g' -> 'unit g', 'flat no 101' -> 'flat 101', '3rd floor' -> '3 floor', '# 3' -> '# 3'.
    """
    m = ORD_FLOOR_RE.search(comp)
    if m:
        val = (m.group(1) or m.group(3)) + " floor"
        return " ".join((comp[:m.start()] + " " + comp[m.end():]).split()), val
    toks = comp.split()
    for k, t in enumerate(toks):
        if t in UNIT_KW:
            val = None; consumed = 0
            for j, nxt in enumerate(toks[k + 1:k + 4]):
                if nxt in UNIT_KW or nxt in UNIT_FILLER:
                    consumed = j + 1
                    continue
                if any(ch.isdigit() for ch in nxt) or (len(nxt) == 1 and nxt.isalpha()):
                    val, consumed = nxt, j + 1
                break
            if val is not None:
                return " ".join(toks[:k] + toks[k + 1 + consumed:]), f"{t} {norm_number(val)}"
            if t in FLOOR_KW:
                return " ".join(toks[:k] + toks[k + 1:]), t
            break
    m = HASH_UNIT_RE.search(comp)
    if m and not comp.startswith("#"):
        return " ".join((comp[:m.start()] + " " + comp[m.end():]).split()), "# " + norm_number(m.group(1))
    return comp, None


def _street_split(comp: str, country: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (house_number, street_name, street_type) from a street component; None where absent."""
    c = comp.strip()
    number = None
    # '28th main', '35th cross', '5th avenue': an ordinal followed by a street word is the street name
    if re.match(r"^\d+(?:st|nd|rd|th)\s+(?:main|cross|avenue|ave|street|st|road|rd)\b", c):
        toks = c.split()
        # 'main' and 'cross' identify Indian streets ('28th main', '35th cross') and stay in the name
        stype = ABBR_TO_TYPE.get(toks[-1]) if len(toks) >= 2 and toks[-1] in ABBR_TO_TYPE and toks[-1] not in ("main", "cross") else None
        name = " ".join(toks[:-1] if stype else toks)
        return None, name, stype
    if country != "India":
        m = LEADING_NUM_RE.match(c)
        if m:
            number, c = norm_number(m.group(1)), m.group(2)
            fb = FR_BIS_RE.match(number + " " + c)
            if fb:
                number = f"{fb.group(1)} {fb.group(2)}"
                c = c[len(fb.group(2)):].strip() if c.startswith(fb.group(2)) else c
    else:
        m = INDIA_NUM_RE.search(" " + c)
        if m:
            number = norm_number(m.group(1))
            c = (c[:max(m.start() - 1, 0)] + " " + c[m.end() - 1:]).strip()
            c = re.sub(INDIA_NUM_RE.pattern.split("(?:")[0], "", c) if False else c
        else:
            m = LEADING_NUM_RE.match(c)
            if m:
                number, c = norm_number(m.group(1)), m.group(2)
    toks = [t for t in c.split() if t]
    # drop the India number keyword remnants ('plot', 'no', 'h', 'door')
    toks = [t for t in toks if t not in ("no", "plot", "h", "hno", "door", "house", "sy", "survey", "kh", "khasra", "gala", "site", "gat", "flat", "shop", "office", "s", "d", "dno", "sno")] if country == "India" else toks
    stype = None
    remainder: List[str] = []
    if toks:
        if toks[-1] in ABBR_TO_TYPE and len(toks) >= 2:
            stype = ABBR_TO_TYPE[toks[-1]]
            if toks[-1] == "st":
                stype = "street"
            toks = toks[:-1]
        elif toks[0] in ABBR_TO_TYPE and country == "France":
            stype = ABBR_TO_TYPE[toks[0]]
            toks = toks[1:]
        elif len(toks) >= 2 and toks[-2] in ABBR_TO_TYPE and toks[-1] in DIRECTIONS:   # 'main st n'
            stype = ABBR_TO_TYPE[toks[-2]]
            toks = toks[:-2] + [DIRECTIONS[toks[-1]]]
        else:
            k = _mid_type_index(toks)
            if k is not None:                                   # 'chincholi bunder road malad west'
                stype, remainder, toks = ABBR_TO_TYPE[toks[k]], toks[k + 1:], toks[:k]
    if toks and toks[0] == "st":
        toks[0] = "saint"
    toks = [DIRECTIONS.get(t, t) if len(t) <= 2 else t for t in toks]
    name = " ".join(toks) if toks else None
    _street_split.remainder = remainder
    return number, name, stype


STRONG_TYPES = {"street", "road", "drive", "avenue", "lane", "court", "circle", "place", "boulevard", "trail", "parkway", "terrace",
                "highway", "way", "alley", "expressway", "freeway", "turnpike", "loop", "run", "path", "pike", "marg", "main", "cross",
                "gali", "rue", "allee", "impasse", "route", "chemin", "quai", "cours", "passage", "faubourg", "promenade", "esplanade"}


def _street_strength(comp: str, country: str) -> int:
    """2: numbered street or a strong street type; 1: building/locality type (square, nagar, sector ...); 0: not a street."""
    toks = comp.split()
    if not toks:
        return 0
    if country != "India" and re.match(r"^#?\d", comp):
        return 2
    types = [ABBR_TO_TYPE[t] for t in toks if t in ABBR_TO_TYPE]
    if country == "India" and INDIA_NUM_RE.search(" " + comp):
        return 2
    if types and any(ch.isdigit() for ch in comp):
        return 2 if any(t in STRONG_TYPES for t in types) else 1
    if len(toks) >= 2 and toks[-1] in ABBR_TO_TYPE:
        return 2 if ABBR_TO_TYPE[toks[-1]] in STRONG_TYPES else 1
    if country == "France" and toks[0] in ABBR_TO_TYPE and ABBR_TO_TYPE[toks[0]] in STRONG_TYPES:
        return 2
    if _mid_type_index(toks) is not None:      # 'chincholi bunder road malad west'
        return 2
    return 0


def _mid_type_index(toks: List[str]) -> Optional[int]:
    """Index of an unambiguous strong street type inside a component (full words only, not first or last)."""
    for k in range(1, len(toks) - 1):
        t = toks[k]
        if t in ABBR_TO_TYPE and ABBR_TO_TYPE[t] in STRONG_TYPES and len(t) >= 4 and t not in ("main", "cross"):
            return k
    return None


def _translit_tokens(text: str) -> List[str]:
    out = []
    for t in text.split():
        if INDIC_RE.search(t):
            tr = "".join(transliterate_indic(t).split())
            out.append(tr or t)
        else:
            out.append(t)
    return out


def parse_address(raw: Optional[str], country: str, vocab: Optional[AddressVocab] = None) -> AddressRep:
    original = raw if raw is not None else ""
    empty = AddressRep(original=original, country=country, house_number=None, street=None, street_type=None, unit=None, city=None,
                       city_norm=None, state=None, state_recognised=False, postal_code=None, landmark=[], locality=[], po_box=None,
                       care_of=None, numbers=[], digit_runs=[], tokens=[], components=[], missing=True, flags={})
    if not original.strip():
        return empty
    flags: Dict[str, bool] = {}
    text = clean_text(original)
    if not text:
        return empty
    if NULL_RE.search(original.casefold()):
        flags["null_token"] = True
    comps = _split_components(text)
    # country words inside the address
    comps = [c for c in comps if c not in COUNTRY_WORDS] or comps
    # PO box / PMB
    po_box = None
    m = PO_RE.search(text)
    if m:
        po_box = norm_number(m.group(1)) if m.group(1) else "po box"
        comps = [" ".join(PO_RE.sub(" ", c).split()) for c in comps]
        comps = [c for c in comps if c]
        flags["po_box"] = True
    # care of
    care_of = None
    kept = []
    for c in comps:
        m = CARE_RE.match(c)
        if m and m.group(2):
            care_of = m.group(2).strip(); flags["care_of"] = True
        else:
            kept.append(c)
    comps = kept or comps
    # state (search from the end; may be first or in the middle)
    state = None; state_idx = None; recognised = False
    if vocab is not None:
        # preference: last component, then first (S2 writes the state first in 6%), then the rightmost middle one;
        # cities named like states ('Indiana, PA', 'Virginia, MN', 'Delaware, OH') would otherwise win
        hits = [(i, vocab.state_of(comps[i], country)) for i in range(len(comps))]
        hits = [(i, s) for i, s in hits if s]
        if hits:
            if hits[-1][0] == len(comps) - 1:
                state_idx, state = hits[-1]
            elif hits[0][0] == 0:
                state_idx, state = hits[0]
            else:
                state_idx, state = hits[-1]
            recognised = True
    # postal code: strictly a trailing 5/6-digit token of one of the last two components (never a house number)
    postal = None
    pre = POSTAL_RE.get(country, POSTAL_RE["US"])
    for i in range(len(comps) - 1, max(len(comps) - 3, -1), -1):
        m = pre.search(comps[i])
        if m and m.end() == len(comps[i]) and (m.start() > 0 or len(comps) >= 3 and i == len(comps) - 1):
            postal = m.group(1); flags["postal_code"] = True
            comps[i] = " ".join(pre.sub(" ", comps[i]).split())
            break
    comps = [c for c in comps if c]
    # landmarks
    landmark: List[str] = []
    rest: List[Tuple[int, str]] = []
    for i, c in enumerate(comps):
        if i == state_idx:
            continue
        m = LANDMARK_RE.match(c)
        if m:
            landmark.append(m.group(2).strip() or m.group(1)); flags["landmark"] = True
        elif c:
            rest.append((i, c))
    # units and street
    unit = None; house_number = None; street = None; street_type = None; street_idx = None
    cleaned: List[Tuple[int, str]] = []
    for i, c in rest:
        c2, u = _extract_unit(c)
        if u and unit is None:
            unit = u; flags["unit"] = True
        if c2:
            cleaned.append((i, c2))
    street_rest: List[str] = []
    for want in (2, 1):
        for i, c in cleaned:
            if _street_strength(c, country) == want:
                n, name, stype = _street_split(c, country)
                if n or name:
                    house_number, street, street_type, street_idx = n, name, stype, i
                    street_rest = list(getattr(_street_split, "remainder", []))
                    break
        if street_idx is not None:
            break
    if house_number is None:
        for i, c in cleaned:
            m = LEADING_NUM_RE.match(c) if country != "India" else INDIA_NUM_RE.search(" " + c)
            if m:
                house_number = norm_number(m.group(1)); street_idx = i if street_idx is None else street_idx
                if street is None:
                    _, name, stype = _street_split(c, country)
                    street, street_type = name, stype
                break
    if house_number is None and country == "India":
        # bare plot / door codes as their own component: '316', 'A-108', 'D-36', '543/77', 'D/1200 B'
        for i, c in cleaned:
            if i == street_idx:
                continue
            if re.fullmatch(r"[a-z]?[-/]?\d[\w/-]*(?:\s[a-z])?", c) and not re.fullmatch(r"\d+(?:st|nd|rd|th)", c):
                house_number = norm_number(c.split()[0]) if " " not in c else norm_number(c)
                cleaned = [(j, x) for j, x in cleaned if j != i]
                break
    # unrecognised admin area (a country absent from the vocabulary): the last digit-free component that is
    # neither the street nor a landmark, when there are at least three components
    if state is None and state_idx is None and len(comps) >= 3:
        for i, c in reversed(cleaned):
            if i != street_idx and not re.search(r"\d", c):
                state, state_idx = c, i
                break
    # city: vocabulary first, then position
    city = None; city_idx = None
    if vocab is not None:
        for i, c in cleaned:
            if i == street_idx:
                continue
            if _city_norm(c) in vocab.cities.get(country, set()):
                city, city_idx = c, i
                if state_idx is not None and i == state_idx - 1:
                    break
    if city is None:
        cands = [(i, c) for i, c in cleaned if i != street_idx and not re.search(r"\d", c)]
        if state_idx is not None:
            before = [(i, c) for i, c in cands if i < state_idx]
            if before:
                city_idx, city = before[-1]
            elif cands:
                city_idx, city = cands[-1]
        elif cands:
            city_idx, city = cands[-1]
    locality = [c for i, c in cleaned if i not in (street_idx, city_idx, state_idx)]
    if street_rest:
        locality.insert(0, " ".join(street_rest))
    all_tokens = [t2 for c in comps for t in c.split() if (t2 := t.strip("#"))]   # '#' is injected before numbers in 10% of S2/S3
    numbers = [norm_number(t) for t in all_tokens if any(ch.isdigit() for ch in t)]
    digit_runs = [re.sub(r"^0+(?=\d)", "", d) for d in re.findall(r"\d+", " ".join(all_tokens))]
    tokens = _translit_tokens(" ".join(all_tokens))
    if any(INDIC_RE.search(t) for t in all_tokens):
        flags["indic"] = True
    if street_idx is not None and street_idx != 0 and country != "India":
        flags["reordered"] = True
    return AddressRep(original=original, country=country, house_number=house_number, street=street, street_type=street_type, unit=unit,
                      city=city, city_norm=_city_norm(city) if city else None, state=state, state_recognised=recognised, postal_code=postal,
                      landmark=landmark, locality=locality, po_box=po_box, care_of=care_of, numbers=numbers, digit_runs=digit_runs,
                      tokens=tokens, components=comps, missing=False, flags=flags)


def parse_addresses(addresses: Iterable[Optional[str]], countries: Iterable[str], vocab: Optional[AddressVocab] = None) -> List[AddressRep]:
    return [parse_address(a, c, vocab) for a, c in zip(addresses, countries)]


# --------------------------------------------------------------------------- #
# Vocabulary builder (training files only)
# --------------------------------------------------------------------------- #
def build_vocab(train_dir: str, out_path: Optional[str] = None, exclude_entity_ids: Optional[Set[str]] = None,
                max_pairs: int = 1_500_000, seed: int = 0) -> AddressVocab:
    """Learn states, aliases and cities from train_source1.tsv and the true pairs in train_ground_truth.tsv.

    ``exclude_entity_ids`` removes the pairs of held-out Source 1 entities (fold safety).  ``max_pairs`` samples
    the pairs used for alias learning (aliases need a 2% share and 20 occurrences; 1.5M pairs are ample).
    """
    import polars as pl
    read = lambda p: pl.read_csv(p, separator="\t", quote_char=None, has_header=True, infer_schema_length=0, encoding="utf8")
    s1 = read(os.path.join(train_dir, "train_source1.tsv"))
    v = AddressVocab()
    v.add_source1(s1["business_address"].to_list(), s1["country"].to_list())
    gt = read(os.path.join(train_dir, "train_ground_truth.tsv"))
    if exclude_entity_ids:
        gt = gt.filter(~pl.col("source1_entity_id").is_in(list(exclude_entity_ids)))
    pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")).explode("m").filter(pl.col("m") != "")
               .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("business_address").alias("a1"), "country"), on="source1_entity_id"))
    if max_pairs and pairs.height > max_pairs:
        pairs = pairs.sample(n=max_pairs, seed=seed)
    s23 = pl.concat([read(os.path.join(train_dir, f"train_source{k}.tsv")).select(pl.col("entity_id").alias("m"), pl.col("business_address").alias("a2")) for k in (2, 3)])
    pairs = pairs.join(s23, on="m")
    v.add_pairs(pairs["a1"].to_list(), pairs["a2"].to_list(), pairs["country"].to_list())
    if out_path:
        v.save(out_path)
    return v
