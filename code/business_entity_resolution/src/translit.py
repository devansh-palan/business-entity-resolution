"""Rule-based Indic -> Latin transliteration, no external resources.

Why: 17% of Indian S2/S3 names contain no Latin letter (Devanagari 10.3%, Telugu 1.5%, Kannada 1.4%,
Tamil 1.3%, Bengali 1.2%, Gujarati 1.2%, Malayalam 0.7%, Oriya 0.3%, Gurmukhi 0.3%) and 6.8% of true
matches pair a Latin Source 1 name with an Indic-script record.  Frequent tokens are handled by a
dictionary learned from training pairs (names.py); this module is the fallback for everything else and
only needs to produce strings that are *close* to the Latin spelling for character n-gram similarity.

How: the nine Indic blocks share the ISCII-derived layout, so a character at offset k in any block is
mapped to the Devanagari character at the same offset and transliterated with one table.  A simplified
schwa rule decides when the inherent 'a' is written: after a consonant that is followed by another
consonant (no virama), or at the end of a word when the consonant closes a conjunct.
"""
from __future__ import annotations

import re
import unicodedata

BLOCKS = {  # base code point of each script block
    "devanagari": 0x0900, "bengali": 0x0980, "gurmukhi": 0x0A00, "gujarati": 0x0A80, "oriya": 0x0B00,
    "tamil": 0x0B80, "telugu": 0x0C00, "kannada": 0x0C80, "malayalam": 0x0D00,
}
INDIC_RE = re.compile(r"[ऀ-෿]")

# offsets inside a block (Devanagari layout)
_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e",
           0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li"}
_MATRAS = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e",
           0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x55: "e", 0x56: "ue", 0x57: "uue", 0x62: "li", 0x63: "li"}
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x29: "n", 0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
    0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_SIGNS = {0x01: "n", 0x02: "n", 0x03: "h", 0x00: "n", 0x04: ""}   # chandrabindu, anusvara, visarga, inverted chandrabindu, ...
_VIRAMA = 0x4D
_NUKTA = 0x3C
_OM = 0x50
_MALAYALAM_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k"}


def script_of(text: str) -> str:
    """Dominant script: 'latin', one of the Indic block names, 'mixed' (Latin + Indic), or 'other'."""
    latin = bool(re.search(r"[A-Za-z]", text))
    counts = {}
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x0DFF:
            for name, base in BLOCKS.items():
                if base <= o < base + 0x80:
                    counts[name] = counts.get(name, 0) + 1
                    break
    if counts and latin:
        return "mixed"
    if counts:
        return max(counts, key=counts.get)
    return "latin" if latin else "other"


def _block_offset(ch: str):
    o = ord(ch)
    if 0x0900 <= o <= 0x0DFF:
        base = (o - 0x0900) // 0x80 * 0x80 + 0x0900
        return o - base
    return None


def transliterate_indic(text: str) -> str:
    """Transliterate every Indic-script run to lowercase Latin; other characters pass through unchanged."""
    if not INDIC_RE.search(text):
        return text
    text = unicodedata.normalize("NFC", text)      # tables expect composed vowel signs
    out = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        o = ord(ch)
        if ch in _MALAYALAM_CHILLU or o in _MALAYALAM_CHILLU:
            out.append(_MALAYALAM_CHILLU[o]); i += 1; continue
        off = _block_offset(ch)
        if off is None:
            out.append(ch); i += 1; continue
        if off in _CONSONANTS:
            cons = _CONSONANTS[off]
            j = i + 1
            if j < n and _block_offset(text[j]) == _NUKTA:      # nukta modifies the consonant
                cons = {"k": "q", "kh": "kh", "g": "g", "j": "z", "d": "r", "dh": "rh", "f": "f", "y": "y"}.get(cons, cons)
                j += 1
            nxt = _block_offset(text[j]) if j < n else None
            out.append(cons)
            if nxt is None:                                        # word end: schwa only after a conjunct
                if i > 0 and _block_offset(text[i - 1]) == _VIRAMA:
                    out.append("a")
                i = j; continue
            if nxt == _VIRAMA:
                i = j + 1; continue
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt]); i = j + 1; continue
            gurmukhi_tippi = nxt == 0x70 and 0x0A00 <= ord(text[j]) < 0x0A80
            if nxt in _CONSONANTS or nxt in _SIGNS or nxt in _VOWELS or gurmukhi_tippi:
                out.append("a")                                    # inherent vowel before another sound
            i = j; continue
        if off in _VOWELS:
            out.append(_VOWELS[off])
        elif off in _MATRAS:
            out.append(_MATRAS[off])
        elif off in _SIGNS:
            out.append(_SIGNS[off])
        elif off == 0x70 and 0x0A00 <= o < 0x0A80:                 # Gurmukhi tippi (nasal)
            out.append("n")
        elif off == _OM:
            out.append("om")
        elif 0x66 <= off <= 0x6F:
            out.append(str(off - 0x66))
        elif off in (0x64, 0x65):                                  # danda
            out.append(" ")
        # virama, nukta on its own, and other marks produce nothing
        i += 1
    return "".join(out)


def transliterate_tokens(text: str) -> str:
    """Transliterate and normalise spacing; Latin parts are left as they are."""
    t = transliterate_indic(unicodedata.normalize("NFC", text))
    return " ".join(t.split())
