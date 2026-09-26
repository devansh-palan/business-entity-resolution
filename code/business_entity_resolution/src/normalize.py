"""Text normalisation shared by validation grouping, blocking and features (Polars expressions).

Everything here is country-agnostic: Unicode NFKD, casefold, punctuation stripped, and a legal-suffix
list that covers US, Indian and French company forms. No external resources are used.
"""
from __future__ import annotations

import polars as pl

LEGAL_SUFFIXES = [
    # United States
    "inc", "incorporated", "llc", "corp", "corporation", "co", "company", "ltd", "limited",
    "lp", "llp", "pc", "pllc", "plc",
    # India
    "pvt", "private", "opc",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scp", "sel", "selarl", "ei", "eirl",
    "cie", "ets", "etablissements",
    # other common forms that may appear in an open-set country
    "gmbh", "ag", "bv", "nv", "srl", "spa",
]


def norm_text(expr: pl.Expr) -> pl.Expr:
    """Lowercase, strip Latin accents, replace every non-alphanumeric run by one space.

    Only the Latin combining-diacritics block (U+0300-036F) is stripped after NFKD: Indic vowel signs and
    viramas are also combining marks, and stripping them shreds Indic words (measured in the name analysis).
    Indic block characters count as word characters for the same reason.
    """
    e = expr.fill_null("").str.to_lowercase().str.normalize("NFKD").str.replace_all(r"[̀-ͯ]+", "")
    return e.str.replace_all(r"[^\p{L}\p{N}ऀ-෿]+", " ").str.strip_chars()


def tokens(expr: pl.Expr) -> pl.Expr:
    return norm_text(expr).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def name_core_tokens(expr: pl.Expr) -> pl.Expr:
    """Name tokens without legal-form words."""
    return tokens(expr).list.eval(pl.element().filter(~pl.element().is_in(LEGAL_SUFFIXES)))


def name_core_sorted(expr: pl.Expr) -> pl.Expr:
    """Order-insensitive core-name key, e.g. 'Summit Inc' and 'Inc Summit' -> 'summit'."""
    return name_core_tokens(expr).list.sort().list.join(" ")


def addr_sorted(expr: pl.Expr) -> pl.Expr:
    """Order-insensitive address key (component reordering is a known noise pattern)."""
    return tokens(expr).list.sort().list.join(" ")
