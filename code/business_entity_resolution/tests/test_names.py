"""Tests for src/names.py using name pairs observed in the training data."""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import names as nm  # noqa: E402
from translit import transliterate_indic, script_of  # noqa: E402


def core(s, d=None):
    return nm.normalize_name(s, d).core


def test_case_accents_spaces_and_punctuation():
    r = nm.normalize_name("Clear Ríver")
    assert r.clean == "clear river" and r.core == "clear river"
    assert core("Mdeod  Cárdiology") == "mdeod cardiology"
    assert core("Bangalore-Planner-Corp") == "bangalore planner"
    assert nm.normalize_name("Bangalore-Planner-Corp").legal_forms == ["corp"]
    assert core("SUMMIT NETWORK  GROUP") == "summit network group"
    assert nm.normalize_name("SUMMIT NETWORK GROUP").flags["all_caps"] is True


def test_ampersand_apostrophe_brackets():
    assert core("Woodfield, Barr & Baker Chase") == "woodfield barr and baker chase"
    assert core("Luke's Grand Tax") == "lukes grand tax"          # joined, as the domain-style matches show
    assert nm.normalize_name("Shri Supreme Consulting Private  (Limited)").legal_forms == ["pvt", "ltd"]
    assert nm.normalize_name("Steele Express [LLC]").legal_forms == ["llc"]
    assert core("Steele Express [LLC]") == "steele express"


def test_legal_forms_canonical_and_removed_but_kept():
    for raw, forms in [("Willow L.L.C.", ["llc"]), ("Interstate Ridge Tsak0s L.L.C.", ["llc"]), ("Chiropractic Care Of Talladega Incorporated", ["inc"]),
                       ("Balaji Management Pvt Ltd", ["pvt", "ltd"]), ("Balaji Management Private Limited", ["pvt", "ltd"]),
                       ("Ebony N. Kraus, D.D.S., DDS PC", ["pc"]), ("SARL Dupont Fils", ["sarl"]), ("Ets Martin", ["ets"])]:
        r = nm.normalize_name(raw)
        assert r.legal_forms == forms, (raw, r.legal_forms)
    assert core("Interstate Ridge Tsak0s L.L.C.") == "interstate ridge tsakos"
    assert core("Willow L.L.C.") == "willow" and core("Willow") == "willow"
    r = nm.normalize_name("MAHARATHI TECVGENISIONS PVT PVT LTD")
    assert r.core == "maharathi tecvgenisions" and r.legal_forms == ["pvt", "pvt", "ltd"]
    # a name that is nothing but forms keeps its tokens
    assert core("Limited") == "limited"


def test_designations_and_comma_order():
    r = nm.normalize_name("Ebony N. Kraus, D.D.S., DDS PC")
    assert r.designations == ["dds", "dds"] and r.core == "ebony n kraus"
    r = nm.normalize_name("Reed, Edyth N., D.D.S.")
    assert r.designations == ["dds"] and r.core_sorted == "edyth n reed"
    r = nm.normalize_name("Monarch, Inc")
    assert r.core == "monarch" and r.legal_forms == ["inc"]
    assert core("Do It Yourself Garage") == "do it yourself garage"   # 'do' only counts after a comma


def test_alias_markers_take_the_name_after_the_marker():
    cases = [("Deltahalo D.B.A. Bay Pinnacle Horizon LP", "bay pinnacle horizon", "deltahalo"),
             ("Zetafluxonyx Sys aka Kestrel PLLC", "kestrel", "zetafluxonyx sys"),
             ("Ectoevocira fka Golden Wellness International LLC", "golden wellness international", "ectoevocira"),
             ("Fayecira t/a Lucky Pilates Inc", "lucky pilates", "fayecira"),
             ("Brixonyx formerly: Monarch, Inc", "monarch", "brixonyx"),
             ("Ectojax One formerly known as Colonial Capital", "colonial capital", "ectojax one")]
    for raw, c, prefix in cases:
        r = nm.normalize_name(raw)
        assert r.core == c and r.flags.get("has_alias") and prefix in r.variants, (raw, r.core, r.variants)


def test_domain_style_names():
    r = nm.normalize_name("Helios.Com")
    assert r.flags["domain_style"] and r.core == "helios" and r.compact == "helios"
    assert nm.normalize_name("CLAIRSPROTT.COM").compact == "clairsprott"
    assert nm.normalize_name("pacificmaterialsdynamics.c0m").compact == "pacificmaterialsdynamics"
    assert nm.normalize_name("#KOKANINDIA").compact == "kokanindia"
    r = nm.normalize_name("Indu-Trust | www.indutrust.com")
    assert r.core == "indu trust" and "indutrust" in r.variants and r.flags.get("url_segment")
    r = nm.normalize_name("CASCADEROJASSFAMILY.COM - 6670506828")
    assert r.compact == "cascaderojassfamily" and r.flags.get("phone_stripped")
    assert nm.normalize_name("Cascade Allied Telecom LLC").compact == "cascadealliedtelecom"


def test_digit_for_letter_fix_is_narrow():
    assert core("Regi0nal Sfhpeior Electronics") == "regional sfhpeior electronics"
    assert core("Capita1 Denta1 5ervices 8right") == "capital dental services bright"
    assert core("24hr Fitness 2nd Street") == "24hr fitness 2nd street"        # legitimate mixed tokens untouched
    assert core("B2B Labs 3D Print") == "b2b labs 3d print"
    assert core("Studio 38 Barbershop LLC") == "studio 38 barbershop"
    assert nm.normalize_name("Studio 38 Barbershop LLC").digits == ["38"]
    assert nm.normalize_name("de1hi Traders").flags.get("digit_fixed") is True


def test_honorifics_and_repeats():
    assert core("Smt FUTURE FUTURE DEVELOPERS LTD") == "future developers"
    r = nm.normalize_name("Shri INTELLECT CLINIC CLINIC")
    assert r.core == "intellect clinic" and r.honorific == "shri" and r.flags.get("repeat_removed")
    assert core("Shree & Co") == "shree and"                                  # too short to strip the honorific
    assert core("M/s Patel Traders") == "patel traders"
    assert core("N N Barreto Properties") == "n n barreto properties"        # single-letter repeats are kept
    assert core("Dr Reddys Labs") == "reddys labs"
    assert core("Dr Reddys") == "dr reddys"


def test_reordering_and_generic_variant():
    a, b = nm.normalize_name("Woodfield, Barr & Baker Chase"), nm.normalize_name("Woodfield, Chase & Baker Barr")
    assert a.core != b.core and a.core_sorted == b.core_sorted
    r = nm.normalize_name("Youth Inc Service")
    assert r.core == "youth service" and "youth" in r.variants
    assert nm.normalize_name("Primary Care Group").acronym == "pcg"
    assert nm.normalize_name("Adroit Advisors Group").acronym == "aag"
    r = nm.normalize_name("AAG")
    assert r.flags["is_acronym"] and r.core == "aag"


def test_spelling_variant():
    assert core("Wellness Centre") == core("Wellness Center") == "wellness center"
    assert nm.normalize_name("Wellness Centre").flags.get("spelling_variant") is True
    # Indian spellings are unified only in the transliteration representation; core stays faithful
    a, b = nm.normalize_name("Laxmi Energy Private Limited"), nm.normalize_name("Lakshmi Energy Pvt Ltd")
    assert a.core != b.core and a.translit_core_sorted == b.translit_core_sorted == "energy lakshmi"
    assert nm.normalize_name("Shree Systems").translit_core == "shri systems" and nm.normalize_name("Shree Systems").core == "shree systems"
    assert nm.normalize_name("Jay Galaxy").translit_core == nm.normalize_name("Jey Galaxy").translit_core == "jai galaxy"


def test_indic_scripts_survive_cleaning_and_transliterate():
    r = nm.normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert r.script == "devanagari"
    assert r.tokens == ["राम", "मार्केटिंग", "प्राइवेट", "लिमिटेड"]           # vowel signs intact
    assert r.legal_forms == ["pvt", "ltd"]                                    # recognised through transliteration
    assert r.core_tokens == ["राम", "मार्केटिंग"] and r.translit_core == "ram marketing"
    d = {"प्राइवेट": "pvt", "लिमिटेड": "ltd", "मार्केटिंग": "marketing", "राम": "ram"}
    assert nm.normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड", d).translit_core == "ram marketing"
    r = nm.normalize_name("ઇન્ટરનેશનલ વિઝન ટેક્નોલોજીસ પ્રા. લિ.")
    assert r.legal_forms == ["pvt", "ltd"] and len(r.core_tokens) == 3
    r = nm.normalize_name("Royal हॉस्पिटैलिटी Private Limited")
    assert r.script == "mixed" and r.translit_core == "royal hospitailiti" and r.core_tokens == ["royal", "हॉस्पिटैलिटी"]
    r = nm.normalize_name("శివ మేనేజ్‌మెంట్ ప్రైవేట్ లిమిటెడ్")
    assert r.legal_forms == ["pvt", "ltd"] and "‌" not in r.clean


def test_learn_translit_dict_positional_and_fuzzy():
    lat = ["Ram Marketing Private Limited", "Ram Marketing Private Limited", "Shakti Consulting Limited", "Prime Food Private Limited", "Shakti Trading Private Limited"]
    ind = ["राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड", "शक्ति कंसल्टिंग लिमिटेड", "प्राइम फूड प्रा. लि.", "शक्ति ट्रेडिंग प्राइवेट लिमिटेड"]
    d = nm.learn_translit_dict(lat, ind, min_count=2)
    assert d["राम"] == "ram" and d["मार्केटिंग"] == "marketing" and d["प्राइवेट"] == "pvt" and d["लिमिटेड"] == "ltd" and d["शक्ति"] == "shakti"
    assert nm.normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड", d).translit_core_sorted == "marketing ram"


def test_representations_are_complete_and_original_kept(tmp_path):
    r = nm.normalize_name("Ectoevocira fka Golden Wellness International LLC")
    d = r.as_dict()
    for k in ("original", "clean", "tokens", "legal_forms", "core_tokens", "core", "core_sorted", "compact", "acronym", "translit_core", "variants", "digits", "script", "flags"):
        assert k in d
    assert d["original"] == "Ectoevocira fka Golden Wellness International LLC"
    assert nm.char_ngrams("ram ma") == ["ram", "amm", "mma"] and nm.char_ngrams("ab") == ["ab"]
    p = tmp_path / "d.json"
    nm.save_translit_dict({"राम": "ram"}, str(p))
    assert nm.load_translit_dict(str(p)) == {"राम": "ram"}
    assert nm.normalize_name("").core == "" and nm.normalize_name(None).original == ""


def test_transliterator_basics():
    assert transliterate_indic("लिमिटेड") == "limited" and transliterate_indic("मार्केटिंग") == "marketing"
    assert transliterate_indic("महाराष्ट्र") == "maharashtra" and transliterate_indic("ગુજરાત") == "gujarat"
    assert transliterate_indic("ਪੰਜਾਬ") == "panjab"
    assert script_of("Summit Inc") == "latin" and script_of("ಕರ್ನಾಟಕ") == "kannada" and script_of("Royal हॉस्पिटैलिटी") == "mixed"
