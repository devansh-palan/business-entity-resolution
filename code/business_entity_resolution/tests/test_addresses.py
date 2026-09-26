"""Tests for src/addresses.py with addresses observed in the training data."""
from __future__ import annotations

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import addresses as ad  # noqa: E402


@pytest.fixture(scope="module")
def vocab():
    v = ad.AddressVocab()
    s1 = [("1795 Westchester Drive, High Point, NC", "US"), ("17560 Ellis Road, Tahlequah, OK", "US"), ("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US"),
          ("1103 Harrison Avenue, Cincinnati, OH", "US"), ("4388 Highland Drive, Unit 4, Millcreek, UT", "US"), ("8353 Folsom Boulevard, Sacramento, CA", "US"),
          ("17814 Tanglewood Court, Hazel Crest, IL", "US"), ("1419 Skywater Road, Gravois Mills, MO", "US"), ("286 Elmwood Drive, Radcliff, KY", "US"),
          ("739 Outer Drive, Lenoir City, TN", "US"), ("475 Buffalo Street, Rowlesburg, WV", "US"), ("6905 Darmstadt Road, Evansville, IN", "US"),
          ("Plot No.9904/13157, Erabang, Nimapara, Puri, Orissa", "India"), ("A-108, 1St Floor Knox Plaza, Chincholi Bunder Road Malad West, Mumbai, Maharashtra", "India"),
          ("No. 18/1, 1St Main Road Govindarajanagar, Bangalore, Karnataka", "India"), ("Flat No. 101, Pavansut-1, Salasar Park-A, Taki Road, Vasai, Thane, Maharashtra", "India"),
          ("D-36, Pocket C, Mayur Vihar Phase II, Delhi, Delhi", "India")] * 25
    v.add_source1([a for a, _ in s1], [c for _, c in s1])
    a1 = ["1795 Westchester Drive, High Point, NC"] * 30 + ["Plot No.9904/13157, Erabang, Nimapara, Puri, Orissa"] * 60 + ["A-108, Mumbai, Maharashtra"] * 60
    a2 = (["1795 WESTCHESTER DR, HIGH POINT, North Carolina"] * 30 + ["PLOT 9904, PURI, Odisha"] * 30 + ["PLOT 9904, PURI, OD"] * 30
          + ["A-108, MUMBAI, MH"] * 30 + ["A-108, MUMBAI, महाराष्ट्र"] * 30)
    v.add_pairs(a1, a2, ["US"] * 30 + ["India"] * 120)
    return v


def test_clean_and_number_normalisation():
    assert ad.clean_text("1795 Westchester Drive, High Point, NC") == "1795 westchester drive, high point, nc"
    assert ad.clean_text("<NULL>, IL, 17814. TANGLEWOOD COURT, HAZEL CREST") == "il, 17814 tanglewood court, hazel crest"
    assert ad.norm_number("001419") == "1419" and ad.norm_number("17814.") == "17814" and ad.norm_number("#1105") == "1105"
    assert ad.norm_number("18/1") == "18/1" and ad.norm_number("1-98/5/2a") == "1-98/5/2a" and ad.norm_number("0") == "0"


def test_us_canonical_and_noisy_forms(vocab):
    r = ad.parse_address("1795 Westchester Drive, High Point, NC", "US", vocab)
    assert (r.house_number, r.street, r.street_type, r.city_norm, r.state, r.state_recognised) == ("1795", "westchester", "drive", "high point", "nc", True)
    assert r.unit is None and r.postal_code is None and r.landmark == [] and r.original == "1795 Westchester Drive, High Point, NC"
    r = ad.parse_address("1795 WESTCHESTER DR, HIGH POINT, North Carolina", "US", vocab)
    assert (r.house_number, r.street, r.street_type, r.city_norm, r.state) == ("1795", "westchester", "drive", "high point", "nc")
    r = ad.parse_address("<NULL>, IL, 17814. TANGLEWOOD COURT, HAZEL CREST", "US", vocab)
    assert (r.house_number, r.street, r.street_type, r.state, r.city_norm) == ("17814", "tanglewood", "court", "il", "hazel crest") and r.flags.get("reordered")
    r = ad.parse_address("001419 SKYWATER ROAD, GRAVOIS MILLS, MO", "US", vocab)
    assert r.house_number == "1419" and r.street == "skywater" and r.street_type == "road"
    r = ad.parse_address("0286 ELMWOOD DRIVE, PO BOX 5961, RADCLIFF, KY", "US", vocab)
    assert r.house_number == "286" and r.po_box == "5961" and r.city_norm == "radcliff" and r.state == "ky"
    r = ad.parse_address("739 Outer Drive, Lenoir City  CDP, TN", "US", vocab)
    assert r.city_norm == "lenoir" and r.state == "tn"
    r = ad.parse_address("6905-C DARMSTADT RD, EVANSVILLE, IN", "US", vocab)
    assert r.house_number == "6905-c" and r.street == "darmstadt" and r.street_type == "road"
    r = ad.parse_address("475 1/2 BUFFALO ST, ROWLESBURG, WV", "US", vocab)
    assert r.house_number == "475 1/2" and r.street == "buffalo" and r.street_type == "street"


def test_us_units_and_saint():
    r = ad.parse_address("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US")
    assert r.house_number == "2100" and r.unit is not None and "g" in r.unit and r.city_norm == "dundalk"
    r = ad.parse_address("4388 Highland Drive, Unit 4, Millcreek, UT", "US")
    assert r.unit == "unit 4" and r.street == "highland"
    r = ad.parse_address("96 ripple way pmb 311 clayton nc", "US")
    assert r.po_box == "311"
    r = ad.parse_address("1041 Mariana Street, Tempe, AZ, Unit 2", "US")
    assert r.unit == "unit 2" and r.city_norm == "tempe"
    r = ad.parse_address("100 Main St, St Louis, MO", "US")
    assert r.street == "main" and r.street_type == "street" and r.city_norm == "saint louis"
    r = ad.parse_address("7578 Orchid Ct, Prince William County, Virginia", "US")
    assert r.street_type == "court" and r.city_norm == "prince william"


def test_india_numbers_units_landmarks(vocab):
    r = ad.parse_address("Plot No.9904/13157, Erabang, Nimapara, Puri, Orissa", "India", vocab)
    assert r.house_number == "9904/13157" and r.state == "orissa" and r.state_recognised and r.city_norm == "puri"
    assert set(r.locality) == {"erabang", "nimapara"}
    r = ad.parse_address("PLOT 9904, PURI, Odisha", "India", vocab)
    assert r.house_number == "9904" and r.state == "orissa"
    r = ad.parse_address("A-108, MUMBAI, महाराष्ट्र", "India", vocab)
    assert r.state == "maharashtra" and r.house_number == "a-108" or r.numbers == ["a-108"]
    r = ad.parse_address("Flat No. 101, Pavansut-1, Salasar Park-A, Taki Road, Vasai, Thane, Maharashtra", "India", vocab)
    assert r.unit == "flat 101" and r.street == "taki" and r.street_type == "road" and r.state == "maharashtra"
    r = ad.parse_address("316, 3Rd Floor, Ratnanjali Square, Near Gloriya Restaurant, Prenatirth Derasar Road, Satellite, Ahmedabad, Gujarat", "India")
    assert r.unit == "3 floor" and r.landmark == ["gloriya restaurant"] and r.street == "prenatirth derasar" and r.street_type == "road"
    r = ad.parse_address("H.No 910 A 3503, Mumbai, महाराष्ट्र", "India", vocab)
    assert r.house_number == "910" and r.state == "maharashtra"
    r = ad.parse_address("Village, Nadiad, Diwali Pol Dabhan Bhagol, Kheda, C/O Patel Hirenkumar Jayantibhai, Gujarat, Nadiad", "India")
    assert r.care_of == "patel hirenkumar jayantibhai"
    r = ad.parse_address("No. 18/1, 1St Main Road Govindarajanagar, Bangalore, Karnataka", "India", vocab)
    assert r.house_number == "18/1" and r.city_norm == "bangalore" and r.state == "karnataka"


def test_reordered_addresses_do_not_make_cities_states():
    v = ad.AddressVocab()
    canon = ["10 Main Street, Portland, OR"] * 100          # Portland is second to last 100 times
    reordered = ["OR, 10 Main Street, Portland"] * 25       # ... and last 25 times (reordered Source 1 rows)
    v.add_source1(canon + reordered, ["US"] * 125)
    assert "or" in v.states["US"] and "portland" not in v.states["US"] and "portland" in v.cities["US"]


def test_state_aliases_and_compatibility(vocab):
    assert vocab.state_of("odisha", "India") == "orissa" and vocab.state_of("od", "India") == "orissa"
    assert vocab.state_of("north carolina", "US") == "nc" and vocab.state_of("mh", "India") == "maharashtra"
    assert vocab.state_of("महाराष्ट्र", "India") == "maharashtra"
    assert vocab.state_of("high point", "US") is None       # cities never become state aliases
    assert vocab.states_compatible("nc", "nc", "US") and not vocab.states_compatible("nc", "ok", "US")


def test_france_generic_fallback():
    r = ad.parse_address("175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine", "France")
    assert r.house_number == "175" and r.street_type == "boulevard" and r.street == "du president franklin roosevelt"
    assert r.city_norm == "bordeaux" and r.state == "nouvelle-aquitaine" and not r.state_recognised
    r = ad.parse_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France")
    assert r.street_type == "rue" and r.street == "de dieppe" and r.city_norm == "lille"
    r = ad.parse_address("12 bis rue Lyderic, Lille", "France")
    assert r.house_number == "12 bis" and r.street == "lyderic" and r.city_norm == "lille" and r.state is None
    v = ad.AddressVocab()
    v.extend_from_source1(["1 Rue X, Lille, Hauts-de-France"] * 25 + ["2 Rue Y, Nantes, Pays de la Loire"] * 25, ["France"] * 50)
    r = ad.parse_address("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France", v)
    assert r.state == "hauts-de-france" and r.state_recognised and r.city_norm == "lille"


def test_missing_and_postal():
    r = ad.parse_address(None, "US")
    assert r.missing and r.tokens == [] and r.original == ""
    r = ad.parse_address("   ", "India")
    assert r.missing
    r = ad.parse_address("10 Main Street, Springfield, IL 62704", "US")
    assert r.postal_code == "62704" and r.house_number == "10"
    r = ad.parse_address("Plot 4, Sector 9, Pune, Maharashtra 411001", "India")
    assert r.postal_code == "411001" and r.house_number == "4"


def test_tokens_numbers_and_translit():
    r = ad.parse_address("D/1200 B, 28TH MAIN, 35TH C, CROSS RD, 4TH T BLOCK, BANGALORE SOUTH, BANGALORE, ಕರ್ನಾಟಕ", "India")
    assert "d/1200" in r.numbers and "1200" in r.digit_runs and "28th" in r.numbers and r.flags.get("indic") and "karnatak" in " ".join(r.tokens)
    assert r.house_number == "d/1200 b" and r.street == "28th main"        # ordinal street names are not house numbers
    r = ad.parse_address("316, 3Rd Floor, Ratnanjali Square, Near Gloriya Restaurant, Prenatirth Derasar Road, Satellite, Ahmedabad, Gujarat", "India")
    assert r.house_number == "316" and "316" not in r.locality
    r = ad.parse_address("A-108, 1St Floor Knox Plaza, Chincholi Bunder Road Malad West, Mumbai, Maharashtra", "India")
    assert r.house_number == "a-108" and r.unit == "1 floor" and r.street_type == "road"


def test_full_state_names_survive_city_collisions():
    v = ad.AddressVocab()
    v.add_source1(["1 Main Street, Texas City, TX"] * 30 + ["2 Oak Road, Houston, TX"] * 30, ["US"] * 60)
    assert "texas" in v.cities["US"] and "tx" in v.states["US"]
    v.add_pairs(["2 Oak Road, Houston, TX"] * 100, ["2 OAK RD, HOUSTON, TX"] * 55 + ["2 Oak Rd, Houston, Texas"] * 40 + ["2 Oak Rd, Texas City, TX, Houston"] * 5, ["US"] * 100)
    assert v.state_of("texas", "US") == "tx" and v.state_of("houston", "US") is None


def test_representation_has_required_fields():
    d = ad.parse_address("1795 Westchester Drive, High Point, NC", "US").as_dict()
    for k in ("house_number", "street", "unit", "city", "state", "postal_code", "country", "landmark", "original"):
        assert k in d
    assert d["country"] == "US" and d["original"] == "1795 Westchester Drive, High Point, NC"


def test_hash_marks_do_not_survive_in_tokens():
    rep = ad.parse_address("##7719 YELLOW PINE DR, HOUSTN, TX", "US")
    assert rep.house_number == "7719"
    assert "7719" in rep.tokens and not any(t.startswith("#") for t in rep.tokens)
    rep2 = ad.parse_address("Apt #5, 12 Main Street, Springfield, IL", "US")
    assert rep2.unit is not None and "5" in rep2.unit and not any("#" in t for t in rep2.tokens)
