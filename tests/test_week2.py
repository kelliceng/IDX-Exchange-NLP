from pathlib import Path

import pandas as pd
import pytest

from scripts.text_cleaning import ABBREVIATIONS, TextCleaner

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner()


# ---------- prices ----------

@pytest.mark.parametrize("raw, expected", [
    ("priced at 450k", "priced at 450000"),
    ("$1.2m home", "$1200000 home"),
    ("$300K in upgrades", "$300000 in upgrades"),
    ("Offered at $48,888,000", "Offered at $48888000"),
    ("sales exceeding $12 million", "sales exceeding $12000000"),
    ("now below $1.15M", "now below $1150000"),
    ("just reduced by 50k!", "just reduced by 50000!"),
])
def test_prices(cleaner, raw, expected):
    assert cleaner.normalize_prices(raw) == expected


@pytest.mark.parametrize("raw", [
    "Residence 4M presents",   # unit name, not a price
    "a 5km drive",             # distance
    "Built in 2019",           # year
])
def test_prices_left_alone(cleaner, raw):
    assert cleaner.normalize_prices(raw) == raw


# ---------- measurements ----------

@pytest.mark.parametrize("raw, expected", [
    ("2,000 sqft", "2000 square feet"),
    ("909 sq. ft. of living", "909 square feet of living"),
    ("1,769 sq ft", "1769 square feet"),
    ("2,940 SF of living area", "2940 square feet of living area"),
    ("3,203 square feet", "3203 square feet"),
    ("1500 square foot", "1500 square feet"),
    ("4,435 interior SF", "4435 square feet"),
    ("0.5 ac lot", "0.5 acres lot"),
    ("12 ft ceilings", "12 feet ceilings"),
    ("2,197+/- sq. ft. home", "2197 square feet home"),
    ("$274 per sq ft", "$274 per square foot"),
    ("the 4,400-square-foot home", "the 4400 square feet home"),
    ("2,520 sq ft. Highlights", "2520 square feet. Highlights"),  # keeps the sentence's period
])
def test_measurements(cleaner, raw, expected):
    assert cleaner.normalize_measurements(raw) == expected


# ---------- abbreviations ----------

@pytest.mark.parametrize("raw, expected", [
    ("3br", "3 bedroom"),
    ("4BR, 3.5BA", "4 bedroom, 3.5 bathroom"),
    ("2 BR/1 BA condo", "2 bedroom/1 bathroom condo"),
    ("3BD/2.5BA", "3 bedroom/2.5 bathroom"),
    ("2 bdrm 1 bath", "2 bedroom 1 bath"),
    ("office w/ 6 bathrooms", "office with 6 bathrooms"),
    ("w/pool", "with pool"),
    ("sold w/o furniture", "sold without furniture"),
    ("w/d hookups", "washer/dryer hookups"),
    ("central A/C", "central air conditioning"),
    ("central AC", "central air conditioning"),
    ("SS appliances", "stainless steel appliances"),
    ("approx. 1.3 acres", "approximately 1.3 acres"),
    ("near the 15 fwy", "near the 15 freeway"),
    ("large mbr suite", "large master bedroom suite"),
    ("4 br. & 2.5 ba. offering", "4 bedroom & 2.5 bathroom offering"),
    ("on Main Ave. The", "on Main avenue. The"),  # keeps the sentence's period
])
def test_abbreviations(cleaner, raw, expected):
    assert cleaner.expand_abbreviations(raw) == expected


@pytest.mark.parametrize("raw", [
    "LG washer and dryer",   # brand name
    "bath and bed linens",   # full words, not abbreviations
    "brand new roof",        # 'br' inside a word
])
def test_abbreviations_left_alone(cleaner, raw):
    assert cleaner.expand_abbreviations(raw) == raw


def test_abbreviation_dictionary_size():
    assert len(ABBREVIATIONS) >= 30


# ---------- number words ----------

@pytest.mark.parametrize("raw, expected", [
    ("three bedrooms", "3 bedrooms"),
    ("Four bedrooms, two bathrooms", "4 bedrooms, 2 bathrooms"),
    ("two-car garage", "2-car garage"),
    ("two additional bedrooms", "2 additional bedrooms"),
    ("2 1/2 baths", "2.5 baths"),
    ("one of a kind", "one of a kind"),   # not a count
    ("one of the bedrooms", "one of the bedrooms"),   # not a count
    ("four and a half bathrooms", "4.5 bathrooms"),
    ("1-and-a-half-bath", "1.5-bath"),
    ("Three ample sized bedrooms", "3 ample sized bedrooms"),
    ("four generously sized bedrooms", "4 generously sized bedrooms"),
])
def test_number_words(cleaner, raw, expected):
    assert cleaner.normalize_number_words(raw) == expected


# ---------- unicode, html, whitespace, case, punctuation ----------

@pytest.mark.parametrize("raw, expected", [
    ("It’s a “gem”", "It's a \"gem\""),
    ("nearly ¾ of an acre", "nearly 3/4 of an acre"),
    ("endless​‌ possibilities", "endless possibilities"),
    ("3,246± sq ft", "3,246 sq ft"),
    ("wow…", "wow..."),
])
def test_unicode(cleaner, raw, expected):
    assert cleaner.normalize_unicode(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("<b>Gorgeous</b> home", "Gorgeous home"),
    ("Pool &amp; spa", "Pool & spa"),
    ("line one<br>line two", "line one line two"),
])
def test_html(cleaner, raw, expected):
    assert cleaner.remove_html(raw) == expected


def test_whitespace(cleaner):
    assert cleaner.normalize_whitespace("Kitchen\r\r\nNew   roof\t ") == "Kitchen New roof"


@pytest.mark.parametrize("raw, expected", [
    ("PRICED AT ONLY $389,000", "priced at only $389,000"),
    ("CORNER home", "corner home"),
    ("Low HOA and new HVAC", "Low HOA and new HVAC"),
])
def test_case(cleaner, raw, expected):
    assert cleaner.normalize_case(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Must see!!!", "Must see!"),
    ("**UPDATE** price", "UPDATE price"),
    ("great location , close to parks", "great location, close to parks"),
])
def test_punctuation(cleaner, raw, expected):
    assert cleaner.normalize_punctuation(raw) == expected


# ---------- full pipeline ----------

def test_clean_text_full(cleaner):
    raw = "GORGEOUS 3BR/2BA w/ pool!!\r\n2,000 sqft, priced at $1.2M — MUST SEE"
    assert cleaner.clean_text(raw) == (
        "gorgeous 3 bedroom/2 bathroom with pool! 2000 square feet, "
        "priced at $1200000 - must see"
    )


@pytest.mark.parametrize("raw", [None, float("nan"), ""])
def test_clean_text_empty(cleaner, raw):
    assert cleaner.clean_text(raw) == ""


def test_clean_text_is_stable(cleaner):
    """Cleaning already-clean text shouldn't change it."""
    once = cleaner.clean_text("3BR/2BA, 2,000 sqft w/ pool, $450k")
    assert cleaner.clean_text(once) == once


# ---------- profiling and cleaned dataset ----------

def test_profiling(cleaner):
    df = pd.DataFrame({"remarks": ["3BR w/ pool <b>nice</b>", None, "$450,000 home"]})
    profile = cleaner.profile_column(df, "remarks")
    assert "null_rate" in profile
    assert "avg_length" in profile
    assert profile["null_rate"] == pytest.approx(1 / 3)
    assert profile["has_html"] == 1
    assert profile["price_mentions"] == 1


def test_cleaned_dataset():
    df = pd.read_csv(DATA / "listing_sample_clean.csv")
    assert len(df) >= 500
    assert df["remarks_clean"].notna().all()
    assert not df["remarks_clean"].str.contains("​|\r|\n").any()
