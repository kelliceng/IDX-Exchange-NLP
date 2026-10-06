from pathlib import Path

import pytest

from scripts.entity_extractor import EntityExtractor
from scripts.evaluate_entities import load_labeled, score

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def extractor():
    return EntityExtractor()


# ---------- bedrooms ----------

@pytest.mark.parametrize("text, expected", [
    ("This 3 bedroom, 2 bath home", 3),
    ("a charming 4-bedroom home", 4),
    ("offering 4 spacious bedrooms, each with views", 4),
    ("Turnkey 4 Bed 3 Bath", 4),
    ("2 bedroom/1 bathroom condo", 2),
    ("primary suite plus 2 additional bedrooms", None),   # not the total
    ("ADU with 1 bedroom and 1 bathroom", None),         # second unit
    ("can be converted into a 5th bedroom", None),       # potential, not actual
    ("2 units, each offering 2 bedrooms", None),         # per unit
    ("No bedroom count here", None),
])
def test_bedrooms(extractor, text, expected):
    assert extractor.extract_bedrooms(text) == expected


# ---------- bathrooms ----------

@pytest.mark.parametrize("text, expected", [
    ("3 bedroom, 2.5 bath home", 2.5),
    ("4 bedrooms and 3 full bathrooms", 3),
    ("2 full bathrooms and 2 half bathrooms", 3),
    ("a 1.75-bathroom family home", 1.75),
    ("a 5-piece bathroom", None),                        # not a count
    ("3 additional bedrooms, 2 full baths", None),       # baths for the extra bedrooms
])
def test_bathrooms(extractor, text, expected):
    assert extractor.extract_bathrooms(text) == expected


# ---------- square feet ----------

@pytest.mark.parametrize("text, expected", [
    ("offering 1800 square feet of living space", 1800),
    ("set on a 7000 square feet lot with a 1500 square feet home", 1500),
    ("a 400 square feet garage", None),
    ("approved plans to add 1877 square feet", None),
    ("a floor plan of over 3900 square feet", 3900),
])
def test_sqft(extractor, text, expected):
    assert extractor.extract_sqft(text) == expected


# ---------- price ----------

@pytest.mark.parametrize("text, expected", [
    ("Now offered at $724900.", 724900),
    ("new price - only $460000!", 460000),
    ("over $50000 of builder upgrades", None),
    ("leased for $2550 a month", None),
])
def test_price(extractor, text, expected):
    assert extractor.extract_price(text) == expected


# ---------- amenities and extract_all ----------

def test_amenities(extractor):
    found = extractor.extract_amenities("Pool, quartz countertops, and a 2-car garage near the beach")
    assert {"pool", "quartz countertops", "garage"} <= set(found)
    assert "near the beach" not in found  # location terms aren't amenities


def test_extract_all_keys(extractor):
    result = extractor.extract_all("3 bedroom, 2 bath, 1500 square feet, offered at $650000, with a pool")
    assert result == {"bedrooms": 3, "bathrooms": 2, "price": 650000, "sqft": 1500,
                      "amenities": ["pool"]}


# ---------- labeled dataset and target score ----------

def test_labeled_dataset_size():
    test_set, dev_set = load_labeled("test"), load_labeled("dev")
    assert 200 <= len(test_set) + len(dev_set) <= 300


def test_f1_target_on_test_set(extractor):
    table, _ = score(load_labeled("test"), extractor)
    assert table.loc["overall", "f1"] >= 0.85
