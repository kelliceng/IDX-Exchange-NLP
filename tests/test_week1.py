import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.taxonomy_builder import CATEGORIES, compile_matchers, match_terms

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

INTENTS = {
    "search_listings",
    "property_details",
    "pricing_valuation",
    "compare_listings",
    "neighborhood_info",
    "schedule_showing",
    "financing_costs",
}


@pytest.fixture(scope="module")
def taxonomy():
    with open(DATA / "taxonomy.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(DATA / "listing_sample.csv")


@pytest.fixture(scope="module")
def queries():
    return pd.read_csv(DATA / "sample_queries.csv")


# ---------- taxonomy ----------

def test_taxonomy_loaded(taxonomy):
    assert len(taxonomy["terms"]) >= 200
    assert all("id" in t and "term" in t for t in taxonomy["terms"])


def test_taxonomy_has_8_categories(taxonomy):
    assert set(taxonomy["categories"]) == set(CATEGORIES)
    used = {t["category"] for t in taxonomy["terms"]}
    assert used == set(CATEGORIES)


def test_taxonomy_ids_and_terms_unique(taxonomy):
    ids = [t["id"] for t in taxonomy["terms"]]
    words = [w for t in taxonomy["terms"] for w in [t["term"], *t["synonyms"]]]
    assert len(ids) == len(set(ids))
    assert len(words) == len(set(words))


# ---------- listing sample ----------

def test_sample_data_quality(sample):
    assert len(sample) >= 500
    assert sample["remarks"].str.len().min() > 50
    assert sample["L_ListingID"].is_unique


# ---------- user queries ----------

def test_sample_queries(queries):
    assert len(queries) >= 50
    assert set(queries["intent"]) <= INTENTS
    assert queries["intent"].value_counts().min() >= 5  # every intent has examples


# ---------- coverage ----------

def test_remark_coverage(taxonomy, sample):
    """At least 30% of remarks mention at least one taxonomy term."""
    matchers = compile_matchers(taxonomy)
    covered = sample["remarks"].apply(lambda r: len(match_terms(r, matchers)) > 0)
    assert covered.mean() >= 0.30


def test_term_coverage(taxonomy, sample):
    """At least 30% of taxonomy terms show up somewhere in the sample."""
    text = " ".join(sample["remarks"])
    matchers = compile_matchers(taxonomy)
    found = match_terms(text, matchers)
    assert len(found) / len(taxonomy["terms"]) >= 0.30
