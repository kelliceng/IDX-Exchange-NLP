import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from scripts.semantic_search import INDEX_DIR, KeywordSearcher, SemanticSearcher, tokenize

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

# Made-up listings so the tests don't depend on company data
LISTINGS = {
    1: "Charming cabin tucked in the pines with peaceful forest views",
    2: "Downtown loft steps from restaurants, nightlife, and shopping",
    3: "Chef's kitchen with a large island, double ovens, and a gas range",
    4: "Horse property on 5 acres with a barn and riding arena",
    5: "Fixer upper that needs TLC, great investment opportunity",
}


@pytest.fixture(scope="module")
def semantic():
    searcher = SemanticSearcher()
    searcher.build_index(list(LISTINGS.values()), list(LISTINGS), show_progress=False)
    return searcher


@pytest.fixture(scope="module")
def keyword():
    searcher = KeywordSearcher()
    searcher.build_index(list(LISTINGS.values()), list(LISTINGS))
    return searcher


# ---------- semantic search ----------

def test_embedding_size(semantic):
    assert semantic.embed(["test"]).shape == (1, 384)


@pytest.mark.parametrize("query, best_match", [
    ("quiet home in the woods", 1),
    ("walk to bars and cafes", 2),
    ("perfect for someone who loves to cook", 3),
    ("equestrian ranch", 4),
    ("needs work", 5),
])
def test_semantic_finds_meaning(semantic, query, best_match):
    assert semantic.search(query, top_k=1)["listing_id"].iloc[0] == best_match


def test_scores_are_sorted(semantic):
    scores = semantic.search("cabin", top_k=5)["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_search_within_filtered_listings(semantic):
    # e.g. only listings that passed the Week 4 filters
    results = semantic.search("quiet home in the woods", top_k=5, only_ids=[2, 3])
    assert set(results["listing_id"]) <= {2, 3}


def test_save_and_load(semantic, tmp_path):
    semantic.save(tmp_path)
    loaded = SemanticSearcher.load(tmp_path, model=semantic.model)
    assert loaded.search("needs work", top_k=1)["listing_id"].iloc[0] == 5


# ---------- keyword search ----------

def test_tokenize_drops_filler_words():
    assert tokenize("The Pool and the Spa") == ["pool", "spa"]


def test_keyword_finds_exact_words(keyword):
    assert keyword.search("barn arena", top_k=1)["listing_id"].iloc[0] == 4


def test_keyword_misses_synonyms(keyword):
    # No listing uses the word "equestrian", so keyword search finds nothing. Semantic search finds the horse property.
    assert keyword.search("equestrian", top_k=5).empty


# ---------- speed ----------

def test_latency_under_100ms_for_10k_listings(semantic):
    vectors = np.random.default_rng(0).normal(size=(10_000, 384)).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(384)
    index.add(vectors)
    semantic.embed(["warm up"])

    times = []
    for _ in range(20):
        start = time.perf_counter()
        query = semantic.embed(["quiet home surrounded by nature"])
        index.search(query, 10)
        times.append(time.perf_counter() - start)
    assert np.median(times) < 0.100


# ---------- full index and evaluation ----------

@pytest.mark.skipif(not (INDEX_DIR / "listings.faiss").exists(), reason="run scripts/semantic_search.py first")
def test_full_index_built():
    searcher = SemanticSearcher.load()
    assert searcher.index.ntotal == len(searcher.listing_ids) > 50_000
    assert searcher.index.d == 384


def test_relevance_judgments():
    judgments = pd.read_csv(DATA / "search_relevance.csv")
    per_method = judgments.groupby("method").size()
    assert per_method["semantic"] >= 50 and per_method["keyword"] >= 50
    assert set(judgments["relevant"]) <= {0, 1}
