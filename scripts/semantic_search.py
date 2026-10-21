"""Search listings by meaning (semantic search) and by keywords (BM25).

Run from the project root to build the search index for every listing:
    python scripts/data_loading.py --all    # every listing with remarks -> all_listings.csv
    python scripts/semantic_search.py       # embeds them and saves the index to data/models/

Then search:
    from scripts.semantic_search import SemanticSearcher
    searcher = SemanticSearcher.load()
    searcher.search("quiet home surrounded by nature", top_k=5)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")      # lets PyTorch and FAISS run together on a Mac
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "scripts." imports work when run directly

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, models

from scripts.text_cleaning import TextCleaner

ROOT = Path(__file__).resolve().parent.parent
LISTINGS_PATH = ROOT / "data" / "processed" / "all_listings.csv"
MODEL_NAME = "all-MiniLM-L6-v2"                             # small, fast model with 384 numbers per text
MODEL_DIR = ROOT / "data" / "models" / MODEL_NAME
INDEX_DIR = ROOT / "data" / "models" / "semantic_index"

STOPWORDS = set("""a an and are as at be by for from has have home in is it its of on or our that the
this to with your you will all this these those into over than""".split())


def load_model():
    """Use the saved copy of the model if there is one, otherwise download it and save a copy."""
    if MODEL_DIR.exists():
        try:
            return SentenceTransformer(str(MODEL_DIR))
        except Exception:
            # Saved by a different sentence-transformers version: rebuild it from the same weights
            word = models.Transformer(str(MODEL_DIR), max_seq_length=256)
            pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
            return SentenceTransformer(modules=[word, pooling, models.Normalize()])
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(MODEL_DIR))
    return model


def load_listings(path=LISTINGS_PATH):
    """All listings with cleaned remarks, one row per listing."""
    df = pd.read_csv(path).drop_duplicates("L_ListingID").reset_index(drop=True)
    df["remarks_clean"] = df["remarks"].apply(TextCleaner().clean_text)
    return df


class SemanticSearcher:
    """Finds listings whose descriptions mean something similar to the query."""

    def __init__(self, model=None):
        self.model = model or load_model()
        self.cleaner = TextCleaner()
        self.index = None
        self.listing_ids = None

    def embed(self, texts, batch_size=64, show_progress=False):
        """Turn each text into a vector (a list of 384 numbers) that captures its meaning.
        Vectors are scaled to length 1, so inner product = cosine similarity."""
        vectors = self.model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                                    convert_to_numpy=True, normalize_embeddings=True)
        return vectors.astype("float32")

    def build_index(self, texts, listing_ids, show_progress=True):
        vectors = self.embed(texts, show_progress=show_progress)
        self.index = faiss.IndexFlatIP(vectors.shape[1])   # exact search by inner product
        self.index.add(vectors)
        self.listing_ids = np.asarray(listing_ids)

    def search(self, query, top_k=10, only_ids=None):
        """Top matches as a table of listing_id and score (1 = same meaning, 0 = unrelated).
        only_ids limits the search to certain listings, e.g. ones that passed the Week 4 filters."""
        query_vector = self.embed([self.cleaner.clean_text(query)])
        params = None
        if only_ids is not None:
            positions = np.flatnonzero(np.isin(self.listing_ids, list(only_ids))).astype("int64")
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(positions))
        scores, positions = self.index.search(query_vector, top_k, params=params)
        keep = positions[0] >= 0
        return pd.DataFrame({"listing_id": self.listing_ids[positions[0][keep]],
                             "score": scores[0][keep].round(4)})

    def save(self, folder=INDEX_DIR):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "listings.faiss"))
        np.save(folder / "listing_ids.npy", self.listing_ids)

    @classmethod
    def load(cls, folder=INDEX_DIR, model=None):
        searcher = cls(model)
        searcher.index = faiss.read_index(str(Path(folder) / "listings.faiss"))
        searcher.listing_ids = np.load(Path(folder) / "listing_ids.npy", allow_pickle=True)
        return searcher


def tokenize(text):
    """Lowercase words without common filler words, for keyword search."""
    return [w for w in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", str(text).lower()) if w not in STOPWORDS]


class KeywordSearcher:
    """BM25 keyword search: ranks listings by how often (and how rarely elsewhere) the query words appear."""

    def __init__(self):
        self.cleaner = TextCleaner()
        self.bm25 = None
        self.listing_ids = None

    def build_index(self, texts, listing_ids):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.listing_ids = np.asarray(listing_ids)

    def search(self, query, top_k=10):
        scores = self.bm25.get_scores(tokenize(self.cleaner.clean_text(query)))
        top = np.argsort(-scores)[:top_k]
        top = top[scores[top] > 0]                           # skip listings with none of the words
        return pd.DataFrame({"listing_id": self.listing_ids[top], "score": scores[top].round(2)})


if __name__ == "__main__":
    start = time.time()
    listings = load_listings()
    print(f"Loaded and cleaned {len(listings):,} listings ({time.time() - start:.0f}s)")

    start = time.time()
    searcher = SemanticSearcher()
    searcher.build_index(listings["remarks_clean"], listings["L_ListingID"])
    searcher.save()
    print(f"Built and saved the semantic index ({time.time() - start:.0f}s) -> {INDEX_DIR.relative_to(ROOT)}")

    print("\nTest search: 'quiet home surrounded by nature'")
    print(searcher.search("quiet home surrounded by nature", top_k=5).to_string(index=False))
