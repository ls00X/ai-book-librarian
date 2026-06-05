"""Central configuration.

Adjust BOOKS_COLUMNS / REVIEWS_COLUMNS to match the exact headers of YOUR csv files,
then everything else flows from here. Nothing else needs editing for a normal run.
"""
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw inputs (put your downloaded Kaggle files here) ------------------------
BOOKS_CSV = RAW_DIR / "Book_Details.csv"    # main metadata + description + genres
REVIEWS_DB = RAW_DIR / "book_reviews.db"    # SQLite: review text for the sentiment feature
REVIEWS_TABLE = "book_reviews"

# Processed artifacts -------------------------------------------------------
CATALOG_PARQUET = PROCESSED_DIR / "catalog.parquet"          # app-facing catalog (+ predicted_rating)
TRAINING_PARQUET = PROCESSED_DIR / "training_table.parquet"  # full table used for model training
DESCRIPTIONS_CACHE = PROCESSED_DIR / "descriptions_cache.json"
SENTIMENT_PARQUET = PROCESSED_DIR / "sentiment_by_title.parquet"

# Model / retrieval artifacts ----------------------------------------------
RATING_MODEL = MODELS_DIR / "rating_model.joblib"
EMBEDDINGS_NPY = MODELS_DIR / "book_embeddings.npy"
EMB_IDS_NPY = MODELS_DIR / "book_embedding_ids.npy"
TFIDF_VECTORIZER = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX = MODELS_DIR / "tfidf_matrix.joblib"

# --- Column mapping: standard_name -> column name in Book_Details.csv ------
BOOKS_COLUMNS = {
    "src_book_id": "book_id",            # native id (kept for the reviews join)
    "title": "book_title",
    "authors": "author",
    "average_rating": "average_rating",  # TARGET
    "num_pages": "num_pages",            # list-string e.g. "['652']" -> parsed
    "ratings_count": "num_ratings",
    "text_reviews_count": "num_reviews",
    "publication_date": "publication_info",  # list-string -> year parsed
    "description": "book_details",       # the blurb
    "genres": "genres",                  # list-string e.g. "['Fantasy', ...]"
    # NOTE: do NOT map rating_distribution -> it is the breakdown of average_rating (leakage).
}

# --- Reviews (SQLite) ------------------------------------------------------
REVIEWS_COLUMNS = {
    "book_id": "book_id",            # joins to BOOKS_COLUMNS["src_book_id"]
    "review_text": "review_content",
    "review_rating": "review_rating",
}

# --- Enrichment (Google Books API) ----------------------------------------
GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"

# >>> PASTE YOUR GOOGLE BOOKS API KEY HERE (between the quotes) <<<
# Works without a key too, just at a lower request quota.
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")

ENRICH_LIMIT = 4000      # cap #books to keep/enrich (deadline-friendly). Set None for all.
ENRICH_SLEEP = 0.2       # seconds between API calls

# --- Modeling --------------------------------------------------------------
TARGET = "average_rating"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TOP_N_GENRES = 15        # number of genre one-hot columns

# --- Embeddings / retrieval ------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CANDIDATE_POOL = 50      # candidates retrieved before re-ranking

# --- Ranking weights -------------------------------------------------------
RANK_WEIGHTS = {"semantic": 0.6, "rating": 0.25, "popularity": 0.15}
GENRE_FIT_BONUS = 0.05

# --- LLM -------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
