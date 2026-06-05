"""Build the processed catalog and training table from the raw sources.

This dataset already contains `description` and `genres`, so the Google Books API
(src/enrich.py) is OPTIONAL — it is only used to fill books whose native description
is empty (and only if you ran it).

Run order: (optional) enrich.py, then sentiment.py, then this.
Output:
  - training_table.parquet : every feature column, used by train.py
  - catalog.parquet        : the app-facing subset (predicted_rating added later by train.py)
"""
import ast
import json
import re

import numpy as np
import pandas as pd

from src import config as cfg


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _parse_list(val):
    """Parse a genres/categories cell into a list of strings.

    Handles list-like strings ("['Fantasy', 'Fiction']"), comma-separated strings,
    real lists, and empties.
    """
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    s = str(val).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            out = ast.literal_eval(s)
            if isinstance(out, (list, tuple)):
                return [str(x).strip() for x in out]
        except Exception:
            pass
    return [t.strip() for t in s.split(",") if t.strip()]


def _first_int(val):
    """Extract the first integer from a list-string like "['652']" or "['912 pages, ...']"."""
    for tok in _parse_list(val):
        m = re.search(r"\d+", str(tok))
        if m:
            return int(m.group())
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else np.nan


def _extract_year(val):
    """Pull a 4-digit year out of e.g. "['First published July 16, 2005']"."""
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", str(val))
    return int(m.group()) if m else np.nan


def book_key(row) -> str:
    """Stable per-book id: native source id, else ISBN-13, else normalized title+author."""
    sid = str(row.get("src_book_id", "")).strip()
    if sid and sid not in ("nan", ""):
        return f"gr:{sid}"
    isbn = str(row.get("isbn13", "")).strip()
    if isbn and isbn not in ("nan", "0", "0.0", ""):
        return f"isbn:{isbn}"
    author = str(row.get("authors", "")).split(",")[0]
    return f"t:{_norm_title(row.get('title', ''))}|{_norm_title(author)}"


def load_books() -> pd.DataFrame:
    """Load Book_Details.csv and normalize to a standard schema."""
    df = pd.read_csv(cfg.BOOKS_CSV, on_bad_lines="skip", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    rename = {raw.strip(): std for std, raw in cfg.BOOKS_COLUMNS.items()
              if raw.strip() in df.columns}
    df = df.rename(columns=rename)

    df["average_rating"] = pd.to_numeric(df.get("average_rating"), errors="coerce")
    for col in ["ratings_count", "text_reviews_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "num_pages" in df.columns:
        df["num_pages"] = df["num_pages"].map(_first_int)
    if "publication_date" in df.columns:
        df["publication_year"] = df["publication_date"].map(_extract_year)

    df["description"] = df.get("description", "").fillna("").astype(str)
    df["categories"] = df.get("genres", "").map(_parse_list)

    df = df.dropna(subset=["title", "average_rating"]).reset_index(drop=True)
    df["book_id"] = df.apply(book_key, axis=1)
    # keep the raw source id (as string) for joining reviews
    if "src_book_id" in df.columns:
        df["src_book_id"] = df["src_book_id"].astype(str)
    df = df.drop_duplicates(subset="book_id").reset_index(drop=True)
    return df


def fill_missing_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """OPTIONAL: fill empty descriptions/genres from the Google Books cache, if it exists."""
    if not cfg.DESCRIPTIONS_CACHE.exists():
        return df
    cache = json.loads(cfg.DESCRIPTIONS_CACHE.read_text())
    empty = df["description"].str.len() == 0
    df.loc[empty, "description"] = df.loc[empty, "book_id"].map(
        lambda k: (cache.get(k) or {}).get("description", "") or ""
    )
    no_genre = df["categories"].map(len) == 0
    df.loc[no_genre, "categories"] = df.loc[no_genre, "book_id"].map(
        lambda k: (cache.get(k) or {}).get("categories", []) or []
    )
    return df


def attach_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if not cfg.SENTIMENT_PARQUET.exists() or "src_book_id" not in df.columns:
        df["sentiment_compound"] = np.nan
        df["n_reviews"] = 0
        return df
    sent = pd.read_parquet(cfg.SENTIMENT_PARQUET)  # src_book_id, sentiment_compound, n_reviews
    sent["src_book_id"] = sent["src_book_id"].astype(str)
    df = df.merge(sent, on="src_book_id", how="left")
    df["n_reviews"] = df["n_reviews"].fillna(0)
    return df


def build() -> pd.DataFrame:
    df = load_books()
    if cfg.ENRICH_LIMIT:
        df = df.head(cfg.ENRICH_LIMIT).copy()

    df = fill_missing_descriptions(df)   # no-op unless you ran src/enrich.py
    df = attach_sentiment(df)            # NaN-imputed later if no reviews

    # readable genre string for the app + LLM context
    df["genre_str"] = df["categories"].map(
        lambda xs: ", ".join(xs) if isinstance(xs, list) else str(xs)
    )

    # genre one-hot features (top-N most frequent categories)
    top_genres = (
        df["categories"].explode().dropna().value_counts().head(cfg.TOP_N_GENRES).index.tolist()
    )
    for g in top_genres:
        df[f"genre_{_slug(g)}"] = df["categories"].map(
            lambda xs, gg=g: int(gg in xs) if isinstance(xs, list) else 0
        )

    df.to_parquet(cfg.TRAINING_PARQUET, index=False)

    catalog_cols = [
        "book_id", "title", "authors", "average_rating", "ratings_count",
        "num_pages", "publication_year", "language_code", "description",
        "genre_str", "sentiment_compound",
    ]
    df[[c for c in catalog_cols if c in df.columns]].to_parquet(cfg.CATALOG_PARQUET, index=False)

    print(f"Built training_table ({len(df)} rows, {df.shape[1]} cols) and catalog.")
    print(f"  with description: {(df['description'].str.len() > 0).sum()}")
    print(f"  with sentiment:   {df['sentiment_compound'].notna().sum()}")
    print(f"  genre features:   {len(top_genres)} -> {top_genres}")
    return df


if __name__ == "__main__":
    build()
