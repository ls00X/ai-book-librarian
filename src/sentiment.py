"""Compute a sentiment score per book from review text (VADER), read from the SQLite reviews DB.

Clean integration: reviews join to Book_Details on book_id (verified high coverage), so the
aggregated compound sentiment becomes a per-book feature in the rating model (NLP -> ML).
Also prints the VADER-vs-star-rating correlation as a quantitative NLP sanity check.

Output: SENTIMENT_PARQUET keyed on `src_book_id` (joined in data_prep.attach_sentiment).

Run:  python -m src.sentiment
"""
import re
import sqlite3

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src import config as cfg


def _parse_rating(val):
    """'Rating 5 out of 5' -> 5.0 ; NaN otherwise."""
    m = re.search(r"\d+", str(val))
    return float(m.group()) if m else np.nan


def main(max_reviews_per_book: int = 200):
    cols = cfg.REVIEWS_COLUMNS
    con = sqlite3.connect(cfg.REVIEWS_DB)
    df = pd.read_sql(
        f"SELECT {cols['book_id']} AS book_id, "
        f"{cols['review_text']} AS text, "
        f"{cols['review_rating']} AS rating FROM {cfg.REVIEWS_TABLE}",
        con,
    )
    con.close()

    df = df.dropna(subset=["text"])
    df["book_id"] = df["book_id"].astype(str)
    df = df.groupby("book_id", group_keys=False).head(max_reviews_per_book)

    sia = SentimentIntensityAnalyzer()
    df["compound"] = (
        df["text"].astype(str).str.slice(0, 1000).map(lambda t: sia.polarity_scores(t)["compound"])
    )

    # quantitative sanity check: does text sentiment track the star rating?
    df["stars"] = df["rating"].map(_parse_rating)
    valid = df.dropna(subset=["stars"])
    if len(valid) > 50:
        r = valid["compound"].corr(valid["stars"])
        print(f"VADER compound vs. star rating: r = {r:.3f} (n={len(valid)})")

    agg = (
        df.groupby("book_id")
        .agg(sentiment_compound=("compound", "mean"), n_reviews=("compound", "size"))
        .reset_index()
        .rename(columns={"book_id": "src_book_id"})
    )
    agg.to_parquet(cfg.SENTIMENT_PARQUET, index=False)
    print(f"Sentiment computed for {len(agg)} books "
          f"(median {agg['n_reviews'].median():.0f} reviews/book) -> {cfg.SENTIMENT_PARQUET}")


if __name__ == "__main__":
    main()
