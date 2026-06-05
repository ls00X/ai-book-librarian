"""Build the two retrieval indices used by the NLP block:

  1. Dense embeddings via sentence-transformers (semantic search).
  2. A TF-IDF baseline (the comparison required for the NLP block: dense vs sparse retrieval).

Both are saved to models/ and loaded at inference time by recommend.py.

Run:  python -m src.embeddings
"""
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config as cfg


def _doc(row) -> str:
    parts = [str(row.get("title", "")), str(row.get("genre_str", "")), str(row.get("description", ""))]
    return ". ".join(p for p in parts if p and p != "nan")


def main():
    cat = pd.read_parquet(cfg.CATALOG_PARQUET)
    docs = cat.apply(_doc, axis=1).tolist()
    ids = cat["book_id"].to_numpy()

    print(f"Embedding {len(docs)} books with {cfg.EMBED_MODEL} ...")
    model = SentenceTransformer(cfg.EMBED_MODEL)
    emb = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(cfg.EMBEDDINGS_NPY, emb.astype("float32"))
    np.save(cfg.EMB_IDS_NPY, ids)

    tfidf = TfidfVectorizer(max_features=20000, stop_words="english")
    mat = tfidf.fit_transform(docs)
    joblib.dump(tfidf, cfg.TFIDF_VECTORIZER)
    joblib.dump(mat, cfg.TFIDF_MATRIX)

    print(f"Saved embeddings {emb.shape} and TF-IDF matrix {mat.shape}")


if __name__ == "__main__":
    main()
