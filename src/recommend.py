"""Inference core shared by the Streamlit app.

Pipeline per query:
  query --(NLP embedding/TF-IDF)--> semantic similarity over the catalog
        --> candidate pool --> re-rank with:
              w_sem * semantic + w_rating * predicted_rating + w_pop * popularity (+ genre-fit bonus)
The predicted_rating term is the ML block's output feeding the ranking decision logic.
"""
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src import config as cfg


def _minmax(x):
    x = np.asarray(x, dtype="float32")
    rng = x.max() - x.min()
    return np.zeros_like(x) if rng < 1e-9 else (x - x.min()) / rng


class Recommender:
    def __init__(self):
        self.emb = np.load(cfg.EMBEDDINGS_NPY)
        self.emb_ids = np.load(cfg.EMB_IDS_NPY, allow_pickle=True)
        self.model = SentenceTransformer(cfg.EMBED_MODEL)
        self.tfidf = joblib.load(cfg.TFIDF_VECTORIZER)
        self.tfidf_mat = joblib.load(cfg.TFIDF_MATRIX)
        # align the catalog row order to the embedding id order
        cat = pd.read_parquet(cfg.CATALOG_PARQUET).set_index("book_id")
        self.catalog = cat.loc[self.emb_ids].reset_index()

    def _semantic(self, query: str, method: str) -> np.ndarray:
        if method == "tfidf":
            qv = self.tfidf.transform([query])
            return cosine_similarity(qv, self.tfidf_mat).ravel()
        qe = self.model.encode([query], normalize_embeddings=True)[0]
        return self.emb @ qe  # both normalized -> cosine similarity

    def recommend(self, query, top_k=5, weights=None, method="embedding") -> pd.DataFrame:
        weights = weights or cfg.RANK_WEIGHTS
        sims = self._semantic(query, method)
        pool = np.argsort(sims)[::-1][:cfg.CANDIDATE_POOL]

        c = self.catalog.iloc[pool].copy()
        c["semantic"] = sims[pool]

        pred = c["predicted_rating"] if "predicted_rating" in c.columns else c["average_rating"]
        pred = pred.fillna(pred.median())
        pop = np.log1p(c["ratings_count"].fillna(0))

        score = (
            weights["semantic"] * _minmax(c["semantic"])
            + weights["rating"] * _minmax(pred)
            + weights["popularity"] * _minmax(pop)
        )
        # light genre-fit bonus: query words appearing in the genre string
        ql = {t for t in query.lower().split() if len(t) > 3}
        gfit = c["genre_str"].fillna("").str.lower().map(lambda g: any(t in g for t in ql))
        score = score + cfg.GENRE_FIT_BONUS * gfit.astype(float).to_numpy()

        c["score"] = score
        return c.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
