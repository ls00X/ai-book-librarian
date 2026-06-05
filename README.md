---
title: AI Book Librarian
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

# 📚 AI Book Librarian

Personalized book recommendation with rating prediction and natural-language explanations.
Combines two AI blocks in one pipeline:

- **NLP** — semantic search over book descriptions (sentence-transformer embeddings + a TF-IDF
  baseline), review-sentiment extraction (VADER), and an LLM that explains each recommendation.
- **ML Numeric** — a regression model that predicts a book's expected rating from its metadata
  + popularity + the NLP-derived sentiment feature.

**Integration:** sentiment (NLP) → feature for the rating model (ML); predicted rating (ML)
→ ranking signal + grounding for the LLM explanation (NLP).

```
query → embedding search → candidate books
                              │  + predicted_rating (ML model)
                              │  + popularity
                              ▼
                      weighted ranking → top-k → LLM explanation
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...            # for explanations (Windows: set OPENAI_API_KEY=...)
# optional, higher enrichment quota:
export GOOGLE_BOOKS_API_KEY=...
```

## 2. Data

Two different data sources from one Goodreads collection on Kaggle (not used in the course):
https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024

1. **Book_Details.csv** → `data/raw/Book_Details.csv` — metadata + description (`book_details`) + genres
   (ML features, the target `average_rating`, and the text embedded for semantic search).
2. **book_reviews.db** (SQLite) → `data/raw/book_reviews.db` — review text → VADER sentiment feature,
   joined to books on `book_id`.

(`books.db` from the same download is not used.) If headers differ, edit `BOOKS_COLUMNS` /
`REVIEWS_COLUMNS` in `src/config.py`. `ENRICH_LIMIT` caps how many books are kept (speed vs. size).
Run `python check_review_join.py` once to confirm the reviews join.

## 3. Build artifacts (training — run once, locally)

```bash
python check_review_join.py   # one-time: confirm reviews join to books
python -m src.sentiment    # VADER sentiment per book from book_reviews.db  (NLP -> ML feature)
python -m src.data_prep    # build catalog.parquet + training_table.parquet
python -m src.train        # compare models, save best, write predicted_rating into catalog
python -m src.embeddings   # build dense + TF-IDF retrieval indices
```

(`src/enrich.py` / Google Books API is optional — only needed if you ever want to fill missing
descriptions. This dataset already has them natively.)

Outputs land in `models/` and `data/processed/`. EDA: open `notebooks/eda.ipynb`.

## 4. Run the app (inference)

```bash
streamlit run app.py
```

The app loads only saved artifacts and never trains (clean train/inference separation).

## 5. Deploy (Hugging Face Spaces)

The repo is Spaces-ready (Streamlit SDK declared in the front-matter above).

1. On huggingface.co: **New Space → SDK: Streamlit**. This creates a git repo.
2. Push your project into that Space repo — including the artifacts the app loads:
   `data/processed/catalog.parquet` and everything in `models/` (raw data stays out, see `.gitignore`).
   Files >10 MB need Git LFS (`git lfs track "*.npy"`), but at ~4k books they should stay small.
3. **Settings → Variables and secrets → New secret**: `OPENAI_API_KEY = sk-...`
4. The Space builds and gives you the public URL → put it in `documentation.md`.

> Submission: the GitHub repo is the deliverable (add collaborators `jasminh` and `bkuehnis`);
> the Hugging Face Space URL is the required deployment link inside the documentation.

(Streamlit Community Cloud works too, with the same artifacts committed and the key set under Secrets.)

## Project layout

```
src/config.py      paths, column maps, all parameters
src/enrich.py      Google Books description/genre enrichment (cached)
src/sentiment.py   VADER review sentiment  (NLP → ML feature)
src/data_prep.py   build catalog + training table
src/train.py       model comparison, error analysis, predicted_rating  (ML block)
src/embeddings.py  dense + TF-IDF indices  (NLP retrieval)
src/recommend.py   query → retrieve → re-rank  (inference core)
src/explain.py     OpenAI explanation, 2 prompt variants  (NLP → output)
app.py             Streamlit UI
notebooks/eda.ipynb  exploratory data analysis
documentation.md   filled course documentation template
```
