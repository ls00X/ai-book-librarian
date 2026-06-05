# AI Applications Project Documentation Template

> Code references use the form `path` + function/section. Replace `<user>/<repo>` in links with
> your GitHub path, e.g. `https://github.com/<user>/<repo>/blob/main/src/train.py#L95-L120`.
> Values marked _[fill after training]_ come from `reports/model_comparison.csv` after you run the pipeline.

## Documentation Hint

References point to the function or section in the code rather than a fixed line, so they stay
valid as the code evolves.

## Project Metadata

- Project title: AI Book Librarian – Personalized Book Recommendation with Rating Prediction and Natural Language Explanations
- Student: Laura Stärk
- GitHub repository URL: <fill>
- Deployment URL: <(https://huggingface.co/spaces/lst0004/AI_Book_Librarian) — Hugging Face Space>
- Submission date: <fill> 2026-06-07

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [ ] Computer Vision

Primary blocks used for core solution (choose 2):

- Primary block 1: NLP (semantic retrieval, review sentiment, LLM explanation)
- Primary block 2: ML Numeric Data (rating prediction)

If a third block is selected, it is documented and graded separately as extra work. — N/A

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- Problem statement: Readers struggle to find their next book from huge catalogs; keyword search ignores mood/theme, and a single global average rating says nothing about fit. There is no tool that combines *what a reader wants in natural language* with *how well-received a book is* and *explains the choice*.
- Goal: Take a free-text request (favourite book / genre / mood), return a ranked shortlist combining semantic fit, a predicted rating and popularity, and produce an honest natural-language explanation.
- Success criteria: (1) rating model beats the mean-rating baseline on RMSE; (2) embedding retrieval returns topically relevant books (qualitative check + vs. TF-IDF baseline); (3) explanations are grounded in the actual retrieved data; (4) a working deployed app.

### 1.2 Integration Logic

- How the selected blocks interact:
  - NLP → ML: review sentiment (VADER) is aggregated per book and used as a **feature** in the rating model — see `attach_sentiment()` in `src/data_prep.py` and the `NUMERIC` feature list in `src/train.py`.
  - ML → NLP: the trained model's **predicted_rating** is written into the catalog (`src/train.py`, end of `main()`), then used both as a ranking term (`Recommender.recommend()` in `src/recommend.py`) and as grounding in the LLM prompt (`_books_block()` in `src/explain.py`).
- Data and output flow between blocks:

```
reviews ─VADER─▶ sentiment feature ─┐
books + Google Books descriptions ──┼─▶ ML rating model ─▶ predicted_rating
                                    │                              │
user query ─embedding search─▶ candidates ─▶ ranking(sem, rating, pop) ─▶ top-k ─▶ LLM explanation
```

---

## 2. Block Documentation

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Book_Details.csv (Kaggle: https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024?select=Book_Details.csv) | Structured CSV | ~16k books | Metadata features + target (`average_rating`) |
| 2 | book_reviews.db (same Kaggle collection: https://www.kaggle.com/datasets/dk123891/books-dataset-goodreadsmay-2024?select=book_reviews.db) | SQLite review text | ~63k reviews | Source of the NLP-derived `sentiment_compound` feature (joined on `book_id`) |

#### 2A.2 Preprocessing and Features

- Cleaning steps: header strip (leading-space `num_pages` bug), numeric coercion, parse `publication_date`→year, drop rows without title/rating, de-duplicate by ISBN/title key — `load_books()` in `src/data_prep.py`.
- Preprocessing steps: median impute + standardize numerics; most-frequent impute + one-hot for `language_code`; log1p on `ratings_count`/`text_reviews_count` — `build_preprocessor()` / `load_xy()` in `src/train.py`.
- Feature engineering and selection: `sentiment_compound` (NLP-derived), `publication_year`, top-15 genre one-hots, log-scaled popularity counts.

#### 2A.3 Model Selection

- Models tested: Ridge regression, Random Forest, Gradient Boosting (and XGBoost if installed).
- Why these models were chosen: Ridge = linear baseline; tree ensembles capture non-linear interactions between popularity, length and sentiment; spanning linear→ensemble shows whether complexity is justified on a narrow target.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Metadata only, no sentiment | Ridge, RF, GB | RMSE _[fill]_ | vs. mean-baseline RMSE _[fill]_ |
| 2 | Add NLP feature | + `sentiment_compound` | same | RMSE _[fill]_ | _[fill]_ |
| 3 | Tune best | + genre one-hots / XGBoost | best of above | RMSE _[fill]_ | _[fill]_ |

> Fill from `reports/model_comparison.csv`. To produce the iteration-1 numbers, run once with the sentiment feature removed from `NUMERIC` in `src/train.py`.

#### 2A.5 Evaluation and Error Analysis

- Metrics used: RMSE, MAE, R² on a 20% hold-out + 5-fold CV (`src/train.py`).
- Final results: 
  Model	RMSE	MAE	R²	CV RMSE
  Ridge Regression	0.226	0.176	0.234	0.226
  Random Forest	0.223	0.170	0.252	0.223
  Gradient Boosting	0.220	0.170	0.273	0.222
  XGBoost	0.222	0.170	0.261	0.220

  Gradient Boosting achieved the best overall performance with the lowest hold-out RMSE (0.220) and the highest R² (0.273). Therefore, it was selected as the final production model and saved as rating_model.joblib.

- Error patterns and likely causes: largest residuals can be inspected in reports/worst_predictions.csv. The largest prediction errors occur mainly for niche books with few ratings and for books whose popularity or reception is not fully captured by the available metadata. The relatively narrow Goodreads rating distribution also limits the achievable R².


#### 2A.6 Integration with Other Block(s)

- Inputs received from other block(s): `sentiment_compound` from the NLP sentiment step.
- Outputs provided to other block(s): `predicted_rating` per book → ranking term and LLM grounding.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Book_Details.csv (`book_details` field) | Text | ~16k descriptions | Documents embedded for semantic search |
| 2 | book_reviews.db (`review_content`) | SQLite text | ~63k reviews | VADER sentiment (feeds ML) + groundable in explanations |
| 3 | User query | Text (runtime) | 1 string | Search query + LLM input |

#### 2B.2 Preprocessing and Prompt Design

- Text preprocessing: build per-book document `title + genres + description` (`_doc()` in `src/embeddings.py`); title normalization for the sentiment join (`_norm_title()`); review text truncated to 1000 chars before VADER (`src/sentiment.py`).
- Prompt design or retrieval setup: dense embeddings with `all-MiniLM-L6-v2`, cosine similarity, top-`CANDIDATE_POOL` then re-rank. Two LLM prompt variants (`concise`, `structured`) grounded in retrieved book data — `PROMPT_VARIANTS` in `src/explain.py`.

#### 2B.3 Approach Selection

- Approach used: transformer sentence-embeddings for retrieval + classical VADER for sentiment + prompt-engineered LLM (light RAG) for explanation.
- Alternatives considered: TF-IDF retrieval (kept as the comparison baseline); transformer sentiment model (VADER chosen for speed/no-download on the deadline).

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Retrieval baseline | TF-IDF | sparse cosine | top-5 relevance (manual) _[fill]_ | — |
| 2 | Dense retrieval | MiniLM embeddings | dense cosine | top-5 relevance _[fill]_ | _[fill]_ |
| 3 | Explanation prompt | concise vs structured | same retrieval | groundedness / usefulness _[fill]_ | _[fill]_ |

#### 2B.5 Evaluation and Error Analysis

- Evaluation strategy: qualitative side-by-side of TF-IDF vs. embedding retrieval on a fixed set of test queries; qualitative review of both prompt variants for groundedness (does it invent facts?) and honesty (does it flag loose matches?).
- Results:
Dense retrieval using all-MiniLM-L6-v2 produced more semantically relevant recommendations than TF-IDF, especially for abstract queries describing moods, themes, or reading preferences rather than exact keywords.

The final system therefore uses dense embedding retrieval as the default recommendation method.

For explanations, the structured prompt variant was selected because it produced clearer and more informative recommendations while remaining grounded in the retrieved book metadata and predicted ratings.

As a validation of the sentiment feature, VADER sentiment scores were compared against user star ratings on 61,233 reviews:

Pearson correlation: r = 0.178
Number of reviews: 61,233

Although the correlation is moderate, it confirms that review sentiment contains useful information and can contribute predictive value to the rating model.
- Error patterns and likely causes: embedding retrieval can occasionally favor highly popular books because they have richer descriptions and metadata. Retrieval quality also depends on description coverage; books with short or missing descriptions provide weaker semantic signals.

#### 2B.6 Integration with Other Block(s)

- Inputs received from other block(s): `predicted_rating` (ML) used in ranking and shown in the explanation prompt.
- Outputs provided to other block(s): `sentiment_compound` feature feeding the ML model.

### 2C. Computer Vision (If selected)

N/A — not selected.

---

## 3. Deployment

- Deployment URL: <fill — Hugging Face Space>
- Main user flow: enter a free-text reading request → view ranked book cards (actual vs. predicted rating, match score) → read the LLM "Why these books?" note. Sidebar controls retrieval method, ranking weights and explanation style.
- Screenshot or short demo: _[add screenshots to `reports/figures/` and reference here]_.

---

## 4. Execution Instructions

- Environment setup: `python -m venv .venv && pip install -r requirements.txt`; set `OPENAI_API_KEY`.
- Data setup: place `books.csv` and `reviews.csv` in `data/raw/` (links in README §2).
- Training command(s):
  ```bash
  python check_review_join.py
  python -m src.sentiment
  python -m src.data_prep
  python -m src.train
  python -m src.embeddings
  ```
- Inference/run command(s): `streamlit run app.py`
- Reproducibility notes: fixed `RANDOM_STATE=42`; all parameters centralized in `src/config.py`; training and inference are fully separated (the app only loads saved artifacts).

---

## 5. Optional Bonus Evidence

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [x] Extended evaluation (TF-IDF vs. dense retrieval; two explanation prompts; VADER-vs-star-rating sanity check)
- [ ] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

Evidence for selected bonus items:

- Retrieval comparison: TF-IDF versus dense transformer embeddings (`all-MiniLM-L6-v2`).
- Explanation comparison: concise versus structured prompt variants.
- Quantitative sentiment validation: VADER sentiment scores compared against 61,233 Goodreads review ratings, yielding a Pearson correlation of r = 0.178.
- Multiple evaluation perspectives combining quantitative metrics (RMSE, MAE, R², correlation analysis) and qualitative assessment (retrieval relevance and explanation quality).

