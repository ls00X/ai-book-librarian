"""Train and compare rating-prediction models (the ML Numeric block).

- Target: average_rating (a book's community reception).
- Features: book metadata + popularity + the NLP-derived sentiment feature + genre one-hots.
- Compares Ridge / RandomForest / GradientBoosting (+ XGBoost if installed).
- Saves: best pipeline (rating_model.joblib), a comparison report, an error-analysis file,
  and writes predicted_rating back into the catalog -> this is the ML output the ranking
  and LLM-explanation blocks consume.

Run:  python -m src.train
"""
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as cfg

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

NUMERIC = ["num_pages", "ratings_count", "text_reviews_count", "publication_year", "sentiment_compound"]
CATEGORICAL = ["language_code"]


def load_xy():
    df = pd.read_parquet(cfg.TRAINING_PARQUET)
    # log-transform highly skewed count features
    for c in ["ratings_count", "text_reviews_count"]:
        if c in df.columns:
            df[c] = np.log1p(df[c])
    df = df.dropna(subset=[cfg.TARGET]).reset_index(drop=True)

    genre_cols = [c for c in df.columns if c.startswith("genre_") and c != "genre_str"]
    features = (
        [c for c in NUMERIC if c in df.columns]
        + [c for c in CATEGORICAL if c in df.columns]
        + genre_cols
    )
    return df, df[features], df[cfg.TARGET], features, genre_cols


def build_preprocessor(features, genre_cols):
    num = [c for c in NUMERIC if c in features]
    cat = [c for c in CATEGORICAL if c in features]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("sc", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("oh", OneHotEncoder(handle_unknown="ignore",
                                                                    sparse_output=False))]), cat))
    if genre_cols:
        transformers.append(("genre", "passthrough", genre_cols))
    return ColumnTransformer(transformers)


def main():
    warnings.filterwarnings("ignore")
    df, X, y, features, genre_cols = load_xy()
    print(f"Training on {len(df)} books, {len(features)} features.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    pre = build_preprocessor(features, genre_cols)

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=cfg.RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=cfg.RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            random_state=cfg.RANDOM_STATE, n_jobs=-1,
        )

    rows, fitted = [], {}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae = float(mean_absolute_error(yte, pred))
        r2 = float(r2_score(yte, pred))
        cv = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
        cv_rmse = float(np.sqrt(-cv.mean()))
        rows.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "CV_RMSE": cv_rmse})
        fitted[name] = pipe
        print(f"{name:16s} RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}  CV_RMSE={cv_rmse:.3f}")

    report = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    report.to_csv(cfg.REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = report.iloc[0]["model"]
    best = fitted[best_name]
    joblib.dump(best, cfg.RATING_MODEL)
    print(f"\nBest model: {best_name}  ->  {cfg.RATING_MODEL}")

    # --- Error analysis: worst predictions + simple residual stats -----------
    pred_te = best.predict(Xte)
    err = pd.DataFrame({"y_true": yte.values, "y_pred": pred_te})
    err["residual"] = err["y_true"] - err["y_pred"]
    err["abs_err"] = err["residual"].abs()
    err.sort_values("abs_err", ascending=False).head(25).to_csv(
        cfg.REPORTS_DIR / "worst_predictions.csv", index=False
    )
    print(f"Residual mean={err['residual'].mean():.3f}  std={err['residual'].std():.3f}")

    # --- Feature importance (if available) -----------------------------------
    try:
        model_obj = best.named_steps["model"]
        if hasattr(model_obj, "feature_importances_"):
            names = best.named_steps["pre"].get_feature_names_out()
            fi = pd.DataFrame({"feature": names, "importance": model_obj.feature_importances_})
            fi.sort_values("importance", ascending=False).to_csv(
                cfg.REPORTS_DIR / "feature_importance.csv", index=False
            )
    except Exception as e:
        print(f"(feature importance skipped: {e})")

    # --- ML output -> other blocks: predicted_rating for every catalog book --
    full = pd.read_parquet(cfg.TRAINING_PARQUET)
    for c in ["ratings_count", "text_reviews_count"]:
        if c in full.columns:
            full[c] = np.log1p(full[c])
    preds = pd.DataFrame({"book_id": full["book_id"], "predicted_rating": best.predict(full[features])})

    catalog = pd.read_parquet(cfg.CATALOG_PARQUET)
    catalog = catalog.drop(columns=["predicted_rating"], errors="ignore").merge(preds, on="book_id", how="left")
    catalog.to_parquet(cfg.CATALOG_PARQUET, index=False)
    print("Wrote predicted_rating into catalog.parquet")


if __name__ == "__main__":
    main()
