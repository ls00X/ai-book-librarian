"""Quick schema inspector.

Put Book_Details.csv, books.db and book_reviews.db into data/raw/, then run:
    python inspect_data.py
Paste the printed output back so the loaders can be mapped to the real names.
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw"


def inspect_csv(p: Path):
    print(f"\n=== CSV: {p.name} ===")
    head = pd.read_csv(p, nrows=3)
    nrows = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
    print("approx rows:", nrows, "| columns:", list(head.columns))
    print(head.to_string(max_colwidth=70))


def inspect_db(p: Path):
    print(f"\n=== SQLite DB: {p.name} ===")
    con = sqlite3.connect(p)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
    print("tables:", tables)
    for t in tables:
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM '{t}'", con)["n"][0]
        cols = pd.read_sql(f"PRAGMA table_info('{t}')", con)["name"].tolist()
        print(f"\n  -- table '{t}'  ({n} rows)")
        print("  columns:", cols)
        print(pd.read_sql(f"SELECT * FROM '{t}' LIMIT 2", con).to_string(max_colwidth=70))
    con.close()


for name in ["Book_Details.csv", "books.db", "book_reviews.db"]:
    p = RAW / name
    if not p.exists():
        print(f"(missing: {p})")
        continue
    (inspect_csv if p.suffix == ".csv" else inspect_db)(p)
