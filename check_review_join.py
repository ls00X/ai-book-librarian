"""Check whether book_reviews.db can be joined to Book_Details.csv on book_id.

Run:  python check_review_join.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent / "data" / "raw"

bd = pd.read_csv(RAW / "Book_Details.csv", usecols=["book_id"])
con = sqlite3.connect(RAW / "book_reviews.db")
rv = pd.read_sql("SELECT DISTINCT book_id FROM book_reviews", con)
con.close()

bd_ids = set(bd["book_id"].astype(str))
rv_ids = set(rv["book_id"].astype(str))
overlap = bd_ids & rv_ids

print(f"Book_Details: {len(bd_ids)} ids, range {bd['book_id'].min()}..{bd['book_id'].max()}")
print(f"Reviews:      {len(rv_ids)} distinct book ids, range {rv['book_id'].min()}..{rv['book_id'].max()}")
print(f"Overlapping ids: {len(overlap)}  ({100*len(overlap)/max(len(bd_ids),1):.1f}% of books)")
print("-> JOIN WORKS — reviews can become a per-book sentiment feature."
      if overlap else
      "-> NO OVERLAP — the files use different id systems; reviews can't be joined by id.")
