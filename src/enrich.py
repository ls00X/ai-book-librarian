"""Enrich books with descriptions + categories from the Google Books API.

This is data source #2 for the NLP block (powers the semantic search and the LLM
context). Results are cached to DESCRIPTIONS_CACHE; re-running skips finished books,
so an interrupted run can simply be restarted.

Run:  python -m src.enrich
Optional: export GOOGLE_BOOKS_API_KEY=...  (higher quota)
"""
import json
import os
import time

import requests
from tqdm import tqdm

from src import config as cfg
from src.data_prep import load_books, book_key


def fetch_one(isbn13, title, author, api_key=None):
    isbn = str(isbn13).strip()
    if isbn and isbn not in ("nan", "0", "0.0", ""):
        params = {"q": f"isbn:{isbn}"}
    else:
        q = f"intitle:{title}"
        if author:
            q += f"+inauthor:{str(author).split(',')[0]}"
        params = {"q": q}
    if api_key:
        params["key"] = api_key
    try:
        r = requests.get(cfg.GOOGLE_BOOKS_API, params=params, timeout=15)
        if r.status_code != 200:
            return None
        items = r.json().get("items")
        if not items:
            return None
        vi = items[0].get("volumeInfo", {})
        return {"description": vi.get("description", ""), "categories": vi.get("categories", [])}
    except requests.RequestException:
        return None


def main():
    api_key = cfg.GOOGLE_BOOKS_API_KEY or os.environ.get("GOOGLE_BOOKS_API_KEY")
    books = load_books()
    if cfg.ENRICH_LIMIT:
        books = books.head(cfg.ENRICH_LIMIT).copy()

    cache = json.loads(cfg.DESCRIPTIONS_CACHE.read_text()) if cfg.DESCRIPTIONS_CACHE.exists() else {}
    todo = [row for _, row in books.iterrows() if book_key(row) not in cache]
    print(f"{len(cache)} already cached, {len(todo)} to fetch "
          f"({'with' if api_key else 'no'} API key)")

    for i, row in enumerate(tqdm(todo)):
        cache[book_key(row)] = fetch_one(
            row.get("isbn13"), row.get("title"), row.get("authors"), api_key
        ) or {"description": "", "categories": []}
        time.sleep(cfg.ENRICH_SLEEP)
        if i % 200 == 0:
            cfg.DESCRIPTIONS_CACHE.write_text(json.dumps(cache))

    cfg.DESCRIPTIONS_CACHE.write_text(json.dumps(cache))
    hit = sum(1 for v in cache.values() if v.get("description"))
    print(f"Saved {len(cache)} entries ({hit} with a description) to {cfg.DESCRIPTIONS_CACHE}")


if __name__ == "__main__":
    main()
