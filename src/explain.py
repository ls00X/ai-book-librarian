"""Natural-language explanation of the recommendations (OpenAI).

This is the NLP block consuming the ML output: the prompt is grounded ("light RAG") in the
actual retrieved books, including the model's predicted rating, so the LLM explains *why*
the numeric/semantic pipeline chose them. Two prompt variants are provided as the required
NLP comparison. If no API key is set, a template fallback keeps the app functional.
"""
import os

import pandas as pd

from src import config as cfg

PROMPT_VARIANTS = {
    "concise": (
        "You are a knowledgeable librarian. In 3-4 sentences, explain to the reader why the "
        "books below fit their request. Be specific and honest; say so if a pick is only a "
        "loose match. Do not invent facts beyond the data given.\n\n"
        "Reader request: {query}\n\nBooks:\n{books}"
    ),
    "structured": (
        "You are a knowledgeable librarian. For the reader request and the candidate books below, "
        "write a short note with three parts:\n"
        "1) Why these fit (reference concrete themes/genres).\n"
        "2) Possible downsides or caveats (length, pacing, or a lower predicted rating).\n"
        "3) One 'if you like these, also try' suggestion.\n"
        "Be honest and specific. Do not invent facts beyond the data given.\n\n"
        "Reader request: {query}\n\nBooks:\n{books}"
    ),
}


def _books_block(results: pd.DataFrame) -> str:
    lines = []
    for _, b in results.iterrows():
        pred = b.get("predicted_rating")
        pred_s = f"{pred:.2f}" if pred == pred else "n/a"  # NaN check
        desc = (b.get("description") or "")[:300]
        lines.append(
            f"- {b['title']} by {b.get('authors', '?')} "
            f"| genres: {b.get('genre_str') or 'n/a'} "
            f"| actual rating: {b.get('average_rating', float('nan')):.2f} "
            f"| predicted: {pred_s} | match: {b.get('semantic', 0):.2f}\n"
            f"  description: {desc}"
        )
    return "\n".join(lines)


def _fallback(query: str, results: pd.DataFrame) -> str:
    top = results.iloc[0]
    return (
        f"Closest match for '{query}': **{top['title']}** by {top.get('authors', '?')} "
        f"(match {top.get('semantic', 0):.2f}). "
        "Set OPENAI_API_KEY to enable the full natural-language explanation."
    )


def explain_recommendation(query, results, variant="concise", model=None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback(query, results)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = PROMPT_VARIANTS[variant].format(query=query, books=_books_block(results))
        resp = client.chat.completions.create(
            model=model or cfg.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return _fallback(query, results) + f"\n\n_(LLM error: {e})_"
