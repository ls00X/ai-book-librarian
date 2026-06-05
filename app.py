"""AI Book Librarian - Streamlit inference app.

Loads only the saved artifacts (model, embeddings, catalog) - it never trains.
Run locally:  streamlit run app.py
"""
import os

import streamlit as st

from src import config as cfg
from src.explain import PROMPT_VARIANTS, explain_recommendation
from src.recommend import Recommender

# Make the OpenAI key available from Streamlit secrets when deployed
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

st.set_page_config(page_title="AI Book Librarian", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_recommender():
    return Recommender()


st.title("📚 AI Book Librarian")
st.caption(
    "Describe what you feel like reading. The app retrieves matching books, predicts their "
    "expected rating with a numeric model, and explains the picks in natural language."
)

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Retrieval method", ["embedding", "tfidf"], index=0,
                          help="Dense semantic search vs. the TF-IDF baseline.")
    top_k = st.slider("Number of recommendations", 3, 10, 5)
    st.markdown("**Ranking weights**")
    w_sem = st.slider("Semantic fit", 0.0, 1.0, 0.60, 0.05)
    w_rat = st.slider("Predicted rating", 0.0, 1.0, 0.25, 0.05)
    w_pop = st.slider("Popularity", 0.0, 1.0, 0.15, 0.05)
    variant = st.selectbox("Explanation style", list(PROMPT_VARIANTS.keys()), index=0)

query = st.text_input(
    "What do you want to read?",
    "A dark fantasy with political intrigue, not too long, similar to The Witcher.",
)

if st.button("Recommend", type="primary") and query.strip():
    rec = get_recommender()
    total = (w_sem + w_rat + w_pop) or 1.0
    weights = {"semantic": w_sem / total, "rating": w_rat / total, "popularity": w_pop / total}

    with st.spinner("Searching the shelves..."):
        results = rec.recommend(query, top_k=top_k, weights=weights, method=method)

    for _, b in results.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(b["title"])
                st.caption(f"by {b.get('authors', '?')}  ·  {b.get('genre_str') or 'genre n/a'}")
                full_desc = b.get("description") or ""
                if full_desc:
                    st.write(full_desc[:400] + ("..." if len(full_desc) > 400 else ""))
            with c2:
                st.metric("Actual rating", f"{b.get('average_rating', float('nan')):.2f}")
                pred = b.get("predicted_rating")
                if pred == pred:  # not NaN
                    st.metric("Predicted", f"{pred:.2f}")
                st.metric("Match", f"{b.get('semantic', 0):.2f}")

    with st.spinner("Writing the librarian's note..."):
        note = explain_recommendation(query, results, variant=variant)
    st.markdown("### Why these books?")
    st.write(note)
