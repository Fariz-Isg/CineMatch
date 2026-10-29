"""CineMatch — Netflix-style Streamlit frontend.

Calls the trained checkpoints in model/checkpoints/ or model/checkpoints_az/
via model/recommender.py.
This file never trains anything and never touches datasets/ directly — it only
reads the exported model artifacts through the Recommender class.

Run:  streamlit run streamlit/app.py   (from the repo root)
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --- make `model/` importable regardless of where streamlit is launched from ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.recommender import Recommender  # noqa: E402

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------------
# Netflix-style theme
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #141414; }
    #MainMenu, footer, header { visibility: hidden; }

    .cm-hero {
        padding: 2.2rem 1rem 1.2rem 1rem;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 1rem;
    }
    .cm-logo {
        color: #E50914; font-size: 2.4rem; font-weight: 800;
        letter-spacing: -1px; margin: 0;
    }
    .cm-sub { color: #b3b3b3; font-size: 0.95rem; margin-top: 0.2rem; }

    .cm-row-title {
        color: #ffffff; font-size: 1.3rem; font-weight: 700;
        margin: 1.6rem 0 0.6rem 0.2rem;
    }

    .cm-card {
        border-radius: 6px;
        padding: 14px 12px;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.6);
        transition: transform 0.15s ease;
        overflow: hidden;
        position: relative;
    }
    .cm-card:hover { transform: scale(1.035); }
    .cm-card-title {
        font-weight: 700; font-size: 0.92rem; line-height: 1.2rem;
        text-shadow: 0 1px 4px rgba(0,0,0,0.85);
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .cm-card-genre {
        font-size: 0.72rem; color: #e8e8e8; margin-top: 4px; opacity: 0.9;
        text-shadow: 0 1px 3px rgba(0,0,0,0.85);
    }
    .cm-badge {
        position: absolute; top: 8px; right: 10px;
        background: rgba(0,0,0,0.65); color: #46d369;
        font-size: 0.72rem; font-weight: 700;
        padding: 2px 7px; border-radius: 3px;
    }
    .cm-empty { color: #808080; font-style: italic; padding: 1rem 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Catalog selector in sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🎬 Catalog")
catalog_choice = st.sidebar.selectbox(
    "Choose Movie Catalog",
    ["🇦🇿 Azerbaijani Cinema", "🌍 Global (MovieLens)"],
    index=0,
    help="Switch between Azerbaijani Cinema and the standard global MovieLens dataset."
)
st.sidebar.divider()

az_checkpoint_dir = ROOT / "model" / "checkpoints_az"
global_checkpoint_dir = ROOT / "model" / "checkpoints"

if "Azerbaijani" in catalog_choice and az_checkpoint_dir.exists():
    active_checkpoint = str(az_checkpoint_dir)
else:
    active_checkpoint = str(global_checkpoint_dir)


# ----------------------------------------------------------------------------
# Model loading (cached — checkpoint is read from disk once per catalog)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CineMatch model…")
def load_model(checkpoint_path: str):
    return Recommender(checkpoint_dir=checkpoint_path)


rec = load_model(active_checkpoint)
meta = rec.meta

# a fixed palette so the same movie always gets the same gradient
PALETTE = [
    ("#E50914", "#831010"), ("#0071EB", "#003a75"), ("#7B2FF7", "#3d0f8f"),
    ("#00A896", "#02565a"), ("#F5A623", "#8a4e00"), ("#D91E5C", "#5c0c26"),
    ("#2E8B57", "#0f3d24"), ("#8E44AD", "#3b1c4a"),
]


def card_gradient(title: str) -> tuple[str, str]:
    h = int(hashlib.md5(title.encode()).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


def render_row(row_title: str, df, score_col: str | None = None):
    st.markdown(f'<div class="cm-row-title">{row_title}</div>', unsafe_allow_html=True)
    if df is None or len(df) == 0:
        st.markdown('<div class="cm-empty">Nothing here yet — try a different selection.</div>',
                    unsafe_allow_html=True)
        return
    cols = st.columns(min(len(df), 6))
    for i, (_, movie) in enumerate(df.iterrows()):
        with cols[i % len(cols)]:
            c1, c2 = card_gradient(movie["title"])
            badge = ""
            if score_col and score_col in movie:
                badge = f'<div class="cm-badge">{int(round(movie[score_col] * 100))}% match</div>'
            genres = movie.get("genres", "")
            genres_display = genres.replace("|", " · ") if isinstance(genres, str) else ""

            poster = movie.get("poster_url")
            has_poster = isinstance(poster, str) and poster.startswith("http")
            if has_poster:
                bg_style = f"background: linear-gradient(to top, rgba(12,12,12,0.92) 0%, rgba(12,12,12,0.45) 55%, rgba(0,0,0,0.15) 100%), url('{poster}') center/cover;"
            else:
                bg_style = f"background: linear-gradient(135deg, {c1}, {c2});"

            display_title = movie["title"]
            orig = movie.get("original_title")
            if isinstance(orig, str) and orig.strip() and orig != movie["title"]:
                display_title = f"{movie['title']} ({orig})"

            card_html = (
                f'<div class="cm-card" style="{bg_style}">'
                f'{badge}'
                f'<div class="cm-card-title">{display_title}</div>'
                f'<div class="cm-card-genre">{genres_display}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="cm-hero">
    <p class="cm-logo">CINEMATCH</p>
    <p class="cm-sub">
        {meta.get('dataset', 'Movie recommender')} · FunkSVD (k={meta['k']}) + genre content · α={meta['best_alpha']} ·
        {meta['n_users_sampled']:,} users · {meta['n_items']:,} titles · test RMSE {meta['test_rmse']}
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# "Who's watching" — profile picker (existing user vs. new user)
# ----------------------------------------------------------------------------
mode = st.sidebar.radio("Who's watching?", ["Existing user", "New user"])
st.sidebar.divider()

if mode == "Existing user":
    users, counts = rec.users_by_activity()
    idx = st.sidebar.selectbox(
        "Profile",
        range(len(users)),
        index=len(users) // 2,
        format_func=lambda i: f"User {users[i]}  ·  {counts[i]} ratings",
    )
    user_id = int(users[idx])
    alpha = st.sidebar.slider(
        "Balance: genres ↔ similar tastes", 0.0, 1.0, float(meta["best_alpha"]), 0.05,
        help="0 = genres only, 1 = collaborative filtering only",
    )
    top_n = st.sidebar.slider("How many to show", 6, 24, 12, 6)

    history = rec.user_history(user_id, 6)
    render_row("Your History (Top Rated)", history)
    render_row("Top Picks For You", rec.recommend(user_id, alpha=alpha, n=top_n), score_col="hybrid")
    render_row("Trending Now", rec.popular(top_n))

    if len(history) > 0:
        seed_title = history.iloc[0]["title"]
        render_row(f"Because you watched: {seed_title}", rec.similar_movies(seed_title, top_n))

else:
    st.sidebar.markdown("Pick a few movies you like — we'll build your taste profile on the fly.")
    all_titles = rec.movies["title"].tolist()
    liked_titles = st.sidebar.multiselect("Movies you liked", all_titles)
    top_n = st.sidebar.slider("How many to show", 6, 24, 12, 6)

    render_row("Trending Now", rec.popular(top_n))

    if liked_titles:
        liked_ids = [int(rec.movies["movieId"].iloc[rec.title_to_pos[t]]) for t in liked_titles]
        render_row("Recommended For You (by genre)",
                   rec.recommend_for_new_user(liked_ids, top_n), score_col="content")
    else:
        st.markdown('<div class="cm-empty">Pick at least one movie on the left '
                    'to get personalized recommendations.</div>', unsafe_allow_html=True)

st.divider()

# ----------------------------------------------------------------------------
# Search / explore: similar movies by title
# ----------------------------------------------------------------------------
st.markdown('<div class="cm-row-title">🔍 Find Similar Movies by Title</div>', unsafe_allow_html=True)

# Pick a sensible default title depending on catalog
default_search_idx = 0
for cand in ["Arshin Takes a Wife", "Don't Be Afraid, I'm with You", "Babek", "Toy Story (1995)"]:
    if cand in rec.title_to_pos:
        default_search_idx = rec.title_to_pos[cand]
        break

search_title = st.selectbox(
    "Find movies similar to…",
    rec.movies["title"].tolist(),
    index=int(default_search_idx)
)
n_sim = st.slider("How many similar titles", 6, 24, 12, 6, key="nsim")
render_row(f"Similar to \u201c{search_title}\u201d", rec.similar_movies(search_title, n_sim), score_col="similarity")

st.divider()
dataset_desc = (
    "Dataset: Azerbaijani native cinema catalog enriched via TMDB with taste-matched ratings. "
    if meta.get("catalog_type") == "az"
    else "Dataset: F. M. Harper & J. A. Konstan (2015), The MovieLens Datasets, ACM TiiS 5(4). "
)
st.caption(
    f"{dataset_desc}Model checkpoint loaded read-only — no training happens here."
)
