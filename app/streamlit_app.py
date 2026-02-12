# app/streamlit_app.py
from __future__ import annotations

import time
from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    SESSIONS_PARQUET,
    METRICS_JSON,
    MODEL_PATH,
    ITEM_POPULARITY_PARQUET,
    COVISIT_PARQUET,
)

# Optional explain (won't crash if missing)
try:
    from src.model import explain_one_prediction
    HAS_EXPLAIN = True
except Exception:
    HAS_EXPLAIN = False

DEMO_DIR = ROOT / "demo_assets"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return p.name


def _pick_assets():
    # full paths
    events_path = Path(SESSIONS_PARQUET)
    pop_path = Path(ITEM_POPULARITY_PARQUET)
    covis_path = Path(COVISIT_PARQUET)
    model_path = Path(MODEL_PATH)
    metrics_path = Path(METRICS_JSON)

    # demo fallbacks
    demo_events = DEMO_DIR / "events_demo.parquet"
    demo_pop = DEMO_DIR / "item_popularity_demo.parquet"
    demo_covis = DEMO_DIR / "covis_topk_demo.parquet"
    demo_model = DEMO_DIR / "scorer_demo.joblib"
    demo_metrics = DEMO_DIR / "metrics_demo.json"

    if not events_path.exists() and demo_events.exists():
        events_path = demo_events
    if not pop_path.exists() and demo_pop.exists():
        pop_path = demo_pop
    if not covis_path.exists() and demo_covis.exists():
        covis_path = demo_covis
    if not model_path.exists() and demo_model.exists():
        model_path = demo_model
    if not metrics_path.exists() and demo_metrics.exists():
        metrics_path = demo_metrics

    return events_path, pop_path, covis_path, model_path, metrics_path


EVENTS_PATH, POP_PATH, COVIS_PATH, MODEL_FILE, METRICS_FILE = _pick_assets()

st.set_page_config(page_title="OTTO Session Recommender", layout="wide")
st.title("OTTO Session Recommender (End-to-End DS-Ops)")
st.caption("Ingest → train → evaluate → serve → interpret (where available).")

# ---- Guardrails
missing = []
for p, label in [
    (EVENTS_PATH, "events parquet"),
    (POP_PATH, "popularity parquet"),
    (COVIS_PATH, "covis parquet"),
    (MODEL_FILE, "model file"),
]:
    if not Path(p).exists():
        missing.append(f"{label}: `{_rel(Path(p))}`")

if missing:
    st.error("Missing required assets:\n\n" + "\n".join([f"- {m}" for m in missing]))
    st.stop()

# ---- Cached loads (keep them LIGHT)
@st.cache_data(show_spinner=False)
def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["session", "aid", "ts"])
    return df

@st.cache_data(show_spinner=False)
def load_pop_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["aid", "pop"])

@st.cache_data(show_spinner=False)
def load_covis_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["aid", "neighbor_aid", "covis"])

@st.cache_data(show_spinner=False)
def load_metrics(path: Path):
    if not path.exists():
        return None
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_bundle(model_path: Path):
    import joblib
    obj = joblib.load(model_path)
    return obj  # {"scaler":..., "model":...}

events = load_events(EVENTS_PATH)
pop_df = load_pop_df(POP_PATH)
covis_df = load_covis_df(COVIS_PATH)
metrics = load_metrics(METRICS_FILE)
model_obj = load_bundle(MODEL_FILE)

# Precompute small, cheap things
POP_TOP = pop_df.sort_values("pop", ascending=False)["aid"].astype(int).tolist()
POP_MAP = dict(zip(pop_df["aid"].astype(int), pop_df["pop"].astype(float)))

def build_small_covis_map(covis_df: pd.DataFrame, seed_items: list[int]) -> dict[int, list[tuple[int, int]]]:
    """Only keep covis rows for the last few items in the session."""
    if not seed_items:
        return {}
    sub = covis_df[covis_df["aid"].isin(seed_items)]
    out: dict[int, list[tuple[int, int]]] = {}
    for aid, g in sub.groupby("aid", sort=False):
        out[int(aid)] = list(
            zip(g["neighbor_aid"].astype(int).tolist(), g["covis"].astype(int).tolist())
        )
    return out

def make_candidates_fast(
    session_aids: list[int],
    pop_top: list[int],
    pop_map: dict[int, float],
    covis_map_small: dict[int, list[tuple[int, int]]],
    max_candidates: int = 100,
    seed_last_n: int = 5,
) -> list[int]:
    if not session_aids:
        return pop_top[:max_candidates]

    seed_items = session_aids[-seed_last_n:]
    scores = Counter()

    # covis neighbors
    for a in seed_items:
        for n, c in covis_map_small.get(int(a), []):
            scores[int(n)] += float(c)

    # popularity fallback
    for aid in pop_top[:200]:
        scores[int(aid)] += 0.01 * float(pop_map.get(int(aid), 0.0))

    # slight boost for recently seen
    for a in set(session_aids[-10:]):
        scores[int(a)] += 0.1

    ranked = [aid for aid, _ in scores.most_common(max_candidates)]
    if len(ranked) < max_candidates:
        ranked += pop_top
        ranked = list(dict.fromkeys(ranked))[:max_candidates]
    return ranked

def score_candidates_fast(session_aids, candidates, pop_map, covis_map_small, model_obj):
    # features MUST match src/model.py feature order
    def featurize(candidate_aid: int) -> np.ndarray:
        pop = np.log1p(pop_map.get(int(candidate_aid), 0.0))

        covis_score = 0.0
        for seed in session_aids[-5:]:
            for n, c in covis_map_small.get(int(seed), []):
                if int(n) == int(candidate_aid):
                    covis_score += float(c)
        covis_score = np.log1p(covis_score)

        recency = 1.0 if int(candidate_aid) in set(map(int, session_aids[-10:])) else 0.0
        seen = 1.0 if int(candidate_aid) in set(map(int, session_aids)) else 0.0

        return np.array([pop, covis_score, recency, seen], dtype=np.float32)

    X = np.vstack([featurize(int(c)) for c in candidates]).astype(np.float32)
    Xs = model_obj["scaler"].transform(X)
    probs = model_obj["model"].predict_proba(Xs)[:, 1]
    return sorted(zip(candidates, probs.tolist()), key=lambda x: x[1], reverse=True)

# ---- Sidebar
st.sidebar.header("Serving Settings")
max_candidates = st.sidebar.slider("Max candidates per session", 30, 200, 100, 10)
top_k = st.sidebar.slider("Top-K recommendations", 5, 50, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Artifacts (relative)")
st.sidebar.write(f"Model: `{_rel(MODEL_FILE)}`")
st.sidebar.write(f"Events: `{_rel(EVENTS_PATH)}`")
st.sidebar.write(f"Popularity: `{_rel(POP_PATH)}`")
st.sidebar.write(f"Covis: `{_rel(COVIS_PATH)}`")
st.sidebar.write(f"Metrics: `{_rel(METRICS_FILE)}`")

tab_reco, tab_metrics, tab_explain = st.tabs(["🔮 Recommend", "📏 Metrics", "🧠 Explain (optional)"])

# ---- Recommend
with tab_reco:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Input")

        input_mode = st.radio(
            "Choose input mode",
            ["Pick a session_id (from subset)", "Paste custom session aids"],
            horizontal=True,
        )

        context_aids: list[int] = []

        if input_mode.startswith("Pick"):
            session_ids = events["session"].drop_duplicates().astype(int).tolist()
            sid = st.selectbox("session_id", session_ids[:5000])
            g = events[events["session"] == int(sid)].sort_values("ts")
            context_aids = g["aid"].astype(int).tolist()
            st.markdown("**Session (last 20 aids)**")
            st.json(context_aids[-20:])
        else:
            txt = st.text_area("Paste aids (comma-separated)", value="1098089,1354785,342507,1120175")
            context_aids = [int(x.strip()) for x in txt.split(",") if x.strip()]

        if st.button("Recommend", type="primary"):
            st.session_state["do_reco"] = True
            st.session_state["context_aids"] = context_aids

    with right:
        st.subheader("Build / Eval Status")
        if metrics:
            st.json(metrics)
        else:
            st.info("No metrics found (metrics json missing/invalid).")

    if st.session_state.get("do_reco", False):
        try:
            with st.spinner("Generating & scoring candidates..."):
                t0 = time.time()
                context_aids = st.session_state.get("context_aids", [])

                seeds = [int(x) for x in context_aids[-5:]]
                covis_map_small = build_small_covis_map(covis_df, seeds)

                cands = make_candidates_fast(
                    session_aids=context_aids,
                    pop_top=POP_TOP,
                    pop_map=POP_MAP,
                    covis_map_small=covis_map_small,
                    max_candidates=max_candidates,
                )

                ranked = score_candidates_fast(
                    session_aids=context_aids,
                    candidates=cands,
                    pop_map=POP_MAP,
                    covis_map_small=covis_map_small,
                    model_obj=model_obj,
                )[:top_k]

                t1 = time.time()

            st.markdown("---")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.subheader("Top Recommendations")
                st.dataframe(pd.DataFrame(ranked, columns=["aid", "score"]), use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Session context (last 20)")
                st.json(context_aids[-20:])

            st.caption(f"Latency: {round(t1 - t0, 2)} seconds (CPU)")

        except Exception as e:
            st.error("Recommend crashed. Here is the full error:")
            st.exception(e)

# ---- Metrics
with tab_metrics:
    st.subheader("Offline Metrics")
    if metrics:
        st.json(metrics)
    else:
        st.info("No metrics available.")

# ---- Explain
with tab_explain:
    st.subheader("SHAP-style explanation (optional)")
    if not HAS_EXPLAIN:
        st.warning("explain_one_prediction() not available in src/model.py")
    else:
        session_ids = events["session"].drop_duplicates().astype(int).tolist()
        sid = st.selectbox("session_id (explain)", session_ids[:5000], key="explain_sid")

        g = events[events["session"] == int(sid)].sort_values("ts")
        context_aids = g["aid"].astype(int).tolist()

        default_cand = str(context_aids[-1]) if context_aids else "0"
        txt = st.text_input("candidate aid to explain", value=default_cand, key="explain_candidate")

        if st.button("Explain", type="primary", key="explain_btn"):
            try:
                exp = explain_one_prediction(context_aids, int(txt.strip()), pop_df, covis_df, None)  # bundle handled inside your function
                st.json(exp)
            except Exception as e:
                st.error("Explain crashed. Full error:")
                st.exception(e)
