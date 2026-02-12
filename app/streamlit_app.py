# app/streamlit_app.py
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    SESSIONS_PARQUET_UI,
    ITEM_POPULARITY_PARQUET_UI,
    COVISIT_PARQUET_UI,
    MODEL_PATH_UI,
    METRICS_JSON_UI,
)

from src.candidates import make_candidates_for_session

# Prefer fast scoring if you have it, else fallback
try:
    from src.model import ScorerBundle, score_candidates_fast as _score_fast
    HAS_FAST = True
except Exception:
    HAS_FAST = False

from src.model import load_model, score_candidates

try:
    from src.model import explain_one_prediction
    HAS_EXPLAIN = True
except Exception:
    HAS_EXPLAIN = False


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return p.name


st.set_page_config(page_title="OTTO Session Recommender", layout="wide")
st.title("OTTO Session Recommender (End-to-End DS-Ops)")
st.caption("Ingest → train → evaluate → serve → interpret (demo assets used on cloud).")


# ---- Guardrails (now checks UI paths which fallback to demo_assets/)
required = [
    (SESSIONS_PARQUET_UI, "events parquet"),
    (ITEM_POPULARITY_PARQUET_UI, "popularity parquet"),
    (COVISIT_PARQUET_UI, "covis parquet"),
    (MODEL_PATH_UI, "model file"),
]
missing = [f"- {name}: `{_rel(p)}`" for p, name in required if not Path(p).exists()]

if missing:
    st.error(
        "Missing required assets:\n\n"
        + "\n".join(missing)
        + "\n\nFix: commit `demo_assets/` to the repo (events_demo/pop/covis/model/metrics)."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["session", "aid", "ts"]].copy()


@st.cache_data(show_spinner=False)
def load_pop(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_covis(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_metrics(path: Path):
    if not Path(path).exists():
        return None
    import json
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


events = load_events(Path(SESSIONS_PARQUET_UI))
pop_df = load_pop(Path(ITEM_POPULARITY_PARQUET_UI))
covis_df = load_covis(Path(COVISIT_PARQUET_UI))
metrics = load_metrics(Path(METRICS_JSON_UI))

bundle = load_model()  # this loads from MODEL_PATH in config; OK if you made MODEL_PATH_UI primary
# If your load_model() always uses MODEL_PATH (not MODEL_PATH_UI), do this instead:
# import joblib
# obj = joblib.load(Path(MODEL_PATH_UI))
# bundle = ScorerBundle(scaler=obj["scaler"], model=obj["model"])

# ---- Sidebar
st.sidebar.header("Serving Settings")
max_candidates = st.sidebar.slider("Max candidates per session", 30, 300, 100, 10)
top_k = st.sidebar.slider("Top-K recommendations", 5, 50, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Artifacts (relative)")
st.sidebar.write(f"Events: `{_rel(Path(SESSIONS_PARQUET_UI))}`")
st.sidebar.write(f"Popularity: `{_rel(Path(ITEM_POPULARITY_PARQUET_UI))}`")
st.sidebar.write(f"Covis: `{_rel(Path(COVISIT_PARQUET_UI))}`")
st.sidebar.write(f"Model: `{_rel(Path(MODEL_PATH_UI))}`")
st.sidebar.write(f"Metrics: `{_rel(Path(METRICS_JSON_UI))}`")


tab_reco, tab_metrics, tab_explain = st.tabs(["🔮 Recommend", "📏 Metrics", "🧠 Explain (optional)"])


# ============================================================
# TAB 1: Recommend
# ============================================================
with tab_reco:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Input")

        input_mode = st.radio(
            "Choose input mode",
            ["Pick a session_id (from subset)", "Paste custom session aids"],
            horizontal=True,
            key="mode",
        )

        context_aids: list[int] = []

        if input_mode.startswith("Pick"):
            session_ids = events["session"].drop_duplicates().astype(int).tolist()
            sid = st.selectbox("session_id", session_ids[:5000], key="sid")
            g = events[events["session"] == int(sid)].sort_values("ts")
            context_aids = g["aid"].astype(int).tolist()
            st.markdown("**Session (last 20 aids)**")
            st.json(context_aids[-20:])
        else:
            txt = st.text_area("Paste aids (comma-separated)", value="1098089,1354785,342507,1120175", key="txt")
            try:
                context_aids = [int(x.strip()) for x in txt.split(",") if x.strip()]
            except Exception:
                st.error("Could not parse aids. Example: 1098089,1354785,342507")
                st.stop()

        run = st.button("Recommend", type="primary", key="recommend_btn")

    with right:
        st.subheader("Build / Eval Status")
        if metrics:
            st.json(metrics)
        else:
            st.info("No metrics found (demo mode can include metrics_demo.json).")

    if run:
        try:
            with st.spinner("Generating & scoring candidates..."):
                t0 = time.time()

                cands = make_candidates_for_session(
                    context_aids, pop_df, covis_df, max_candidates=max_candidates
                )

                ranked = score_candidates(context_aids, cands, pop_df, covis_df, bundle)[:top_k]

                t1 = time.time()

            st.markdown("---")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.subheader("Top Recommendations")
                st.dataframe(pd.DataFrame(ranked, columns=["aid", "score"]),
                             use_container_width=True, hide_index=True)
            with c2:
                st.subheader("Session context (last 20)")
                st.json(context_aids[-20:])

            st.caption(f"Latency: {round(t1 - t0, 2)} seconds (CPU)")

        except Exception as e:
            st.error("Recommend failed. Full error below.")
            st.exception(e)


# ============================================================
# TAB 2: Metrics
# ============================================================
with tab_metrics:
    st.subheader("Offline Metrics")
    if metrics:
        st.json(metrics)
    else:
        st.info("No metrics available.")


# ============================================================
# TAB 3: Explain
# ============================================================
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
        txt = st.text_input("candidate aid to explain", value=default_cand, key="cand")

        try:
            cand = int(txt.strip())
        except Exception:
            st.error("candidate aid must be an int")
            st.stop()

        if st.button("Explain", type="primary", key="explain_btn"):
            try:
                with st.spinner("Computing explanation..."):
                    exp = explain_one_prediction(context_aids, cand, pop_df, covis_df, bundle)

                contrib = np.array(exp["contrib"], dtype=float)
                feature_names = list(exp["feature_names"])

                order = np.argsort(np.abs(contrib))[::-1]
                contrib = contrib[order]
                feature_names = [feature_names[i] for i in order]

                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.barh(feature_names[::-1], contrib[::-1])
                plt.title(f"SHAP-style contributions (method: {exp.get('method', 'linear')})")
                st.pyplot(fig, clear_figure=True)

                st.caption(
                    f"Pred prob: {float(exp.get('prob', 0.0)):.4f} | "
                    f"candidate_aid: {int(exp.get('candidate_aid', cand))} | "
                    f"logit: {float(exp.get('score_logit', 0.0)):.3f}"
                )

            except Exception as e:
                st.error("Explain failed. Full error below.")
                st.exception(e)
