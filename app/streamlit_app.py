# app/streamlit_app.py
from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Make `import src...` work when running: streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SESSIONS_PARQUET, METRICS_JSON, MODEL_PATH
from src.candidates import load_popularity, load_covis_topk, make_candidates_for_session
from src.model import load_model, score_candidates

# If you don't have explain_one_prediction implemented, we gracefully disable it.
try:
    from src.model import explain_one_prediction
    HAS_EXPLAIN = True
except Exception:
    HAS_EXPLAIN = False


st.set_page_config(page_title="OTTO Session Recommender", layout="wide")
st.title("OTTO Session Recommender (End-to-End DS-Ops)")
st.caption("Ingest → train → evaluate → serve → interpret (where available).")

# -----------------------------
# Cached loads
# -----------------------------
@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    return pd.read_parquet(SESSIONS_PARQUET)

@st.cache_data(show_spinner=False)
def load_build_metrics():
    if METRICS_JSON.exists():
        import json
        return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    return None

@st.cache_data(show_spinner=False)
def load_candidate_sources():
    pop = load_popularity()
    covis = load_covis_topk()
    return pop, covis


# -----------------------------
# Guardrails: required files
# -----------------------------
if not SESSIONS_PARQUET.exists():
    st.error(f"Missing events parquet: {SESSIONS_PARQUET}\nRun: python -m scripts.01_build_data")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Missing model file: {MODEL_PATH}\nRun: python -m scripts.02_train_model")
    st.stop()


events = load_events()
build_metrics = load_build_metrics()
pop_df, covis_df = load_candidate_sources()
bundle = load_model()

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Serving Settings")
max_candidates = st.sidebar.slider("Max candidates per session", 30, 300, 100, 10)
top_k = st.sidebar.slider("Top-K recommendations", 5, 50, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Artifacts (relative)")
def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT.resolve()))
    except Exception:
        # fallback: only show filename
        return Path(p).name

st.sidebar.write(f"Model: `{_rel(MODEL_PATH)}`")
st.sidebar.write(f"Events: `{_rel(SESSIONS_PARQUET)}`")
st.sidebar.write(f"Metrics: `{_rel(METRICS_JSON)}`")


# -----------------------------
# Layout
# -----------------------------
tab_reco, tab_metrics, tab_explain = st.tabs(
    ["🔮 Recommend", "📏 Metrics", "🧠 Explain (optional)"]
)

# ============================================================
# TAB 1: Recommend
# ============================================================
with tab_reco:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Input")

        input_mode = st.radio(
            "Choose input mode",
            ["Pick a session_id (from ingested subset)", "Paste custom session aids"],
            horizontal=True,
        )

        context_aids: list[int] = []

        if input_mode.startswith("Pick"):
            session_ids = events["session"].drop_duplicates().astype(int).tolist()
            session_id = st.selectbox("session_id", session_ids[:5000])
            g = events[events["session"] == int(session_id)].sort_values("ts")
            context_aids = g["aid"].astype(int).tolist()
            st.markdown("**Session (last events)**")
            st.json(context_aids[-20:])
        else:
            txt = st.text_area("Paste aids (comma-separated)", value="1098089,1354785,342507,1120175")
            try:
                context_aids = [int(x.strip()) for x in txt.split(",") if x.strip()]
            except Exception:
                st.error("Could not parse aids. Example: 1098089,1354785,342507")
                st.stop()

        run = st.button("Recommend", type="primary")

    with right:
        st.subheader("Build / Eval Status")
        if build_metrics:
            st.json(build_metrics)
        else:
            st.info("No metrics found yet. Run: python -m scripts.03_eval")

        st.subheader("How to interpret outputs (quick)")
        st.markdown(
            """
- **aid**: item id recommended
- **score**: model probability (higher = more likely based on patterns learned)
- **Latency**: serving performance (critical for real-time recommender UX)
"""
        )

    if run:
        t0 = time.time()
        cands = make_candidates_for_session(context_aids, pop_df, covis_df, max_candidates=max_candidates)
        ranked = score_candidates(context_aids, cands, pop_df, covis_df, bundle)[:top_k]
        t1 = time.time()

        st.markdown("---")
        c1, c2 = st.columns([1, 1], gap="large")

        with c1:
            st.subheader("Top Recommendations")
            df_rank = pd.DataFrame(ranked, columns=["aid", "score"])
            st.dataframe(df_rank, use_container_width=True, hide_index=True)

        with c2:
            st.subheader("Session context (last 20 aids)")
            st.json(context_aids[-20:])

        st.caption(f"Latency: {round(t1 - t0, 2)} seconds (CPU)")

# ============================================================
# TAB 2: Metrics
# ============================================================
with tab_metrics:
    st.subheader("Offline Metrics (from metrics.json)")
    if build_metrics:
        st.json(build_metrics)
    else:
        st.info("Run: python -m scripts.03_eval")

# ============================================================
# TAB 3: Explain (optional)
# ============================================================
with tab_explain:
    st.subheader("SHAP-style explanation (optional)")

    if not HAS_EXPLAIN:
        st.info("Explainer is not available. Ensure `explain_one_prediction()` exists in src/model.py.")
        st.stop()

    st.markdown("Pick a session and a candidate item to explain.")
    session_ids = events["session"].drop_duplicates().astype(int).tolist()
    sid = st.selectbox("session_id (explain)", session_ids[:5000], key="explain_sid")

    g = events[events["session"] == int(sid)].sort_values("ts")
    context_aids = g["aid"].astype(int).tolist()

    default_cand = str(context_aids[-1]) if context_aids else "0"
    txt = st.text_input("candidate aid to explain", value=default_cand, key="explain_candidate")

    try:
        cand = int(txt.strip())
    except Exception:
        st.error("candidate aid must be an int")
        st.stop()

    if st.button("Explain", type="primary", key="explain_btn"):
        exp = explain_one_prediction(context_aids, cand, pop_df, covis_df, bundle)

        # Defensive: show payload keys if schema mismatch
        if "contrib" not in exp or "feature_names" not in exp:
            st.error(f"Explainer returned unexpected schema. Keys: {list(exp.keys())}")
            st.json(exp)
            st.stop()

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
            f"Pred prob: {float(exp.get('prob', 0.0)):.4f} | candidate_aid: {exp.get('candidate_aid', cand)} | "
            f"logit: {float(exp.get('score_logit', 0.0)):.3f}"
        )


