# app/streamlit_app.py
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Path bootstrap: allow `import src...` reliably in hosted envs
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    SESSIONS_PARQUET,
    ITEM_POPULARITY_PARQUET,
    COVISIT_PARQUET,
    MODEL_PATH,
    METRICS_JSON,
)
from src.candidates import make_candidates_for_session
from src.model import ScorerBundle, score_candidates

# Optional explain
try:
    from src.model import explain_one_prediction  # type: ignore
    HAS_EXPLAIN = True
except Exception:
    HAS_EXPLAIN = False

DEMO_DIR = ROOT / "demo_assets"


# -----------------------------
# helpers
# -----------------------------
def _rel(p: Path) -> str:
    """Show only repo-relative paths (no local machine paths)."""
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return p.name


def _pick_assets() -> Tuple[Path, Path, Path, Path, Path]:
    """Prefer full assets; fall back to demo_assets/* if present."""
    events_path = SESSIONS_PARQUET
    pop_path = ITEM_POPULARITY_PARQUET
    covis_path = COVISIT_PARQUET
    model_path = MODEL_PATH
    metrics_path = METRICS_JSON

    demo_events = DEMO_DIR / "events_demo.parquet"
    demo_pop = DEMO_DIR / "item_popularity_demo.parquet"
    demo_covis = DEMO_DIR / "covis_topk_demo.parquet"
    demo_model = DEMO_DIR / "scorer_demo.joblib"
    demo_metrics = DEMO_DIR / "metrics_demo.json"

    if (not events_path.exists()) and demo_events.exists():
        events_path = demo_events
    if (not pop_path.exists()) and demo_pop.exists():
        pop_path = demo_pop
    if (not covis_path.exists()) and demo_covis.exists():
        covis_path = demo_covis
    if (not model_path.exists()) and demo_model.exists():
        model_path = demo_model
    if (not metrics_path.exists()) and demo_metrics.exists():
        metrics_path = demo_metrics

    return events_path, pop_path, covis_path, model_path, metrics_path


def _file_info(p: Path) -> str:
    if not p.exists():
        return f"❌ missing: `{_rel(p)}`"
    sz = p.stat().st_size
    return f"✅ `{_rel(p)}` ({sz/1024:.1f} KB)"


def _safe_json_read(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="OTTO Session Recommender", layout="wide")
st.title("OTTO Session Recommender (End-to-End DS-Ops)")
st.caption("Ingest → train → evaluate → serve → interpret (where available).")

EVENTS_PATH, POP_PATH, COVIS_PATH, MODEL_FILE, METRICS_FILE = _pick_assets()

# -----------------------------
# Guardrails (ALWAYS show what's missing)
# -----------------------------
missing = []
for p, label in [
    (EVENTS_PATH, "events parquet"),
    (POP_PATH, "popularity parquet"),
    (COVIS_PATH, "covis parquet"),
    (MODEL_FILE, "model file"),
]:
    if not p.exists():
        missing.append(f"- {label}: `{_rel(p)}`")

if missing:
    st.error(
        "Missing required assets:\n\n"
        + "\n".join(missing)
        + "\n\nMake sure you uploaded `demo_assets/*` (events_demo, item_popularity_demo, covis_topk_demo, scorer_demo)."
    )
    st.stop()


# -----------------------------
# Cached loads (keep them simple)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_events(path_str: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(path_str))
    df = df[["session", "aid", "ts"]].copy()
    df["session"] = df["session"].astype(int)
    df["aid"] = df["aid"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_pop(path_str: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(path_str))
    df["aid"] = df["aid"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_covis(path_str: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(path_str))
    df["aid"] = df["aid"].astype(int)
    df["neighbor_aid"] = df["neighbor_aid"].astype(int)
    return df


@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str) -> ScorerBundle:
    import joblib
    obj = joblib.load(Path(path_str))
    return ScorerBundle(scaler=obj["scaler"], model=obj["model"])


events = load_events(str(EVENTS_PATH))
pop_df = load_pop(str(POP_PATH))
covis_df = load_covis(str(COVIS_PATH))
bundle = load_bundle(str(MODEL_FILE))
metrics = _safe_json_read(METRICS_FILE)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Serving Settings")
max_candidates = st.sidebar.slider("Max candidates per session", 30, 300, 100, 10)
top_k = st.sidebar.slider("Top-K recommendations", 5, 50, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Artifacts (relative)")
st.sidebar.write(_file_info(MODEL_FILE))
st.sidebar.write(_file_info(EVENTS_PATH))
st.sidebar.write(_file_info(POP_PATH))
st.sidebar.write(_file_info(COVIS_PATH))
if METRICS_FILE.exists():
    st.sidebar.write(_file_info(METRICS_FILE))

# -----------------------------
# Tabs
# -----------------------------
tab_reco, tab_metrics, tab_explain, tab_diag = st.tabs(
    ["🔮 Recommend", "📏 Metrics", "🧠 Explain (optional)", "🧪 Diagnostics"]
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
            ["Pick a session_id (from subset)", "Paste custom session aids"],
            horizontal=True,
        )

        # Use a FORM to avoid unstable rerun behavior on hosted platforms
        with st.form("reco_form", clear_on_submit=False):
            context_aids: List[int] = []

            if input_mode.startswith("Pick"):
                session_ids = events["session"].drop_duplicates().astype(int).tolist()
                sid = st.selectbox("session_id", session_ids[:5000])
                g = events[events["session"] == int(sid)].sort_values("ts")
                context_aids = g["aid"].astype(int).tolist()
                st.markdown("**Session (last 20 aids)**")
                st.json(context_aids[-20:])
            else:
                txt = st.text_area(
                    "Paste aids (comma-separated)",
                    value="1098089,1354785,342507,1120175",
                )
                try:
                    context_aids = [int(x.strip()) for x in txt.split(",") if x.strip()]
                except Exception:
                    st.error("Could not parse aids. Example: 1098089,1354785,342507")
                    st.stop()

            submitted = st.form_submit_button("Recommend")

        if submitted:
            st.session_state["last_context_aids"] = context_aids
            st.session_state["do_reco"] = True

    with right:
        st.subheader("Build / Eval Status")
        if metrics:
            st.json(metrics)
        else:
            st.info("No metrics found (metrics json missing/invalid).")

    # --- Run recommendation ONLY if requested
    if st.session_state.get("do_reco", False):
        # IMPORTANT: flip flag OFF immediately to prevent rerun loops
        st.session_state["do_reco"] = False

        try:
            context_aids = st.session_state.get("last_context_aids", [])
            with st.spinner("Generating & scoring candidates..."):
                t0 = time.time()
                cands = make_candidates_for_session(
                    context_aids, pop_df, covis_df, max_candidates=max_candidates
                )
                ranked = score_candidates(
                    context_aids, cands, pop_df, covis_df, bundle
                )[:top_k]
                t1 = time.time()

            st.session_state["last_ranked"] = ranked
            st.session_state["last_latency"] = round(t1 - t0, 2)

        except Exception as e:
            st.session_state["last_exception"] = traceback.format_exc()
            st.error("Recommend crashed. Full error below.")
            st.exception(e)

    # --- Render last results (persist across reruns)
    if "last_ranked" in st.session_state:
        st.markdown("---")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Top Recommendations")
            st.dataframe(
                pd.DataFrame(st.session_state["last_ranked"], columns=["aid", "score"]),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.subheader("Session context (last 20)")
            st.json(st.session_state.get("last_context_aids", [])[-20:])

        st.caption(f"Latency: {st.session_state.get('last_latency', 0)} seconds (CPU)")

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

        with st.form("explain_form", clear_on_submit=False):
            default_cand = str(context_aids[-1]) if context_aids else "0"
            txt = st.text_input("candidate aid to explain", value=default_cand)
            exp_submit = st.form_submit_button("Explain")

        if exp_submit:
            try:
                cand = int(txt.strip())
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
                st.session_state["last_exception"] = traceback.format_exc()
                st.error("Explain crashed. Full error below.")
                st.exception(e)

# ============================================================
# TAB 4: Diagnostics
# ============================================================
with tab_diag:
    st.subheader("Diagnostics (use this when the UI goes blank)")

    st.markdown("### Assets picked")
    st.write("Events:", _file_info(EVENTS_PATH))
    st.write("Popularity:", _file_info(POP_PATH))
    st.write("Covis:", _file_info(COVIS_PATH))
    st.write("Model:", _file_info(MODEL_FILE))
    st.write("Metrics:", _file_info(METRICS_FILE))

    st.markdown("### Data sanity")
    st.write("events rows:", int(len(events)))
    st.write("unique sessions:", int(events["session"].nunique()))
    st.write("pop rows:", int(len(pop_df)))
    st.write("covis rows:", int(len(covis_df)))

    st.markdown("### Last exception (if any)")
    if "last_exception" in st.session_state:
        st.code(st.session_state["last_exception"])
    else:
        st.info("No exception captured yet.")

    st.markdown("### Quick self-test")
    if st.button("Run quick recommend self-test"):
        try:
            # pick a tiny context
            sid0 = int(events["session"].iloc[0])
            g0 = events[events["session"] == sid0].sort_values("ts")
            context = g0["aid"].astype(int).tolist()[:10]

            t0 = time.time()
            cands = make_candidates_for_session(context, pop_df, covis_df, max_candidates=50)
            ranked = score_candidates(context, cands, pop_df, covis_df, bundle)[:10]
            t1 = time.time()

            st.success(f"Self-test OK in {round(t1 - t0, 3)}s")
            st.dataframe(pd.DataFrame(ranked, columns=["aid", "score"]), hide_index=True)
        except Exception:
            st.session_state["last_exception"] = traceback.format_exc()
            st.error("Self-test failed.")
            st.code(st.session_state["last_exception"])
