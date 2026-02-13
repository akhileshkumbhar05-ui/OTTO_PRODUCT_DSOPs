from __future__ import annotations

import json
import time
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Streamlit must be first UI call
# -------------------------
st.set_page_config(page_title="OTTO Session Recommender", layout="wide")
st.title("OTTO Session Recommender (End-to-End DS-Ops)")
st.caption("Serve recommendations from small demo assets (Cloud-safe).")

# Show full errors on Cloud (helps avoid “blank”)
try:
    st.set_option("client.showErrorDetails", True)
except Exception:
    pass

# -------------------------
# Paths / imports
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_DIR = ROOT / "demo_assets"

# Import after sys.path fix
from src.candidates import make_candidates_for_session
from src.model import ScorerBundle, score_candidates_fast  # requires your added function


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(p)


def pick_demo_assets():
    # Hard-pick demo assets only (prevents accidentally reading missing full artifacts)
    events = DEMO_DIR / "events_demo.parquet"
    pop = DEMO_DIR / "item_popularity_demo.parquet"
    covis = DEMO_DIR / "covis_topk_demo.parquet"
    model = DEMO_DIR / "scorer_demo.joblib"
    metrics = DEMO_DIR / "metrics_demo.json"
    return events, pop, covis, model, metrics


EVENTS_PATH, POP_PATH, COVIS_PATH, MODEL_PATH, METRICS_PATH = pick_demo_assets()


# -------------------------
# Guardrails (render, don’t crash)
# -------------------------
missing = [p for p in [EVENTS_PATH, POP_PATH, COVIS_PATH, MODEL_PATH] if not p.exists()]
if missing:
    st.error("Missing required demo assets in `demo_assets/`:")
    for p in missing:
        st.write(f"- `{_rel(p)}`")
    st.info("Fix: commit demo_assets/*.parquet and demo_assets/scorer_demo.joblib to GitHub, then reboot app.")
    st.stop()


# -------------------------
# Cached loaders (but not executed unless called)
# -------------------------
@st.cache_data(show_spinner=False)
def load_events(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["session", "aid", "ts"]].copy()


@st.cache_data(show_spinner=False)
def load_pop(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_covis(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> ScorerBundle:
    import joblib
    obj = joblib.load(Path(path))
    return ScorerBundle(scaler=obj["scaler"], model=obj["model"])


@st.cache_resource(show_spinner=False)
def build_pop_map(pop_df: pd.DataFrame) -> dict[int, float]:
    return dict(zip(pop_df["aid"].astype(int), pop_df["pop"].astype(float)))


def build_small_covis_map(covis_df: pd.DataFrame, seed_items: list[int]) -> dict[int, list[tuple[int, int]]]:
    # Only build for the last few seed items
    if not seed_items:
        return {}
    sub = covis_df[covis_df["aid"].isin(seed_items)]
    out: dict[int, list[tuple[int, int]]] = {}
    for aid, g in sub.groupby("aid", sort=False):
        out[int(aid)] = list(
            zip(
                g["neighbor_aid"].astype(int).tolist(),
                g["covis"].astype(int).tolist(),
            )
        )
    return out


def safe_read_metrics(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# -------------------------
# Sidebar (NO heavy compute)
# -------------------------
st.sidebar.header("Serving Settings")
max_candidates = st.sidebar.slider("Max candidates per session", 30, 300, 100, 10)
top_k = st.sidebar.slider("Top-K recommendations", 5, 50, 20, 5)

st.sidebar.divider()
st.sidebar.subheader("Demo assets being used")
st.sidebar.write(f"Events: `{_rel(EVENTS_PATH)}`")
st.sidebar.write(f"Popularity: `{_rel(POP_PATH)}`")
st.sidebar.write(f"Covis: `{_rel(COVIS_PATH)}`")
st.sidebar.write(f"Model: `{_rel(MODEL_PATH)}`")

metrics = safe_read_metrics(METRICS_PATH)
if metrics:
    st.sidebar.subheader("Demo metrics")
    st.sidebar.json(metrics)


# -------------------------
# Load light objects ONCE (cached) — safe even on rerun
# -------------------------
events = load_events(str(EVENTS_PATH))
pop_df = load_pop(str(POP_PATH))
covis_df = load_covis(str(COVIS_PATH))
bundle = load_bundle(str(MODEL_PATH))
pop_map = build_pop_map(pop_df)

# Sessions list
session_ids = events["session"].drop_duplicates().astype(int).tolist()


# -------------------------
# UI
# -------------------------
tab1, tab2 = st.tabs(["🔮 Recommend", "🧪 Debug"])

with tab1:
    st.subheader("Input")

    input_mode = st.radio(
        "Choose input mode",
        ["Pick a session_id (demo subset)", "Paste custom session aids"],
        horizontal=True,
    )

    context_aids: list[int] = []

    if input_mode.startswith("Pick"):
        sid = st.selectbox("session_id", session_ids[:5000])
        g = events[events["session"] == int(sid)].sort_values("ts")
        context_aids = g["aid"].astype(int).tolist()
        st.markdown("**Session (last 20 aids)**")
        st.json(context_aids[-20:])
    else:
        txt = st.text_area("Paste aids (comma-separated)", value="1098089,1354785,342507,1120175")
        try:
            context_aids = [int(x.strip()) for x in txt.split(",") if x.strip()]
        except Exception:
            st.error("Could not parse aids. Example: 1098089,1354785,342507")
            st.stop()

    # Use a FORM so slider changes don’t “auto trigger” any compute behavior
    with st.form("reco_form", clear_on_submit=False):
        submitted = st.form_submit_button("Recommend")

    if submitted:
        st.session_state["last_context_aids"] = context_aids
        st.session_state["last_run_settings"] = {"max_candidates": max_candidates, "top_k": top_k}
        st.session_state["last_error"] = None
        st.session_state["last_ranked"] = None
        st.session_state["last_latency"] = None

        try:
            with st.spinner("Generating candidates + scoring..."):
                t0 = time.time()

                # 1) candidates (uses DF; OK for demo assets)
                cands = make_candidates_for_session(
                    context_aids,
                    pop_df,
                    covis_df,
                    max_candidates=max_candidates,
                )

                # 2) covis map only for last few seeds
                seeds = [int(x) for x in context_aids[-5:]]
                covis_map_small = build_small_covis_map(covis_df, seeds)

                # 3) score
                ranked = score_candidates_fast(
                    session_aids=context_aids,
                    candidates=cands,
                    pop_map=pop_map,
                    covis_map=covis_map_small,
                    bundle=bundle,
                )[:top_k]

                t1 = time.time()

            st.session_state["last_ranked"] = ranked
            st.session_state["last_latency"] = round(t1 - t0, 3)

        except Exception as e:
            st.session_state["last_error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))

    # Render results (persist across reruns)
    if st.session_state.get("last_error"):
        st.error("Recommend crashed. Full traceback:")
        st.code(st.session_state["last_error"])
    elif st.session_state.get("last_ranked") is not None:
        ranked = st.session_state["last_ranked"]
        st.success(f"Done in {st.session_state.get('last_latency', '?')}s")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Top Recommendations")
            st.dataframe(pd.DataFrame(ranked, columns=["aid", "score"]), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Context (last 20)")
            st.json(st.session_state.get("last_context_aids", [])[-20:])

with tab2:
    st.subheader("Debug snapshot")
    st.write("Python:", sys.version)
    st.write("ROOT:", str(ROOT))
    st.write("events shape:", events.shape)
    st.write("pop_df shape:", pop_df.shape)
    st.write("covis_df shape:", covis_df.shape)
    st.write("sample session_ids:", session_ids[:5])
    st.write("Last settings:", st.session_state.get("last_run_settings"))
    st.write("Last latency:", st.session_state.get("last_latency"))
