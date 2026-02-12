# src/config.py
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------
# Standard (full) locations
# ------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

RAW_TRAIN_JSONL = RAW_DIR / "train.jsonl"

SESSIONS_PARQUET = PROCESSED_DIR / "events.parquet"
SESSION_INDEX_PARQUET = PROCESSED_DIR / "sessions_index.parquet"

ITEM_POPULARITY_PARQUET = ARTIFACTS_DIR / "item_popularity.parquet"
COVISIT_PARQUET = ARTIFACTS_DIR / "covis_topk.parquet"

MODEL_PATH = MODELS_DIR / "scorer.joblib"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"

# ------------------------------------------------------------
# Demo assets (portable for deployments)
# ------------------------------------------------------------
DEMO_DIR = ROOT / "demo_assets"
DEMO_EVENTS = DEMO_DIR / "events_demo.parquet"
DEMO_SESSION_INDEX = DEMO_DIR / "sessions_index_demo.parquet"
DEMO_POP = DEMO_DIR / "item_popularity_demo.parquet"
DEMO_COVIS = DEMO_DIR / "covis_topk_demo.parquet"
DEMO_MODEL = DEMO_DIR / "scorer_demo.joblib"
DEMO_METRICS = DEMO_DIR / "metrics_demo.json"

# Force demo mode via env if you want (optional)
# Set in Streamlit Cloud secrets or settings: USE_DEMO=1
USE_DEMO = os.getenv("USE_DEMO", "0").strip() == "1"

def pick_path(primary: Path, demo: Path) -> Path:
    """
    Use primary if it exists, else demo if it exists.
    If USE_DEMO=1, prefer demo when available.
    """
    if USE_DEMO and demo.exists():
        return demo
    if primary.exists():
        return primary
    if demo.exists():
        return demo
    return primary  # for error messages

# These are the paths your app should use
SESSIONS_PARQUET_UI = pick_path(SESSIONS_PARQUET, DEMO_EVENTS)
SESSION_INDEX_PARQUET_UI = pick_path(SESSION_INDEX_PARQUET, DEMO_SESSION_INDEX)
ITEM_POPULARITY_PARQUET_UI = pick_path(ITEM_POPULARITY_PARQUET, DEMO_POP)
COVISIT_PARQUET_UI = pick_path(COVISIT_PARQUET, DEMO_COVIS)
MODEL_PATH_UI = pick_path(MODEL_PATH, DEMO_MODEL)
METRICS_JSON_UI = pick_path(METRICS_JSON, DEMO_METRICS)

# ------------------------------------------------------------
# OTTO covis settings
# ------------------------------------------------------------
COVIS_WINDOW = 20
COVIS_TOPK_PER_ITEM = 50

# ------------------------------------------------------------
# Runtime / resource caps (CPU-safe)
# ------------------------------------------------------------
RANDOM_SEED = 42
MAX_SESSIONS = 200_000
MAX_EVENTS_PER_SESSION = 50
MAX_CANDIDATES_PER_SESSION = 100
MAX_TRAIN_ROWS = 500_000

# Backward-compatible aliases
POPULARITY_PARQUET = ITEM_POPULARITY_PARQUET
COVIS_TOPK_PARQUET = COVISIT_PARQUET
COVIS_TOPK = COVIS_TOPK_PER_ITEM
