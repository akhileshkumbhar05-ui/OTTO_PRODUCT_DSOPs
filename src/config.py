# src/config.py
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Demo assets (small, repo-commit friendly)
DEMO_DIR = ROOT / "demo_assets"

# Detect Hugging Face Spaces runtime
IN_HF_SPACE = os.environ.get("SPACE_ID") is not None

# Ensure dirs exist (safe to call repeatedly)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
# Local expected dataset path (not used on HF unless you mount it)
RAW_TRAIN_JSONL = RAW_DIR / "train.jsonl"

# ------------------------------------------------------------
# OTTO co-visitation settings
# ------------------------------------------------------------
COVIS_WINDOW = 20          # lookback window inside each session
COVIS_TOPK_PER_ITEM = 50   # keep only top-k neighbors per aid

# ------------------------------------------------------------
# Runtime / resource caps (CPU-safe)
# ------------------------------------------------------------
RANDOM_SEED = 42

# Ingest caps (keep laptop safe). Set None to use all.
MAX_SESSIONS = 200_000
MAX_EVENTS_PER_SESSION = 50

# Candidate cap per session during train/eval/serve
MAX_CANDIDATES_PER_SESSION = 100

# Training row cap (prevents crashing)
MAX_TRAIN_ROWS = 500_000

# ------------------------------------------------------------
# Runtime path switching (Local vs Hugging Face Spaces)
# ------------------------------------------------------------
if IN_HF_SPACE:
    # Use demo assets committed to the repo
    SESSIONS_PARQUET = DEMO_DIR / "events_demo.parquet"
    SESSION_INDEX_PARQUET = DEMO_DIR / "sessions_index_demo.parquet"

    ITEM_POPULARITY_PARQUET = DEMO_DIR / "item_popularity_demo.parquet"
    COVISIT_PARQUET = DEMO_DIR / "covis_topk_demo.parquet"

    MODEL_PATH = DEMO_DIR / "scorer_demo.joblib"
    METRICS_JSON = DEMO_DIR / "metrics_demo.json"

else:
    # Local full pipeline artifacts
    SESSIONS_PARQUET = PROCESSED_DIR / "events.parquet"
    SESSION_INDEX_PARQUET = PROCESSED_DIR / "sessions_index.parquet"

    ITEM_POPULARITY_PARQUET = ARTIFACTS_DIR / "item_popularity.parquet"
    COVISIT_PARQUET = ARTIFACTS_DIR / "covis_topk.parquet"

    MODEL_PATH = MODELS_DIR / "scorer.joblib"
    METRICS_JSON = ARTIFACTS_DIR / "metrics.json"

# ------------------------------------------------------------
# Backward-compatible aliases (older imports won’t break)
# ------------------------------------------------------------
POPULARITY_PARQUET = ITEM_POPULARITY_PARQUET
COVIS_TOPK_PARQUET = COVISIT_PARQUET
COVIS_TOPK = COVIS_TOPK_PER_ITEM
