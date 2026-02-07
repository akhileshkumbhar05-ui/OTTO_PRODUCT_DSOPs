# src/config.py
from __future__ import annotations

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

# Ensure dirs exist (safe to call repeatedly)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
# You have train.jsonl locally. Default expected location:
#   data/raw/train.jsonl
RAW_TRAIN_JSONL = RAW_DIR / "train.jsonl"

# ------------------------------------------------------------
# Processed outputs
# ------------------------------------------------------------
# Event-level table: [session, aid, ts, type(optional)]
SESSIONS_PARQUET = PROCESSED_DIR / "events.parquet"
SESSION_INDEX_PARQUET = PROCESSED_DIR / "sessions_index.parquet"

# ------------------------------------------------------------
# Candidate artifacts (names expected by src/candidates.py)
# ------------------------------------------------------------
# Popularity baseline: [aid, pop]
ITEM_POPULARITY_PARQUET = ARTIFACTS_DIR / "item_popularity.parquet"

# Co-visitation topk neighbors: [aid, neighbor_aid, covis]
COVISIT_PARQUET = ARTIFACTS_DIR / "covis_topk.parquet"

# OTTO covis settings
COVIS_WINDOW = 20          # lookback window inside each session
COVIS_TOPK_PER_ITEM = 50   # keep only top-k neighbors per aid

# ------------------------------------------------------------
# Model + evaluation artifacts
# ------------------------------------------------------------
# Default model path (your scripts can version this if you want)
MODEL_PATH = MODELS_DIR / "scorer.joblib"

# Metrics written by eval scripts / UI actions
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"

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
# Backward-compatible aliases (so other older imports don’t break)
# ------------------------------------------------------------
# If any file still imports these older names, they will still work.
POPULARITY_PARQUET = ITEM_POPULARITY_PARQUET
COVIS_TOPK_PARQUET = COVISIT_PARQUET
COVIS_TOPK = COVIS_TOPK_PER_ITEM
