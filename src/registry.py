# src/registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from .config import MODELS_DIR, DEFAULT_MODEL_NAME


@dataclass
class ModelRecord:
    name: str
    path: Path


def list_models() -> List[ModelRecord]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(MODELS_DIR.glob("*.joblib"), key=lambda p: p.name)
    return [ModelRecord(name=p.name, path=p) for p in paths]


def ensure_default_model_exists(legacy_model_path: Optional[Path] = None) -> None:
    """
    If you previously saved a model to some MODEL_PATH, you can pass it here and
    we will copy it into the registry folder as DEFAULT_MODEL_NAME.
    Otherwise does nothing.
    """
    default_path = MODELS_DIR / DEFAULT_MODEL_NAME
    if default_path.exists():
        return

    if legacy_model_path and legacy_model_path.exists():
        obj = joblib.load(legacy_model_path)
        joblib.dump(obj, default_path)


def load_model_by_name(name: str):
    p = MODELS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    obj = joblib.load(p)
    return obj


def save_model_by_name(obj: Dict, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = MODELS_DIR / name
    joblib.dump(obj, p)
    return p
