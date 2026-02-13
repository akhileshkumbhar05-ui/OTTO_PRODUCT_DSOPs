# src/model.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .config import (
    MODEL_PATH,
    RANDOM_SEED,
    MAX_CANDIDATES_PER_SESSION,
    MAX_TRAIN_ROWS,
)
from .candidates import make_candidates_for_session


@dataclass
class ScorerBundle:
    scaler: StandardScaler
    model: SGDClassifier


def _build_fast_maps(
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame
) -> Tuple[Dict[int, float], Dict[int, List[Tuple[int, int]]]]:
    pop_map = dict(zip(pop_df["aid"].astype(int), pop_df["pop"].astype(float)))

    covis_map: Dict[int, List[Tuple[int, int]]] = {}
    if len(covis_df) > 0:
        for aid, g in covis_df.groupby("aid", sort=False):
            covis_map[int(aid)] = list(
                zip(
                    g["neighbor_aid"].astype(int).tolist(),
                    g["covis"].astype(int).tolist(),
                )
            )
    return pop_map, covis_map


def _features_for_candidate(
    candidate_aid: int,
    session_aids: List[int],
    pop_map: Dict[int, float],
    covis_map: Dict[int, List[Tuple[int, int]]],
) -> np.ndarray:
    """
    Lightweight, CPU-safe features:
      f0: log1p(popularity)
      f1: log1p(covis sum from last 5 session items)
      f2: recency flag (in last 10)
      f3: seen flag (appears anywhere in session)
    """
    pop = np.log1p(pop_map.get(int(candidate_aid), 0.0))

    covis_score = 0.0
    for seed in session_aids[-5:]:
        for n, c in covis_map.get(int(seed), []):
            if int(n) == int(candidate_aid):
                covis_score += float(c)
    covis_score = np.log1p(covis_score)

    recency = 1.0 if int(candidate_aid) in set(map(int, session_aids[-10:])) else 0.0
    seen = 1.0 if int(candidate_aid) in set(map(int, session_aids)) else 0.0

    return np.array([pop, covis_score, recency, seen], dtype=np.float32)


def build_training_rows(
    events: pd.DataFrame,
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    train_sessions: int,
    log_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU-safe training row builder:
      - sample sessions first
      - filter events to sampled sessions BEFORE groupby
      - split 80/20 timeline (context/future)
      - candidates from context
      - label=1 if candidate appears in future
    capped by MAX_TRAIN_ROWS
    """
    rng = np.random.default_rng(RANDOM_SEED)

    events = events[["session", "aid", "ts"]].copy()
    pop_map, covis_map = _build_fast_maps(pop_df, covis_df)

    all_sessions = events["session"].drop_duplicates().to_numpy()
    if train_sessions < len(all_sessions):
        sampled = rng.choice(all_sessions, size=train_sessions, replace=False)
    else:
        sampled = all_sessions

    events_small = events[events["session"].isin(set(map(int, sampled)))].copy()
    events_small.sort_values(["session", "ts"], inplace=True)

    grouped = events_small.groupby("session", sort=False)

    X: List[np.ndarray] = []
    y: List[int] = []

    sids = events_small["session"].drop_duplicates().tolist()

    for i, sid in enumerate(sids, start=1):
        g = grouped.get_group(sid)
        aids = g["aid"].astype(int).tolist()
        if len(aids) < 5:
            continue

        split = max(1, int(0.8 * len(aids)))
        context = aids[:split]
        future = set(aids[split:])

        cands = make_candidates_for_session(
            context, pop_df, covis_df, max_candidates=MAX_CANDIDATES_PER_SESSION
        )

        for cand in cands:
            X.append(_features_for_candidate(int(cand), context, pop_map, covis_map))
            y.append(1 if int(cand) in future else 0)

        if len(y) >= MAX_TRAIN_ROWS:
            break

        if log_every and (i % log_every == 0):
            print(f"[TRAIN] sessions: {i}/{len(sids)} | rows: {len(y)}")

    if not X:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int32)


def train_scorer(X: np.ndarray, y: np.ndarray) -> ScorerBundle:
    """
    Lightweight classifier (won’t crash laptop).
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(Xs, y)

    return ScorerBundle(scaler=scaler, model=clf)


def save_model(bundle: ScorerBundle) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": bundle.scaler, "model": bundle.model}, MODEL_PATH)


def load_model() -> ScorerBundle:
    """
    IMPORTANT: this name must exist (src.evaluate imports it).
    """
    obj = joblib.load(MODEL_PATH)
    return ScorerBundle(scaler=obj["scaler"], model=obj["model"])


def score_candidates(
    session_aids: List[int],
    candidates: List[int],
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    bundle: ScorerBundle,
) -> List[Tuple[int, float]]:
    pop_map, covis_map = _build_fast_maps(pop_df, covis_df)

    feats = np.vstack(
        [_features_for_candidate(int(c), session_aids, pop_map, covis_map) for c in candidates]
    )
    feats_s = bundle.scaler.transform(feats)
    probs = bundle.model.predict_proba(feats_s)[:, 1]

    ranked = sorted(zip(candidates, probs.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

# -----------------------------
# Model versioning + explanation helpers
# -----------------------------
from pathlib import Path
from typing import Optional

def _resolve_model_path(version: Optional[str] = None) -> Path:
    """
    If version is provided, load from artifacts/<version>/model.joblib (convention),
    else fall back to MODEL_PATH.
    """
    if version:
        p = Path("artifacts") / version / "model.joblib"
        if p.exists():
            return p
    return MODEL_PATH


def load_model_version(version: Optional[str] = None) -> ScorerBundle:
    """
    Loader used by Streamlit UI.
    version=None -> default MODEL_PATH
    """
    p = _resolve_model_path(version)
    obj = joblib.load(p)
    return ScorerBundle(scaler=obj["scaler"], model=obj["model"])


from typing import Any

def explain_one_prediction(
    session_aids: List[int],
    candidate_aid: int,
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    bundle: ScorerBundle,
) -> Dict[str, Any]:
    """
    SHAP-style attribution for linear models:
      contribution(feature) = standardized_feature_value * weight

    Returns a stable schema used by Streamlit:
      {
        "candidate_aid": int,
        "prob": float,
        "method": "linear",
        "feature_names": [str, ...],
        "contrib": [float, ...],
        "bias": float,
        "score_logit": float
      }
    """
    pop_map, covis_map = _build_fast_maps(pop_df, covis_df)
    x = _features_for_candidate(int(candidate_aid), session_aids, pop_map, covis_map).reshape(1, -1)
    xs = bundle.scaler.transform(x)[0]

    w = bundle.model.coef_.ravel()
    b = float(bundle.model.intercept_.ravel()[0]) if hasattr(bundle.model, "intercept_") else 0.0

    contrib = (xs * w).astype(float)
    score_logit = float(b + contrib.sum())

    # sigmoid -> probability
    prob = float(1.0 / (1.0 + np.exp(-score_logit)))

    feature_names = [
        "popularity(log1p)",
        "covis(log1p_sum_last5)",
        "recency(in_last10)",
        "seen_in_session",
    ]

    return {
        "candidate_aid": int(candidate_aid),
        "prob": prob,
        "method": "linear",
        "feature_names": feature_names,
        "contrib": contrib.tolist(),
        "bias": float(b),
        "score_logit": score_logit,
    }
def score_candidates_fast(
    session_aids: List[int],
    candidates: List[int],
    pop_map: Dict[int, float],
    covis_map: Dict[int, List[Tuple[int, int]]],
    bundle: ScorerBundle,
) -> List[Tuple[int, float]]:
    """
    Faster scorer that avoids building maps from DataFrames every call.
    Expects pop_map + covis_map already prepared by the caller.

    Returns: list of (aid, probability) sorted desc.
    """
    if not candidates:
        return []

    feats = np.vstack(
        [
            _features_for_candidate(int(c), session_aids, pop_map, covis_map)
            for c in candidates
        ]
    )
    feats_s = bundle.scaler.transform(feats)
    probs = bundle.model.predict_proba(feats_s)[:, 1]

    ranked = sorted(
        zip([int(c) for c in candidates], probs.astype(float).tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked
