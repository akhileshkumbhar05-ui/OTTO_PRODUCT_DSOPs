# src/analytics.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import EVENT_TYPE_ID_TO_NAME
from .candidates import make_candidates_for_session
from .model import load_model_version, score_candidates, _build_fast_maps
from .evaluate import recall_at_k, hitrate_at_k


def compute_session_funnel(events: pd.DataFrame) -> Dict:
    """
    Funnel: clicks -> carts -> orders, based on event 'type' column if present.
    If your events parquet has no 'type', we fallback to a session-length funnel.
    """
    out = {}

    if "type" in events.columns:
        # counts by type across all events
        type_counts = events["type"].value_counts().to_dict()
        named = {EVENT_TYPE_ID_TO_NAME.get(int(k), str(k)): int(v) for k, v in type_counts.items()}

        out["event_counts"] = named

        # session-level: proportion of sessions with >=1 cart/order
        sess = events.groupby("session", sort=False)["type"].apply(list)
        has_click = float(np.mean([0 in s for s in sess]))
        has_cart = float(np.mean([1 in s for s in sess]))
        has_order = float(np.mean([2 in s for s in sess]))
        out["session_rates"] = {"has_click": has_click, "has_cart": has_cart, "has_order": has_order}
    else:
        # fallback: session length distribution
        lens = events.groupby("session", sort=False)["aid"].size()
        out["session_len"] = {
            "p50": int(lens.quantile(0.50)),
            "p75": int(lens.quantile(0.75)),
            "p90": int(lens.quantile(0.90)),
            "max": int(lens.max()),
        }

    return out


def offline_eval(
    events: pd.DataFrame,
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    model_name: str,
    n_sessions: int = 500,
    k: int = 20,
    max_candidates: int = 100,
) -> Dict:
    """
    Offline: split session timeline into context/future, measure recall@k & hitrate@k.
    """
    rng = np.random.default_rng(42)
    sessions = events["session"].drop_duplicates().to_numpy()
    if n_sessions < len(sessions):
        sessions = rng.choice(sessions, size=n_sessions, replace=False)

    events_small = events[events["session"].isin(set(map(int, sessions)))].copy()
    events_small.sort_values(["session", "ts"], inplace=True)
    grouped = events_small.groupby("session", sort=False)

    bundle = load_model_version(model_name)

    recalls, hits = [], []
    t0 = time.time()

    for sid in sessions:
        g = grouped.get_group(int(sid))
        aids = g["aid"].astype(int).tolist()
        if len(aids) < 5:
            continue

        split = max(1, int(0.8 * len(aids)))
        context = aids[:split]
        future = set(aids[split:])

        cands = make_candidates_for_session(context, pop_df, covis_df, max_candidates=max_candidates)
        ranked = score_candidates(context, cands, pop_df, covis_df, bundle)
        pred = [aid for aid, _ in ranked]

        recalls.append(recall_at_k(future, pred, k))
        hits.append(hitrate_at_k(future, pred, k))

    t1 = time.time()

    return {
        "model": model_name,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hitrate@{k}": float(np.mean(hits)) if hits else 0.0,
        "n_eval_sessions": int(len(recalls)),
        "max_candidates": int(max_candidates),
        "elapsed_seconds": float(round(t1 - t0, 2)),
    }


def online_proxy_eval(
    events: pd.DataFrame,
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    model_name: str,
    n_sessions: int = 500,
    k: int = 20,
    max_candidates: int = 100,
) -> Dict:
    """
    "Online proxy" for a session recommender:
    For each session, pick a random cut point, predict next items from context,
    and measure if the next 1 item appears in top-k (CTR-like proxy).
    """
    rng = np.random.default_rng(123)
    sessions = events["session"].drop_duplicates().to_numpy()
    if n_sessions < len(sessions):
        sessions = rng.choice(sessions, size=n_sessions, replace=False)

    events_small = events[events["session"].isin(set(map(int, sessions)))].copy()
    events_small.sort_values(["session", "ts"], inplace=True)
    grouped = events_small.groupby("session", sort=False)

    bundle = load_model_version(model_name)
    pop_map, covis_map = _build_fast_maps(pop_df, covis_df)

    hits = []
    t0 = time.time()

    for sid in sessions:
        g = grouped.get_group(int(sid))
        aids = g["aid"].astype(int).tolist()
        if len(aids) < 6:
            continue

        # random cut (must leave at least 1 "next" item)
        cut = rng.integers(low=3, high=len(aids) - 1)
        context = aids[:cut]
        next_item = aids[cut]  # immediate next step

        cands = make_candidates_for_session(context, pop_df, covis_df, max_candidates=max_candidates)

        ranked = score_candidates(context, cands, pop_map, covis_map, bundle)
        pred = [aid for aid, _ in ranked][:k]

        hits.append(1.0 if int(next_item) in set(pred) else 0.0)

    t1 = time.time()

    return {
        "model": model_name,
        f"next_item_hit@{k}": float(np.mean(hits)) if hits else 0.0,
        "n_eval_sessions": int(len(hits)),
        "max_candidates": int(max_candidates),
        "elapsed_seconds": float(round(t1 - t0, 2)),
    }


def ab_compare(
    events: pd.DataFrame,
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    n_sessions: int = 500,
    k: int = 20,
    max_candidates: int = 100,
) -> Dict:
    """
    A/B simulation:
      - run offline_eval for both on the same sampled sessions (same seed)
      - return deltas
    """
    # to keep the same sampled sessions, just reuse offline_eval seed logic by calling both
    a = offline_eval(events, pop_df, covis_df, model_a, n_sessions=n_sessions, k=k, max_candidates=max_candidates)
    b = offline_eval(events, pop_df, covis_df, model_b, n_sessions=n_sessions, k=k, max_candidates=max_candidates)

    return {
        "A": a,
        "B": b,
        "delta_recall": b.get(f"recall@{k}", 0.0) - a.get(f"recall@{k}", 0.0),
        "delta_hitrate": b.get(f"hitrate@{k}", 0.0) - a.get(f"hitrate@{k}", 0.0),
    }
