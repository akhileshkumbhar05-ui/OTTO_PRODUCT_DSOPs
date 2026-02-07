# src/evaluate.py
import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import METRICS_JSON, MAX_CANDIDATES_PER_SESSION
from .candidates import make_candidates_for_session
from .model import load_model, score_candidates


def recall_at_k(y_true_set: set, y_pred_list: List[int], k: int) -> float:
    if not y_true_set:
        return 0.0
    pred = set(y_pred_list[:k])
    return len(pred & y_true_set) / float(len(y_true_set))


def hitrate_at_k(y_true_set: set, y_pred_list: List[int], k: int) -> float:
    pred = set(y_pred_list[:k])
    return 1.0 if len(pred & y_true_set) > 0 else 0.0


def evaluate_sample(
    events: pd.DataFrame,
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    n_sessions: int = 2000,
    k: int = 20,
    max_candidates: int | None = None,
    seed: int = 42,
    log_every: int = 50,
) -> Dict[str, float]:
    """
    Optimized evaluation:
      - sample sessions first
      - filter events to sampled sessions
      - build pop/covis dicts once
      - use fast candidate generation + fast scoring
    """
    import time
    from .candidates import build_fast_structures, make_candidates_for_session_fast
    from .model import load_model, score_candidates_fast

    t0 = time.time()

    events = events[["session", "aid", "ts"]].copy()
    rng = np.random.default_rng(seed)

    all_sessions = events["session"].drop_duplicates().to_numpy()
    if n_sessions < len(all_sessions):
        sampled = rng.choice(all_sessions, size=n_sessions, replace=False)
    else:
        sampled = all_sessions

    sampled_set = set(map(int, sampled))
    events_small = events[events["session"].isin(sampled_set)].copy()
    events_small.sort_values(["session", "ts"], inplace=True)

    grouped = events_small.groupby("session", sort=False)
    bundle = load_model()

    # build once
    pop_top, pop_map, covis_map = build_fast_structures(pop_df, covis_df)

    used_max_cands = int(max_candidates) if max_candidates is not None else int(MAX_CANDIDATES_PER_SESSION)

    recalls = []
    hits = []

    sids = events_small["session"].drop_duplicates().tolist()

    for i, sid in enumerate(sids, start=1):
        g = grouped.get_group(sid)
        aids = g["aid"].astype(int).tolist()
        if len(aids) < 5:
            continue

        split = max(1, int(0.8 * len(aids)))
        context = aids[:split]
        future = set(aids[split:])

        cands = make_candidates_for_session_fast(
            context,
            pop_top=pop_top,
            pop_map=pop_map,
            covis_map=covis_map,
            max_candidates=used_max_cands,
        )

        ranked = score_candidates_fast(context, cands, pop_map, covis_map, bundle)
        pred_list = [aid for aid, _ in ranked]

        recalls.append(recall_at_k(future, pred_list, k))
        hits.append(hitrate_at_k(future, pred_list, k))

        if log_every and (i % log_every == 0):
            print(f"[EVAL] {i}/{len(sids)} sessions | elapsed={time.time()-t0:.1f}s")

    metrics = {
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hitrate@{k}": float(np.mean(hits)) if hits else 0.0,
        "n_eval_sessions": int(len(recalls)),
        "max_candidates": int(used_max_cands),
        "elapsed_seconds": float(round(time.time() - t0, 2)),
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

