# src/insights.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Timing:
    candidate_ms: float
    scoring_ms: float
    total_ms: float


@dataclass
class ExplainRow:
    aid: int
    score: float
    pop_score: float
    covis_score: float
    recency_score: float


def session_summary(events: pd.DataFrame, session_id: int, last_n: int = 20) -> Dict:
    g = events.loc[events["session"] == session_id].sort_values("ts")
    aids = g["aid"].astype(int).tolist()
    etypes = g["type"].astype(int).tolist() if "type" in g.columns else None

    uniq = len(set(aids))
    rep_rate = 0.0 if len(aids) == 0 else (len(aids) - uniq) / max(1, len(aids))
    out = {
        "session_id": int(session_id),
        "n_events": int(len(aids)),
        "n_unique_items": int(uniq),
        "repeat_rate": float(rep_rate),
        "last_items": aids[-last_n:],
    }
    if etypes is not None:
        out["type_counts"] = {
            "clicks": int(sum(1 for t in etypes if t == 0)),
            "carts": int(sum(1 for t in etypes if t == 1)),
            "orders": int(sum(1 for t in etypes if t == 2)),
        }
    return out


def timed_recommend(
    context_aids: List[int],
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    make_candidates_fn,
    score_candidates_fn,
    model_bundle,
    max_candidates: int,
) -> Tuple[List[Tuple[int, float]], Timing]:
    t0 = time.perf_counter()
    cands = make_candidates_fn(context_aids, pop_df, covis_df, max_candidates=max_candidates)
    t1 = time.perf_counter()
    ranked = score_candidates_fn(context_aids, cands, pop_df, covis_df, model_bundle)
    t2 = time.perf_counter()

    return ranked, Timing(
        candidate_ms=(t1 - t0) * 1000.0,
        scoring_ms=(t2 - t1) * 1000.0,
        total_ms=(t2 - t0) * 1000.0,
    )


def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def explain_topk_simple(
    context_aids: List[int],
    top_aids_scores: List[Tuple[int, float]],
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    alpha_pop: float = 0.35,
    alpha_covis: float = 0.50,
    alpha_recency: float = 0.15,
) -> pd.DataFrame:
    # popularity column
    pop_col = "pop" if "pop" in pop_df.columns else ("score" if "score" in pop_df.columns else None)
    pop_map = dict(zip(pop_df["aid"].astype(int), pop_df[pop_col].astype(float))) if pop_col else {}

    # covis columns (best effort)
    cols = covis_df.columns.tolist()
    src_col = "src" if "src" in cols else ("aid_x" if "aid_x" in cols else ("a" if "a" in cols else None))
    dst_col = "dst" if "dst" in cols else ("aid_y" if "aid_y" in cols else ("b" if "b" in cols else None))
    w_col = "w" if "w" in cols else ("weight" if "weight" in cols else ("score" if "score" in cols else None))

    covis_map = {}
    if src_col and dst_col and w_col:
        covis_map = {(int(r[src_col]), int(r[dst_col])): float(r[w_col]) for _, r in covis_df.iterrows()}

    # recency weights
    rec_w = np.linspace(0.2, 1.0, num=len(context_aids), dtype="float32") if context_aids else np.array([])

    tmp_rows: List[ExplainRow] = []
    for aid, final_score in top_aids_scores:
        aid = int(aid)
        pop_score = float(pop_map.get(aid, 0.0))

        covis_vals = [float(covis_map.get((int(c), aid), 0.0)) for c in context_aids[-50:]]
        covis_score = float(np.sum(covis_vals)) if covis_vals else 0.0

        if covis_vals and len(rec_w) >= len(covis_vals):
            recency_score = float(np.sum(np.array(covis_vals, dtype="float32") * rec_w[-len(covis_vals):]))
        else:
            recency_score = 0.0

        tmp_rows.append(ExplainRow(aid=aid, score=float(final_score),
                                  pop_score=pop_score, covis_score=covis_score, recency_score=recency_score))

    # normalize for display
    p = _safe_norm(np.array([r.pop_score for r in tmp_rows], dtype="float32"))
    c = _safe_norm(np.array([r.covis_score for r in tmp_rows], dtype="float32"))
    rr = _safe_norm(np.array([r.recency_score for r in tmp_rows], dtype="float32"))

    out = []
    for i, r in enumerate(tmp_rows):
        out.append({
            "aid": r.aid,
            "final_score": round(r.score, 6),
            "popularity_signal": round(float(alpha_pop * p[i]), 4),
            "covis_signal": round(float(alpha_covis * c[i]), 4),
            "recency_signal": round(float(alpha_recency * rr[i]), 4),
        })

    df = pd.DataFrame(out).sort_values("final_score", ascending=False)
    return df
