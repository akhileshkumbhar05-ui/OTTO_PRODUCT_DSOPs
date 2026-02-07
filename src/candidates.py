from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from .config import (
    COVIS_WINDOW,
    COVIS_TOPK_PER_ITEM,
    ITEM_POPULARITY_PARQUET,
    COVISIT_PARQUET,
)


def build_item_popularity(events: pd.DataFrame) -> pd.DataFrame:
    """
    Popularity = count of aids across all events (simple and strong baseline).
    Returns df with [aid, pop].
    """
    pop = events["aid"].value_counts().rename_axis("aid").reset_index(name="pop")
    pop.sort_values("pop", ascending=False, inplace=True)
    return pop


def build_covis_topk(events: pd.DataFrame) -> pd.DataFrame:
    """
    Build co-visitation counts within each session using a sliding lookback window.
    Stores top-k neighbors for each item to stay bounded.
    Output df columns: [aid, neighbor_aid, covis]
    """
    # events must be sorted by session, ts
    covis_counts: Dict[int, Counter] = defaultdict(Counter)

    # group by session without materializing giant lists of all data at once
    for session_id, g in tqdm(events.groupby("session", sort=False), desc="Building covis"):
        aids = g["aid"].tolist()
        # sliding window: for each position i, connect to previous up to window
        for i in range(len(aids)):
            a = aids[i]
            start = max(0, i - COVIS_WINDOW)
            for j in range(start, i):
                b = aids[j]
                if a == b:
                    continue
                covis_counts[a][b] += 1
                covis_counts[b][a] += 1

    rows = []
    for aid, counter in covis_counts.items():
        for neighbor_aid, c in counter.most_common(COVIS_TOPK_PER_ITEM):
            rows.append({"aid": int(aid), "neighbor_aid": int(neighbor_aid), "covis": int(c)})

    out = pd.DataFrame(rows)
    return out


def write_candidate_artifacts(events: pd.DataFrame) -> None:
    pop = build_item_popularity(events)
    covis = build_covis_topk(events)

    ITEM_POPULARITY_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    pop.to_parquet(ITEM_POPULARITY_PARQUET, index=False)
    covis.to_parquet(COVISIT_PARQUET, index=False)


def load_popularity() -> pd.DataFrame:
    return pd.read_parquet(ITEM_POPULARITY_PARQUET)


def load_covis_topk() -> pd.DataFrame:
    return pd.read_parquet(COVISIT_PARQUET)


def make_candidates_for_session(
    session_aids: List[int],
    pop_df: pd.DataFrame,
    covis_df: pd.DataFrame,
    max_candidates: int = 50,
) -> List[int]:
    """
    Candidate set = union of:
      - covis neighbors of last few items
      - top popular items
    Then ranked by heuristic score.
    """
    if not session_aids:
        return pop_df["aid"].head(max_candidates).tolist()

    # quick lookups
    pop_map = dict(zip(pop_df["aid"].astype(int), pop_df["pop"].astype(float)))

    # covis_df may be large; we rely on it being topk-bounded
    # build a mapping for fast lookup: aid -> list of (neighbor, covis)
    # For app speed, do this once and pass in pre-built dict (app does that).
    # Here: build on the fly (ok for training batch size), but not ideal for huge loops.
    grouped = covis_df.groupby("aid", sort=False)

    seed_items = session_aids[-5:]  # last 5 events
    scores = Counter()

    for a in seed_items:
        if a in grouped.indices:
            sub = covis_df.iloc[grouped.indices[a]]
            for n, c in zip(sub["neighbor_aid"].tolist(), sub["covis"].tolist()):
                scores[int(n)] += float(c)

    # add popularity fallback
    for aid in pop_df["aid"].head(200).tolist():
        scores[int(aid)] += 0.01 * float(pop_map.get(int(aid), 0.0))

    # boost seen items slightly (often repeats happen)
    for a in session_aids[-10:]:
        scores[int(a)] += 0.1

    ranked = [aid for aid, _ in scores.most_common(max_candidates)]
    # ensure exactly max_candidates
    if len(ranked) < max_candidates:
        ranked += pop_df["aid"].head(max_candidates).tolist()
        ranked = list(dict.fromkeys(ranked))[:max_candidates]
    return ranked

# --- FAST STRUCTURES (build once, reuse) ---

from typing import Optional

def build_fast_structures(pop_df: pd.DataFrame, covis_df: pd.DataFrame):
    """
    Build dictionaries for O(1) lookup during eval/training.
    Returns:
      pop_top: list[int] of popular aids (descending)
      pop_map: dict[int, float] aid -> pop
      covis_map: dict[int, list[tuple[int,int]]] aid -> [(neighbor, covis), ...]
    """
    pop_top = pop_df["aid"].astype(int).tolist()
    pop_map = dict(zip(pop_df["aid"].astype(int), pop_df["pop"].astype(float)))

    covis_map = {}
    # covis_df is already top-k bounded; safe to keep in memory
    for aid, g in covis_df.groupby("aid", sort=False):
        covis_map[int(aid)] = list(
            zip(g["neighbor_aid"].astype(int).tolist(), g["covis"].astype(int).tolist())
        )

    return pop_top, pop_map, covis_map


def make_candidates_for_session_fast(
    session_aids: List[int],
    pop_top: List[int],
    pop_map: Dict[int, float],
    covis_map: Dict[int, List[Tuple[int, int]]],
    max_candidates: int = 50,
    seed_last_n: int = 5,
) -> List[int]:
    """
    Fast candidate generation using dict lookups only (no DataFrame ops).
    """
    if not session_aids:
        return pop_top[:max_candidates]

    seed_items = session_aids[-seed_last_n:]
    scores = Counter()

    for a in seed_items:
        for n, c in covis_map.get(int(a), []):
            scores[int(n)] += float(c)

    # popularity fallback
    for aid in pop_top[:200]:
        scores[int(aid)] += 0.01 * float(pop_map.get(int(aid), 0.0))

    # slight boost to recently seen items
    recent = set(session_aids[-10:])
    for a in recent:
        scores[int(a)] += 0.1

    ranked = [aid for aid, _ in scores.most_common(max_candidates)]
    if len(ranked) < max_candidates:
        # fill with popular
        ranked += pop_top
        ranked = list(dict.fromkeys(ranked))[:max_candidates]
    return ranked
