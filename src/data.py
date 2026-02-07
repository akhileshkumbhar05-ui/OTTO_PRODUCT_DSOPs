import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from .config import (
    RAW_TRAIN_JSONL,
    PROCESSED_DIR,
    SESSIONS_PARQUET,
    SESSION_INDEX_PARQUET,
    MAX_SESSIONS,
    MAX_EVENTS_PER_SESSION,
)

# OTTO event types typically: clicks=0, carts=1, orders=2
TYPE_MAP = {"clicks": 0, "carts": 1, "orders": 2}


@dataclass
class Event:
    session: int
    aid: int
    ts: int
    etype: int


def _iter_events_from_jsonl(jsonl_path: Path) -> Iterable[Tuple[int, List[dict]]]:
    """
    Yields (session_id, events_list) from train.jsonl.
    Each line: {"session":..., "events":[{"aid":..,"ts":..,"type":..}, ...]}
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield int(obj["session"]), obj["events"]


def build_events_parquet(
    jsonl_path: Path = RAW_TRAIN_JSONL,
    out_events_parquet: Path = SESSIONS_PARQUET,
    out_sessions_index: Path = SESSION_INDEX_PARQUET,
) -> Dict[str, int]:
    """
    Reads JSONL, caps sessions/events, writes:
      - events.parquet: columns [session, aid, ts, type]
      - sessions_index.parquet: one column [session] (for UI dropdown)
    Returns stats dict.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    sessions = []
    n_sessions = 0
    n_events = 0

    for session_id, events in tqdm(_iter_events_from_jsonl(jsonl_path), desc="Reading JSONL"):
        sessions.append(session_id)
        n_sessions += 1

        # keep only last MAX_EVENTS_PER_SESSION (recency)
        if MAX_EVENTS_PER_SESSION and len(events) > MAX_EVENTS_PER_SESSION:
            events = events[-MAX_EVENTS_PER_SESSION:]

        for e in events:
            rows.append(
                {
                    "session": session_id,
                    "aid": int(e["aid"]),
                    "ts": int(e["ts"]),
                    "type": TYPE_MAP[e["type"]],
                }
            )
        n_events += len(events)

        if MAX_SESSIONS is not None and n_sessions >= MAX_SESSIONS:
            break

    df = pd.DataFrame(rows)
    # Ensure stable ordering
    df.sort_values(["session", "ts"], inplace=True)

    df.to_parquet(out_events_parquet, index=False)
    pd.DataFrame({"session": sessions}).drop_duplicates().to_parquet(out_sessions_index, index=False)

    return {"sessions": n_sessions, "events": n_events, "path": str(out_events_parquet)}
