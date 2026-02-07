import pandas as pd

from src.config import RAW_TRAIN_JSONL, SESSIONS_PARQUET
from src.data import build_events_parquet
from src.candidates import write_candidate_artifacts

def main():
    if not RAW_TRAIN_JSONL.exists():
        raise FileNotFoundError(f"Put train.jsonl at: {RAW_TRAIN_JSONL}")

    stats = build_events_parquet()
    print("[OK] Built parquet:", stats)

    events = pd.read_parquet(SESSIONS_PARQUET)
    write_candidate_artifacts(events)
    print("[OK] Built candidate artifacts (popularity + covis).")

if __name__ == "__main__":
    main()
