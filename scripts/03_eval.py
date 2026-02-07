# scripts/03_eval.py
import time
import pandas as pd

from src.config import SESSIONS_PARQUET
from src.candidates import load_popularity, load_covis_topk
from src.evaluate import evaluate_sample


def main():
    t0 = time.time()
    print("[EVAL] Loading events parquet:", SESSIONS_PARQUET)

    # Load only required columns to reduce RAM
    events = pd.read_parquet(SESSIONS_PARQUET, columns=["session", "aid", "ts"])
    print(f"[EVAL] Loaded events: shape={events.shape} | elapsed={time.time()-t0:.1f}s")

    print("[EVAL] Loading candidate sources...")
    pop = load_popularity()
    covis = load_covis_topk()
    print(f"[EVAL] Loaded pop/covis | elapsed={time.time()-t0:.1f}s")

    metrics = evaluate_sample(
        events,
        pop,
        covis,
        n_sessions=500,
        k=20,
        max_candidates=100,   # IMPORTANT: cap candidates for speed
        log_every=50,
    )
    print("[OK] Metrics:", metrics)


if __name__ == "__main__":
    main()
