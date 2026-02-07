import pandas as pd

from src.config import SESSIONS_PARQUET, TRAIN_SESSIONS
from src.candidates import load_popularity, load_covis_topk
from src.model import build_training_rows, train_scorer, save_model

def main():
    events = pd.read_parquet(SESSIONS_PARQUET)
    pop = load_popularity()
    covis = load_covis_topk()

    print("[INFO] Building training data (capped)...")
    X, y = build_training_rows(events, pop, covis, train_sessions=TRAIN_SESSIONS)
    print("[INFO] X shape:", X.shape, "positives:", int(y.sum()), "negatives:", int((y == 0).sum()))

    if X.shape[0] == 0:
        raise RuntimeError("No training rows created. Increase MAX_SESSIONS or TRAIN_SESSIONS, or check data.")

    print("[INFO] Training lightweight scorer...")
    bundle = train_scorer(X, y)

    save_model(bundle)
    print("[OK] Saved model.")

if __name__ == "__main__":
    main()
