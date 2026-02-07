# src/experiments.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mlflow


@dataclass
class RunConfig:
    run_name: str
    params: Dict
    metrics: Dict
    artifacts: Optional[Dict[str, Path]] = None


def log_run(cfg: RunConfig, tracking_uri: str = "file:./mlruns") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("otto-session-recommender")

    with mlflow.start_run(run_name=cfg.run_name) as run:
        for k, v in cfg.params.items():
            mlflow.log_param(k, v)
        for k, v in cfg.metrics.items():
            mlflow.log_metric(k, float(v))

        # Also store a metrics.json per run (useful for grading)
        tmp = Path("artifacts") / "runs"
        tmp.mkdir(parents=True, exist_ok=True)
        out = tmp / f"{run.info.run_id}_metrics.json"
        out.write_text(json.dumps(cfg.metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(out))

        if cfg.artifacts:
            for _, path in cfg.artifacts.items():
                if path.exists():
                    mlflow.log_artifact(str(path))

        return run.info.run_id
