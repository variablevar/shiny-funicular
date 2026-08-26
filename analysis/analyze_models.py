#!/usr/bin/env python3
"""
Day 3 analysis: per-timeframe feature importance from trained FreqAI models,
logged to the MLflow tracking server.

For each FreqAI identifier (5m-primary, 15m-target, 1h-target), loads the
latest trained LightGBM model per pair, extracts gain importances, and groups
them by timeframe suffix (_5m/_15m/_30m/_1h/_4h) to verify the TF hierarchy:
micro features should matter for short horizons, macro for long.

MLflow logging uses the REST API directly (requests) to avoid the
mlflow<->ccxt cryptography dependency conflict in the venv.

Usage:
    source venv/bin/activate
    python analysis/analyze_models.py
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

MODELS_ROOT = Path("ft_userdata/user_data/models")
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT = "gtquant"

TIMEFRAMES = ["_5m", "_15m", "_30m", "_1h", "_4h"]


# --------------------------------------------------------------------- #
# MLflow minimal REST client
# --------------------------------------------------------------------- #

class MLflowREST:
    def __init__(self, uri: str):
        self.uri = uri.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.uri}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = requests.get(f"{self.uri}{path}", params=params or {}, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_or_create_experiment(self, name: str) -> str:
        try:
            res = self._get("/api/2.0/mlflow/experiments/get-by-name", {"experiment_name": name})
            return res["experiment"]["experiment_id"]
        except requests.HTTPError:
            res = self._post("/api/2.0/mlflow/experiments/create", {"name": name})
            return res["experiment_id"]

    def create_run(self, experiment_id: str, run_name: str) -> str:
        res = self._post(
            "/api/2.0/mlflow/runs/create",
            {
                "experiment_id": experiment_id,
                "run_name": run_name,
                "start_time": int(time.time() * 1000),
                "tags": [{"key": "project", "value": "gtquant"}],
            },
        )
        return res["run"]["info"]["run_id"]

    def log_param(self, run_id: str, key: str, value) -> None:
        self._post("/api/2.0/mlflow/runs/log-parameter",
                   {"run_id": run_id, "key": key, "value": str(value)})

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self._post("/api/2.0/mlflow/runs/log-metric",
                   {"run_id": run_id, "key": key, "value": float(value),
                    "timestamp": int(time.time() * 1000), "step": step})

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self._post("/api/2.0/mlflow/runs/set-tag",
                   {"run_id": run_id, "key": key, "value": value})


# --------------------------------------------------------------------- #
# Feature importance analysis
# --------------------------------------------------------------------- #

def tf_of_feature(feature: str) -> str:
    """Map a FreqAI feature name to its timeframe suffix."""
    for tf in TIMEFRAMES:
        if feature.endswith(tf):
            return tf.lstrip("_")
    return "base"


def latest_models(identifier_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return [(pair, model_path, metadata_path)] for the latest sub-train per pair."""
    out = []
    for pair_dir in sorted(identifier_dir.glob("sub-train-*")):
        pair = pair_dir.name.replace("sub-train-", "")
        models = sorted(pair_dir.glob("*_model.joblib"))
        metas = sorted(pair_dir.glob("*_metadata.json"))
        if models and metas:
            out.append((pair, models[-1], metas[-1]))
    return out


def analyze_identifier(identifier: str, mlflow: MLflowREST, exp_id: str) -> pd.DataFrame:
    id_dir = MODELS_ROOT / identifier
    if not id_dir.exists():
        print(f"  {identifier}: no models found, skipping")
        return pd.DataFrame()

    for pair, model_path, meta_path in latest_models(id_dir):
        meta = json.loads(meta_path.read_text())
        features = meta["training_features_list"]
        model = joblib.load(model_path)

        # LightGBM sklearn wrapper -> gain importances, aligned by the
        # model's own feature names (LGBM drops constant features at train
        # time, so len(importances) can differ from the metadata list).
        booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
        if booster is not None:
            importances = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
            names = list(booster.feature_name())
            if len(names) != len(importances):
                names = features[: len(importances)]
        else:
            importances = np.asarray(getattr(model, "feature_importances_", np.zeros(len(features))), dtype=float)
            names = features[: len(importances)]

        imp = pd.DataFrame({"feature": names, "gain": importances})
        imp["gain_norm"] = imp["gain"] / imp["gain"].sum() if imp["gain"].sum() > 0 else 0.0
        imp["tf"] = imp["feature"].map(tf_of_feature)

        tf_share = imp.groupby("tf")["gain_norm"].sum().sort_values(ascending=False)
        top = imp.nlargest(15, "gain_norm")

        print(f"\n=== {identifier} / {pair} ({model_path.parent.name}) ===")
        print("TF importance share:")
        for tf, share in tf_share.items():
            print(f"  {tf:6s} {share:7.2%}")
        print("Top 10 features:")
        for _, row in top.head(10).iterrows():
            print(f"  {row['feature'][:60]:60s} {row['gain_norm']:.4f}")

        # Log to MLflow.
        run_id = mlflow.create_run(exp_id, f"{identifier}/{pair}")
        mlflow.log_param(run_id, "identifier", identifier)
        mlflow.log_param(run_id, "pair", pair)
        mlflow.log_param(run_id, "model_dir", model_path.parent.name)
        mlflow.log_param(run_id, "n_features", len(features))
        mlflow.log_param(run_id, "label", ",".join(meta.get("label_list", [])))
        for tf, share in tf_share.items():
            mlflow.log_metric(run_id, f"tf_share_{tf}", share)
        for i, (_, row) in enumerate(top.head(10).iterrows(), 1):
            mlflow.log_param(run_id, f"top_feature_{i}", row["feature"])
            mlflow.log_metric(run_id, f"top_gain_{i}", row["gain_norm"])
        mlflow.set_tag(run_id, "stage", "day3-baseline")
        print(f"  -> logged to MLflow run {run_id[:8]}…")

        imp.to_csv(MODELS_ROOT / identifier / f"feature_importance_{pair}.csv", index=False)

    return imp


def main() -> None:
    mlflow = MLflowREST(MLFLOW_URI)
    exp_id = mlflow.get_or_create_experiment(EXPERIMENT)
    print(f"MLflow experiment '{EXPERIMENT}' id={exp_id}")

    for identifier in ["gtquant-v0.1-5m-primary", "gtquant-v0.1-15m-target", "gtquant-v0.1-1h-target"]:
        analyze_identifier(identifier, mlflow, exp_id)

    print("\nDone. MLflow UI: http://localhost:5000")


if __name__ == "__main__":
    main()
