"""
Regime Classifier for GT-Quant Milestone 2.

Trains an unsupervised K-Means model on trend/volatility/momentum features and maps
the resulting clusters to interpretable market regimes:

    - STRONG_BULL : aggressively upward trending, high momentum
    - BULL        : upward trending
    - RANGE       : low volatility, directionless
    - BEAR        : downward trending
    - HIGH_VOL    : highest realized volatility regardless of direction

The fitted scaler and KMeans model can be persisted so that the same regime
mapping is applied during walk-forward testing and live operation.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Regime(Enum):
    """Named market regimes produced by the classifier."""

    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    RANGE = "RANGE"
    BEAR = "BEAR"
    HIGH_VOL = "HIGH_VOL"


@dataclass
class RegimeProfile:
    """Summary statistics for a single cluster/regime."""

    regime: str
    n_bars: int
    pct_time: float
    avg_trend_score: float
    avg_volatility: float
    avg_return_24h: float
    avg_rsi: float
    avg_funding_zscore: float
    avg_volume_zscore: float


class RegimeClassifier:
    """
    K-Means based regime classifier.

    Parameters
    ----------
    feature_cols : list[str], optional
        Columns used for clustering. Defaults are chosen to capture
        trend, volatility, momentum and derivatives flow.
    n_clusters : int, optional
        Number of K-Means clusters. Default is 5 to match the five regimes.
    random_state : int, optional
        Seed for reproducible clustering.
    """

    # Features used by default. All are contemporaneous / past-only; no lookahead.
    DEFAULT_FEATURES: List[str] = [
        "return_24h",
        "ema_9_dist",
        "ema_50_dist",
        "ema_200_dist",
        "rsi_14",
        "macd_hist",
        "realized_vol_24h",
        "volume_zscore",
        "funding_zscore",
    ]

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        n_clusters: int = 5,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols or self.DEFAULT_FEATURES.copy()
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[KMeans] = None
        self.cluster_regime_map: Optional[dict[int, Regime]] = None
        self.regime_profiles: Optional[List[RegimeProfile]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> RegimeClassifier:
        """
        Fit scaler + KMeans and assign human-readable regime labels to clusters.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe (e.g. output from FeatureEngine).
        """
        logger.info(f"Fitting RegimeClassifier on {len(df)} rows...")
        df = self._prepare(df)
        X = self._get_feature_matrix(df)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = self.model.fit_predict(X_scaled)

        self.cluster_regime_map = self._map_clusters_to_regimes(df, labels)
        self.regime_profiles = self._build_profiles(df, labels)

        logger.info("RegimeClassifier fit complete.")
        for profile in self.regime_profiles:
            logger.info(
                f"  {profile.regime:12s} | bars={profile.n_bars:6d} "
                f"| pct={profile.pct_time:.2%} | trend={profile.avg_trend_score:+.4f} "
                f"| vol={profile.avg_volatility:.4f}"
            )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append a ``regime`` column to ``df`` using the fitted model.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with added ``regime`` and ``cluster`` columns.
        """
        if self.model is None or self.scaler is None or self.cluster_regime_map is None:
            raise RuntimeError("RegimeClassifier must be fit before transform().")

        df = self._prepare(df)
        X = self._get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)

        clusters = self.model.predict(X_scaled)
        out = df.copy()
        out["cluster"] = clusters
        out["regime"] = [self.cluster_regime_map[c].value for c in clusters]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(df)
        return self.transform(df)

    def predict_latest(self, df: pd.DataFrame) -> Regime:
        """
        Return the regime for the most recent row.

        Useful for online/live decision making.
        """
        out = self.transform(df)
        return Regime(out["regime"].iloc[-1])

    def get_regime_profiles(self) -> List[RegimeProfile]:
        """Return summary profile for each regime."""
        if self.regime_profiles is None:
            raise RuntimeError("RegimeClassifier has not been fit yet.")
        return self.regime_profiles

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Union[str, Path]) -> None:
        """
        Persist fitted artifacts to disk.

        Stores scaler, KMeans model, cluster->regime mapping and feature list.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "scaler": self.scaler,
            "model": self.model,
            "cluster_regime_map": self.cluster_regime_map,
            "feature_cols": self.feature_cols,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
        }
        with open(path / "regime_artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f)

        # Also store a human-readable JSON summary of the mapping.
        summary = {
            "feature_cols": self.feature_cols,
            "n_clusters": self.n_clusters,
            "cluster_regime_map": {
                str(k): v.value for k, v in (self.cluster_regime_map or {}).items()
            },
            "profiles": [
                {
                    "regime": p.regime,
                    "n_bars": p.n_bars,
                    "pct_time": p.pct_time,
                    "avg_trend_score": p.avg_trend_score,
                    "avg_volatility": p.avg_volatility,
                    "avg_return_24h": p.avg_return_24h,
                    "avg_rsi": p.avg_rsi,
                    "avg_funding_zscore": p.avg_funding_zscore,
                    "avg_volume_zscore": p.avg_volume_zscore,
                }
                for p in (self.regime_profiles or [])
            ],
        }
        with open(path / "regime_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"RegimeClassifier artifacts saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> RegimeClassifier:
        """Load a previously saved RegimeClassifier."""
        path = Path(path)
        with open(path / "regime_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)

        instance = cls(
            feature_cols=artifacts["feature_cols"],
            n_clusters=artifacts["n_clusters"],
            random_state=artifacts["random_state"],
        )
        instance.scaler = artifacts["scaler"]
        instance.model = artifacts["model"]
        instance.cluster_regime_map = artifacts["cluster_regime_map"]
        return instance

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required columns exist.

        Computes ``realized_vol_24h`` if missing. Drops rows with NaNs in
        the modeling features.
        """
        df = df.copy()

        if "realized_vol_24h" not in df.columns:
            df["realized_vol_24h"] = df["close"].pct_change().rolling(24).std()

        # Ensure every feature we need is present.
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns required by RegimeClassifier: {missing}")

        # Drop rows with NaNs in modeling features; keep other columns intact.
        return df.dropna(subset=self.feature_cols)

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Return the standardized feature matrix."""
        return df[self.feature_cols].astype(float).to_numpy()

    def _map_clusters_to_regimes(self, df: pd.DataFrame, labels: np.ndarray) -> dict[int, Regime]:
        """
        Assign each K-Means cluster to one of the five named regimes.

        The heuristic ranks clusters by trend, volatility and momentum and then
        assigns labels deterministically.
        """
        df = df.copy()
        df["cluster"] = labels

        cluster_stats: dict[int, dict[str, float]] = {}
        for c in np.unique(labels):
            sub = df[df["cluster"] == c]
            trend_score = self._trend_score(sub)
            cluster_stats[c] = {
                "trend_score": trend_score,
                "volatility": sub["realized_vol_24h"].mean(),
                "rsi": sub["rsi_14"].mean(),
                "return_24h": sub["return_24h"].mean(),
                "volume_zscore": sub["volume_zscore"].mean(),
            }

        # Sort clusters by different criteria.
        by_trend = sorted(cluster_stats.items(), key=lambda x: x[1]["trend_score"], reverse=True)
        by_vol = sorted(cluster_stats.items(), key=lambda x: x[1]["volatility"], reverse=True)

        assigned: set[int] = set()
        mapping: dict[int, Regime] = {}

        # 1. Highest volatility cluster -> HIGH_VOL.
        high_vol_cluster = by_vol[0][0]
        mapping[high_vol_cluster] = Regime.HIGH_VOL
        assigned.add(high_vol_cluster)

        # 2. Among remaining clusters, highest trend -> STRONG_BULL, lowest -> BEAR,
        #    middle -> BULL / RANGE.
        remaining = [item for item in by_trend if item[0] not in assigned]

        # If we have exactly 4 remaining clusters (the normal 5-cluster case):
        if len(remaining) == 4:
            mapping[remaining[0][0]] = Regime.STRONG_BULL
            mapping[remaining[1][0]] = Regime.BULL
            mapping[remaining[2][0]] = Regime.RANGE
            mapping[remaining[3][0]] = Regime.BEAR
        else:
            # Fallback for any other n_clusters configuration: label by trend rank.
            labels_by_trend = [Regime.STRONG_BULL, Regime.BULL, Regime.RANGE, Regime.BEAR]
            for i, (c, _) in enumerate(remaining):
                mapping[c] = labels_by_trend[min(i, len(labels_by_trend) - 1)]

        return mapping

    def _build_profiles(self, df: pd.DataFrame, labels: np.ndarray) -> List[RegimeProfile]:
        """Build RegimeProfile objects for each named regime."""
        df = df.copy()
        df["cluster"] = labels
        df["regime"] = [self.cluster_regime_map[c].value for c in labels]

        profiles: List[RegimeProfile] = []
        total = len(df)
        for regime in Regime:
            sub = df[df["regime"] == regime.value]
            if sub.empty:
                profiles.append(
                    RegimeProfile(
                        regime=regime.value,
                        n_bars=0,
                        pct_time=0.0,
                        avg_trend_score=0.0,
                        avg_volatility=0.0,
                        avg_return_24h=0.0,
                        avg_rsi=0.0,
                        avg_funding_zscore=0.0,
                        avg_volume_zscore=0.0,
                    )
                )
                continue

            profiles.append(
                RegimeProfile(
                    regime=regime.value,
                    n_bars=len(sub),
                    pct_time=len(sub) / total if total else 0.0,
                    avg_trend_score=self._trend_score(sub),
                    avg_volatility=sub["realized_vol_24h"].mean(),
                    avg_return_24h=sub["return_24h"].mean(),
                    avg_rsi=sub["rsi_14"].mean(),
                    avg_funding_zscore=sub["funding_zscore"].mean()
                    if "funding_zscore" in sub.columns
                    else 0.0,
                    avg_volume_zscore=sub["volume_zscore"].mean(),
                )
            )

        return profiles

    @staticmethod
    def _trend_score(df: pd.DataFrame) -> float:
        """
        Composite trend score for a group of bars.

        Combines EMA distances, recent return and RSI momentum. Positive = bullish.
        """
        ema_9 = df["ema_9_dist"].mean() if "ema_9_dist" in df.columns else 0.0
        ema_50 = df["ema_50_dist"].mean() if "ema_50_dist" in df.columns else 0.0
        ema_200 = df["ema_200_dist"].mean() if "ema_200_dist" in df.columns else 0.0
        ret_24h = df["return_24h"].mean()
        rsi = df["rsi_14"].mean() - 50.0  # center around 0
        macd = df["macd_hist"].mean() if "macd_hist" in df.columns else 0.0

        # Weights chosen so that price structure dominates momentum.
        return float(
            0.25 * ema_9
            + 0.30 * ema_50
            + 0.20 * ema_200
            + 0.15 * ret_24h
            + 0.05 * rsi / 100.0
            + 0.05 * macd
        )


# ---------------------------------------------------------------------- #
# Convenience function used by main.py / notebooks
# ---------------------------------------------------------------------- #

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any regime-specific derived features that are not in FeatureEngine.

    Currently this only adds ``realized_vol_24h`` if missing.
    """
    out = df.copy()
    if "realized_vol_24h" not in out.columns:
        out["realized_vol_24h"] = out["close"].pct_change().rolling(24).std()
    return out
