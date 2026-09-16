"""
GTQuantRegimeGated — GTQuantMultiTF + KMeans regime gate (v0.2 Phase 1,
docs/v0.2-strategy.md §Layer 1).

Wires the previously-unused KMeans regime classifier (core/regime_classifier.py,
artifact in user_data/models/regime/) into the entry path as a pre-ML trade
gate: when the 1h regime maps to VOLATILE or QUIET, no new entries are taken
— the FreqAI prediction is not consulted for direction. Target: ~50% fewer
trades at same-or-better net P&L.

Design decisions
----------------
* **Recompute in-strategy, not offline parquet.** The regime is computed on
  the 1h informative dataframe via strategies/regime_features.py — the same
  pure-pandas builder used to fit the artifact (analysis/train_regime_model.py).
  One code path serves backtest and live identically; a precomputed parquet
  would need a separate live path and its own merge/alignment audit.

* **No lookahead by construction.** The regime lives on the 1h informative
  frame, which Freqtrade merges onto the 5m frame shifted one informative
  candle forward: the value on 5m bar t comes from the 1h bar that closed at
  or before t. All features in regime_features.py are causal on top of that.

* **Fallback.** If the artifact is missing, unreadable, or older than
  ``regime_max_age_days``, the gate is disabled and entries fall back to the
  parent's rule-based regime gate (BLOCKED_REGIMES = RANGE/HIGH_VOL). Bars
  whose features are still warming up (~first 200 1h bars) get action UNKNOWN
  and are blocked — fail closed, never fail open.

* The parent's rule-based ``regime`` column is left untouched for position
  sizing, audit logging and the LLM shadow context.
"""
import json
import pickle
import sys
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

import pandas as pd
from pandas import DataFrame

_STRAT_DIR = str(Path(__file__).resolve().parent)
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

import regime_features as rf  # noqa: E402
from GTQuantMultiTF import GTQuantMultiTF  # noqa: E402

from freqtrade.strategy import informative  # noqa: E402

import logging
logger = logging.getLogger(__name__)


class GTQuantRegimeGated(GTQuantMultiTF):
    """GTQuantMultiTF with the KMeans regime gate replacing the entry filter."""

    # v0.2 §Layer 1: Volatile/Quiet -> no new trades.
    KMEANS_BLOCKED_ACTIONS = ("VOLATILE", "QUIET")
    KMEANS_TRADABLE_ACTIONS = ("TRENDING", "MEAN_REVERTING")

    use_kmeans_gate = True

    # Artifact location relative to user_data_dir; absolute path overrides
    # (tests point this at a tmp_path fixture).
    regime_artifact_dir: str | None = None

    # Artifact older than this is treated as stale -> rule-based fallback.
    regime_max_age_days = 45

    _kmeans_cache: dict | None = None
    _kmeans_failed: bool = False

    # ------------------------------------------------------------------ #
    # Artifact loading
    # ------------------------------------------------------------------ #

    def _artifact_dir(self) -> Path:
        if self.regime_artifact_dir:
            return Path(self.regime_artifact_dir)
        user_data = Path(getattr(self, "config", {}).get("user_data_dir", "user_data"))
        return user_data / "models" / "regime"

    def _load_kmeans(self) -> dict | None:
        """
        Load scaler + KMeans + cluster->action map once per process.

        Returns None (and disables the gate) when the artifact is missing,
        unpickleable, or older than regime_max_age_days.
        """
        if self._kmeans_cache is not None:
            return self._kmeans_cache
        if self._kmeans_failed:
            return None
        try:
            base = self._artifact_dir()
            summary = json.loads((base / "regime_summary.json").read_text())
            trained_at = datetime.fromisoformat(str(summary["trained_at"]))
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - trained_at).days
            if age_days > self.regime_max_age_days:
                logger.warning(
                    f"KMeans regime artifact is {age_days}d old "
                    f"(> {self.regime_max_age_days}d) — falling back to rule-based gate")
                self._kmeans_failed = True
                return None
            with open(base / "regime_runtime.pkl", "rb") as f:
                art = pickle.load(f)
            self._kmeans_cache = {
                "scaler": art["scaler"],
                "model": art["model"],
                "cluster_action_map": {
                    int(k): v for k, v in summary["cluster_action_map"].items()
                },
                # JSON string map (cluster_regime_map in regime_artifacts.pkl
                # holds core.* enums that don't exist inside the container).
                "cluster_regime_map": {
                    int(k): v for k, v in summary["cluster_regime_map"].items()
                },
            }
            logger.info(
                f"KMeans regime gate armed (artifact {age_days}d old, "
                f"map={self._kmeans_cache['cluster_action_map']})")
            return self._kmeans_cache
        except Exception as e:
            logger.warning(f"KMeans regime artifact unavailable ({e}) — "
                           "falling back to rule-based regime gate")
            self._kmeans_failed = True
            return None

    # ------------------------------------------------------------------ #
    # 1h informative: rule-based features + KMeans regime
    # ------------------------------------------------------------------ #

    @informative("1h")
    def populate_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_informative_1h(dataframe, metadata)
        kmeans = self._load_kmeans() if self.use_kmeans_gate else None
        if kmeans is None:
            return dataframe  # columns absent -> populate_entry_trend falls back

        feats = rf.build_regime_features(dataframe)
        valid = feats.notna().all(axis=1)
        action = pd.Series("UNKNOWN", index=dataframe.index, dtype=object)
        regime_name = pd.Series("UNKNOWN", index=dataframe.index, dtype=object)
        if valid.any():
            X = kmeans["scaler"].transform(
                feats.loc[valid, rf.FEATURE_COLS].astype(float).to_numpy())
            clusters = kmeans["model"].predict(X)
            action.loc[valid] = [
                kmeans["cluster_action_map"].get(int(c), "UNKNOWN") for c in clusters]
            regime_name.loc[valid] = [
                kmeans["cluster_regime_map"].get(int(c), "UNKNOWN") for c in clusters]
        dataframe["regime_action"] = action
        dataframe["regime_kmeans"] = regime_name
        return dataframe

    # ------------------------------------------------------------------ #
    # Entry gate
    # ------------------------------------------------------------------ #

    def _kmeans_gate_active(self, dataframe: DataFrame) -> bool:
        return (
            self.use_kmeans_gate
            and self._load_kmeans() is not None
            and "regime_action_1h" in dataframe.columns
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Pre-ML gate: when the KMeans action on the latest closed 1h bar is
        VOLATILE or QUIET, no entries at all — HOLD before consulting FreqAI.
        TRENDING / MEAN_REVERTING allow both directions (direction still comes
        from the model + 1h slope). UNKNOWN (warmup) is blocked.
        """
        if not self._kmeans_gate_active(dataframe):
            return super().populate_entry_trend(dataframe, metadata)

        prediction = self.prediction_col
        if prediction not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        tradable = dataframe["regime_action_1h"].isin(self.KMEANS_TRADABLE_ACTIONS)

        long_cond = [
            dataframe["do_predict"] == 1,
            dataframe[prediction] > self.entry_threshold.value,
            dataframe["ema_50_slope_1h"] > 0,
            dataframe["volume_zscore"] > -1.0,
            tradable,
        ]
        short_cond = [
            dataframe["do_predict"] == 1,
            dataframe[prediction] < -self.entry_threshold.value,
            dataframe["ema_50_slope_1h"] < 0,
            dataframe["volume_zscore"] > -1.0,
            tradable,
        ]

        dataframe.loc[reduce(lambda x, y: x & y, long_cond),
                      ["enter_long", "enter_tag"]] = (1, "freqai_long_regime")
        dataframe.loc[reduce(lambda x, y: x & y, short_cond),
                      ["enter_short", "enter_tag"]] = (1, "freqai_short_regime")
        return dataframe
