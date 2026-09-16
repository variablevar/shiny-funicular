"""
regime_features.py — Shared 1h feature builder for the KMeans regime gate.

Single source of truth used by BOTH:
  - analysis/train_regime_model.py  (offline fit on feather history)
  - GTQuantRegimeGated.py           (in-strategy, on the 1h informative frame)

Keeping the math in one module guarantees the live/backtest feature frame is
bit-identical to what the scaler + KMeans artifact was fit on.

All features are causal (computed from the current and past bars only) and
scale-invariant (distances/ratios, MACD normalized by close) so one artifact
serves BTC and ETH alike.

``funding_zscore`` from core/regime_classifier.DEFAULT_FEATURES is deliberately
dropped: funding data is not available inside Freqtrade informative frames in
backtests, and a feature that exists at fit time but not at runtime is a
lookahead/consistency hazard. The classifier is fit with FEATURE_COLS below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Feature list handed to RegimeClassifier(feature_cols=...) at fit time and
# re-built here at runtime. Order matters: the scaler/KMeans expect it.
FEATURE_COLS = [
    "return_24h",
    "ema_9_dist",
    "ema_50_dist",
    "ema_200_dist",
    "rsi_14",
    "macd_hist",
    "realized_vol_24h",
    "volume_zscore",
]

# First ~24 rows are NaN (24-bar rolling windows); EMA200 takes ~200 bars to
# fully stabilize. Treat any NaN-feature bar as UNKNOWN regime.
WARMUP_BARS = 200


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, pure pandas (matches ta.RSI semantics closely enough for
    clustering; consistency between fit and runtime is what matters)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> RSI 100 (pure up-move streak)
    return rsi.where(avg_loss != 0, 100.0)


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the KMeans feature frame from a 1h OHLCV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``close`` and ``volume`` columns, indexed or
        ordered by time ascending.

    Returns
    -------
    pd.DataFrame
        Frame aligned to ``df.index`` with FEATURE_COLS columns. The first
        ~WARMUP_BARS rows contain NaNs (EMA/rolling warmup) — callers must
        treat those bars as UNKNOWN regime, never backfill them.
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    ret_1h = close.pct_change()

    out = pd.DataFrame(index=df.index)
    out["return_24h"] = close.pct_change(24)

    for span in (9, 50, 200):
        ema = close.ewm(span=span, adjust=False).mean()
        out[f"ema_{span}_dist"] = (close - ema) / ema

    out["rsi_14"] = _rsi_wilder(close, 14)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    # Normalized by close: raw MACD scale differs 30x between BTC and ETH,
    # which would make the clusters pair-dominated instead of regime-dominated.
    out["macd_hist"] = (macd - signal) / close

    out["realized_vol_24h"] = ret_1h.rolling(24).std()

    vol_mean = volume.rolling(24).mean()
    vol_std = volume.rolling(24).std()
    out["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)

    return out[FEATURE_COLS]
