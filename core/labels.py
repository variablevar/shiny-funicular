"""
Multi-horizon label generator (Day 2).

Builds forward-looking labels from an OHLCV dataframe with STRICT
no-lookahead semantics: the label at row ``i`` uses only rows ``> i``.
The final ``horizon`` rows of every label are NaN by construction.

Label set per plan Day 2:
    future_return_{h}       close-to-close return over h candles
    future_volatility_{h}   std of 1-candle returns over the next h candles
    max_adverse_excursion_{h}   worst drawdown vs entry over next h candles
    max_favorable_excursion_{h} best run-up vs entry over next h candles

These are used for offline analysis, dataset curation, and as targets for
secondary FreqAI identifiers (one label per identifier — see
docs/freqai-setup-notes.md gotcha #4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def future_return(close: pd.Series, horizon: int) -> pd.Series:
    """Close-to-close return ``horizon`` candles ahead."""
    return close.shift(-horizon) / close - 1.0


def future_volatility(close: pd.Series, horizon: int) -> pd.Series:
    """
    Realized volatility (std of 1-candle log returns) over the NEXT
    ``horizon`` candles. Computed backward-looking then shifted so row ``i``
    covers candles (i, i+horizon].
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(horizon).std().shift(-horizon)


def max_favorable_excursion(high: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    """Best run-up from current close over the next ``horizon`` candles."""
    future_max = high.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    return future_max / close - 1.0


def max_adverse_excursion(low: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    """Worst drawdown from current close over the next ``horizon`` candles."""
    future_min = low.iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1].shift(-1)
    return future_min / close - 1.0


def build_labels(df: pd.DataFrame, horizon: int, prefix: str = "") -> pd.DataFrame:
    """
    Append the full label set for one horizon to ``df`` (returns a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high``, ``low``, ``close``.
    horizon : int
        Candle lookahead.
    prefix : str
        Column name prefix, e.g. "5m_" -> "5m_future_return".
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    missing = {"high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = df.copy()
    out[f"{prefix}future_return"] = future_return(df["close"], horizon)
    out[f"{prefix}future_volatility"] = future_volatility(df["close"], horizon)
    out[f"{prefix}max_favorable_excursion"] = max_favorable_excursion(df["high"], df["close"], horizon)
    out[f"{prefix}max_adverse_excursion"] = max_adverse_excursion(df["low"], df["close"], horizon)
    return out


def build_labels_multi(df: pd.DataFrame, horizons: dict[str, int]) -> pd.DataFrame:
    """
    Append labels for several named horizons.

    horizons : {"1m": 1, "5m": 5, "1h": 12} -> columns like "1m_future_return".
    """
    out = df.copy()
    for name, h in horizons.items():
        out = build_labels(out, h, prefix=f"{name}_")
    return out
