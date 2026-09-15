"""
1m microstructure -> 5m FreqAI feature plumbing (Day 14).

Pure-pandas helpers (no freqtrade imports) so the host test-suite can import
this module directly. Used by GTQuantMultiTFMicro.

Data source: parquet files written by collectors/historical_micro.py into
<user_data>/data/micro/{PAIR}-1m-micro.parquet, plus (live/dry_run only) a
tail read from TimescaleDB ohlcv_1m for bars newer than the parquet.

LOOKAHEAD CONTRACT
------------------
Freqtrade timestamps candles by OPEN time. The micro aggregates attached to
the 5m row with open time `t` are computed from 1m bars in [t-5m, t) — i.e.
only 1m bars whose CLOSE time is <= t. The features are therefore fully known
at the open of the 5m bar, strictly before any decision made on that bar.
This is deliberately one 5m bucket more conservative than the candle's own
OHLCV (which is known only at the bar's close). See
tests/test_micro_features.py for the leakage property tests.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PAIR_TO_SYMBOL = {"BTC/USDT:USDT": "BTCUSDT", "ETH/USDT:USDT": "ETHUSDT"}

DB_DSN = ("dbname=gtquant user=gtquant password=gtquant_local "
          "host=host.docker.internal port=5432")

# Raw per-bucket aggregates produced by aggregate_to_5m.
BUCKET_COLS = [
    "m_volume", "m_vol_delta", "m_trade_count",
    "m_tick_momentum", "m_rv", "m_bars",
]

# Derived stationary features exposed to FreqAI (%- prefixed by the caller).
MICRO_FEATURE_COLS = [
    "micro_vol_delta_ratio",     # order-flow imbalance: Σdelta / Σvol
    "micro_tick_mom_ratio",      # uptick-downtick balance per trade
    "micro_rv_rel",              # realized variance vs trailing 1h mean
    "micro_trade_intensity",     # trade count vs trailing 1h mean
    "micro_trade_size_rel",      # avg trade size vs trailing 1h mean
    "micro_ofi_1h",              # 1h cumulative delta / 1h volume
]

_parquet_cache: dict = {}


def _parquet_path(user_data_dir: Path, pair: str) -> Path:
    fname = pair.replace("/", "_").replace(":", "_") + "-1m-micro.parquet"
    return Path(user_data_dir) / "data" / "micro" / fname


def _read_parquet_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    hit = _parquet_cache.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    df = pd.read_parquet(path)
    _parquet_cache[str(path)] = (mtime, df)
    return df


def _db_tail(symbol: str, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """Recent 1m bars from TimescaleDB (live/dry_run tail beyond the parquet)."""
    import psycopg2

    conn = psycopg2.connect(DB_DSN, connect_timeout=3)
    try:
        q = """
            SELECT time, open, high, low, close, volume,
                   volume_delta, tick_momentum, realized_variance, trade_count
            FROM ohlcv_1m
            WHERE symbol = %s AND time > %s AND time <= %s
            ORDER BY time
        """
        df = pd.read_sql(q, conn, params=(symbol, since, until))
    finally:
        conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["time"], utc=True)
    return df


def load_micro_1m(pair: str, start: pd.Timestamp, end: pd.Timestamp,
                  user_data_dir: Path, runmode: str = "backtest") -> pd.DataFrame:
    """
    Load 1m micro bars for `pair` covering [start, end]. Backtest/hyperopt:
    parquet only (reproducible, no DB coupling). Live/dry_run: parquet plus
    the DB tail for bars newer than the parquet snapshot.
    """
    symbol = PAIR_TO_SYMBOL.get(pair)
    if symbol is None:
        return pd.DataFrame()

    frames = []
    df = _read_parquet_cached(_parquet_path(user_data_dir, pair))
    if df is not None and not df.empty:
        frames.append(df)

    if runmode in ("live", "dry_run"):
        try:
            parquet_end = df["date"].max() if df is not None and not df.empty \
                else start - pd.Timedelta(days=1)
            tail = _db_tail(symbol, max(start, parquet_end), end)
            if not tail.empty:
                frames.append(tail)
        except Exception as e:
            logger.warning(f"micro DB tail unavailable ({e}); parquet only")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return out[(out["date"] >= start) & (out["date"] <= end)].reset_index(drop=True)


def aggregate_to_5m(micro: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1m micro bars into 5m buckets. The bucket is keyed by the OPEN
    time of the FOLLOWING 5m bar (bucket_end = floor5(date) + 5min), so that
    merging on dataframe['date'] == bucket_end implements the lookahead
    contract (only 1m bars closing at or before the 5m row's open time).
    """
    if micro.empty:
        return pd.DataFrame(columns=["date"] + BUCKET_COLS)

    d = micro.copy()
    bucket_end = d["date"].dt.floor("5min") + pd.Timedelta(minutes=5)
    d = d.assign(bucket_end=bucket_end)

    grp = d.groupby("bucket_end", sort=True)
    out = pd.DataFrame({
        "m_volume": grp["volume"].sum(),
        "m_vol_delta": grp["volume_delta"].sum(),
        "m_trade_count": grp["trade_count"].sum(),
        "m_tick_momentum": grp["tick_momentum"].sum(),
        "m_rv": grp["realized_variance"].sum(),
        "m_bars": grp["date"].count(),
    }).reset_index().rename(columns={"bucket_end": "date"})
    return out


def _rolling_rel(series: pd.Series, window: int) -> pd.Series:
    base = series.rolling(window, min_periods=3).mean()
    return series / base


def compute_micro_features(df5m: pd.DataFrame) -> pd.DataFrame:
    """
    Derive stationary micro features from the merged m_* bucket columns.
    All rolling windows look strictly backward; input columns already satisfy
    the lookahead contract. Rows without micro coverage become 0 (LightGBM
    treats them as a distinct, learnable value).
    """
    vol = df5m["m_volume"].where(df5m["m_volume"] > 0)
    tc = df5m["m_trade_count"].where(df5m["m_trade_count"] > 0)

    df5m["micro_vol_delta_ratio"] = df5m["m_vol_delta"] / vol
    df5m["micro_tick_mom_ratio"] = df5m["m_tick_momentum"] / tc
    df5m["micro_rv_rel"] = _rolling_rel(df5m["m_rv"], 12)
    df5m["micro_trade_intensity"] = _rolling_rel(df5m["m_trade_count"], 12)
    df5m["micro_trade_size_rel"] = _rolling_rel(df5m["m_volume"] / tc, 12)
    df5m["micro_ofi_1h"] = (
        df5m["m_vol_delta"].rolling(12, min_periods=3).sum()
        / df5m["m_volume"].rolling(12, min_periods=3).sum()
    )
    df5m[MICRO_FEATURE_COLS] = df5m[MICRO_FEATURE_COLS].fillna(0.0)
    return df5m


def attach_micro_features(df5m: pd.DataFrame, pair: str,
                          user_data_dir: Path,
                          runmode: str = "backtest") -> pd.DataFrame:
    """
    Entry point called from populate_indicators: merge 5m-bucketed micro
    aggregates onto the 5m dataframe and derive the micro_* feature columns.
    Fails open (returns the dataframe with zero-filled features) when no
    micro data exists for the pair/window.
    """
    if df5m.empty:
        return df5m
    start = df5m["date"].min()
    end = df5m["date"].max()
    micro = load_micro_1m(pair, start - pd.Timedelta(minutes=5),
                          end, user_data_dir, runmode)
    buckets = aggregate_to_5m(micro)

    df5m = df5m.merge(buckets, on="date", how="left", sort=False)
    for col in BUCKET_COLS:
        df5m[col] = df5m[col].fillna(0.0)
    return compute_micro_features(df5m)
