"""
tests/adversarial/conftest.py — Shared fixtures for adversarial test suite.

Provides:
    strategy  — GTQuantMultiTF instance (fresh per test, no state)
    config    — dict config with dry_run=true, empty keys
    mock_dp   — DataHandler that returns injected dataframes
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock


# --------------------------------------------------------------------------- #
# Helper: synthetic OHLCV dataframe
# --------------------------------------------------------------------------- #
def make_ohlcv(n_bars: int = 100, start_ts: int = 1_700_000_000,
               tf_minutes: int = 5, spread_pct: float = 0.0002) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame with all columns the strategy needs."""
    n = n_bars
    dates = pd.date_range(start=pd.Timestamp(start_ts, unit="s", tz="UTC"),
                          periods=n, freq=f"{tf_minutes}T")
    close = 50_000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low  = close - np.abs(np.random.randn(n) * 30)
    open_ = low + np.random.rand(n) * (high - low)
    vol   = np.random.rand(n) * 1000 + 500
    df = pd.DataFrame({
        "date":     dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": vol,
    })
    # TimescaleDB columns (for micro filter)
    df["spread"]       = spread_pct
    df["volume_delta"] = 0.0
    return df


# --------------------------------------------------------------------------- #
# Mock DataProvider
# --------------------------------------------------------------------------- #
class MockDataProvider:
    """
    Stand-in for freqtrade.strategy.IStrategy.dp.
    Returns either a clean or an injected dataframe per pair/timeframe.
    """
    def __init__(self, injected_data: dict | None = None):
        # { (pair, timeframe): DataFrame }
        self._data   = injected_data or {}
        self.whitelist = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        self.runmode  = "live"

    def get_analyzed_dataframe(self, pair, timeframe):
        key = (pair, timeframe)
        df  = self._data.get(key)
        if df is None:
            df = make_ohlcv()
        return df, pd.Timestamp.utcnow()

    def current_whitelist(self):
        return self.whitelist

    def get_pair_dataframe(self, pair, timeframe):
        key = (pair, timeframe)
        return self._data.get(key) or make_ohlcv()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def config():
    """Dry-run config matching ft_userdata/config.json."""
    return {
        "dry_run": True,
        "dry_run_wallet": 1000,
        "max_open_trades": 3,
        "stake_amount": 100,
        "timeframe": "5m",
        "exchange": {"name": "binance", "key": "", "secret": ""},
    }


@pytest.fixture
def strategy(config):
    """GTQuantMultiTF with a mock dp injected."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "GTQuantMultiTF",
        "/home/gt/Desktop/gt-quant/ft_userdata/user_data/strategies/GTQuantMultiTF.py",
    )
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["GTQuantMultiTF"] = mod
    spec.loader.exec_module(mod)
    S = mod.GTQuantMultiTF()
    S.dp = MockDataProvider()
    return S


@pytest.fixture
def injected_5m(strategy) -> pd.DataFrame:
    """5m base dataframe with all strategy columns populated."""
    df = make_ohlcv(n_bars=300, start_ts=1_700_000_000)
    df["ema_9"]  = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    import talib.abstract as ta
    df["rsi_14"]  = ta.RSI(df, timeperiod=14)
    macd          = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd_hist"] = macd["macdhist"]
    df["atr_14"]  = ta.ATR(df, timeperiod=14)
    df["bb_lower"] = df["close"] * 0.98
    df["bb_upper"] = df["close"] * 1.02
    df["bb_pct"]   = 0.5
    df["volume_zscore"] = 0.0
    # Regime columns (informative)
    df["ema_50_slope_1h"]  = 0.001
    df["adx_1h"]           = 30.0
    df["ema_50_200_dist_30m"] = 0.01
    df["bull_structure_4h"] = 1
    # FreqAI columns
    df["do_predict"] = 1
    df["&-s-future_return_5m"] = 0.001
    df["regime"] = "BULL"
    return df


@pytest.fixture
def injected_1m_high_spread() -> pd.DataFrame:
    """TimescaleDB 1m rows with wide spread and negative volume delta."""
    rows = []
    for i in range(60):
        rows.append({
            "symbol":       "BTCUSDT",
            "spread":       0.001,   # 0.1% — exceeds micro_max_spread (0.0005)
            "volume_delta": -500,    # negative — would trigger vol-delta rejection
        })
    return pd.DataFrame(rows)
