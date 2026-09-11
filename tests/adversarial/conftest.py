"""
tests/adversarial/conftest.py — Shared fixtures for adversarial test suite.

Loads the REAL strategy class from
ft_userdata/user_data/strategies/GTQuantMultiTF.py so tests exercise actual
strategy code, not local reimplementations.

The local venv has no `freqtrade`/`technical` (backtests run in the FreqAI
Docker image), so minimal import stubs are installed ONLY when the real
packages are unavailable. The stubs provide just enough surface
(IStrategy / DecimalParameter / informative / qtpylib) to import the module
and call its pure-dataframe methods; all tests stay hermetic — no network,
no database, no docker.

Provides:
    strategy  — GTQuantMultiTF instance (fresh per test, mock dp injected)
    config    — dict config with dry_run=true, empty keys
    mock_dp   — DataHandler that returns injected dataframes
"""
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = REPO_ROOT / "ft_userdata" / "user_data" / "strategies" / "GTQuantMultiTF.py"

PAIR = "BTC/USDT:USDT"


# --------------------------------------------------------------------------- #
# Import stubs for freqtrade / technical (only when the real ones are absent)
# --------------------------------------------------------------------------- #
def _install_freqtrade_stubs():
    try:
        import freqtrade.strategy  # noqa: F401
        import technical  # noqa: F401
        return  # real packages present (e.g. inside the FreqAI docker image)
    except ModuleNotFoundError:
        pass

    class _DecimalParameter:
        """Stand-in exposing `.value` = the declared default."""

        def __init__(self, low, high, default=None, decimals=8, space="",
                     optimize=True, load=True):
            self.low, self.high = low, high
            self.value = default if default is not None else low

    class _IStrategy:
        INTERFACE_VERSION = 3

        def __init__(self, *args, **kwargs):
            pass

    def _informative(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

    ft_strategy = types.ModuleType("freqtrade.strategy")
    ft_strategy.IStrategy = _IStrategy
    ft_strategy.DecimalParameter = _DecimalParameter
    ft_strategy.informative = _informative
    ft_pkg = types.ModuleType("freqtrade")
    ft_pkg.strategy = ft_strategy

    qtpylib = types.ModuleType("technical.qtpylib")
    technical = types.ModuleType("technical")
    technical.qtpylib = qtpylib

    sys.modules.setdefault("freqtrade", ft_pkg)
    sys.modules.setdefault("freqtrade.strategy", ft_strategy)
    sys.modules.setdefault("technical", technical)
    sys.modules.setdefault("technical.qtpylib", qtpylib)


_install_freqtrade_stubs()


def load_strategy_class():
    """Import the real GTQuantMultiTF class by path (fresh module object)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("GTQuantMultiTF", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["GTQuantMultiTF"] = mod
    spec.loader.exec_module(mod)
    return mod.GTQuantMultiTF


# --------------------------------------------------------------------------- #
# Helper: synthetic OHLCV dataframe
# --------------------------------------------------------------------------- #
def make_ohlcv(n_bars: int = 100, start_ts: int = 1_700_000_000,
               tf_minutes: int = 5, spread_pct: float = 0.0002,
               seed: int = 42) -> pd.DataFrame:
    """Return a clean OHLCV DataFrame with all columns the strategy needs."""
    rng = np.random.default_rng(seed)
    n = n_bars
    dates = pd.date_range(start=pd.Timestamp(start_ts, unit="s", tz="UTC"),
                          periods=n, freq=f"{tf_minutes}min")
    close = 50_000 + np.cumsum(rng.standard_normal(n) * 50)
    high = close + np.abs(rng.standard_normal(n) * 30)
    low = close - np.abs(rng.standard_normal(n) * 30)
    open_ = low + rng.random(n) * (high - low)
    vol = rng.random(n) * 1000 + 500
    df = pd.DataFrame({
        "date":     dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": vol,
    })
    # TimescaleDB columns (for micro filter)
    df["spread"] = spread_pct
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
        self._data = injected_data or {}
        self.whitelist = [PAIR, "ETH/USDT:USDT"]
        self.runmode = "live"

    def get_analyzed_dataframe(self, pair, timeframe):
        key = (pair, timeframe)
        df = self._data.get(key)
        if df is None:
            df = make_ohlcv()
        return df, pd.Timestamp.now(tz="UTC")

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
    """Real GTQuantMultiTF instance with a mock dp injected."""
    cls = load_strategy_class()
    try:
        s = cls()          # stubbed IStrategy: no config needed
    except TypeError:
        s = cls(config)    # real freqtrade IStrategy requires config
    s.dp = MockDataProvider()
    return s


@pytest.fixture
def injected_5m(strategy) -> pd.DataFrame:
    """5m base dataframe with all strategy columns populated."""
    df = make_ohlcv(n_bars=300, start_ts=1_700_000_000)
    df["ema_9"] = df["close"].ewm(span=9).mean()
    df["ema_21"] = df["close"].ewm(span=21).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    import talib.abstract as ta
    df["rsi_14"] = ta.RSI(df, timeperiod=14)
    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd_hist"] = macd["macdhist"]
    df["atr_14"] = ta.ATR(df, timeperiod=14)
    df["bb_lower"] = df["close"] * 0.98
    df["bb_upper"] = df["close"] * 1.02
    df["bb_pct"] = 0.5
    df["volume_zscore"] = 0.0
    # Regime columns (informative)
    df["ema_50_slope_1h"] = 0.001
    df["adx_1h"] = 30.0
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
