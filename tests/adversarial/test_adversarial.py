"""
tests/adversarial/test_adversarial.py — Day 12 adversarial tests.

Every test here calls the REAL GTQuantMultiTF strategy code (loaded from
ft_userdata/user_data/strategies/GTQuantMultiTF.py via the conftest
`strategy` fixture). There are deliberately NO local reimplementations of
strategy logic — if the strategy changes, these tests follow it or fail.

Hermetic: no network, no database, no docker. psycopg2 / requests are
stubbed in sys.modules where the strategy would otherwise touch them.
"""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from conftest import PAIR, MockDataProvider, load_strategy_class

PRED_COL = "&-s-future_return_5m"  # must match GTQuantMultiTF.prediction_col
METADATA = {"pair": PAIR}


# --------------------------------------------------------------------------- #
# Dataframe builders (inputs only — all decisions come from the strategy)
# --------------------------------------------------------------------------- #
def entry_df(**kw) -> pd.DataFrame:
    """One-row 5m frame holding only the columns populate_entry_trend reads."""
    return pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01", tz="UTC")],
        "close": [kw.get("close", 50_000.0)],
        "regime": [kw.get("regime", "BULL")],
        "do_predict": [kw.get("do_predict", 1)],
        PRED_COL: [kw.get("pred", 0.0)],
        "ema_50_slope_1h": [kw.get("slope", 0.0)],
        "volume_zscore": [kw.get("vol_z", 0.0)],
    })


def regime_df(slope: float, adx: float, n: int = 60,
              close: np.ndarray | None = None) -> pd.DataFrame:
    """Frame with constant 1h informative columns for _compute_regime.

    n < 288 keeps the rv rolling-median NaN so HIGH_VOL never fires unless
    the caller engineers a vol spike via `close`.
    """
    if close is None:
        close = 50_000 + np.arange(n) * 10.0  # near-constant returns, no vol spike
    return pd.DataFrame({
        "close": close,
        "ema_50_slope_1h": slope,
        "adx_1h": adx,
    })


def strong_downtrend_df(n: int = 300) -> pd.DataFrame:
    """Synthetic 5m frame whose last bar the real classifier must label STRONG_BEAR.

    A small deterministic sinusoid keeps realized vol well-defined and stable
    so the HIGH_VOL override cannot fire on floating-point noise.
    """
    trend = np.exp(np.linspace(0, -0.05, n))            # smooth, steady decline
    wiggle = 1.0 + 0.001 * np.sin(np.arange(n) / 3.0)   # stable, bounded rv
    close = 60_000 * trend * wiggle
    df = regime_df(slope=-0.005, adx=35.0, n=n, close=close)
    df["do_predict"] = 1
    df[PRED_COL] = -0.005          # model predicts further downside
    df["volume_zscore"] = 0.0
    return df


def entered(df: pd.DataFrame, col: str) -> int:
    return int(df[col].fillna(0).iloc[-1]) if col in df.columns else 0


# --------------------------------------------------------------------------- #
# Real regime classification
# --------------------------------------------------------------------------- #
class TestRegimeClassification:
    """Thresholds are read off the strategy so a strategy-side retune is
    reflected here — the assertions pin the *decision structure*, not
    hardcoded copies of the constants."""

    def test_prediction_col_matches_fixture(self, strategy):
        assert strategy.prediction_col == PRED_COL

    def test_strong_bull_upgrade(self, strategy):
        r = strategy._compute_regime(regime_df(
            slope=strategy.regime_slope_strong * 3,
            adx=strategy.regime_adx_trend_min + 10))
        assert (r == "STRONG_BULL").all()

    def test_strong_bear_downgrade(self, strategy):
        r = strategy._compute_regime(regime_df(
            slope=-strategy.regime_slope_strong * 3,
            adx=strategy.regime_adx_trend_min + 10))
        assert (r == "STRONG_BEAR").all()

    def test_mild_positive_slope_is_bull_not_strong(self, strategy):
        r = strategy._compute_regime(regime_df(
            slope=strategy.regime_slope_strong / 2,
            adx=strategy.regime_adx_trend_min + 10))
        assert (r == "BULL").all()

    def test_mild_negative_slope_is_bear(self, strategy):
        r = strategy._compute_regime(regime_df(
            slope=-strategy.regime_slope_strong / 2,
            adx=strategy.regime_adx_trend_min + 10))
        assert (r == "BEAR").all()

    def test_very_low_adx_is_range(self, strategy):
        r = strategy._compute_regime(regime_df(
            slope=strategy.regime_slope_strong * 3,
            adx=strategy.regime_adx_range_max - 5))
        assert (r == "RANGE").all()

    def test_adx_between_range_and_trend_keeps_direction(self, strategy):
        """adx_range_max <= adx < adx_trend_min: not chop, but no STRONG upgrade."""
        adx = (strategy.regime_adx_range_max + strategy.regime_adx_trend_min) / 2
        r = strategy._compute_regime(
            regime_df(slope=strategy.regime_slope_strong * 3, adx=adx))
        assert (r == "BULL").all()  # directional, but never STRONG_*

    def test_high_vol_overrides_trend(self, strategy):
        n = 300
        close = 50_000 + np.arange(n) * 10.0
        close[-1] = close[-2] * 1.10  # +10% bar -> realized-vol spike
        r = strategy._compute_regime(regime_df(
            slope=strategy.regime_slope_strong * 3,
            adx=strategy.regime_adx_trend_min + 10, n=n, close=close))
        assert r.iloc[-1] == "HIGH_VOL"
        assert (r.iloc[:-13] != "HIGH_VOL").all()  # spike only affects its window

    def test_normal_vol_never_highvol(self, strategy):
        r = strategy._compute_regime(regime_df(slope=0.001, adx=30.0))
        assert (r != "HIGH_VOL").all()


# --------------------------------------------------------------------------- #
# Real entry gating per regime
# --------------------------------------------------------------------------- #
class TestRegimeGating:
    def test_strong_bull_allows_long(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BULL", pred=0.005, slope=0.005), METADATA)
        assert entered(df, "enter_long") == 1
        assert entered(df, "enter_short") == 0
        assert df["enter_tag"].iloc[-1] == "freqai_long"

    def test_range_blocks_all(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="RANGE", pred=0.005, slope=0.005), METADATA)
        assert entered(df, "enter_long") == 0
        df = strategy.populate_entry_trend(
            entry_df(regime="RANGE", pred=-0.005, slope=-0.005), METADATA)
        assert entered(df, "enter_short") == 0

    def test_highvol_blocks_all(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="HIGH_VOL", pred=0.005, slope=0.005), METADATA)
        assert entered(df, "enter_long") == 0
        df = strategy.populate_entry_trend(
            entry_df(regime="HIGH_VOL", pred=-0.005, slope=-0.005), METADATA)
        assert entered(df, "enter_short") == 0

    def test_long_blocked_in_bear(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="BEAR", pred=0.005, slope=-0.001), METADATA)
        assert entered(df, "enter_long") == 0

    def test_bear_allows_short(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="BEAR", pred=-0.005, slope=-0.001), METADATA)
        assert entered(df, "enter_short") == 1
        assert entered(df, "enter_long") == 0

    def test_short_blocked_in_bull(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="BULL", pred=-0.005, slope=0.001), METADATA)
        assert entered(df, "enter_short") == 0

    def test_blocked_regimes_constant(self, strategy):
        assert set(strategy.BLOCKED_REGIMES) == {"RANGE", "HIGH_VOL"}


class TestStrongBearRegression:
    """Regression for the Day-12 fix: `short_regime_ok` must include
    STRONG_BEAR (previously only BEAR, which blocked shorts exactly when
    the downtrend was strongest)."""

    def test_strong_bear_permits_short(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BEAR", pred=-0.005, slope=-0.005), METADATA)
        assert entered(df, "enter_short") == 1
        assert df["enter_tag"].iloc[-1] == "freqai_short"

    def test_strong_bear_still_blocks_long(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BEAR", pred=0.005, slope=0.005), METADATA)
        assert entered(df, "enter_long") == 0

    def test_end_to_end_strong_downtrend_shorts(self, strategy):
        """Full pipeline: real classifier -> STRONG_BEAR -> real entry logic
        must permit a short and no long on the final bar."""
        df = strong_downtrend_df()
        df["regime"] = strategy._compute_regime(df)
        assert df["regime"].iloc[-1] == "STRONG_BEAR"
        df = strategy.populate_entry_trend(df, METADATA)
        assert entered(df, "enter_short") == 1
        assert entered(df, "enter_long") == 0


class TestFreqAIFilter:
    def test_no_entry_without_prediction(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BULL", do_predict=0, pred=0.01, slope=0.005),
            METADATA)
        assert entered(df, "enter_long") == 0
        assert entered(df, "enter_short") == 0

    def test_weak_signal_below_threshold_blocked(self, strategy):
        weak = strategy.entry_threshold.value / 2
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BULL", pred=weak, slope=0.005), METADATA)
        assert entered(df, "enter_long") == 0

    def test_signal_just_above_threshold_enters(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BULL",
                     pred=strategy.entry_threshold.value * 2, slope=0.005),
            METADATA)
        assert entered(df, "enter_long") == 1

    def test_missing_prediction_columns_is_noop(self, strategy):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        out = strategy.populate_entry_trend(df, METADATA)
        assert "enter_long" not in out.columns and "enter_short" not in out.columns

    def test_dead_bar_blocked(self, strategy):
        df = strategy.populate_entry_trend(
            entry_df(regime="STRONG_BULL", pred=0.005, slope=0.005, vol_z=-2.0),
            METADATA)
        assert entered(df, "enter_long") == 0


class TestExitOnModelFlip:
    def test_prediction_below_exit_threshold_exits_long(self, strategy):
        df = strategy.populate_exit_trend(
            entry_df(pred=strategy.exit_threshold.value * 2), METADATA)
        assert entered(df, "exit_long") == 1
        assert df["exit_tag"].iloc[-1] == "model_flip_down"

    def test_prediction_above_minus_threshold_exits_short(self, strategy):
        df = strategy.populate_exit_trend(
            entry_df(pred=-strategy.exit_threshold.value * 2), METADATA)
        assert entered(df, "exit_short") == 1
        assert df["exit_tag"].iloc[-1] == "model_flip_up"

    def test_mid_range_prediction_no_exit(self, strategy):
        df = strategy.populate_exit_trend(entry_df(pred=0.0), METADATA)
        assert entered(df, "exit_long") == 0
        assert entered(df, "exit_short") == 0

    def test_no_exit_without_do_predict(self, strategy):
        df = strategy.populate_exit_trend(
            entry_df(do_predict=0, pred=-0.01), METADATA)
        assert entered(df, "exit_long") == 0


# --------------------------------------------------------------------------- #
# Real 1m microstructure filter (confirm_trade_entry)
# --------------------------------------------------------------------------- #
def confirm(strategy, pair=PAIR, side="long"):
    return strategy.confirm_trade_entry(
        pair=pair, order_type="limit", amount=0.01, rate=50_000.0,
        time_in_force="GTC", current_time=pd.Timestamp("2026-01-01", tz="UTC"),
        entry_tag=None, side=side)


class TestMicroFilter:
    @pytest.fixture(autouse=True)
    def _no_audit_io(self, strategy, monkeypatch):
        """Audit logging must never touch a real DB from tests."""
        self.audit = MagicMock()
        monkeypatch.setattr(strategy, "_audit", self.audit)

    def _set_micro(self, strategy, monkeypatch, value):
        monkeypatch.setattr(strategy, "_latest_1m_micro",
                            lambda pair, lookback_bars=60: value)

    def test_missing_micro_data_fails_open(self, strategy, monkeypatch):
        self._set_micro(strategy, monkeypatch, None)
        assert confirm(strategy) is True
        self.audit.assert_not_called()  # fail-open path returns before auditing

    def test_wide_spread_rejects(self, strategy, monkeypatch):
        self._set_micro(strategy, monkeypatch, (0.001, 0.0, 200.0))
        assert confirm(strategy) is False
        assert self.audit.call_args[0][3] == "rejected_spread"

    def test_spread_at_limit_approves(self, strategy, monkeypatch):
        self._set_micro(strategy, monkeypatch,
                        (strategy.micro_max_spread, 0.0, 200.0))
        assert confirm(strategy) is True  # strict `>` comparison

    def test_negative_vol_delta_rejects(self, strategy, monkeypatch):
        sigma = strategy.micro_vol_delta_sigma
        self._set_micro(strategy, monkeypatch, (0.0001, -(sigma + 0.5) * 200.0, 200.0))
        assert confirm(strategy) is False
        assert self.audit.call_args[0][3] == "rejected_volume_delta"

    def test_vol_delta_at_boundary_approves(self, strategy, monkeypatch):
        sigma = strategy.micro_vol_delta_sigma
        self._set_micro(strategy, monkeypatch, (0.0001, -sigma * 200.0, 200.0))
        assert confirm(strategy) is True  # boundary is exclusive (strict `<`)

    def test_normal_conditions_approve(self, strategy, monkeypatch):
        self._set_micro(strategy, monkeypatch, (0.0001, 100.0, 200.0))
        assert confirm(strategy) is True
        assert self.audit.call_args[0][3] == "approved"

    def test_backtest_runmode_never_gated(self, strategy, monkeypatch):
        strategy.dp.runmode = "backtest"
        self._set_micro(strategy, monkeypatch, (0.01, -1e6, 1.0))
        assert confirm(strategy) is True

    def test_filter_disabled_approves(self, strategy, monkeypatch):
        strategy.use_micro_filter = False
        self._set_micro(strategy, monkeypatch, (0.01, -1e6, 1.0))
        assert confirm(strategy) is True


# --------------------------------------------------------------------------- #
# Real _latest_1m_micro against a stubbed psycopg2 (replaces the old
# trivially-passing "DB unreachable" test)
# --------------------------------------------------------------------------- #
class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *args, **kwargs):
        pass

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        pass


class TestMicroDataFetch:
    def _fake_psycopg2(self, monkeypatch, connect):
        fake = types.ModuleType("psycopg2")
        fake.connect = connect
        monkeypatch.setitem(sys.modules, "psycopg2", fake)

    def test_db_error_returns_none_so_caller_fails_open(self, strategy, monkeypatch):
        def boom(**kwargs):
            raise ConnectionError("no db in tests")
        self._fake_psycopg2(monkeypatch, boom)
        assert strategy._latest_1m_micro(PAIR) is None
        # ...and the real entry gate therefore fails open
        monkeypatch.setattr(strategy, "_audit", MagicMock())
        assert confirm(strategy) is True

    def test_unknown_pair_returns_none(self, strategy):
        assert strategy._latest_1m_micro("DOGE/USDT:USDT") is None

    def test_empty_rows_returns_none(self, strategy, monkeypatch):
        self._fake_psycopg2(monkeypatch, lambda **kw: _FakeConn([]))
        assert strategy._latest_1m_micro(PAIR) is None

    def test_parses_rows_and_computes_std(self, strategy, monkeypatch):
        rows = [(0.0002 + i * 1e-5, 10.0 * ((-1) ** i) + i) for i in range(10)]
        self._fake_psycopg2(monkeypatch, lambda **kw: _FakeConn(rows))
        spread, vol_delta, vd_std = strategy._latest_1m_micro(PAIR)
        assert spread == rows[0][0]          # latest row first (ORDER BY time DESC)
        assert vol_delta == rows[0][1]
        assert vd_std == pytest.approx(float(np.std([r[1] for r in rows])))

    def test_std_none_with_too_few_rows(self, strategy, monkeypatch):
        rows = [(0.0002, 5.0)] * 3
        self._fake_psycopg2(monkeypatch, lambda **kw: _FakeConn(rows))
        assert strategy._latest_1m_micro(PAIR) == (0.0002, 5.0, None)


# --------------------------------------------------------------------------- #
# Circuit breakers: assert the REAL protections config, not literals
# --------------------------------------------------------------------------- #
class TestCircuitBreaker:
    def test_max_drawdown_protection(self, strategy):
        dd = [p for p in strategy.protections if p["method"] == "MaxDrawdown"]
        assert len(dd) == 1, "strategy must define exactly one MaxDrawdown guard"
        assert 0 < dd[0]["max_allowed_drawdown"] <= 0.15

    def test_stoploss_guard(self, strategy):
        sg = [p for p in strategy.protections if p["method"] == "StoplossGuard"]
        assert len(sg) == 1
        assert sg[0]["trade_limit"] >= 1
        assert sg[0]["stop_duration_candles"] >= 1

    def test_cooldown_period_present(self, strategy):
        cp = [p for p in strategy.protections if p["method"] == "CooldownPeriod"]
        assert len(cp) == 1
        assert cp[0]["stop_duration_candles"] >= 1


# --------------------------------------------------------------------------- #
# Regime-aware sizing
# --------------------------------------------------------------------------- #
class TestCustomStakeAmount:
    def _stake(self, strategy, regime, proposed=100.0):
        df = pd.DataFrame({"regime": [regime]})
        strategy.dp = MockDataProvider({(PAIR, strategy.timeframe): df})
        return strategy.custom_stake_amount(
            pair=PAIR, current_time=pd.Timestamp("2026-01-01", tz="UTC"),
            current_rate=50_000.0, proposed_stake=proposed,
            min_stake=10.0, max_stake=10_000.0, leverage=1.0,
            entry_tag=None, side="long")

    def test_strong_trend_sizes_up(self, strategy):
        mult = strategy.regime_size_mult["STRONG_BULL"]
        assert self._stake(strategy, "STRONG_BULL") == pytest.approx(100.0 * mult)

    def test_mild_regime_base_size(self, strategy):
        assert self._stake(strategy, "BULL") == pytest.approx(100.0)

    def test_unknown_regime_falls_back_to_base(self, strategy):
        assert self._stake(strategy, "WEIRD") == pytest.approx(100.0)

    def test_dp_failure_returns_proposed(self, strategy):
        strategy.dp = MagicMock()
        strategy.dp.get_analyzed_dataframe.side_effect = RuntimeError("boom")
        stake = strategy.custom_stake_amount(
            pair=PAIR, current_time=None, current_rate=1.0, proposed_stake=100.0,
            min_stake=10.0, max_stake=10_000.0, leverage=1.0,
            entry_tag=None, side="long")
        assert stake == pytest.approx(100.0)

    def test_capped_at_max_stake(self, strategy):
        assert self._stake(strategy, "STRONG_BULL", proposed=9_000.0) == 10_000.0


# --------------------------------------------------------------------------- #
# Idempotency + shadow mode hermetics
# --------------------------------------------------------------------------- #
class TestIdempotency:
    def test_double_entry_call_same_result(self, strategy):
        kw = dict(regime="STRONG_BULL", pred=0.005, slope=0.005)
        a = strategy.populate_entry_trend(entry_df(**kw), METADATA)
        b = strategy.populate_entry_trend(entry_df(**kw), METADATA)
        cols = ["enter_long", "enter_short"]
        assert a[cols].fillna(0).equals(b[cols].fillna(0))


class TestShadowMode:
    def test_non_live_runmode_makes_no_network_calls(self, strategy, monkeypatch):
        strategy.dp.runmode = "backtest"
        fake = MagicMock()
        monkeypatch.setitem(sys.modules, "requests", fake)
        strategy.bot_loop_start(pd.Timestamp("2026-01-01", tz="UTC"))
        fake.post.assert_not_called()

    def test_llm_service_failure_is_swallowed(self, strategy, injected_5m, monkeypatch):
        strategy.dp = MockDataProvider({(PAIR, "5m"): injected_5m})
        strategy.dp.runmode = "live"
        fake = types.ModuleType("requests")

        def boom(*args, **kwargs):
            raise ConnectionError("llm service down")

        fake.post = boom
        monkeypatch.setitem(sys.modules, "requests", fake)
        # live runmode -> it DOES try, but any failure must be non-fatal
        strategy.bot_loop_start(pd.Timestamp("2026-01-01", tz="UTC"))

    def test_shadow_log_db_failure_is_swallowed(self, strategy, monkeypatch):
        def boom(**kwargs):
            raise ConnectionError("no db in tests")
        fake = types.ModuleType("psycopg2")
        fake.connect = boom
        monkeypatch.setitem(sys.modules, "psycopg2", fake)
        strategy._log_shadow(PAIR, pd.Timestamp("2026-01-01", tz="UTC"), 1,
                             {"bias": "long", "confidence": 0.9})


# --------------------------------------------------------------------------- #
# Sanity: the fixture really is the real strategy class from disk
# --------------------------------------------------------------------------- #
class TestRealStrategyLoaded:
    def test_class_origin(self, strategy):
        assert type(strategy).__name__ == "GTQuantMultiTF"
        assert "ft_userdata" in sys.modules[type(strategy).__module__].__file__

    def test_local_module_agrees_with_fixture(self, strategy):
        import inspect
        from conftest import STRATEGY_PATH
        assert inspect.getsourcefile(load_strategy_class()) == str(STRATEGY_PATH)
        assert inspect.getsourcefile(type(strategy)) == str(STRATEGY_PATH)
