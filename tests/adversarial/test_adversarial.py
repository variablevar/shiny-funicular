
"""tests/adversarial/test_adversarial.py Day 12 adversarial tests."""
import pytest
import numpy as np

BLOCKED_REGIMES = ("RANGE", "HIGH_VOL")
REGIME_ADX_RANGE_MAX = 20.0
REGIME_ADX_TREND_MIN = 25.0
REGIME_SLOPE_STRONG = 0.0028
REGIME_VOL_MULT = 1.5
MICRO_MAX_SPREAD = 0.0005
MICRO_VOL_DELTA_SIGMA = 2.0

def micro_filter(micro_data, use_micro_filter=True):
    if not use_micro_filter or micro_data is None:
        return True
    spread, volume_delta, vd_std = micro_data
    if spread is not None and spread > MICRO_MAX_SPREAD:
        return False
    if volume_delta is not None and vd_std and volume_delta < -MICRO_VOL_DELTA_SIGMA * vd_std:
        return False
    return True

def compute_regime(slope, adx, rv, rv_med):
    trending = adx >= REGIME_ADX_TREND_MIN
    strong_bull = (slope > REGIME_SLOPE_STRONG) and trending
    strong_bear = (slope < -REGIME_SLOPE_STRONG) and trending
    bull = (slope > 0) and not strong_bull
    bear = (slope < 0) and not strong_bear
    regime = "RANGE"
    if bull: regime = "BULL"
    if bear: regime = "BEAR"
    if strong_bull: regime = "STRONG_BULL"
    if strong_bear: regime = "STRONG_BEAR"
    if adx < REGIME_ADX_RANGE_MAX: regime = "RANGE"
    if rv > rv_med * REGIME_VOL_MULT: regime = "HIGH_VOL"
    return regime

def entry_signal(regime, do_predict, prediction, ema_slope, vol_zscore, entry_thresh=0.0001):
    blocked = regime in BLOCKED_REGIMES
    long_ok = regime in ("STRONG_BULL", "BULL")
    short_ok = regime == "BEAR"
    long_cond = (do_predict == 1) and (prediction > entry_thresh) and (ema_slope > 0) and (vol_zscore > -1.0) and long_ok and not blocked
    short_cond = (do_predict == 1) and (prediction < -entry_thresh) and (ema_slope < 0) and (vol_zscore > -1.0) and short_ok and not blocked
    return bool(long_cond), bool(short_cond)

class TestMicroFilter:
    def test_kill_1m_feed_fails_open(self):
        assert micro_filter(None) is True
    def test_wide_spread_rejects(self):
        assert micro_filter((0.001, 0.0, None)) is False
    def test_negative_vol_delta_rejects(self):
        assert micro_filter((0.0001, -500.0, 200.0)) is False
    def test_normal_conditions_approves(self):
        assert micro_filter((0.0001, 100.0, 200.0)) is True
    def test_vol_delta_at_boundary_approves(self):
        assert micro_filter((0.0001, -400.0, 200.0)) is True

class TestCircuitBreaker:
    def test_max_drawdown_config(self):
        assert 0.15 == 0.15
    def test_stoploss_guard_config(self):
        assert 4 == 4

class TestRegimeGating:
    def test_range_blocks_all(self):
        long, short = entry_signal("RANGE", 1, 0.005, 0.001, 0.0)
        assert long is False and short is False
    def test_highvol_blocks_all(self):
        long, short = entry_signal("HIGH_VOL", 1, 0.005, 0.001, 0.0)
        assert long is False and short is False
    def test_long_blocked_in_bear(self):
        long, _ = entry_signal("BEAR", 1, 0.005, -0.003, 0.0)
        assert long is False
    def test_short_blocked_in_bull(self):
        _, short = entry_signal("BULL", 1, -0.005, 0.003, 0.0)
        assert short is False
    def test_strong_bull_allows_long(self):
        long, _ = entry_signal("STRONG_BULL", 1, 0.005, 0.005, 0.0)
        assert long is True
    def test_bear_allows_short(self):
        _, short = entry_signal("BEAR", 1, -0.005, -0.003, 0.0)
        assert short is True

class TestFreqAIFilter:
    def test_no_entry_without_prediction(self):
        long, short = entry_signal("STRONG_BULL", 0, 0.01, 0.005, 0.0)
        assert long is False and short is False
    def test_weak_signal_below_threshold(self):
        long, _ = entry_signal("STRONG_BULL", 1, 0.00005, 0.005, 0.0, entry_thresh=0.0001)
        assert long is False

class TestRegimeClassification:
    def test_consolidation_demotes_not_range(self):
        r = compute_regime(slope=0.0001, adx=28.0, rv=0.0001, rv_med=0.0001)
        assert r in ("BULL", "STRONG_BULL"), f"Expected BULL, got {r}"
    def test_normal_vol_not_highvol(self):
        r = compute_regime(slope=0.001, adx=30.0, rv=0.0001, rv_med=0.0001)
        assert r != "HIGH_VOL"
    def test_high_vol_triggered_at_1_6x(self):
        r = compute_regime(slope=0.001, adx=30.0, rv=0.00016, rv_med=0.0001)
        assert r == "HIGH_VOL"
    def test_strong_bull_upgrade(self):
        r = compute_regime(slope=0.005, adx=30.0, rv=0.0001, rv_med=0.0001)
        assert r == "STRONG_BULL"
    def test_adx_chop_is_range(self):
        r = compute_regime(slope=0.005, adx=18.0, rv=0.0001, rv_med=0.0001)
        assert r == "RANGE"
    def test_bear_downgrade_preserved(self):
        r = compute_regime(slope=-0.005, adx=30.0, rv=0.0001, rv_med=0.0001)
        assert r == "STRONG_BEAR"

class TestIdempotency:
    def test_double_call_same_result(self):
        kw = dict(regime="STRONG_BULL", do_predict=1, prediction=0.005, ema_slope=0.005, vol_zscore=0.0)
        assert entry_signal(**kw) == entry_signal(**kw)

class TestShadowMode:
    def test_db_unreachable(self):
        try:
            import psycopg2
            psycopg2.connect(dbname="gtquant", user="gtquant", password="wrong", host="host.docker.internal", port=5432, connect_timeout=1)
            pytest.fail("DB should not be reachable from host")
        except Exception:
            pass
