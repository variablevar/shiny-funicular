"""
Unit tests for the TF-aware RiskConfig and TFRiskEngine (Day 1).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.risk_config import RiskConfig, TFRiskEngine, TRADED_TIMEFRAMES


@pytest.fixture
def cfg():
    return RiskConfig()


@pytest.fixture
def engine(cfg):
    return TFRiskEngine(cfg)


class TestRiskConfig:
    def test_tf_size_multipliers_match_plan(self, cfg):
        # Plan: 1m 0.5x, 5m 1.0x, 15m 1.5x, 1h 2.0x
        assert cfg.size_multiplier("1m") == 0.5
        assert cfg.size_multiplier("5m") == 1.0
        assert cfg.size_multiplier("15m") == 1.5
        assert cfg.size_multiplier("1h") == 2.0

    def test_max_notional_scales_with_multiplier(self, cfg):
        # base_stake 1000 -> 5m cap 1000, 1h cap 2000
        assert cfg.max_notional("5m") == 1000.0
        assert cfg.max_notional("1h") == 2000.0

    def test_unknown_timeframe_raises(self, cfg):
        with pytest.raises(ValueError):
            cfg.size_multiplier("3m")
        with pytest.raises(ValueError):
            cfg.stale_threshold("3m")

    def test_stale_thresholds_ordered_by_tf(self, cfg):
        # Faster TFs must have tighter staleness limits.
        assert cfg.stale_threshold("1m") < cfg.stale_threshold("5m")
        assert cfg.stale_threshold("5m") < cfg.stale_threshold("15m")
        assert cfg.stale_threshold("15m") < cfg.stale_threshold("1h")


class TestTFRiskEngine:
    def test_approves_valid_trade(self, engine):
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=1, notional=500.0, equity=1000.0
        )
        assert ok, reason

    def test_rejects_untradeable_timeframe(self, engine):
        ok, reason = engine.validate_trade(
            timeframe="1m", direction=1, notional=100.0, equity=1000.0
        )
        assert not ok
        assert "not tradeable" in reason

    def test_rejects_oversized_position_per_tf(self, engine):
        # 5m cap is 1000 (1.0x of base_stake); 1500 must be rejected.
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=1, notional=1500.0, equity=10000.0
        )
        assert not ok
        assert "exceeds" in reason

    def test_rejects_when_max_open_positions(self, engine):
        for _ in range(engine.cfg.max_open_positions):
            engine.on_position_opened()
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=1, notional=100.0, equity=1000.0
        )
        assert not ok
        assert "open positions" in reason

    def test_rejects_after_daily_loss_limit(self, engine):
        # Lose 4% of 1000 in a day (limit is 3%).
        engine.update_state(pnl=-40.0, equity=960.0)
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=1, notional=100.0, equity=960.0
        )
        assert not ok
        assert "daily loss" in reason

    def test_funding_filter_blocks_expensive_longs(self, engine):
        # Funding above +0.01% blocks new longs.
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=1, notional=100.0,
            equity=1000.0, funding_rate=0.0005,
        )
        assert not ok
        assert "funding" in reason

    def test_funding_filter_blocks_expensive_shorts(self, engine):
        # Funding below -0.01% blocks new shorts.
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=-1, notional=100.0,
            equity=1000.0, funding_rate=-0.0005,
        )
        assert not ok
        assert "funding" in reason

    def test_halt_and_resume(self, engine):
        engine.trigger_halt("API failure")
        ok, reason = engine.validate_trade(
            timeframe="5m", direction=1, notional=100.0, equity=1000.0
        )
        assert not ok
        assert "halted" in reason

        engine.resume()
        ok, _ = engine.validate_trade(
            timeframe="5m", direction=1, notional=100.0, equity=1000.0
        )
        assert ok

    def test_drawdown_check(self, engine):
        engine.update_state(pnl=0.0, equity=1000.0)
        assert engine.check_drawdown(960.0) is True    # 4% dd, under 5% limit
        assert engine.check_drawdown(900.0) is False   # 10% dd, over limit

    def test_stale_data_detection(self, engine):
        assert engine.check_stale_data("5m", 300) is True    # 5m old, under 600s
        assert engine.check_stale_data("5m", 900) is False   # 15m old, stale
        assert engine.check_stale_data("1m", 300) is False   # stale for 1m feed

    def test_traded_timeframes_constant(self):
        # The plan trades 5m/15m/1h only; 1m is microstructure input.
        assert "1m" not in TRADED_TIMEFRAMES
        assert set(TRADED_TIMEFRAMES) == {"5m", "15m", "1h"}
