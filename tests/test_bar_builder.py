"""
Unit tests for the 1m BarBuilder (Day 2 collector core).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from collectors.bar_builder import BarBuilder, BarAccumulator, minute_floor


class TestMinuteFloor:
    def test_floors_to_minute(self):
        assert minute_floor(61_234) == 60_000
        assert minute_floor(119_999) == 60_000
        assert minute_floor(120_000) == 120_000


class TestBarAccumulator:
    def test_ohlcv_aggregation(self):
        bar = BarAccumulator(symbol="BTCUSDT", minute_ms=0)
        bar.add_trade(100.0, 1.0, taker_buy=True)
        bar.add_trade(105.0, 2.0, taker_buy=True)
        bar.add_trade(95.0, 1.5, taker_buy=False)
        bar.add_trade(102.0, 0.5, taker_buy=False)

        out = bar.finalize()
        assert out["open"] == 100.0
        assert out["high"] == 105.0
        assert out["low"] == 95.0
        assert out["close"] == 102.0
        assert out["volume"] == 5.0
        assert out["trade_count"] == 4

    def test_volume_delta_and_tick_momentum(self):
        bar = BarAccumulator(symbol="BTCUSDT", minute_ms=0)
        bar.add_trade(100.0, 2.0, taker_buy=True)    # buy 2
        bar.add_trade(101.0, 3.0, taker_buy=True)    # buy 3, uptick
        bar.add_trade(100.0, 4.0, taker_buy=False)   # sell 4, downtick
        out = bar.finalize()
        assert out["volume_delta"] == 1.0     # 5 buy - 4 sell
        assert out["tick_momentum"] == 0      # 1 up - 1 down

    def test_realized_variance_positive(self):
        bar = BarAccumulator(symbol="BTCUSDT", minute_ms=0)
        bar.add_trade(100.0, 1.0, taker_buy=True)
        bar.add_trade(101.0, 1.0, taker_buy=True)
        bar.add_trade(100.0, 1.0, taker_buy=False)
        out = bar.finalize()
        assert out["realized_variance"] > 0

    def test_book_features(self):
        bar = BarAccumulator(symbol="BTCUSDT", minute_ms=0)
        bar.add_trade(100.0, 1.0, taker_buy=True)
        bar.add_book(bid=99.9, ask=100.1, bid_size=3.0, ask_size=1.0)
        out = bar.finalize()
        assert abs(out["spread"] - 0.2 / 100.0) < 1e-9
        assert abs(out["bid_ask_imbalance"] - 0.5) < 1e-9   # (3-1)/(3+1)

    def test_empty_bar_returns_none(self):
        bar = BarAccumulator(symbol="BTCUSDT", minute_ms=0)
        assert bar.finalize() is None


class TestBarBuilder:
    def test_bars_split_by_minute(self):
        bb = BarBuilder()
        bb.add_trade("BTCUSDT", ts_ms=1_000, price=100.0, amount=1.0, taker_buy=True)
        bb.add_trade("BTCUSDT", ts_ms=61_000, price=101.0, amount=1.0, taker_buy=True)
        assert bb.pending() == 2

    def test_flush_due_only_completed_minutes(self):
        bb = BarBuilder()
        bb.add_trade("BTCUSDT", ts_ms=1_000, price=100.0, amount=1.0, taker_buy=True)
        bb.add_trade("BTCUSDT", ts_ms=61_000, price=101.0, amount=1.0, taker_buy=True)
        bb.add_trade("BTCUSDT", ts_ms=121_000, price=102.0, amount=1.0, taker_buy=True)

        done = bb.flush_due(now_ms=125_000)
        assert len(done) == 2                        # minutes 0 and 60000
        assert bb.pending() == 1                     # minute 120000 still open
        assert {b["close"] for b in done} == {100.0, 101.0}

    def test_symbols_tracked_independently(self):
        bb = BarBuilder()
        bb.add_trade("BTCUSDT", ts_ms=1_000, price=100.0, amount=1.0, taker_buy=True)
        bb.add_trade("ETHUSDT", ts_ms=1_000, price=50.0, amount=2.0, taker_buy=True)
        done = bb.flush_due(now_ms=61_000)
        assert len(done) == 2
        assert {b["symbol"] for b in done} == {"BTCUSDT", "ETHUSDT"}

    def test_flush_all_on_shutdown(self):
        bb = BarBuilder()
        bb.add_trade("BTCUSDT", ts_ms=121_000, price=102.0, amount=1.0, taker_buy=True)
        done = bb.flush_all()
        assert len(done) == 1
        assert bb.pending() == 0
