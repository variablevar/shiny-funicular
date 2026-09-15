"""
Unit tests for the historical aggTrades -> 1m micro bar builder.

Key property: collectors.historical_micro.bars_from_trades (vectorized,
used on multi-GB daily archives) must produce exactly the bars that
collectors.bar_builder.BarBuilder (the live collector's logic) produces
when fed the same trades in time order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from collectors.bar_builder import BarBuilder
from collectors.historical_micro import bars_from_trades


def _bars_via_bar_builder(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Feed trades through the live BarBuilder and collect flushed bars."""
    bb = BarBuilder()
    for r in trades.itertuples():
        bb.add_trade(symbol, int(r.ts_ms), float(r.price), float(r.quantity),
                     bool(r.taker_buy))
    bars = bb.flush_all()
    df = pd.DataFrame(bars)
    return df.sort_values("time").reset_index(drop=True)


def _compare(bars_fast: pd.DataFrame, bars_ref: pd.DataFrame) -> None:
    assert len(bars_fast) == len(bars_ref)
    cols = ["time", "open", "high", "low", "close", "volume",
            "volume_delta", "tick_momentum", "realized_variance", "trade_count"]
    for col in cols:
        a = bars_fast[col].to_numpy(dtype=float)
        b = bars_ref[col].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12), f"column {col} differs"
    # Historical bars have no book data.
    assert bars_fast["spread"].isna().all()
    assert bars_fast["bid_ask_imbalance"].isna().all()


def _synthetic_day(seed: int = 7) -> pd.DataFrame:
    """Random-walk trade stream over ~2 hours with mixed taker sides."""
    rng = np.random.default_rng(seed)
    n = 5000
    start_ms = 1_757_750_400_000  # some minute boundary
    ts = start_ms + np.sort(rng.integers(0, 2 * 3600 * 1000, size=n))
    rets = rng.normal(0, 0.0002, size=n)
    price = 50_000 * np.exp(np.cumsum(rets))
    qty = np.abs(rng.normal(0.05, 0.05, size=n)) + 0.001
    taker_buy = rng.random(n) < 0.5
    return pd.DataFrame({"ts_ms": ts.astype(np.int64), "price": price,
                         "quantity": qty, "taker_buy": taker_buy})


class TestBarBuilderEquivalence:
    def test_matches_live_bar_builder(self):
        trades = _synthetic_day()
        fast = bars_from_trades(trades, "BTCUSDT")
        ref = _bars_via_bar_builder(trades, "BTCUSDT")
        _compare(fast, ref)

    def test_matches_with_duplicate_timestamps(self):
        """Several trades in the same millisecond must aggregate identically."""
        trades = _synthetic_day(seed=11)
        trades["ts_ms"] = (trades["ts_ms"] // 1000) * 1000  # force dup ms
        fast = bars_from_trades(trades, "ETHUSDT")
        ref = _bars_via_bar_builder(trades, "ETHUSDT")
        _compare(fast, ref)

    def test_cross_minute_jump_excluded_from_variance(self):
        """The price jump between minutes is not part of either minute's RV."""
        trades = pd.DataFrame({
            "ts_ms": [0, 30_000, 60_000, 90_000],
            "price": [100.0, 101.0, 200.0, 201.0],
            "quantity": [1.0, 1.0, 1.0, 1.0],
            "taker_buy": [True, True, False, False],
        })
        bars = bars_from_trades(trades, "BTCUSDT")
        assert len(bars) == 2
        rv0 = bars.loc[bars["time"] == 0, "realized_variance"].iloc[0]
        rv1 = bars.loc[bars["time"] == 60_000, "realized_variance"].iloc[0]
        assert rv0 == pytest.approx(np.log(101 / 100) ** 2)
        assert rv1 == pytest.approx(np.log(201 / 200) ** 2)
        # tick momentum: minute 0 has one uptick, minute 1 one uptick
        # (the 101 -> 200 jump across minutes counts for neither).
        assert bars["tick_momentum"].tolist() == [1, 1]

    def test_volume_delta_sign(self):
        trades = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000],
            "price": [100.0, 100.0, 100.0],
            "quantity": [3.0, 1.0, 2.0],
            "taker_buy": [True, False, True],
        })
        bars = bars_from_trades(trades, "BTCUSDT")
        assert bars["volume"].iloc[0] == pytest.approx(6.0)
        assert bars["volume_delta"].iloc[0] == pytest.approx(5.0 - 1.0)
        assert bars["trade_count"].iloc[0] == 3

    def test_empty_trades(self):
        bars = bars_from_trades(
            pd.DataFrame(columns=["ts_ms", "price", "quantity", "taker_buy"]),
            "BTCUSDT")
        assert bars.empty
