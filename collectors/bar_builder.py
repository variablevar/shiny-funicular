"""
1m bar builder: aggregates tick-level events into 1-minute bars with
microstructure features (Day 2).

Pure, network-free logic — unit-tested in tests/test_bar_builder.py.
The live Cryptofeed collector (cryptofeed_collector.py) feeds it trades and
book-ticker updates; on each minute close it emits a bar dict matching the
TimescaleDB ``ohlcv_1m`` schema:

    time, symbol, open, high, low, close, volume,
    spread, bid_ask_imbalance, volume_delta,
    tick_momentum, realized_variance, trade_count
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def minute_floor(ts_ms: int) -> int:
    """Floor a millisecond timestamp to its minute boundary."""
    return (ts_ms // 60_000) * 60_000


@dataclass
class BarAccumulator:
    """Accumulates one symbol's ticks for the current minute."""

    symbol: str
    minute_ms: int
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: float = 0.0
    buy_volume: float = 0.0      # taker buy (aggressor = buy)
    sell_volume: float = 0.0     # taker sell
    trade_count: int = 0
    upticks: int = 0
    downticks: int = 0
    last_price: Optional[float] = None
    last_spread: Optional[float] = None
    last_imbalance: Optional[float] = None
    # per-tick log returns for realized variance
    _returns: List[float] = field(default_factory=list)

    def add_trade(self, price: float, amount: float, taker_buy: bool) -> None:
        """Fold one trade tick into the bar."""
        if self.open is None:
            self.open = price
        self.high = price if self.high is None else max(self.high, price)
        self.low = price if self.low is None else min(self.low, price)
        self.close = price
        self.volume += amount
        self.trade_count += 1

        if taker_buy:
            self.buy_volume += amount
        else:
            self.sell_volume += amount

        if self.last_price is not None and self.last_price > 0:
            ret = math.log(price / self.last_price)
            self._returns.append(ret)
            if price > self.last_price:
                self.upticks += 1
            elif price < self.last_price:
                self.downticks += 1
        self.last_price = price

    def add_book(self, bid: float, ask: float, bid_size: float, ask_size: float) -> None:
        """Record the latest top-of-book (spread + imbalance)."""
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            self.last_spread = (ask - bid) / mid
            total = bid_size + ask_size
            self.last_imbalance = (bid_size - ask_size) / total if total > 0 else 0.0

    def finalize(self) -> Optional[dict]:
        """Emit the completed bar, or None if no trades occurred."""
        if self.trade_count == 0 or self.open is None:
            return None
        variance = sum(r * r for r in self._returns) if self._returns else 0.0
        return {
            "time": self.minute_ms,
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "spread": self.last_spread,
            "bid_ask_imbalance": self.last_imbalance,
            "volume_delta": self.buy_volume - self.sell_volume,
            "tick_momentum": self.upticks - self.downticks,
            "realized_variance": variance,
            "trade_count": self.trade_count,
        }


class BarBuilder:
    """
    Aggregates ticks into 1m bars for any number of symbols.

    Feed events with timestamps; call ``flush_due(now_ms)`` to emit all bars
    whose minute has closed.
    """

    def __init__(self):
        self._bars: Dict[Tuple[str, int], BarAccumulator] = {}

    def add_trade(self, symbol: str, ts_ms: int, price: float, amount: float, taker_buy: bool) -> None:
        minute = minute_floor(ts_ms)
        key = (symbol, minute)
        if key not in self._bars:
            self._bars[key] = BarAccumulator(symbol=symbol, minute_ms=minute)
        self._bars[key].add_trade(price, amount, taker_buy)

    def add_book(self, symbol: str, ts_ms: int, bid: float, ask: float, bid_size: float, ask_size: float) -> None:
        minute = minute_floor(ts_ms)
        key = (symbol, minute)
        if key not in self._bars:
            self._bars[key] = BarAccumulator(symbol=symbol, minute_ms=minute)
        self._bars[key].add_book(bid, ask, bid_size, ask_size)

    def flush_due(self, now_ms: int) -> List[dict]:
        """
        Finalize and remove all bars from minutes strictly before the current
        minute. Returns the list of completed bars (may be empty).
        """
        current_minute = minute_floor(now_ms)
        done_keys = [k for k in self._bars if k[1] < current_minute]
        out: List[dict] = []
        for key in done_keys:
            bar = self._bars.pop(key).finalize()
            if bar is not None:
                out.append(bar)
        return out

    def pending(self) -> int:
        """Number of minutes currently being accumulated."""
        return len(self._bars)

    def flush_all(self) -> List[dict]:
        """Finalize everything (used on shutdown)."""
        out: List[dict] = []
        for key in list(self._bars):
            bar = self._bars.pop(key).finalize()
            if bar is not None:
                out.append(bar)
        return out
