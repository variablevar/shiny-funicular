#!/usr/bin/env python3
"""
GT-Quant live collector (Day 2): Binance futures REST -> 1m bars -> TimescaleDB.

NOTE ON IMPLEMENTATION: the plan specifies Cryptofeed websockets, but Binance
futures websocket hosts (fstream.binance.com) are region-blocked from this
node (UK) — connections open but no data flows. The futures REST API
(fapi.binance.com) works, so this collector polls it instead:

  - /fapi/v1/aggTrades        -> 1m OHLCV + volume delta + tick momentum + RV
  - /fapi/v1/ticker/bookTicker -> top-of-book spread + imbalance
  - /fapi/v1/premiumIndex      -> funding rate -> funding_rates hypertable
  - /fapi/v1/openInterest      -> OI -> open_interest hypertable

Completed 1m bars flush to the ohlcv_1m hypertable on each poll cycle.

Usage:
    source venv/bin/activate
    python collectors/live_collector.py                 # run forever
    python collectors/live_collector.py --duration 180  # 3-min test run
"""
from __future__ import annotations

import argparse
import signal
import time
from typing import Dict, List, Optional

import psycopg2
import psycopg2.extras
import requests
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from collectors.bar_builder import BarBuilder

BASE_URL = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

DB_DSN = "dbname=gtquant user=gtquant password=gtquant_local host=localhost port=5432"

TRADE_POLL_SECONDS = 5      # aggTrades + bookTicker cadence
SLOW_POLL_SECONDS = 60      # funding + OI cadence

BAR_INSERT = """
INSERT INTO ohlcv_1m (time, symbol, open, high, low, close, volume,
                      spread, bid_ask_imbalance, volume_delta,
                      tick_momentum, realized_variance, trade_count)
VALUES %s
ON CONFLICT (time, symbol) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume,
    spread = EXCLUDED.spread, bid_ask_imbalance = EXCLUDED.bid_ask_imbalance,
    volume_delta = EXCLUDED.volume_delta, tick_momentum = EXCLUDED.tick_momentum,
    realized_variance = EXCLUDED.realized_variance, trade_count = EXCLUDED.trade_count
"""

FUNDING_INSERT = """
INSERT INTO funding_rates (time, symbol, funding_rate, mark_price)
VALUES %s
ON CONFLICT (time, symbol) DO NOTHING
"""

OI_INSERT = """
INSERT INTO open_interest (time, symbol, open_interest)
VALUES %s
ON CONFLICT (time, symbol) DO UPDATE SET open_interest = EXCLUDED.open_interest
"""


class CollectorDB:
    """Thin psycopg2 wrapper with batched writes."""

    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True

    def write_bars(self, bars: List[dict]) -> int:
        if not bars:
            return 0
        import datetime
        rows = [
            (
                datetime.datetime.utcfromtimestamp(b["time"] / 1000),
                b["symbol"], b["open"], b["high"], b["low"], b["close"], b["volume"],
                b["spread"], b["bid_ask_imbalance"], b["volume_delta"],
                b["tick_momentum"], b["realized_variance"], b["trade_count"],
            )
            for b in bars
        ]
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(cur, BAR_INSERT, rows)
        return len(rows)

    def write_funding(self, symbol: str, ts_ms: int, rate: float, mark: Optional[float]) -> None:
        import datetime
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, FUNDING_INSERT,
                [(datetime.datetime.utcfromtimestamp(ts_ms / 1000), symbol, rate, mark)],
            )

    def write_oi(self, symbol: str, ts_ms: int, oi: float) -> None:
        import datetime
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, OI_INSERT,
                [(datetime.datetime.utcfromtimestamp(ts_ms / 1000), symbol, oi)],
            )

    def close(self) -> None:
        self.conn.close()


class LiveCollector:
    """Polls Binance futures REST and builds 1m microstructure bars."""

    def __init__(self, dsn: str = DB_DSN):
        self.builder = BarBuilder()
        self.db = CollectorDB(dsn)
        self.session = requests.Session()
        self.last_trade_id: Dict[str, Optional[int]] = {s: None for s in SYMBOLS}
        self.bars_written = 0
        self._running = True

    # ------------------------------------------------------------------ #
    # REST fetchers
    # ------------------------------------------------------------------ #

    def _get(self, path: str, params: dict) -> list | dict:
        r = self.session.get(f"{BASE_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def poll_trades(self, symbol: str) -> None:
        """Fetch new aggTrades since the last seen trade id and feed the builder."""
        params: dict = {"symbol": symbol, "limit": 1000}
        if self.last_trade_id[symbol] is not None:
            params["fromId"] = self.last_trade_id[symbol] + 1
        try:
            trades = self._get("/fapi/v1/aggTrades", params)
        except Exception as e:
            logger.warning(f"{symbol} aggTrades fetch failed: {e}")
            return

        for t in trades:
            # 'm' True means buyer is maker -> aggressor is the SELLER.
            self.builder.add_trade(
                symbol=symbol,
                ts_ms=int(t["T"]),
                price=float(t["p"]),
                amount=float(t["q"]),
                taker_buy=not t["m"],
            )
        if trades:
            self.last_trade_id[symbol] = int(trades[-1]["a"])

    def poll_book(self, symbol: str) -> None:
        """Snapshot top-of-book for spread + imbalance."""
        try:
            book = self._get("/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        except Exception as e:
            logger.warning(f"{symbol} bookTicker fetch failed: {e}")
            return
        self.builder.add_book(
            symbol=symbol,
            ts_ms=int(book.get("time", time.time() * 1000)),
            bid=float(book["bidPrice"]),
            ask=float(book["askPrice"]),
            bid_size=float(book["bidQty"]),
            ask_size=float(book["askQty"]),
        )

    def poll_funding(self, symbol: str) -> None:
        try:
            info = self._get("/fapi/v1/premiumIndex", {"symbol": symbol})
            self.db.write_funding(
                symbol,
                int(info["time"]),
                float(info["lastFundingRate"]),
                float(info["markPrice"]),
            )
        except Exception as e:
            logger.warning(f"{symbol} premiumIndex fetch failed: {e}")

    def poll_open_interest(self, symbol: str) -> None:
        try:
            info = self._get("/fapi/v1/openInterest", {"symbol": symbol})
            self.db.write_oi(symbol, int(info["time"]), float(info["openInterest"]))
        except Exception as e:
            logger.warning(f"{symbol} openInterest fetch failed: {e}")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def stop(self, *_args) -> None:
        self._running = False

    def run(self, duration: int = 0) -> None:
        start = time.time()
        last_slow = 0.0
        logger.info(f"live collector started for {SYMBOLS} (REST polling, {TRADE_POLL_SECONDS}s)")

        while self._running:
            now = time.time()
            if duration > 0 and now - start >= duration:
                break

            for symbol in SYMBOLS:
                self.poll_trades(symbol)
                self.poll_book(symbol)

            # Flush any completed 1m bars.
            bars = self.builder.flush_due(int(now * 1000))
            if bars:
                try:
                    n = self.db.write_bars(bars)
                    self.bars_written += n
                    logger.info(f"flushed {n} bars (total {self.bars_written}), pending {self.builder.pending()}")
                except Exception as e:
                    logger.error(f"bar write failed: {e}; dropping {len(bars)}")

            if now - last_slow >= SLOW_POLL_SECONDS:
                for symbol in SYMBOLS:
                    self.poll_funding(symbol)
                    self.poll_open_interest(symbol)
                last_slow = now

            time.sleep(TRADE_POLL_SECONDS)

        # Final flush on exit.
        bars = self.builder.flush_all()
        if bars:
            n = self.db.write_bars(bars)
            self.bars_written += n
            logger.info(f"final flush: {n} bars (total {self.bars_written})")
        self.db.close()
        logger.info("live collector stopped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=0,
                        help="Seconds to run before shutdown (0 = forever)")
    args = parser.parse_args()

    collector = LiveCollector()
    signal.signal(signal.SIGINT, collector.stop)
    signal.signal(signal.SIGTERM, collector.stop)
    collector.run(duration=args.duration)


if __name__ == "__main__":
    main()
