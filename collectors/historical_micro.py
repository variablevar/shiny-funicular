#!/usr/bin/env python3
"""
Historical 1m microstructure bar builder (Day 14).

Replays Binance UM-futures aggTrades daily archives (data.binance.vision)
through the same aggregation logic as the live collector
(collectors/bar_builder.py) to reconstruct the 1m microstructure columns
that the live collector only has since its last restart:

    volume_delta, tick_momentum, realized_variance, trade_count

spread / bid_ask_imbalance are NOT recoverable from aggTrades (they need
bookTicker history, which is impractically large), so they are left NULL —
matching the schema comment "NULL for exchange-official backfill bars".

Outputs per day:
    data/raw/micro_1m/{SYMBOL}/{SYMBOL}_{YYYY-MM-DD}.parquet

and optionally bulk-loads into TimescaleDB `ohlcv_1m` (--load-db), and
merges everything into the single per-pair parquet the strategy reads
(--merge -> ft_userdata/user_data/data/micro/).

The downloader is resumable: days with an existing parquet are skipped.

Usage:
    source venv/bin/activate
    python collectors/historical_micro.py --start 2026-06-01 --end 2026-09-11
    python collectors/historical_micro.py --start 2026-06-01 --end 2026-09-11 \
        --load-db --merge
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ARCHIVE_URL = (
    "https://data.binance.vision/data/futures/um/daily/aggTrades/"
    "{symbol}/{symbol}-aggTrades-{day}.zip"
)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Freqtrade pair name <-> Binance symbol
PAIR_TO_SYMBOL = {"BTC/USDT:USDT": "BTCUSDT", "ETH/USDT:USDT": "ETHUSDT"}
SYMBOL_TO_PAIR = {v: k for k, v in PAIR_TO_SYMBOL.items()}

RAW_DIR = ROOT / "data" / "raw" / "micro_1m"
STRATEGY_DIR = ROOT / "ft_userdata" / "user_data" / "data" / "micro"

DB_DSN = "dbname=gtquant user=gtquant password=gtquant_local host=localhost port=5432"

# aggTrades CSV column order used by Binance when no header row is present.
AGGTRADE_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]

MICRO_BAR_COLS = [
    "time", "symbol", "open", "high", "low", "close", "volume",
    "spread", "bid_ask_imbalance", "volume_delta",
    "tick_momentum", "realized_variance", "trade_count",
]

# Historical rows carry no book data: never clobber live spread/imbalance.
DB_UPSERT = """
INSERT INTO ohlcv_1m (time, symbol, open, high, low, close, volume,
                      spread, bid_ask_imbalance, volume_delta,
                      tick_momentum, realized_variance, trade_count)
VALUES %s
ON CONFLICT (time, symbol) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume,
    volume_delta = EXCLUDED.volume_delta,
    tick_momentum = EXCLUDED.tick_momentum,
    realized_variance = EXCLUDED.realized_variance,
    trade_count = EXCLUDED.trade_count,
    spread = COALESCE(ohlcv_1m.spread, EXCLUDED.spread),
    bid_ask_imbalance = COALESCE(ohlcv_1m.bid_ask_imbalance, EXCLUDED.bid_ask_imbalance)
"""


def download_day(symbol: str, day: str, tmp_dir: Path) -> Path | None:
    """Download one daily aggTrades zip. Returns the path, or None on 404."""
    url = ARCHIVE_URL.format(symbol=symbol, day=day)
    dest = tmp_dir / f"{symbol}-aggTrades-{day}.zip"
    with requests.get(url, stream=True, timeout=120) as resp:
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return dest


def read_aggtrades(zip_path: Path) -> pd.DataFrame:
    """
    Parse a daily aggTrades zip into a DataFrame with columns
    [ts_ms, price, quantity, taker_buy], sorted by (ts, agg_trade_id).

    taker_buy: True when the aggressor was a buyer. Binance records
    is_buyer_maker; the aggressor is the *taker*, so
    taker_buy = not is_buyer_maker.
    """
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"no csv in {zip_path}")
        raw = z.read(names[0])

    # Detect a header row by probing the first field of the first line.
    first_line = raw.split(b"\n", 1)[0].decode("utf-8", "replace")
    has_header = not first_line.split(",")[0].strip().lstrip("-").isdigit()

    df = pd.read_csv(
        io.BytesIO(raw), header=0 if has_header else None,
        names=None if has_header else AGGTRADE_COLS,
    )
    # Normalize column names regardless of header spelling variants.
    rename = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("transact_time", "timestamp", "time"):
            rename[col] = "transact_time"
        elif c == "is_buyer_maker":
            rename[col] = "is_buyer_maker"
    df = df.rename(columns=rename)

    out = pd.DataFrame({
        "ts_ms": df["transact_time"].astype("int64"),
        "price": df["price"].astype("float64"),
        "quantity": df["quantity"].astype("float64"),
        "taker_buy": ~df["is_buyer_maker"].astype(bool),
    })
    out = out.sort_values(["ts_ms"], kind="mergesort", ignore_index=True)
    return out


def bars_from_trades(trades: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Vectorized equivalent of collectors.bar_builder.BarBuilder over one day
    of trades. Produces the exact ohlcv_1m micro schema. Equivalence with
    BarBuilder is covered by tests/test_historical_micro.py.

    Per-minute tick statistics exclude the cross-minute price jump, matching
    BarAccumulator (each minute starts with last_price=None).
    """
    if trades.empty:
        return pd.DataFrame(columns=MICRO_BAR_COLS)

    t = trades
    minute_ms = (t["ts_ms"] // 60_000) * 60_000

    price = t["price"]
    prev_price = price.shift(1)
    same_minute = minute_ms.eq(minute_ms.shift(1))

    log_ret = np.where(same_minute & (prev_price > 0),
                       np.log(price / prev_price), 0.0)
    tick_dir = np.where(~same_minute, 0, np.sign(price - prev_price)).astype(np.int64)
    buy_qty = t["quantity"].where(t["taker_buy"], 0.0)

    work = pd.DataFrame({
        "minute_ms": minute_ms,
        "price": price,
        "quantity": t["quantity"],
        "buy_qty": buy_qty,
        "ret_sq": log_ret * log_ret,
        "tick_dir": tick_dir,
    })

    grp = work.groupby("minute_ms", sort=True)
    bars = pd.DataFrame({
        "time": grp["minute_ms"].first(),
        "open": grp["price"].first(),
        "high": grp["price"].max(),
        "low": grp["price"].min(),
        "close": grp["price"].last(),
        "volume": grp["quantity"].sum(),
        "buy_volume": grp["buy_qty"].sum(),
        "trade_count": grp["price"].count().astype("int64"),
        "tick_momentum": grp["tick_dir"].sum(),
        "realized_variance": grp["ret_sq"].sum(),
    }).reset_index(drop=True)

    bars["symbol"] = symbol
    bars["volume_delta"] = 2.0 * bars["buy_volume"] - bars["volume"]
    bars["spread"] = np.nan
    bars["bid_ask_imbalance"] = np.nan
    return bars[MICRO_BAR_COLS]


def load_day_to_db(bars: pd.DataFrame, conn) -> int:
    """Upsert one day of bars into ohlcv_1m. Returns rows written."""
    import psycopg2.extras

    rows = [
        (
            pd.Timestamp(int(r.time), unit="ms", tz="UTC").to_pydatetime(),
            r.symbol, r.open, r.high, r.low, r.close, r.volume,
            None if pd.isna(r.spread) else r.spread,
            None if pd.isna(r.bid_ask_imbalance) else r.bid_ask_imbalance,
            r.volume_delta, r.tick_momentum, r.realized_variance,
            int(r.trade_count),
        )
        for r in bars.itertuples()
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, DB_UPSERT, rows, page_size=1000)
    conn.commit()
    return len(rows)


def merge_to_strategy_parquet(symbols: list[str], raw_dir: Path,
                              out_dir: Path) -> None:
    """Concatenate daily parquets into the per-pair file the strategy reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        files = sorted((raw_dir / symbol).glob(f"{symbol}_*.parquet"))
        if not files:
            logger.warning(f"{symbol}: no daily parquets found, skipping merge")
            continue
        df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
        df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time")
        df["date"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        pair = SYMBOL_TO_PAIR.get(symbol, symbol)
        fname = pair.replace("/", "_").replace(":", "_") + "-1m-micro.parquet"
        out = out_dir / fname
        df.to_parquet(out, index=False)
        logger.info(
            f"{symbol}: merged {len(df)} 1m bars "
            f"({df['date'].min()} -> {df['date'].max()}) -> {out}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last day (inclusive), YYYY-MM-DD")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=STRATEGY_DIR)
    ap.add_argument("--load-db", action="store_true", help="upsert into TimescaleDB")
    ap.add_argument("--merge", action="store_true",
                    help="write per-pair merged parquet for the strategy")
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()

    days = pd.date_range(args.start, args.end, freq="D", tz="UTC")
    tmp_dir = args.raw_dir / "_zips"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    conn = None
    if args.load_db:
        import psycopg2
        conn = psycopg2.connect(DB_DSN)

    try:
        for symbol in args.symbols:
            day_dir = args.raw_dir / symbol
            day_dir.mkdir(parents=True, exist_ok=True)
            for day in days:
                day_str = day.strftime("%Y-%m-%d")
                parquet_path = day_dir / f"{symbol}_{day_str}.parquet"
                if parquet_path.exists():
                    logger.debug(f"{symbol} {day_str}: parquet exists, skipping")
                    continue
                try:
                    zip_path = download_day(symbol, day_str, tmp_dir)
                except requests.RequestException as e:
                    logger.error(f"{symbol} {day_str}: download failed: {e}")
                    continue
                if zip_path is None:
                    logger.warning(f"{symbol} {day_str}: not in archive (404)")
                    continue
                t0 = time.time()
                trades = read_aggtrades(zip_path)
                bars = bars_from_trades(trades, symbol)
                bars.to_parquet(parquet_path, index=False)
                if conn is not None and not bars.empty:
                    load_day_to_db(bars, conn)
                logger.info(
                    f"{symbol} {day_str}: {len(trades)} trades -> {len(bars)} bars "
                    f"in {time.time() - t0:.1f}s"
                )
                if not args.keep_zips:
                    zip_path.unlink()
    finally:
        if conn is not None:
            conn.close()

    if args.merge:
        merge_to_strategy_parquet(args.symbols, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
