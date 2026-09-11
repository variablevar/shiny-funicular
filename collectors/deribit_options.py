#!/usr/bin/env python3
"""
GT-Quant collector (Day 10 follow-up): Deribit public options API.

Fetches BTC + ETH option chain summary and writes to TimescaleDB:
  - options_chain_snapshot: instruments + greeks + mark_iv per (currency, expiry)
  - options_atm_iv:        25-delta skew + ATM IV per (currency, timestamp)

The Deribit v2 public API is completely open - no API key, no auth, no rate
limit blocking (rate limit 100 req/10s for public endpoints). Endpoints used:

  GET /public/get_instruments?currency=BTC&kind=option&expired=false
  GET /public/get_book_summary_by_currency?currency=BTC&kind=option
  GET /public/get_volatility_index_data?currency=BTC   # DVOL index

Key derived metrics for the strategy:
  - ATM_IV (forward-looking 30d implied vol)
  - 25-delta skew (put IV - call IV at 25-delta strikes; risk-appetite proxy)
  - Term structure (front vs back month IV ratio)
  - DVOL (Deribit's own vol index, free)

Output goes to TimescaleDB so FreqAI's @informative can read it as 1h/4h
columns. Schema is defined in db/schema.sql.

Usage:
    source venv/bin/activate
    python collectors/deribit_options.py                 # run forever
    python collectors/deribit_options.py --duration 180  # 3-min test
"""
from __future__ import annotations

import argparse
import os
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
import psycopg2.extras
import requests
from loguru import logger

# Deribit public API (no auth needed)
DERIBIT_BASE = "https://www.deribit.com/api/v2"
DB_DSN = os.getenv("DB_DSN", "host=localhost dbname=gtquant user=gtquant password=gtquant_local")

POLL_INTERVAL = 60            # seconds between snapshots
RATE_LIMIT_BACKOFF = 10         # seconds if we hit 429
CURRENCIES = ["BTC", "ETH"]

logger.info(f"Deribit options collector starting (poll every {POLL_INTERVAL}s)")


# ─── Deribit API helpers ─────────────────────────────────────────────── #

def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """GET from Deribit public API with rate-limit handling."""
    url = f"{DERIBIT_BASE}{path}"
    while True:
        try:
            r = requests.get(url, params=params or {}, timeout=10)
            if r.status_code == 429:
                logger.warning(f"Rate limited on {path}; sleeping {RATE_LIMIT_BACKOFF}s")
                time.sleep(RATE_LIMIT_BACKOFF)
                continue
            r.raise_for_status()
            data = r.json()
            if "result" not in data:
                logger.error(f"Deribit error on {path}: {data}")
                return None
            result = data["result"]
            # Some endpoints (e.g., volatility_index_data) nest arrays under "data"
            if isinstance(result, dict) and "data" in result:
                return result["data"]
            return result
        except requests.RequestException as e:
            logger.warning(f"Request error {path}: {e}; retrying in {RATE_LIMIT_BACKOFF}s")
            time.sleep(RATE_LIMIT_BACKOFF)


def get_instruments(currency: str) -> List[Dict[str, Any]]:
    """List all active option instruments for a currency."""
    return _get("/public/get_instruments", {"currency": currency, "kind": "option"}) or []


def get_book_summary(currency: str) -> List[Dict[str, Any]]:
    """Get summary (greeks, IV, mark price, OI, volume) for all options."""
    return _get("/public/get_book_summary_by_currency", {"currency": currency, "kind": "option"}) or []


def get_volatility_index(currency: str) -> List[Dict[str, Any]]:
    """DVOL-style realized vol index (last 24h, 1h resolution)."""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 24 * 3600 * 1000  # last 24h
    res = _get("/public/get_volatility_index_data", {
        "currency": currency,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": "60",  # 1h
    })
    # DVOL data: list of [timestamp, open, high, low, close] (5-tuples)
    # Also defensively skip any error markers (None, -1, dict)
    if res:
        res = [r for r in res if isinstance(r, (list, tuple)) and len(r) >= 2
               and isinstance(r[0], int) and r[0] > 0]
    return res or []


# ─── Database helpers ─────────────────────────────────────────────────── #

def _connect():
    return psycopg2.connect(DB_DSN)


def ensure_schema():
    """Create options tables if not present (idempotent)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS options_atm_iv (
                time           TIMESTAMPTZ NOT NULL,
                currency       TEXT NOT NULL,
                atm_iv_pct     DOUBLE PRECISION,
                dvol_pct       DOUBLE PRECISION,
                skew_25d       DOUBLE PRECISION,
                term_ratio     DOUBLE PRECISION,
                front_iv_pct   DOUBLE PRECISION,
                back_iv_pct    DOUBLE PRECISION,
                PRIMARY KEY (time, currency)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS options_atm_iv_currency_time_idx
                ON options_atm_iv (currency, time DESC);
        """)
        # Per-instrument snapshot (for deep dives, not FreqAI features)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS options_chain_snapshot (
                time           TIMESTAMPTZ NOT NULL,
                currency       TEXT NOT NULL,
                instrument     TEXT NOT NULL,
                strike         DOUBLE PRECISION,
                option_type    TEXT,
                expiry         TIMESTAMPTZ,
                mark_iv_pct    DOUBLE PRECISION,
                mark_price     DOUBLE PRECISION,
                underlying     DOUBLE PRECISION,
                open_interest  DOUBLE PRECISION,
                volume_24h     DOUBLE PRECISION,
                delta          DOUBLE PRECISION,
                PRIMARY KEY (time, instrument)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS options_chain_snapshot_currency_time_idx
                ON options_chain_snapshot (currency, time DESC);
        """)
        conn.commit()


def _nearest_expiry(instruments: List[Dict[str, Any]]) -> str:
    """Return YYYY-MM-DD of the nearest expiry (excluding 0DTE if > 7 days out)."""
    if not instruments:
        return ""
    today = datetime.now(timezone.utc).date()
    expiries = sorted({
        datetime.fromtimestamp(i["expiration_timestamp"] / 1000, tz=timezone.utc).date()
        for i in instruments
        if datetime.fromtimestamp(i["expiration_timestamp"] / 1000, tz=timezone.utc).date() >= today
    })
    if not expiries:
        return ""
    return expiries[0].isoformat()


def _find_atm_strike(marks: List[Dict[str, Any]], underlying: float) -> Optional[float]:
    """Find the strike closest to ATM (closest-to-spot call or put)."""
    if not marks or underlying <= 0:
        return None
    best = None
    best_diff = float("inf")
    for m in marks:
        strike = m.get("strike")
        if strike is None:
            continue
        diff = abs(strike - underlying)
        if diff < best_diff:
            best_diff = diff
            best = strike
    return best


def _find_25d_strikes(marks: List[Dict[str, Any]], underlying: float) -> Dict[str, Optional[float]]:
    """Find strikes closest to 25-delta for puts and calls (proxy at +/- 5% OTM)."""
    if not marks or underlying <= 0:
        return {"put_25d_strike": None, "call_25d_strike": None}
    put_target = underlying * 0.97  # 3% OTM put
    call_target = underlying * 1.03  # 3% OTM call
    put_best, put_diff = None, float("inf")
    call_best, call_diff = None, float("inf")
    for m in marks:
        s = m.get("strike")
        opt = m.get("option_type", "")
        if s is None:
            continue
        if opt == "put":
            diff = abs(s - put_target)
            if diff < put_diff:
                put_diff = diff
                put_best = s
        elif opt == "call":
            diff = abs(s - call_target)
            if diff < call_diff:
                call_diff = diff
                call_best = s
    return {"put_25d_strike": put_best, "call_25d_strike": call_best}


def _parse_instrument(name: str) -> Dict[str, Any]:
    """Parse Deribit instrument name like 'BTC-25SEP26-105000-C' into pieces."""
    # Format: <CCY>-<DDMMMYY>-<STRIKE>-<C|P>
    import re
    m = re.match(r"^([A-Z]+)-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])$", name)
    if not m:
        return {}
    ccy, exp_str, strike, opt = m.groups()
    return {"ccy": ccy, "expiry_str": exp_str, "strike": float(strike), "option_type": "call" if opt == "C" else "put"}


def _filter_chain(summary: list, underlying: float) -> list:
    """Parse instrument_name and keep only liquid ATM-ish options with sane IV."""
    out = []
    for m in summary:
        name = m.get("instrument_name", "")
        pat = re.match(r"([A-Z]+)-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])$", name)
        if not pat:
            continue
        strike_s = pat.group(3)
        iv = m.get("mark_iv", 0)
        opt_type = "call" if pat.group(4) == "C" else "put"
        # Filter for sane IV (10-200%) and active liquidity
        bid = m.get("bid_price", 0) or 0
        ask = m.get("ask_price", 0) or 0
        if iv is None or iv < 10 or iv > 200 or bid == 0 or ask == 0:
            continue
        out.append({
            "strike": float(strike_s),
            "iv": float(iv),
            "type": opt_type,
            "mid": (bid + ask) / 2,
        })
    return out


def compute_metrics(currency: str) -> Dict[str, Any]:
    """Pull live option chain and compute ATM IV + 25-delta skew + term ratio."""
    summary = get_book_summary(currency)
    if not summary:
        logger.warning(f"{currency}: no book summary returned")
        return {}

    # Filter to liquid options with sane IV (10-200%, has bid/ask)
    chain = _filter_chain(summary, 0)  # underlying not needed here
    if not chain:
        logger.warning(f"{currency}: no liquid options with sane IV")
        return {}

    # Underlying price from any summary entry
    underlying = summary[0].get("underlying_price") or 0
    if not underlying:
        for m in summary:
            if m.get("underlying_price"):
                underlying = m["underlying_price"]
                break
    if not underlying:
        logger.warning(f"{currency}: no underlying price")
        return {}

    # ATM IV: median of strikes within ±1% of underlying
    atm_strike_lo = underlying * 0.99
    atm_strike_hi = underlying * 1.01
    atm_ivs = [c["iv"] for c in chain if atm_strike_lo <= c["strike"] <= atm_strike_hi]
    atm_iv = float(np.median(atm_ivs)) if atm_ivs else None

    # 25-delta skew: 3% OTM put IV - 3% OTM call IV
    put_target = underlying * 0.97
    call_target = underlying * 1.03
    near_puts = sorted([c for c in chain if c["type"] == "put" and 0.92 * underlying < c["strike"] < underlying],
                       key=lambda c: abs(c["strike"] - put_target))
    near_calls = sorted([c for c in chain if c["type"] == "call" and underlying < c["strike"] < 1.08 * underlying],
                        key=lambda c: abs(c["strike"] - call_target))
    put_iv = near_puts[0]["iv"] if near_puts else None
    call_iv = near_calls[0]["iv"] if near_calls else None
    skew_25d = (put_iv - call_iv) if (put_iv and call_iv) else None

    # Term structure: front month (nearest expiry) vs back month (30-60d)
    instruments = get_instruments(currency)
    exp_ts_map = {i["instrument_name"]: i["expiration_timestamp"] for i in instruments}
    today = datetime.now(timezone.utc).date()
    by_exp: dict = {}
    for m in summary:
        ts = exp_ts_map.get(m.get("instrument_name"))
        if ts is None:
            continue
        iv = m.get("mark_iv")
        bid = m.get("bid_price", 0) or 0
        ask = m.get("ask_price", 0) or 0
        if iv is None or iv < 10 or iv > 200 or bid == 0 or ask == 0:
            continue
        exp_date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        by_exp.setdefault(dte, []).append(iv)
    front_iv = float(np.median(by_exp[min(by_exp)])) if by_exp else None
    back_window = [d for d in by_exp if 30 <= d <= 60]
    back_iv = float(np.median(by_exp[min(back_window)])) if back_window else None
    term_ratio = (front_iv / back_iv) if (front_iv and back_iv and back_iv > 0) else None

    # DVOL index (last closing value, [time, open, high, low, close])
    dvol = None
    try:
        dvol_series = get_volatility_index(currency)
        if dvol_series:
            last = dvol_series[-1]
            # DVOL data structure: [timestamp, open, high, low, close]
            if isinstance(last, (list, tuple)) and len(last) >= 5:
                dvol = last[4]  # close value
            elif isinstance(last, (list, tuple)) and len(last) >= 2:
                dvol = last[1]
    except (KeyError, IndexError, TypeError):
        dvol = None

    return {
        "currency": currency,
        "underlying": underlying,
        "atm_iv_pct": atm_iv if atm_iv is not None else None,

        "dvol_pct": dvol if dvol is not None else None,

        "skew_25d": skew_25d,
        "term_ratio": term_ratio,
        "front_iv_pct": front_iv if front_iv is not None else None,

        "back_iv_pct": back_iv if back_iv is not None else None,
        "n_instruments": len(summary),
    }


def write_metrics(metrics: Dict[str, Any]):
    """Upsert a single row per currency into options_atm_iv."""
    if not metrics:
        return
    now = datetime.now(timezone.utc)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO options_atm_iv
                (time, currency, atm_iv_pct, dvol_pct, skew_25d,
                 term_ratio, front_iv_pct, back_iv_pct)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, currency) DO UPDATE SET
                atm_iv_pct = EXCLUDED.atm_iv_pct,
                dvol_pct   = EXCLUDED.dvol_pct,
                skew_25d   = EXCLUDED.skew_25d,
                term_ratio = EXCLUDED.term_ratio,
                front_iv_pct = EXCLUDED.front_iv_pct,
                back_iv_pct  = EXCLUDED.back_iv_pct
        """, (now, metrics["currency"],
              metrics.get("atm_iv_pct"), metrics.get("dvol_pct"),
              metrics.get("skew_25d"), metrics.get("term_ratio"),
              metrics.get("front_iv_pct"), metrics.get("back_iv_pct")))
        conn.commit()


# ─── Main loop ──────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=0, help="Seconds to run (0=forever)")
    parser.add_argument("--once", action="store_true", help="Single poll then exit")
    args = parser.parse_args()

    ensure_schema()
    logger.info("Schema ensured. Starting polling loop.")

    deadline = time.time() + args.duration if args.duration else None
    n_cycles = 0
    while True:
        for ccy in CURRENCIES:
            try:
                t0 = time.time()
                m = compute_metrics(ccy)
                if m:
                    write_metrics(m)
                    # Safe log: handle None values gracefully
                    atm = m.get('atm_iv_pct') or 0
                    dvol = m.get('dvol_pct') or 0
                    skew = m.get('skew_25d') or 0
                    term = m.get('term_ratio') or 1
                    logger.info(
                        f"{ccy}: ATM={atm:.1f}% "
                        f"DVOL={dvol:.1f}% "
                        f"skew={skew:+.3f} "
                        f"term_ratio={term:.2f} "
                        f"({m.get('n_instruments', 0)} instruments, "
                        f"{time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                logger.error(f"{ccy} poll failed: {e}")
        n_cycles += 1
        if args.once:
            break
        if deadline and time.time() >= deadline:
            logger.info(f"Duration reached; exiting after {n_cycles} cycles")
            break
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
