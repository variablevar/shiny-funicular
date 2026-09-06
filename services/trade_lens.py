#!/usr/bin/env python3
"""
Day 12 follow-up: Multi-Timeframe Trade Lens service.

Provides per-trade OHLCV snapshots across all timeframes (1m/5m/15m/30m/1h/4h/1d)
so the Grafana dashboard can show what each timeframe looked like during the
trade's life. Also exposes entry/exit markers, predicted FreqAI signal,
regime state, and 1m microstructure.

Endpoints:
    GET  /health
    GET  /trades?limit=N                -> recent trades (from Freqtrade)
    GET  /trade/{trade_id}              -> trade detail + per-TF candles
    GET  /candles?pair=BTC_USDT&tf=1m&from=...&to=...
                                        -> OHLCV candles for a window
    GET  /dashboard-data?limit=5        -> bundled data for Grafana panel

Data sources:
    - Freqtrade REST API (localhost:8080) for trades + open positions
    - FreqAI audit_log in TimescaleDB for entry micro-context
    - ohlcv_{1m,5m,15m,30m,1h,4h,1d} feathers in TimescaleDB (preferred)
    - Fallback: Freqtrade local data/ feather files (when DB empty)
    - FreqAI predictions table (when populated) for prediction overlay

Run:
    source venv/bin/activate
    uvicorn services.trade_lens:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

# Freqtrade REST API (in-network via Docker hostname or localhost from host)
FREQTRADE_URL = os.getenv("FREQTRADE_URL", "http://localhost:8080")
FREQTRADE_AUTH = ("gtquant", "gtquant")

# TimescaleDB (read-only queries for candles)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "gtquant")
DB_USER = os.getenv("DB_USER", "gtquant")
DB_PASS = os.getenv("DB_PASS", "gtquant_local")

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
SYMBOL_MAP = {"BTC/USDT:USDT": "BTCUSDT", "ETH/USDT:USDT": "ETHUSDT"}

app = FastAPI(
    title="GT-Quant Trade Lens",
    description="Multi-timeframe trade visualization service",
    version="0.1.0",
)


# ─── helpers ───
def _db_connect():
    import psycopg2
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS, connect_timeout=5,
    )


def _freqtrade(path: str, **params) -> Any:
    r = requests.get(f"{FREQTRADE_URL}{path}", auth=FREQTRADE_AUTH,
                    params=params, timeout=5)
    r.raise_for_status()
    return r.json()


def _iso(ts: str) -> str:
    """Normalize a Freqtrade timestamp to ISO 8601."""
    try:
        # Freqtrade returns 'YYYY-MM-DD HH:MM:SS'
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        return ts


def _candles_feather(symbol: str, tf: str, ts_from: str, ts_to: str) -> list[dict]:
    """Fallback: read candles from Freqtrade local feather files."""
    # symbol: BTCUSDT -> filename: BTC_USDT_USDT-{tf}-futures.feather
    parts = {"BTCUSDT": "BTC_USDT_USDT", "ETHUSDT": "ETH_USDT_USDT"}
    fname_part = parts.get(symbol)
    if not fname_part:
        return []
    path = Path("/app/ft_userdata/user_data/data/binance/futures") / f"{fname_part}-{tf}-futures.feather"
    if not path.exists():
        # Try alternate mount path (inside Docker container)
        for alt in [Path("/freqtrade/user_data/data/binance/futures") / f"{fname_part}-{tf}-futures.feather",
                    Path(f"/app/ft_userdata/user_data/data/binance/futures") / f"{fname_part}-{tf}-futures.feather"]:
            if alt.exists():
                path = alt
                break
    if not path.exists():
        return []
    try:
        df = pd.read_feather(path)
        if "date" not in df.columns:
            return []
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None) if df["date"].dt.tz is None else pd.to_datetime(df["date"]).dt.tz_convert(None)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
        ts_from_dt = pd.to_datetime(ts_from).tz_localize("UTC") if pd.to_datetime(ts_from).tz is None else pd.to_datetime(ts_from)
        ts_to_dt = pd.to_datetime(ts_to).tz_localize("UTC") if pd.to_datetime(ts_to).tz is None else pd.to_datetime(ts_to)
        df = df[(df["date"] >= ts_from_dt) & (df["date"] <= ts_to_dt)]
        if df.empty:
            return []
        out = df.rename(columns={"date": "time", "volume": "volume"})[["time", "open", "high", "low", "close", "volume"]].copy()
        out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return out.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"candles_feather {path}: {e}")
        return []


def _candles_db(symbol: str, tf: str, ts_from: str, ts_to: str) -> list[dict]:
    """Fetch OHLCV candles. Tries TimescaleDB first, falls back to local feathers."""
    table = f"ohlcv_{tf}"
    rows = []
    try:
        with _db_connect() as conn:
            df = pd.read_sql(
                f"""
                SELECT time, open, high, low, close, volume
                FROM {table}
                WHERE symbol = %s AND time >= %s AND time <= %s
                ORDER BY time ASC
                """,
                conn, params=(symbol, ts_from, ts_to),
            )
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            rows = df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"candles_db {table} {symbol}: {e}")

    if not rows:
        rows = _candles_feather(symbol, tf, ts_from, ts_to)
    return rows


def _audit_for_trade(symbol: str, ts_from: str, ts_to: str) -> list[dict]:
    """Fetch audit_log entries during a trade window."""
    try:
        with _db_connect() as conn:
            df = pd.read_sql(
                """
                SELECT time, signal_type, regime, spread_1m, volume_delta_1m,
                       risk_decision, fill_price
                FROM audit_log
                WHERE pair = %s AND time >= %s AND time <= %s
                ORDER BY time ASC
                """,
                conn, params=(symbol, ts_from, ts_to),
            )
        if df.empty:
            return []
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning(f"audit {symbol}: {e}")
        return []


# ─── endpoints ───
@app.get("/health")
def health():
    """Health check + DB connectivity test."""
    db_ok = False
    try:
        with _db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.warning(f"DB ping failed: {e}")
    freqtrade_ok = False
    try:
        _freqtrade("/api/v1/ping")
        freqtrade_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "freqtrade": freqtrade_ok,
        "timeframes_supported": TIMEFRAMES,
    }


@app.get("/trades")
def list_trades(limit: int = Query(20, ge=1, le=200)):
    """List recent trades from Freqtrade REST API."""
    try:
        body = _freqtrade("/api/v1/trades", limit=limit)
        trades = body.get("trades", body) if isinstance(body, dict) else body
        return {"count": len(trades), "trades": trades}
    except Exception as e:
        raise HTTPException(500, f"freqtrade /trades failed: {e}")


@app.get("/trade/{trade_id}")
def trade_detail(trade_id: int, timeframes: str = "1m,5m,15m,1h,4h"):
    """Per-trade detail: metadata + per-TF candles around the trade window.

    Returns 1.5x trade duration of candles on each side so the entry/exit
    markers are visible in context.
    """
    try:
        body = _freqtrade(f"/api/v1/trades", limit=200)
        trades = body.get("trades", body) if isinstance(body, dict) else body
        trade = next((t for t in trades if t.get("trade_id") == trade_id), None)
        if not trade:
            raise HTTPException(404, f"trade {trade_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"freqtrade /trades failed: {e}")

    # Pull symbol + window
    symbol = SYMBOL_MAP.get(trade["pair"])
    if not symbol:
        raise HTTPException(400, f"unknown pair {trade['pair']}")

    open_ts = trade["open_date"]               # 'YYYY-MM-DD HH:MM:SS'
    close_ts = trade.get("close_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    duration_min = trade.get("trade_duration", 0)
    if duration_min <= 0:
        duration_min = 30  # fallback for open trades

    # Pre/post window for context: 50% on each side, min 30min, max 4h
    pad_min = max(30, min(240, duration_min // 2))
    open_dt = datetime.strptime(open_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    close_dt = datetime.strptime(close_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    from_dt = open_dt - pd.Timedelta(minutes=pad_min)
    to_dt = close_dt + pd.Timedelta(minutes=pad_min)

    # Per-TF candles
    tfs = [t.strip() for t in timeframes.split(",") if t.strip() in TIMEFRAMES]
    candles = {}
    for tf in tfs:
        candles[tf] = _candles_db(
            symbol, tf,
            from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

    # Audit entries during the trade
    audit = _audit_for_trade(
        symbol,
        from_dt.strftime("%Y-%m-%d %H:%M:%S"),
        to_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Trade markers for Grafana overlay
    markers = [
        {
            "type": "entry",
            "time": _iso(open_ts),
            "price": trade["open_rate"],
            "side": "short" if trade.get("is_short") else "long",
            "tag": trade.get("enter_tag"),
        },
        {
            "type": "exit",
            "time": _iso(close_ts),
            "price": trade.get("close_rate"),
            "reason": trade.get("exit_reason"),
            "profit_pct": trade.get("profit_pct"),
            "profit_abs": trade.get("profit_abs"),
        },
    ]

    return {
        "trade_id": trade_id,
        "pair": trade["pair"],
        "symbol": symbol,
        "side": "short" if trade.get("is_short") else "long",
        "open_time": _iso(open_ts),
        "close_time": _iso(close_ts),
        "duration_min": duration_min,
        "window": {
            "from": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "entry": {
            "time": _iso(open_ts),
            "price": trade["open_rate"],
            "tag": trade.get("enter_tag"),
        },
        "exit": {
            "time": _iso(close_ts),
            "price": trade.get("close_rate"),
            "reason": trade.get("exit_reason"),
            "profit_pct": trade.get("profit_pct"),
            "profit_abs": trade.get("profit_abs"),
        },
        "markers": markers,
        "candles": candles,
        "audit_log": audit,
    }


@app.get("/candles")
def candles(
    pair: str = Query(..., description="BTC_USDT or ETH_USDT"),
    tf: str = Query("5m", description="1m|5m|15m|30m|1h|4h|1d"),
    ts_from: Optional[str] = Query(None, alias="from"),
    ts_to: Optional[str] = Query(None, alias="to"),
    limit: int = Query(500, ge=1, le=5000),
):
    """Raw OHLCV candles for a pair/timeframe."""
    symbol = pair.upper()
    if tf not in TIMEFRAMES:
        raise HTTPException(400, f"tf must be one of {TIMEFRAMES}")

    # Default window: last 24h
    if not ts_to:
        ts_to = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if not ts_from:
        # 24h ago
        from datetime import timedelta
        ts_from = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

    rows = _candles_db(symbol, tf, ts_from, ts_to)
    # Cap to limit
    rows = rows[-limit:] if len(rows) > limit else rows
    return {"symbol": symbol, "tf": tf, "count": len(rows), "candles": rows}


@app.get("/dashboard-data")
def dashboard_data(limit: int = Query(5, ge=1, le=20)):
    """
    Bundled data for the Grafana 'Trade Lens' dashboard.
    Returns a compact summary of recent trades with per-TF OHLCV
    snapshots in a Grafana-friendly shape.
    """
    trades_body = _freqtrade("/api/v1/trades", limit=limit)
    trades = trades_body.get("trades", trades_body) if isinstance(trades_body, dict) else trades_body

    out = []
    for t in trades:
        tid = t["trade_id"]
        try:
            detail = trade_detail(tid)
            out.append(detail)
        except Exception as e:
            logger.warning(f"trade {tid} detail failed: {e}")
            out.append({
                "trade_id": tid,
                "pair": t.get("pair"),
                "error": str(e),
            })

    return {"count": len(out), "trades": out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
