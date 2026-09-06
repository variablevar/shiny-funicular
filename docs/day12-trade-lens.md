# Day 12 (follow-up): Multi-TF Trade Lens

See what each timeframe looked like during every trade, not just the 5m primary.

## Components

### 1. `services/trade_lens.py` — FastAPI microservice (port 8001)
Returns per-trade OHLCV candles at every timeframe, plus entry/exit markers,
FreqAI regime state, and micro-context from `audit_log`.

**Endpoints:**
- `GET /health` — health check
- `GET /trades?limit=N` — list recent trades
- `GET /trade/{trade_id}?timeframes=1m,5m,15m,1h,4h` — full detail
- `GET /candles?pair=BTC_USDT&tf=5m&from=...&to=...` — raw OHLCV
- `GET /dashboard-data?limit=5` — bundled data for dashboards

**Data sources (in order):**
1. Freqtrade REST API (`localhost:8080`)
2. TimescaleDB `ohlcv_*` tables
3. Local Freqtrade data/ feather files (fallback)

### 2. `monitoring/web/index.html` — Single-page UI (port 8088)
Dark-themed grid showing:
- Trade cards (left): side, pair, P&L, duration
- TF panels (right): 1m, 5m, 15m, 30m, 1h, 4h candles with entry/exit markers
- Audit log at bottom (regime, spread, volume_delta at entry)

Auto-refreshes every 30s.

## URLs
- **Web UI**: http://localhost:8088
- **Trade Lens API**: http://localhost:8001
- **OpenAPI docs**: http://localhost:8001/docs

## Setup
Already deployed via docker-compose. To restart:
```bash
cd /home/gt/Desktop/gt-quant
docker compose up -d trade_lens trade_lens_web
```

## Verified against Trade #1 (ETH long, 44min, -0.13%)
- 1m: 105 candles (every minute during trade + 30min padding)
- 5m: 21 candles
- 15m: 7 candles
- 1h: 1 candle
- 4h: 1 candle
- Audit: 1 entry showing BULL regime, 4bps spread, -60 volume delta at $2479.03

## Notes
- 1m data needs to be downloaded frequently (feather only goes to date of last download). 
  Run `docker exec gtquant-freqtrade freqtrade download-data --timeframes 1m --days 14` periodically.
- The web UI is plain HTML+JS — works in any browser, no plugins needed.
- Use the Infinity Grafana datasource if you want to render inside Grafana panels.
