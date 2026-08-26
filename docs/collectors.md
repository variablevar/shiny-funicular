# GT-Quant Live Collectors (Day 2)

Feeds real-time market data into TimescaleDB for the 1m microstructure layer.

## Components

| File | Role |
|---|---|
| `collectors/bar_builder.py` | Pure tick → 1m bar aggregation (OHLCV, volume delta, tick momentum, realized variance, spread, imbalance). Unit-tested, network-free. |
| `collectors/live_collector.py` | Production collector: Binance futures REST → BarBuilder → TimescaleDB. |

## Why REST polling instead of Cryptofeed websockets

The plan specifies Cryptofeed websockets. On this node (UK IP), Binance
**futures websocket hosts are region-blocked**: `wss://fstream.binance.com`
accepts the TCP/TLS handshake but never delivers a single message (verified
2026-08-26 for `aggTrade`, `depth`, combined `/stream`). Spot websockets
(`wss://stream.binance.com:9443`) work fine — this is the Binance UK
futures restriction.

The futures REST API (`https://fapi.binance.com`) is not blocked, so
`live_collector.py` polls it every 5s:

- `/fapi/v1/aggTrades` (fromId pagination, no gaps) → 1m bars
- `/fapi/v1/ticker/bookTicker` → spread + bid/ask imbalance
- `/fapi/v1/premiumIndex` (60s) → `funding_rates`
- `/fapi/v1/openInterest` (60s) → `open_interest`

If the production node moves to a non-blocked region, swap the trade/book
polling for Cryptofeed websockets; the BarBuilder and DB writer stay as-is.

## Validation (2026-08-26)

Built bars vs exchange-official 1m klines, BTCUSDT 11:17 UTC:

| Field | Official | Collector | Match |
|---|---|---|---|
| Open | 78700.00 | 78700.00 | exact |
| High | 78703.70 | 78703.70 | exact |
| Low | 78668.70 | 78668.70 | exact |
| Close | 78680.00 | 78680.00 | exact |
| Volume | 39.18 | 39.18 | exact |
| Trades | 1737 | 759 | n/a — aggTrades pre-aggregates |

Meets the plan's Day 6 criterion (OHLCV within 0.1% of exchange data).

## Running

```bash
source venv/bin/activate
python collectors/live_collector.py                 # forever
python collectors/live_collector.py --duration 180  # 3-min test

# 2h burn-in (Day 2 exit criterion):
nohup python collectors/live_collector.py > logs/collector.log 2>&1 &
```

Verify bars landing:

```sql
SELECT time, symbol, close, volume, trade_count, spread, volume_delta
FROM ohlcv_1m ORDER BY time DESC LIMIT 10;
```

## Rate-limit budget (per symbol, per minute)

- aggTrades: 12 calls × weight 20 = 240
- bookTicker: 12 × weight 2 = 24
- premiumIndex + openInterest: 2 × weight ~10 = 20

Total ~284/min of the 2400/min futures limit — ~12%, comfortable.

## Known limits / next steps

- 5s polling means spread/imbalance are point samples per poll, not per tick.
  Acceptable for 1m bars; websocket depth would improve this on unblocked IPs.
- Dropped-bar retry is not implemented on DB write failure (logged + dropped).
- Day 6 will add the CCXT cross-check as a standing audit job.
