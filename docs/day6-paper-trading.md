# Day 6 — Paper Trading Prep + 1m Microstructure Filter

## Status: offline parts done; live burn-in BLOCKED on testnet keys

Day 6's core is a 4-hour paper-trading session on Binance testnet. That needs
`exchange.key`/`exchange.secret` in `ft_userdata/user_data/config.json` —
create them at https://testnet.binancefuture.com and drop them in (never
commit them; `config.json` is safe because the keys land in your local file,
but rotate before pushing anywhere public).

## What was built offline

### Custom FreqAI image (`ft_userdata/Dockerfile`)

`gtquant-freqai:latest` = `freqtradeorg/freqtrade:stable_freqai` +
`psycopg2-binary`. The strategy layer can now read TimescaleDB directly
(host: `host.docker.internal:5432`). Build: `docker-compose build freqtrade`.

### 1m microstructure entry filter (`confirm_trade_entry`)

Before any entry, the strategy reads the latest 1m bar from `ohlcv_1m`
(written by `collectors/live_collector.py`) and rejects the entry when:

- **spread > 0.05%** — poor execution conditions, or
- **volume delta < -2σ** of the last 60 1m bars — sell-pressure spike

A rejection makes Freqtrade retry next candle → the 1–2 candle entry delay
the plan specifies. **Fails open** when no micro data is available (per the
Day 12 adversarial test: micro filter failing must not halt trading).

**Live/dry-run only.** The hook returns early in backtest/hyperopt — the DB
holds *live* data, so consulting it during a backtest would be lookahead bias.

### Audit logging (`_audit`)

Every entry decision (approved / rejected_spread / rejected_volume_delta)
writes a row to the TimescaleDB `audit_log` hypertable with pair, timeframe,
side, regime, 1m spread, 1m volume delta, decision, and price. Exit rows +
full P&L join once paper trading runs.

## Verify (once testnet keys are in)

```bash
# 1. collector running (feeds ohlcv_1m)
python collectors/live_collector.py &

# 2. dry-run the strategy
cd ft_userdata && docker-compose up -d freqtrade

# 3. watch decisions
docker exec gtquant-db psql -U gtquant -d gtquant \
  -c "SELECT time, pair, risk_decision, spread_1m, regime FROM audit_log ORDER BY time DESC LIMIT 10;"
```

## Day 6 exit criteria (from plan)

- [ ] Paper account balance correct after trades
- [ ] Audit trail complete (1m/5m/regime context per trade)
- [ ] 1m bar builder validated (done in Day 2 — OHLCV exact match)
- [ ] 4h stable run without intervention
