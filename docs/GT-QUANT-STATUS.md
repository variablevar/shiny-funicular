# GT-QUANT — Status vs. Single-4080 Build Plan

**As of 2026-09-15.** Checked against `docs/GT-QUANT_Single-4080_Build_Plan.md`.
Legend: ✅ done & verified · 🟡 partial / in progress · ❌ not done · ⚠️ done but not effective

---

## 1. Scoreboard

| Plan item | Status | Effectiveness |
|---|---|---|
| §1–2 Stack + hardware lock (one 4080, GPU = LLM only) | ✅ | Working as designed — trading spine is 100% CPU, GPU only does LLM |
| §3 Multi-TF architecture (5m primary, 15m/30m/1h/4h informative) | ✅ | Live in FreqAI config; **1m not yet a training feature** (see §4 Day 2) |
| §4 Day 1 — Infra (Freqtrade, TimescaleDB, Redis, Grafana) | ✅ | All containers up; Redis deployed but unused ⚠️ |
| §4 Day 2 — Features, collectors, labels | 🟡 | Collectors + labels done; 1m micro features built but not yet in FreqAI |
| §4 Day 3 — FreqAI baselines per TF | ✅ | 3 identifiers trained; 960 features; but models don't beat fees ⚠️ |
| §4 Day 4 — Backtests + fee realism | ✅ | Done; results are the core problem (see §6 below) ⚠️ |
| §4 Day 5 — Regime model + risk | 🟡 | Rule-based regime live; KMeans classifier built but not wired; Nautilus not integrated |
| §4 Day 6 — Paper trading | ✅ | Dry-run live since 2026-09-05 (real Binance futures data, simulated orders) |
| §4 Day 7 — Week-1 checkpoint (burn-in, decision log, tag) | ❌ | Never done |
| §5 Day 8 — LLM dataset + fine-tune | ✅ | Done twice: Qwen2.5-7B v0.1/v0.2, then Qwen3-8B per plan |
| §5 Day 9 — Eval + serving | ✅ | Frozen benchmark operational; Ollama serving (vLLM deferred — pin conflict) |
| §5 Day 10 — Macro / on-chain / options features | 🟡 | Deribit options collector live; FRED macro + on-chain missing |
| §5 Day 11 — LLM↔quant fusion + shadow mode | 🟡 | Fusion logic + shadow logging live; fusion has no trade authority yet |
| §5 Day 12 — Adversarial tests | ✅ | 112 tests, now exercise real strategy code |
| §5 Day 13 — 24h burn-in + Telegram alerts | ❌ | Not done |
| §5 Day 14 — Code freeze `v0.1-paper` + TF decision log | ❌ | Not done |
| §6 Phase 3 — 70B burst rental | ❌ | Not applicable yet (14B/8B gate not passed) |

---

## 2. What's running right now (all verified live)

| Component | State |
|---|---|
| `gtquant-freqtrade` | Dry-run, 5m FreqAI/LightGBM, BTC+ETH, 100 USDT stakes — RUNNING |
| `gtquant-db` (TimescaleDB) | Healthy; all 7 TF hypertables + funding/OI + audit/shadow |
| `gtquant-live-collector` | 1m bars flowing (324k rows, fresh today) |
| `gtquant-deribit-options` | Options skew/IV collecting |
| `llm_service` :8000 | Serving **Qwen3-8B** (`gtquant-8b-qwen3-v0.1`), ~0.5s latency |
| Ollama | User-level process (not systemd — see §7 ops notes) |
| Shadow mode | **107k rows logged** — LLM-vs-quant agreement accumulating |

---

## 3. Fixed in this round (commits 981872d → 5c66e66)

1. **STRONG_BEAR short gate** — shorts were blocked in strong downtrends (`981872d`).
2. **Shadow mode never worked** — `int(NaN)` crash swallowed by fail-open; 0 rows ever until fixed (`981872d`).
3. **Collectors dockerized**, DB_DSN via env (`0fe9e48`).
4. **Adversarial suite rewritten** against real strategy code; 112 tests pass; STRONG_BEAR regression pinned (`6d4608e`).
5. **v0.3 dataset placeholders fixed** — real funding rates + real 5m RSI (`ce59548`).
6. **Qwen3-8B fine-tuned + deployed shadow-only** (`5c66e66`):
   - Benchmark: **86.4% overall** vs v0.2's 77.8%; 6/7 archetypes at 100%
   - strong_trend 36% → **59%** (still under the 70% authority gate → LLM stays advisory)

---

## 4. In progress (agents timed out, artifacts on disk, uncommitted)

**A. Aggression / fee tuning** (`GTQuantMultiTFTune`, `config_tune*.json`):
- Hyperopt landed: entry_threshold 0.00144, exit −0.00345, stoploss −15.6%, ROI ladder.
- Walk-forward (monthly OOS, fee-realistic):

| Config | Jun–Jul | Jul–Aug | Aug–Sep | Trades/fold |
|---|---|---|---|---|
| 5m current | — | −2.41% (223 trades) | — | high |
| 5m tuned | −2.29% | −1.47% | −0.15% | ~125 |
| **1h target** | **+0.41%** | −0.99% | **+0.24%** | ~34 |

- Trend: tuned 5m is improving (−2.3 → −0.15) but still negative. **1h is the only config with positive folds (2/3).**

**B. 1m microstructure features into FreqAI** (`collectors/historical_micro.py`, `strategies/micro_features.py`, `GTQuantMultiTFMicro.py`, tests written):
- Historical 1m micro bars built (20 MB parquet in `data/raw/micro_1m/`).
- A/B harness ready (`run_ab_micro_backtest.sh`, `analysis/ab_compare.py`, `micro_importance_check.py`).
- **A/B result not yet produced.**

---

## 5. Live trading record (dry-run, since 2026-09-05)

- **39 trades** (18 short / 21 long), +3.90 USDT total on ~100 USDT stakes (~+0.4% of wallet)
- Shorts active since the STRONG_BEAR fix (7 shorts on Sep 13 alone)
- Last 20 closed: 40% win rate, +0.20 USDT — many exits at −0.01%…−0.2% on model flips (fee churn)

---

## 6. The core problem, honestly

**Gross edge per trade (~0.04%) is below round-trip fees (~0.1%).** Day-4 numbers:

| Identifier | Trades | Gross PnL | Fees | Net PnL | Fee erosion |
|---|---|---|---|---|---|
| 5m primary | 473 | +17.5 | 38.9 | −21.4 | **222%** |
| 1h target | 1591 | +13.8 | 135.8 | −122.1 | **987%** |

Every tuning lever so far (thresholds, regime gates, hyperopt) trades frequency against this fee wall. The two open hopes for a real 5m edge: (a) 1m microstructure features improving prediction quality (A/B pending), (b) LLM fusion improving selectivity (shadow data accumulating, not yet authoritative).

**The plan's own gate said: don't touch the LLM until the quant layer shows positive after-fee Sharpe. That gate was never passed.** The LLM work is done anyway (fine — it's shadow-only), but the quant edge problem is still the ballgame.

---

## 7. Ops notes / known gaps

- Ollama runs as a user process, not systemd. On reboot: kill user instance, `sudo systemctl start ollama`, recreate models in system store (system store only has v0.2).
- `llm_service` is nohup'd uvicorn — not supervised, won't survive reboot.
- Freqtrade POSTs to the LLM every ~5s per pair (not per candle close) — noisy but harmless.
- Redis deployed, unused. `liquidations`/`orderbook_snapshots` tables have no writers.
- Adversarial tests now cover the real strategy, but circuit-breaker behavior is only unit-tested, not live-tested.
- KMeans regime classifier (`core/regime_classifier.py`) built, never wired in.

## 8. Recommended next steps (priority order)

1. **Finish the 1m-micro A/B** — run `run_ab_micro_backtest.sh`, check feature importance. This decides whether 5m can carry real edge.
2. **Decide primary TF with data** — 1h walk-forward is the only positive config; write `docs/day14-tf-decision.md`.
3. **Wire FRED macro features** (Day 10 remainder — cheap, CPU-only).
4. **Promote/demote LLM** after ~1 week of shadow data: agreement rate >75% gate.
5. **Day 13/14**: 24h burn-in with Telegram alerts, then `v0.1-paper` tag.
6. Commit the pending tuning + micro work (currently uncommitted on disk).
