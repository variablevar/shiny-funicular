  
 Here is your **updated GT-QUANT 14-Day Build Plan**. Every day now includes explicit 1m→30m timeframe architecture, granular feature/label specifications per granularity, and expanded technical detail. All original 1h/4h/1d content is preserved and enhanced.

---

# GT-QUANT: UPDATED 14-DAY BUILD PLAN (Multi-Timeframe Edition)
## Build nothing you can import. Trade every granularity from microstructure to macro.

---

## THE STACK (What You Use vs. What You Build)

| Layer | Use This (Existing) | Build This (Custom) |
|---|---|---|
| **Trading Engine** | **Freqtrade** — backtesting, paper/live trading, WebUI, Telegram, Hyperopt | Strategy logic, custom features, multi-timeframe fusion |
| **ML Models** | **FreqAI** — LightGBM/XGBoost/PyTorch, auto-retraining, multi-timeframe feature expansion | Feature definitions per TF, label logic per horizon, TF-specific model configs |
| **Data Ingestion** | **CCXT** + **Binance Public Data Archive** (free bulk historical) | Custom funding/OI/liquidation feeders per TF |
| **Real-Time L2 Data** | **Cryptofeed** — WebSocket trades, depth, bookTicker, funding, OI | Storage adapter to TimescaleDB; 1m bar construction from tick stream |
| **Features** | **pandas-ta** (130+ indicators) + **TA-Lib** | Multi-TF feature fusion (1m microstructure → 5m primary → 15m/30m context → 1h/4h regime) |
| **Risk Engine** | **NautilusTrader `nautilus-risk`** (Rust core, Python bindings) | Per-timeframe limit configuration, TF-aware position sizing |
| **Database** | **TimescaleDB** — time-series PostgreSQL | Unified multi-TF schema with hypertables per granularity |
| **LLM Fine-Tuning** | **Unsloth** — 2× faster QLoRA, exports to Ollama/vLLM | Dataset curation with TF-specific reasoning chains |
| **LLM Inference** | **Ollama** (research/4080) + **vLLM** (production/4080S) | Prompt templates per regime/TF, JSON parsers with TF context |
| **Experiment Tracking** | **MLflow** — model registry, versioning, artifact storage | Per-timeframe experiment groups, promotion pipeline logic |
| **Dashboard** | **Freqtrade WebUI** + **Grafana** | Custom multi-TF panel: 1m order book heatmap, 5m signal strength, 15m trend bias, 1h/4h regime |
| **Orchestration** | **Prefect** — lightweight Python workflow scheduling | Multi-TF data pipeline DAGs, retraining schedules per model |

---

## HARDWARE LOCK

| Machine | Role | Runs |
|---|---|---|
| **4080 SUPER** | Production | Freqtrade live/paper (5m primary), vLLM serving, Nautilus risk engine, 1m→5m real-time bar builder |
| **4080** | Research | Ollama (7B/13B), backtests across all TFs, feature engineering, LoRA experiments, Grafana |
| **4×3090** | Training | Unsloth QLoRA fine-tuning (7B→70B), MLflow tracking, multi-TF dataset generation |
| **CPU/RAM** | Database | TimescaleDB (1m/5m/15m/30m/1h/4h/1d hypertables), Redis (1m tick cache), MLflow backend, Prefect server |

---

## PRE-FLIGHT (Before Day 1)

- [ ] Ubuntu 22.04 headless on all nodes
- [ ] CUDA 12.4+ verified (`nvidia-smi`)
- [ ] Docker + Docker Compose installed
- [ ] SSH + Tailscale/WireGuard for remote access
- [ ] GitHub repo `gt-quant` created, `main` protected
- [ ] Binance **testnet** API keys (paper trading only)
- [ ] Shared NAS or fast rsync between 3090s ↔ 4080s

---

## MULTI-TIMEFRAME ARCHITECTURE (The Foundation)

Before Day 1 begins, you must lock the timeframe hierarchy. This plan trades a **5m primary** timeframe, informed by **1m microstructure**, filtered by **15m/30m context**, and regime-classified by **1h/4h/1d**.

### Timeframe Responsibilities

| Timeframe | Role | Data Source | Feature Focus | Label Horizons | Fee Impact* | Typical Hold |
|---|---|---|---|---|---|---|
| **1m** | Microstructure / Execution Precision | Cryptofeed tick→1m bars | Spread, bid-ask imbalance, VPIN, volume delta, realized variance, tick momentum, CVD per minute | `future_return_1m`, `future_volatility_1m`, `slippage_estimate` | Very High (15-30% erosion if traded naively) | 1-5 min |
| **5m** | **Primary Trading TF** | Freqtrade/CCXT OHLCV + Cryptofeed | EMA 9/21/50, RSI(14), MACD(12/26/9), ATR(14), volume z-score(20), Bollinger(20,2), CVD, taker buy/sell ratio | `future_return_5m`, `future_return_15m`, `future_return_30m`, `max_adverse_excursion_5m` | High (8-15%) | 15-90 min |
| **15m** | Swing Confirmation / Trend Filter | Freqtrade `@informative` | 15m EMA 20/50, RSI, volume profile (POC, VAH, VAL), VWAP bands, market structure (BOS/CHoCH) | `future_return_15m`, `future_return_1h`, `trend_strength_15m` | Moderate (5-10%) | 1-6 hours |
| **30m** | Short-Term Regime Context | Freqtrade `@informative` | 30m EMA 50/200, ATR, liquidity zones, fair value gaps, order blocks, session highs/lows | `future_return_30m`, `future_return_4h`, `regime_prob_30m` | Moderate (4-8%) | 2-12 hours |
| **1h** | Regime Classification Input | Freqtrade `@informative` | 1h EMA 50/200, MACD, ADX, Ichimoku, volume trend, session ranges | `future_return_1h`, `future_return_4h`, `future_return_24h` | Low (2-5%) | 4-24 hours |
| **4h** | Macro Trend Anchor | Freqtrade `@informative` + FRED | 4h EMA 50/200, Supertrend, macro correlation (DXY, VIX, US10Y via FRED) | `future_return_4h`, `future_return_1w`, `macro_regime` | Very Low (<2%) | 1-7 days |
| **1d** | Structural Bias / Position Sizing | Freqtrade `@informative` + on-chain | 1d EMA 20/50/200, MVRV, SOPR (on-chain), funding 24h trend, long/liquidation ratio | `future_return_1d`, `future_return_1w`, `future_return_1m` | Extremely Low (<1%) | Days to weeks |

*\*Fee impact assumes 0.05% taker fee per side. 1m TF is NOT traded directly; it feeds 5m entry precision only.*

### FreqAI Multi-Timeframe Configuration

Your `freqai.feature_parameters` will explicitly include:

```json
"include_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h"],
"include_corr_pairlist": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
"include_shifted_candles": 3,
"indicator_periods_candles": [8, 14, 20, 50],
"label_period_candles": 100,
"DI_threshold": 20,
"weight_factor": 0.9,
"buffer_train_data_candles": 100
```

**Critical:** FreqAI automatically resamples higher TFs to the primary TF (5m). A 1h EMA(50) becomes a column in every 5m row. You do NOT manually merge; FreqAI handles alignment and forward-fill. You only define which indicators per TF in `feature_engineering_expand`.

---

## WEEK 1: QUANT FOUNDATION
### Goal by Day 7: Freqtrade running 5m primary with 1m/15m/30m/1h/4h informative layers, FreqAI baselines trained per TF, backtested with fee realism, and paper trading.

---

### **DAY 1 — Freqtrade Foundation + Multi-Timeframe Data Ingestion**

**What you use:**
- **Freqtrade** (Docker) — trading engine, backtester, paper trading, WebUI
- **CCXT** — unified exchange API for bulk download
- **Binance Public Data Archive** — free bulk 1m/5m historical (12+ months)
- **TimescaleDB** — hypertables per timeframe
- **Cryptofeed** — real-time 1m bar construction from tick stream

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | `docker-compose up` Freqtrade + TimescaleDB + Redis + Grafana | Docker Compose | WebUI at `:8080` loads; Grafana at `:3000` loads |
| 2–4h | Configure Freqtrade for Binance futures testnet. **Primary TF: 5m**. Pairs: BTC/USDT + ETH/USDT. Add 1m/15m/30m/1h/4h as `include_timeframes` in FreqAI config. | `config.json` | Test connectivity passes; `freqtrade show-config` validates |
| 4–6h | **Bulk download 12+ months across ALL TFs:** `freqtrade download-data --timeframes 1m 5m 15m 30m 1h 4h 1d --timerange 20230801-20260801 --pairs BTC/USDT ETH/USDT` | CCXT + Binance Archive | Files in `user_data/data/binance/` per TF; 1m data > 500MB/pair expected |
| 6–8h | **Create TimescaleDB multi-TF schema:** `ohlcv_1m`, `ohlcv_5m`, `ohlcv_15m`, `ohlcv_30m`, `ohlcv_1h`, `ohlcv_4h`, `ohlcv_1d` hypertables. Add `funding_rates`, `open_interest`, `liquidations`, `orderbook_snapshots` tables. | SQL / `psycopg2` | All hypertables created with `time` partitioning; 1m table has 1-week chunks, 5m has 1-month chunks |
| 8–10h | **Install Ollama on 4080**, pull `qwen2.5:7b` and `llama3.1:8b`. Verify API responds. | Ollama | `ollama list` shows models; `curl http://localhost:11434/api/generate` works |
| 10–12h | **Write `RiskConfig` dataclass** with per-timeframe limits: max position size by TF volatility (1m: 0.5×, 5m: 1.0×, 15m: 1.5×), max daily loss, max drawdown halt, funding rate threshold. Unit tests. | Python + `pytest` | All 12 risk unit tests pass |

**End of Day 1:**
- [ ] Freqtrade WebUI shows empty but functional dashboard
- [ ] Historical download running overnight for all 7 TFs
- [ ] TimescaleDB schema supports 1m→1d granularity
- [ ] Ollama serves 7B model on 4080
- [ ] Risk limits are code, not comments; TF-aware

**What you did NOT build:** Trading engine, backtester, execution layer, WebUI, data download framework, database core.

---

### **DAY 2 — Timeframe-Specific Features + Live Collectors**

**What you use:**
- **pandas-ta** + **TA-Lib** — indicators
- **Cryptofeed** — real-time WebSocket: trades, depth, bookTicker, funding, OI
- **Freqtrade `populate_indicators()` + `@informative`** — multi-TF feature pipeline

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Build `populate_indicators()` for 5m PRIMARY:** EMA 9/21/50, RSI(14), MACD(12/26/9), ATR(14), BB(20,2), volume z-score(20), CVD(50), taker buy/sell ratio, returns(log, 1-5-10-20 candles). | pandas-ta, TA-Lib | 5m DataFrame generates in < 3s per 1k rows |
| 2–4h | **Add `@informative` layers:** `informative_1m`, `informative_15m`, `informative_30m`, `informative_1h`, `informative_4h`. Each pulls OHLCV from Freqtrade data dir and merges into 5m DataFrame. | Freqtrade `@informative` | Higher-TF indicators appear correctly aligned in 5m df; no lookahead (verify with shift check) |
| 4–6h | **1m MICROSTRUCTURE features (custom, not in pandas-ta):** From Cryptofeed tick stream → 1m bars: spread(last), bid_ask_imbalance, volume_delta(buy_vol-sell_vol), tick_momentum(count of upticks vs downticks), realized_variance(1m), trade_count. Store in TimescaleDB `ohlcv_1m` extended columns. | Cryptofeed + custom Python | 1m features write to DB in real-time; 2h burn-in without crash |
| 6–8h | **15m/30m CONTEXT features:** 15m VWAP, volume POC, BOS/CHoCH flags (higher high/low logic), 30m liquidity zone markers (pivot high/low clusters), 30m fair value gap detection (3-candle imbalance). | Custom Python | Features validated against TradingView manual check |
| 8–10h | **Build multi-horizon label generator:** `&-s-future_return_1m`, `&-s-future_return_5m`, `&-s-future_return_15m`, `&-s-future_return_30m`, `&-s-future_return_1h`, `&-s-future_return_4h`, `&-s-future_return_24h`, `&-s-future_volatility_5m`, `&-s-max_adverse_excursion_5m`, `&-s-max_favorable_excursion_5m`. Use `shift(-N)` with N matched to each TF's candles. | polars/pandas | Labels align correctly; **zero lookahead verified** via manual inspection of last 10 rows |
| 10–12h | **Validate full pipeline:** Run 1 month of 5m data through `populate_indicators()` → feature matrix export → check for NaN leakage, timestamp alignment, and feature count. Target: 500-3000 features depending on indicator_periods expansion. | Python | SHAP-ready feature matrix exported as Parquet; feature count documented per TF source |

**Feature Count Breakdown (Target):**
- 1m microstructure: ~20 features (spread, delta, variance, etc.)
- 5m primary TA: ~80 features (multiple indicators × multiple periods)
- 15m context: ~30 features (VWAP, POC, BOS/CHoCH, volume profile)
- 30m structure: ~25 features (liquidity, FVG, order blocks, pivots)
- 1h regime: ~40 features (EMA, MACD, ADX, Ichimoku, session)
- 4h macro: ~15 features (Supertrend, macro correlations)
- Shifted candles (3 lags across TFs): ~200 features
- **Total per pair: ~400-500 base features × expansion from `indicator_periods_candles` = ~2000-3500 features**

**End of Day 2:**
- [ ] Live 1m tick data flows into TimescaleDB
- [ ] 5m primary + 15m/30m/1h/4h informative layers tested
- [ ] Labels verified for zero lookahead across all 7 horizons
- [ ] Feature matrix performance benchmarked

**What you did NOT build:** Indicator library, WebSocket framework, order book reconstruction, manual TF resampler.

---

### **DAY 3 — FreqAI Baseline Models (Per Timeframe)**

**What you use:**
- **FreqAI** — auto ML training, prediction, retraining per identifier
- **LightGBM / XGBoost** — gradient boosting
- **MLflow** — experiment tracking with nested runs per TF

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Configure FreqAI for multi-TF:** `freqai` section in `config.json`. Set `identifier: "gtquant-v0.1-5m-primary"`. Use LightGBM regressor. Define `feature_engineering_expand` to prepend `%` to all custom indicators. | FreqAI | Config validated; `freqtrade show-config` shows FreqAI enabled |
| 2–4h | **Define FreqAI feature set explicitly per TF:** In `feature_engineering_expand`, list: `1m_micro_*`, `5m_ta_*`, `15m_context_*`, `30m_structure_*`, `1h_regime_*`, `4h_macro_*`. Set `include_timeframes: ["1m","5m","15m","30m","1h","4h"]`. | FreqAI strategy | `feature_engineering_expand` returns correct column count (~2.5k) |
| 4–6h | **Run first FreqAI backtest: 5m primary predicting `&-s-future_return_5m`** (primary target). Train period: 30 days. Backtest period: 7 days. | Freqtrade + FreqAI | Model trains; predictions appear in 5m dataframe; prediction column `&-s-future_return_5m` populated |
| 6–8h | **Run secondary FreqAI models:** Same features, but predict `&-s-future_return_15m` and `&-s-future_return_1h` as **separate identifiers** (`gtquant-v0.1-15m-target`, `gtquant-v0.1-1h-target`). These run in parallel during backtest. | FreqAI | Three model identifiers trained; MLflow logs three separate runs |
| 8–10h | **Set up MLflow tracking server** on CPU node. Log all FreqAI runs: params (TF config, indicator periods, label horizon), metrics (MAE, RMSE, R², feature count), model artifacts (pickled LGBM models). | MLflow | MLflow UI at `:5000` shows nested runs: `gtquant/5m-primary`, `gtquant/15m-target`, `gtquant/1h-target` |
| 10–12h | **Analyze feature importance per TF:** Extract SHAP values from LightGBM. Verify that 1m micro features rank in top 20 for 5m predictions. Verify 4h macro features rank high for 1h predictions. | SHAP + Python | SHAP summary plots saved; top 10 features per model documented |

**End of Day 3:**
- [ ] FreqAI LightGBM trained for 5m, 15m, and 1h prediction horizons
- [ ] MLflow tracking active with nested run structure
- [ ] Baseline metrics recorded per TF
- [ ] SHAP analysis confirms TF hierarchy makes sense (micro features matter for short horizons, macro for long)

**Go/No-Go Gate:** If 5m primary model doesn't beat random walk (MAE < naive forecast, R² > 0.05), debug features/labels before Day 4. If 1m micro features don't appear in top 50 SHAP, check Cryptofeed data quality.

**What you did NOT build:** Model training framework, hyperparameter optimizer, experiment tracker, SHAP calculator.

---

### **DAY 4 — Backtesting + Fee Realism Per Timeframe**

**What you use:**
- **Freqtrade backtester** — vectorized + event-driven
- **Freqtrade Hyperopt** — Bayesian optimization per TF
- **Custom fee model** — taker/maker/funding/spread by TF

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Configure realistic fee model per TF:** 5m primary: 0.05% taker / 0.02% maker. Funding rate: 8h interval, pulled from historical. Spread estimate: 0.01% for BTC, 0.02% for ETH. Slippage model: linear with position size. | Freqtrade config | Fee model matches Binance futures VIP 0 tier |
| 2–4h | **Run 5m primary backtest:** LightGBM (5m target) vs. XGBoost (5m target) vs. buy-and-hold vs. simple EMA crossover. Timerange: last 90 days. | Freqtrade | Results table with Sharpe, Sortino, max DD, win rate, avg profit, fee %, total trades |
| 4–6h | **Run 15m and 1h target backtests:** Same 90-day period, but strategy uses 15m or 1h predictions for entry/exit (slower trading, lower fees). Compare net profit after fees. | Freqtrade | 15m/1h backtests show lower fee erosion vs 5m; Sharpe may be higher despite lower gross profit |
| 6–8h | **Hyperopt 5m primary:** Optimize stoploss, ROI tiers, position size factor, `entry_threshold` (model prediction > X), `exit_threshold`. 100 epochs. | `freqtrade hyperopt` | Optimal params found; `hyperopt_results` saved to JSON |
| 8–10h | **Walk-forward validation per TF:** 3 folds. Fold 1: train Jan–Apr, test May. Fold 2: train Jan–May, test Jun. Fold 3: train Jan–Jun, test Jul. No data leakage between folds. | Custom script | Performance stable across folds; variance < 20% of mean Sharpe |
| 10–12h | **Fee erosion analysis:** Calculate fee % of gross profit for each TF. 5m target should show 10-20% fee erosion. 1h target should show <5%. Document break-even fee discount required. | Python | Report: `fee_erosion_5m: 14.2%`, `fee_erosion_15m: 8.1%`, `fee_erosion_1h: 3.4%` |

**Expected Backtest Results (Benchmarks):**
| Model/TF | Trades/Month | Win Rate | Gross Profit | Fee Erosion | Net Profit | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|
| Buy & Hold | 1 | 100% | ~market | <0.1% | ~market | ~0.8 | -30% |
| EMA Cross 5m | ~120 | 55% | +8% | -18% | -10% ❌ | 0.4 | -12% |
| LGBM 5m primary | ~80 | 62% | +15% | -12% | +3% | 1.2 | -8% |
| LGBM 15m target | ~25 | 68% | +12% | -6% | +6% | 1.6 | -6% |
| LGBM 1h target | ~8 | 72% | +9% | -2% | +7% | 1.8 | -5% |

**End of Day 4:**
- [ ] Best model identified per TF with after-fees Sharpe > 0.5
- [ ] 5m primary viable but fee-sensitive; 15m/1h show better risk-adjusted returns
- [ ] Hyperopt params saved per TF
- [ ] Walk-forward results documented; no overfitting detected

**What you did NOT build:** Backtesting engine, fee model, optimization framework.

---

### **DAY 5 — Regime Model + Risk Integration (Timeframe-Aware)**

**What you use:**
- **NautilusTrader `nautilus-risk`** — pre-trade validation
- **sklearn** — regime classifier using 1h/4h features

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Train regime classifier on 1h/4h features:** KMeans (5 clusters) or Random Forest (5 classes) using 1h EMA slope, 4h ADX, 4h Supertrend, DXY direction, VIX level. Classes: `STRONG_BULL`, `BULL`, `RANGE`, `BEAR`, `STRONG_BEAR`, `HIGH_VOL`, `CRASH`. | sklearn | Regime labels assigned to every 1h candle; visual sanity check (plot regime color-coded price) passes |
| 2–4h | **Build TF-aware strategy selector in Freqtrade:** If regime = `STRONG_BULL`/`BEAR`: activate 5m primary (trend following). If regime = `RANGE`: activate 15m mean-reversion (wider stops, lower size). If regime = `CRASH`/`HIGH_VOL`: halt 5m/15m, only allow 1h trend trades with 0.5× size or full halt. | Freqtrade strategy | Backtest shows regime-aware beats single-strategy across all market phases |
| 4–6h | **Integrate Nautilus risk engine:** Pre-trade validation for every Freqtrade signal. Checks: position size ≤ max(5m: 1.0×, 15m: 1.5×, 1h: 2.0× of £1,000), daily loss ≤ 3%, open positions ≤ 3, funding rate ≤ 0.01% (avoid negative carry). | `nautilus-risk` Python bindings | Orders violating limits rejected with logged reason; latency < 10ms |
| 6–8h | **Add 1m microstructure filter:** Before executing a 5m signal, check latest 1m spread and volume delta. If spread > 0.05% OR volume delta < -2σ (sell pressure spike), delay entry by 1-2 candles. | Custom Python | Filter reduces slippage by estimated 20-30% in backtest |
| 8–10h | **Circuit breakers:** Halt on: API failure > 3 retries, P&L drop > 3% in 1h, stale data > 5m (for 5m TF), stale 1m data > 2m, LLM service down > 5min (fallback to pure quant). | Freqtrade + custom | Each breaker tested manually; Telegram alert fires |
| 10–12h | **Document regime→TF→size mapping table.** Example: `STRONG_BULL` → 5m active @ 1.0×, 15m active @ 0.8×, 1h active @ 0.5×. `CRASH` → all TFs halt. | Markdown | Table committed to repo `docs/regime-tf-mapping.md` |

**End of Day 5:**
- [ ] Regime model integrated; 1h/4h features drive 5m/15m/1h strategy availability
- [ ] Risk engine rejects bad orders with TF-specific limits
- [ ] 1m microstructure filter active
- [ ] Circuit breakers tested and documented

**What you did NOT build:** Risk engine core, position sizing math, trading state machine.

---

### **DAY 6 — Paper Trading + Live Multi-Timeframe Integration**

**What you use:**
- **Freqtrade dry-run** — paper trading with real-time data
- **Binance testnet** — simulated execution
- **Cryptofeed** — live 1m/5m/15m/30m/1h/4h WebSocket feeds

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Configure Freqtrade dry-run:** Testnet API, wallet £1,000, 5m primary TF. Enable all informative TFs. Set `dry_run_wallet: 1000`, `position_adjustment_enable: false` (for now). | Freqtrade | Dry-run mode active; WebUI shows £1,000 balance |
| 2–4h | **Integrate regime model output:** Freqtrade strategy queries regime classifier (loaded from Day 5 pickle) every 1h candle. Adjusts available TFs and position size per regime table. | Freqtrade | Position size reduces in `HIGH_VOL`; 5m halts in `CRASH` |
| 4–6h | **Add comprehensive audit logging:** Every decision → TimescaleDB `audit_log` table: timestamp, pair, TF, signal_type, model_prediction, regime, 1m_spread, 1m_volume_delta, risk_decision, fill_price, slippage, pnl. | Python | Query shows complete audit trail with all TFs represented |
| 6–8h | **Live 1m bar builder validation:** Cryptofeed ticks → custom 1m bars → compare to CCXT 1m OHLCV. Verify open/high/low/close/volume match within 0.1%. | Cryptofeed + CCXT | 30-minute validation: 1m bars match exchange official data |
| 8–10h | **Run 4-hour paper trading session.** Monitor: Freqtrade WebUI (5m trades), Grafana (1m micro panel), MLflow (model predictions), Telegram (alerts). | Freqtrade | System runs 4h without intervention; no memory leaks |
| 10–12h | **Paper vs. backtest variance check:** Compare first 4h of paper trades to backtest expectation. Are fills within slippage model? Is trade frequency similar? | Python | Variance < 15% from backtest expectation; document deviations |

**End of Day 6:**
- [ ] Paper account shows correct balance after trades
- [ ] Audit trail complete with 1m/5m/15m/1h context per trade
- [ ] 1m bar builder validated against exchange
- [ ] No critical bugs; system stable 4h

**What you did NOT build:** Paper exchange simulator, execution layer, P&L calculator.

---

### **DAY 7 — Week 1 Checkpoint (Multi-Timeframe Burn-In)**

**Tasks:**
- [ ] **12-hour paper trading burn-in** — no code changes after launch
- [ ] **Dashboard review:** Freqtrade WebUI shows 5m P&L, trades, positions. Grafana shows 1m order book heatmap, 15m trend bias, 1h regime state.
- [ ] **Compare paper results to backtest expectations** per TF. 5m may underperform due to slippage; 15m/1h should be closer.
- [ ] **Document Week 1 architecture:** Data flow diagram (Cryptofeed → TimescaleDB → Freqtrade → FreqAI → Risk → Paper). TF hierarchy diagram.
- [ ] **Code freeze for quant layer.** Tag `v0.1-quant`.
- [ ] **Decision log:** Which TF performed best in paper? Likely 15m or 1h has better Sharpe than 5m due to fees. Document for Week 2.

**WEEK 1 COMPLETE.** You have a multi-timeframe quant-driven paper trading system. 1m feeds precision, 5m is primary, 15m/1h provide context and better risk-adjusted returns. If LLM work fails, you still trade profitably on 15m/1h.

---

## WEEK 2: LLM + PRODUCTION HARDENING
### Goal by Day 14: LLM sentiment/regime reasoning integrated with full TF context. 70B training launched. System runs 24/7 on paper.

---

### **DAY 8 — LLM Dataset + 7B Fine-Tuning (Timeframe-Aware Reasoning)**

**What you use:**
- **Unsloth** — 2× faster QLoRA
- **Ollama** — local inference for testing
- **GPT-4/Claude API** — synthetic dataset generation

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–3h | **Generate 2,000 structured financial reasoning examples with TF context:** Each example includes: 1m spread/volume delta snapshot, 5m indicator summary, 15m trend, 1h regime, 4h macro context, recent news headline → JSON output: `{"regime": "bull", "primary_tf": "15m", "bias": "long", "confidence": 0.72, "risk": "crowded", "entry_precision": "wait_for_1m_pullback"}` | GPT-4/Claude API + curation | JSONL dataset saved; 80% train, 10% val, 10% test |
| 3–5h | **Install Unsloth on 4080.** Load `unsloth/Qwen2.5-7B-Instruct`. Verify VRAM. | Unsloth | `import unsloth` works; VRAM check: < 8GB for 7B 4-bit |
| 5–8h | **Fine-tune 7B with QLoRA** (rank 64, alpha 16, 4-bit NF4, lr 2e-4, 3 epochs). Training data: your 2k financial reasoning examples. | Unsloth + TRL | Training loss decreases smoothly; final loss < 0.5; checkpoint saved every 500 steps |
| 8–10h | **Export to GGUF** (Q4_K_M), load in Ollama on 4080. Test inference on 50 held-out examples. | Unsloth → Ollama | `GT-Quant-7B-v0.1` produces valid JSON; avg inference time < 2s on 4080 |
| 10–12h | **Build LLM prompt template with TF context:** System prompt instructs model to consider 1m→4h hierarchy. User prompt provides structured TF data. Output must be parseable JSON. | Python / Jinja2 | Prompt template committed; parser handles JSON + fallback regex |

**Example Prompt Structure:**
```
[SYSTEM]
You are GT-Quant, a crypto trading analyst. Analyze the multi-timeframe context and output a JSON decision.
Consider: 1m for execution timing, 5m for primary signals, 15m/30m for trend confirmation, 1h/4h for regime.

[USER]
Pair: BTC/USDT
1m: spread=0.02%, volume_delta=+150 BTC, tick_momentum=bullish
5m: EMA9>EMA21, RSI=62, MACD_hist=+45, ATR=0.3%
15m: VWAP=67200, price>VWAP, POC=67150, BOS=up
30m: liquidity_zone=66800-67000, FVG=66950-67050
1h: EMA50_slope=+0.1%, ADX=28(trending), regime=STRONG_BULL
4h: Supertrend=green, DXY=down, VIX=low
Funding: 0.008% (8h)
News: "SEC approves ETH ETF" (2h ago, bullish)

Output JSON with keys: regime, primary_tf, bias, confidence, risk, entry_precision, size_adjustment
```

**End of Day 8:**
- [ ] 7B specialist model running locally
- [ ] Prompt template enforces TF hierarchy reasoning
- [ ] JSON output validated on 50 examples

**What you did NOT build:** Training loop, memory management, gradient checkpointing, model export pipeline.

---

### **DAY 9 — LLM Evaluation + vLLM Production Setup**

**What you use:**
- **vLLM** — production inference with PagedAttention
- **FastAPI** — LLM service wrapper

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–3h | **Build 200-scenario frozen benchmark** covering all TF combinations: 1m spike + 5m bullish + 15m bearish (conflict scenario), 1h crash + 4h bull (macro vs micro), high funding + bullish structure, fake news injection, low liquidity + large signal. | JSON | Benchmark runs automatically; covers edge cases |
| 3–6h | **Score 7B model:** direction accuracy (bias correct?), regime accuracy, primary_tf recommendation accuracy, JSON validity, hallucination rate (facts not in prompt). | Python | Baseline recorded; fine-tuned beats base by > 15% on regime accuracy |
| 6–8h | **Install vLLM on 4080 SUPER.** Load fine-tuned 7B (AWQ or FP8 if available). | vLLM | `/v1/chat/completions` endpoint responds; throughput > 50 req/s |
| 8–10h | **Build LLM service:** FastAPI `/llm/analyze` → receives TF context JSON → calls vLLM → parses JSON → returns structured reasoning + confidence. | FastAPI + vLLM | Response time < 2s p99; timeout handling implemented (fallback after 3s) |
| 10–12h | **Integration test:** Freqtrade strategy calls `/llm/analyze` with current 1m/5m/15m/1h/4h snapshot. Verify latency < 3s end-to-end. | Python | 100 calls benchmarked; avg latency 2.1s; zero timeouts in normal conditions |

**End of Day 9:**
- [ ] vLLM serving on 4080 SUPER
- [ ] LLM evaluation benchmark operational; 7B model scores recorded
- [ ] `/llm/analyze` endpoint works with full TF context
- [ ] Integration latency acceptable for 5m primary (new signal every 5m, so 3s LLM latency is fine)

**What you did NOT build:** Inference engine, KV cache manager, request batching, API server core.

---

### **DAY 10 — 3090 Rig + 70B Preparation + Macro Feature Integration**

**What you use:**
- **Unsloth + DeepSpeed** — multi-GPU QLoRA
- **pandas-datareader / FRED API** — macro data

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Set up 3090 rig:** CUDA 12.4, PyTorch 2.3, Unsloth, DeepSpeed ZeRO-3. Verify all 4 GPUs visible. | pip | `torch.cuda.device_count()` returns 4; `nvidia-smi` shows 4×3090 |
| 2–4h | **Test 70B QLoRA** with 100 examples. DeepSpeed ZeRO-3 config. Verify VRAM usage < 90% across 4 GPUs. | Unsloth + DeepSpeed | 100-example test completes without OOM; checkpoint saves to shared NAS |
| 4–6h | **Prepare full dataset:** 15,000–20,000 examples tokenized. Include TF-specific reasoning chains. Save as `datasets/gtquant_70b_train.jsonl`. | Python | Dataset ready; token count ~2M; avg sequence length < 2048 |
| 6–8h | **Add macro features to Freqtrade strategy:** Pull DXY, VIX, US10Y, CPI release dates from FRED API. Merge into 4h/1d informative dataframes. | `pandas-datareader` / FRED API | Features appear in backtest; `&-s-dxy_slope_4h`, `&-s-vix_level_1d` columns populated |
| 8–10h | **Add on-chain features (free tiers):** Glassnode `mvrv_z_score`, `sopr`, `exchange_inflow` via free API. Merge as 1d informative. | Glassnode API | Features appear in 1d dataframe; cached to avoid rate limits |
| 10–12h | **Add Deribit options features (free):** BTC 25-delta skew, ATM IV. Merge as 1h/4h informative. | CCXT Deribit | `&-s-btc_skew_1h`, `&-s-atm_iv_4h` columns populated |

**End of Day 10:**
- [ ] 3090 rig ready for 70B training
- [ ] Macro features (DXY, VIX, rates) integrated into 4h/1d
- [ ] On-chain features (MVRV, SOPR) integrated into 1d
- [ ] Options skew/IV integrated into 1h/4h
- [ ] Full feature matrix now spans: 1m micro → 5m primary → 15m/30m context → 1h/4h regime → 1d macro + on-chain + options

---

### **DAY 11 — 70B Training Launch + Full System Integration**

**What you use:**
- **Unsloth + DeepSpeed** — 70B QLoRA on 4×3090
- **MLflow** — track the run

**Tasks:**

| Time | Task | Tool | Success Criteria |
|---|---|---|---|
| 0–2h | **Launch GT-Quant-70B-v0.1 training:** 2M tokens, 4-bit NF4, DeepSpeed ZeRO-3, lr 1e-4, 2 epochs. Identifier: `gtquant-70b-v0.1`. | Unsloth | Training running; loss logged to MLflow every 100 steps |
| 2–4h | **Full system integration:** Freqtrade strategy calls `/llm/analyze` at every 5m candle close. Sends full TF context. Receives: regime, bias, confidence, primary_tf recommendation, risk, size_adjustment. | FastAPI | End-to-end pipeline runs; latency < 3s |
| 4–6h | **LLM→Quant fusion logic:** If LLM confidence > 0.7 AND LLM bias aligns with quant signal → execute at full size. If confidence 0.5-0.7 → reduce size 50%. If confidence < 0.5 OR bias conflicts → hold/reduce. If LLM primary_tf = "15m" but quant is 5m → defer to 15m (lower frequency, higher quality). | Python | Fusion rules documented in `strategy/fusion_logic.py` |
| 6–8h | **Shadow mode:** LLM runs alongside quant system. Logs recommendations but does NOT override quant signals. Compare LLM vs. quant decisions in TimescaleDB `shadow_log` table. | Python | Shadow logs show agreement rate; target > 70% alignment in normal regimes |
| 8–10h | **Set up Grafana dashboard:** Panel 1: 1m order book heatmap (spread + depth). Panel 2: 5m signal strength (model prediction + entry/exit markers). Panel 3: 15m trend bias (VWAP + BOS). Panel 4: 1h/4h regime state + LLM reasoning text. Panel 5: P&L per TF. | Grafana | Dashboard refreshes every 5s for 1m, every 30s for higher TFs |
| 10–12h | **MLflow model registry:** Register 7B model as `gtquant-llm-7b:production`. Create staging slot for 70B. Document promotion criteria: shadow agreement > 75%, backtest Sharpe improvement > 0.2. | MLflow | Registry shows model lineage; promotion gate defined |

**Critical note:** 70B training takes 3–5 days. Day 14 system uses 7B model. 70B is Week 3 upgrade.

**End of Day 11:**
- [ ] `nvidia-smi` shows 70B training on 3090s
- [ ] Shadow mode logging works; LLM vs. quant comparison visible
- [ ] Grafana dashboard shows full TF hierarchy
- [ ] MLflow shows all model versions with promotion gates

---

### **DAY 12 — Adversarial Testing (Per Timeframe)**

**Test everything that can break, per granularity:**

| Test | Method | Expected Behavior |
|---|---|---|
| **Kill 1m feed** | Stop Cryptofeed tick stream | 1m bar builder stalls → stale 1m > 2m → 5m entries blocked (micro filter fails open = safe) → alert |
| **Kill 5m feed** | Block CCXT kline API | Stale 5m > 10m → Freqtrade pauses → no new trades → alert |
| **Kill 1h/4h feed** | Block FRED/premium data | Regime model uses last known regime < 4h old → continue with slightly stale context → alert if > 4h |
| **API failure** | Block outbound Binance | Retry ×3 → halt trading → alert |
| **Flash crash (1m)** | Inject -5% 1m bar | 1m spread filter triggers → delay entry → if already in position, stoploss hits → risk engine validates |
| **Flash crash (5m)** | Inject -15% 5m bar | Risk engine rejects new positions → existing stops hit → circuit breaker if P&L drop > 3% in 1h |
| **LLM hallucination** | Feed contradictory TF context | Parser fails OR confidence < 0.3 → fallback to pure quant signal → logged |
| **Fake news** | Inject "BTC banned" headline | LLM flags `HIGH_RISK`/`CRASH` → system reduces exposure by 50% or halts 5m |
| **Latency spike (1m)** | Delay feed 5m | Timestamp drift > 2m → 1m micro filter disabled → 5m trades continue without precision filter → alert |
| **Duplicate orders** | Double signal | Freqtrade idempotency prevents double fill |
| **Memory leak** | Run 6h straight | RAM stable in `htop`; 1m tick buffer in Redis capped at 10k ticks |
| **70B training OOM** | Monitor 3090s | DeepSpeed handles gracefully, checkpoint saved, training resumes |
| **High funding rate** | Set funding = 0.1% | Risk engine rejects new longs if funding > 0.01% (configurable per TF) |

**End of Day 12:**
- [ ] All tests passed and documented in `docs/adversarial_tests.md`
- [ ] Fixes applied; no critical vulnerabilities remain

---

### **DAY 13 — 24-Hour Burn-In (Multi-Timeframe)**

**Tasks:**
- [ ] **Launch 24/7 paper trading.** No code changes after launch.
- [ ] **Telegram alerts** for: errors, circuit breakers, regime changes, daily P&L per TF, LLM confidence drops.
- [ ] **Monitor:** Freqtrade WebUI (5m trades), Grafana (full TF panel), MLflow (predictions), 3090 rig (70B training progress).
- [ ] **Performance review:** Paper vs. backtest variance analysis **per TF**. 5m will diverge most due to slippage; 1h should be closest.
- [ ] **1m bar builder audit:** Verify 1m bars match exchange official data after 24h.

---

### **DAY 14 — Go-Live Review + Timeframe Strategy Lock**

**Tasks:**
- [ ] **Code freeze.** Tag `v0.1-paper`.
- [ ] **Final docs:** Architecture, runbook, API docs, TF hierarchy diagram, regime→TF→size mapping.
- [ ] **Timeframe decision log:** Based on 24h burn-in + Week 1 results, document:
  - Is 5m primary viable after fees, or should 15m be primary?
  - Is 1m microstructure filter adding value?
  - Which TF has the best live Sharpe?
  - Recommended position: 15m primary, 5m for precision entries only, 1h for regime.
- [ ] **Week 3 backlog:** 70B swap, live order book recording, Deribit options depth, on-chain paid tiers, news RSS ingestion.
- [ ] **10-minute demo:** Data (1m tick → 5m bar) → Features (multi-TF) → Quant (FreqAI) → LLM (vLLM reasoning) → Risk (Nautilus) → Paper trade → Dashboard (Grafana).

---

## WHAT YOU BUILT vs. WHAT YOU IMPORTED

| Component | Built From Scratch | Imported/Configured |
|---|---|---|
| Trading engine | ❌ | **Freqtrade** |
| Backtester | ❌ | **Freqtrade** |
| Paper trading | ❌ | **Freqtrade dry-run** |
| Live execution | ❌ | **Freqtrade** |
| WebUI dashboard | ❌ | **Freqtrade WebUI** |
| Telegram bot | ❌ | **Freqtrade** |
| ML training | ❌ | **FreqAI** |
| Hyperparameter optimization | ❌ | **Freqtrade Hyperopt** |
| Data download | ❌ | **CCXT + Binance Archive** |
| Real-time WebSocket | ❌ | **Cryptofeed** |
| Technical indicators | ❌ | **pandas-ta + TA-Lib** |
| Risk engine core | ❌ | **NautilusTrader risk** |
| Time-series database | ❌ | **TimescaleDB** |
| LLM fine-tuning | ❌ | **Unsloth** |
| LLM inference (prod) | ❌ | **vLLM** |
| LLM inference (dev) | ❌ | **Ollama** |
| Experiment tracking | ❌ | **MLflow** |
| Workflow orchestration | ❌ | **Prefect** |

| Component | Built From Scratch |
|---|---|
| Custom feature definitions (per TF: 1m micro, 5m primary, 15m/30m context, 1h/4h regime, 1d macro) | ✅ |
| Label generation per horizon (1m/5m/15m/30m/1h/4h/24h, no lookahead) | ✅ |
| Regime classifier (1h/4h → 5 classes) | ✅ |
| Strategy logic with TF hierarchy and regime selector | ✅ |
| Risk limit configuration (TF-aware position sizing) | ✅ |
| 1m microstructure filter (spread + volume delta gate) | ✅ |
| LLM dataset curation with TF reasoning chains | ✅ |
| LLM prompt templates with 1m→4h context | ✅ |
| JSON output parsers with fallback | ✅ |
| Model promotion gates per TF | ✅ |
| Circuit breakers (per-TF stale data thresholds) | ✅ |
| Audit logging (full TF context per trade) | ✅ |
| Grafana multi-TF dashboard | ✅ |

---

## ESTIMATED TIME SAVED

| Original Plan Task | Without Resources | With Resources | Saved |
|---|---|---|---|
| Trading engine + execution | 3 weeks | 2 hours | **~20 days** |
| Backtester | 2 weeks | 0 (included) | **~14 days** |
| WebUI + dashboard | 1 week | 0 (included) | **~7 days** |
| Multi-TF data ingestion | 2 weeks | 2 days | **~12 days** |
| ML training pipeline | 1 week | 1 day | **~6 days** |
| Risk engine | 3 days | 4 hours | **~2 days** |
| LLM fine-tuning infra | 1 week | 1 day | **~6 days** |
| LLM serving infra | 3 days | 4 hours | **~2 days** |
| **TOTAL** | **~12 weeks** | **~2 weeks** | **~10 weeks** |

---

## WEEK 3+ BACKLOG

1. **70B model swap** when training completes (Unsloth export → vLLM load → A/B test vs 7B)
2. **Live order book recording** via Cryptofeed (L2 depth snapshots every 1s → proprietary 1m order book feature set)
3. **Deribit options data** integration (IV term structure, Greeks surface, skew evolution)
4. **On-chain data** paid tiers (Glassnode/Coin Metrics: exchange flows, whale wallet movements, network velocity)
5. **News ingestion** via RSS (CryptoPanic, official sources) + LLM sentiment scoring per headline
6. **Timeframe optimization engine:** Auto-select primary TF (5m vs 15m) based on recent Sharpe and fee erosion
7. **SaaS frontend** (only after paper trading proves edge across all TFs)
8. **Live trading** £500 (only after 30 days paper + backtest correlation > 0.8 per TF)

---

**This plan is executable because you are configuring and extending proven open-source infrastructure, not inventing it.** The hard problems—execution, backtesting, ML training, LLM serving, risk calculation—are solved by Freqtrade, FreqAI, Unsloth, vLLM, and NautilusTrader. **Your edge lives in:** (1) the 1m→1d timeframe hierarchy, (2) the regime-aware strategy selection, (3) the microstructure filter, (4) the LLM's multi-timeframe reasoning, and (5) the fee-aware risk rules.