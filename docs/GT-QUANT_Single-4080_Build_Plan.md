# GT-QUANT on One 4080

**14-Day Build Plan — Re-scoped for a Single RTX 4080 (16 GB VRAM)**

*Build nothing you can import. Trade every granularity from microstructure to macro. September 2026.*

- 0. What Changed and Why (Hardware Reality)
- 1. The Stack (Use vs. Build) — Single-Machine Edition
- 2. Hardware Lock — One Machine, One GPU
- 3. Multi-Timeframe Architecture (Unchanged)
- 4. Week 1 — Quant Foundation (CPU-Bound, Unchanged)
- 5. Week 2 — LLM + Hardening (Re-scoped to 16 GB)
- 6. Phase 3 — The 70B Path via Burst Rental
- 7. Configs: Freqtrade, Unsloth, Model Choices
- 8. VRAM Budget & The One-GPU Constraint
- 9. Built vs. Imported (Unchanged)
- 10. Honest Limitations & Disclaimer

## 0. What Changed and Why (Hardware Reality)

The original plan assumed a four-machine rig: 4× RTX 3090 (training) + 4080 Super (production) + 4080 (research) + CPU database node. You now have one RTX 4080 with 16 GB VRAM and one CPU. Three dependencies in the old plan are impossible on this hardware:

**What the single-4080 reality breaks — and the fix**

| Original Plan | Needed | On One 4080 | Revised Approach |
|---|---|---|---|
| Day 10–11: 70B QLoRA training | ~48 GB VRAM (4× 3090) | Impossible — 16 GB | Move to Phase 3 burst rental (A6000 48GB, ~$0.5–0.8/hr) |
| 70B inference/serving | ~40 GB at 4-bit | Too slow to be usable (CPU offload) | Serve 8B–14B locally; 70B only on rental, Phase 3 |
| Split prod/research across two 4080s | 2 GPUs | 1 GPU shared | Merge roles; the GPU does LLM only — Freqtrade/FreqAI are CPU |
| vLLM "production" + Ollama "research" on separate cards | 2 GPUs | 1 GPU | One serving stack (vLLM), gated by schedule so it never fights trading |

> The Single-GPU Principle
> Your one 4080 does
> LLM work only
> . The entire money-making spine — Freqtrade, FreqAI (LightGBM), the risk engine, TimescaleDB — is
> CPU-bound
> and never touches the GPU. This is why a single 4080 is enough to start: the GPU is a pure LLM co-processor, and trading never waits on it. The hard 70B work is deferred to cheap burst rental, not bought.

What does NOT change: the entire Week 1 quant foundation, the multi-timeframe architecture, the feature/label specifications, the risk engine, and the fee-realism discipline. Those are the parts that actually generate edge, and they are 100% CPU. If the quant layer has no edge, no LLM will save it — so we build and prove that first, exactly as the original plan intended.

## 1. The Stack (Use vs. Build) — Single-Machine Edition

**The stack — what survives, what changes**

| Layer | Use This (Existing) | Runs On | Build This (Custom) |
|---|---|---|---|
| Trading Engine | Freqtrade — backtest, paper/live, WebUI, Telegram, Hyperopt | CPU | Strategy logic, custom features, multi-TF fusion |
| Quant ML | FreqAI — LightGBM/XGBoost, auto-retrain, multi-TF expansion | CPU | Feature definitions per TF, label logic per horizon |
| Data Ingestion | CCXT + Binance Public Data Archive | CPU | Funding/OI/liquidation feeders per TF |
| Real-Time L2 | Cryptofeed — trades, depth, bookTicker, funding, OI | CPU | TimescaleDB adapter; 1m bar construction from ticks |
| Features | pandas-ta (130+) + TA-Lib | CPU | Multi-TF fusion (1m micro → 5m primary → 15m/30m → 1h/4h regime) |
| Risk Engine | NautilusTrader nautilus-risk | CPU | Per-TF limits, TF-aware sizing |
| Database | TimescaleDB (PostgreSQL) + Redis tick cache | CPU/RAM | Unified multi-TF schema, hypertables per granularity |
| LLM Fine-Tuning | Unsloth — QLoRA (8B/14B locally; 32B/70B on rental) | GPU 4080 | Dataset curation with TF reasoning chains |
| LLM Inference | Ollama (research) + vLLM (serving) — same card, scheduled | GPU 4080 | Prompt templates per regime/TF, JSON parsers |
| Experiment Tracking | MLflow — registry, versioning | CPU | Per-TF experiment groups, promotion gates |
| Dashboard | Freqtrade WebUI + Grafana | CPU | Multi-TF panels |
| Orchestration | Prefect | CPU | Multi-TF DAGs, retrain schedules |

Key insight: notice the "Runs On" column. Only two layers touch the GPU — LLM fine-tuning and LLM inference. Everything else is CPU. That is the entire reason this plan works on one card.

## 2. Hardware Lock — One Machine, One GPU

**Single-machine hardware lock**

| Resource | Role | Runs |
|---|---|---|
| RTX 4080 (16 GB) | LLM only — research + serving, scheduled | Unsloth QLoRA (8B/14B), Ollama (research), vLLM (serving 14B 4-bit) |
| CPU (16+ threads rec.) | Trading + ML + orchestration | Freqtrade, FreqAI/LightGBM, Nautilus risk, Cryptofeed, Prefect, 1m bar builder |
| RAM (64 GB rec., 32 GB min) | Data + tracking | TimescaleDB (all hypertables), Redis (1m tick cache), MLflow backend, Grafana |
| NVMe SSD (2 TB rec.) | Bulk data | 12+ months of 7 timeframes × 2 pairs, feature matrices (Parquet), model checkpoints |

RAM matters more than before: with one machine, TimescaleDB, Redis, MLflow, Grafana, Freqtrade and the 1m tick buffer all share RAM. 64 GB is strongly recommended; 32 GB is the workable floor with Redis tick buffer capped aggressively (see §8).

## 3. Multi-Timeframe Architecture (Unchanged)

This is the heart of the edge and it is untouched. You trade a 5m primary timeframe, informed by 1m microstructure , filtered by 15m/30m context , and regime-classified by 1h/4h/1d . All of this is CPU work (Freqtrade + FreqAI resampling).

**Timeframe responsibilities (identical to original plan)**

| TF | Role | Label Horizons | Fee Impact | Hold |
|---|---|---|---|---|
| 1m | Microstructure / execution precision | future_return_1m , slippage_estimate | Very High — not traded directly | 1–5 min |
| 5m | Primary trading TF | future_return_5m/15m/30m , MAE_5m | High (8–15%) | 15–90 min |
| 15m | Swing confirmation / trend filter | future_return_15m/1h , trend_strength_15m | Moderate (5–10%) | 1–6 h |
| 30m | Short-term regime context | future_return_30m/4h , regime_prob_30m | Moderate (4–8%) | 2–12 h |
| 1h | Regime classification input | future_return_1h/4h/24h | Low (2–5%) | 4–24 h |
| 4h | Macro trend anchor | future_return_4h/1w , macro_regime | Very Low (<2%) | 1–7 d |
| 1d | Structural bias / sizing | future_return_1d/1w/1m | Extremely Low (<1%) | Days–weeks |

FreqAI auto-resamples higher TFs onto the 5m row — you only define indicators per TF in feature_engineering_expand . The include_timeframes config is in §7.

## 4. Week 1 — Quant Foundation (CPU-Bound, Unchanged)

Goal by Day 7: Freqtrade running 5m primary with 1m/15m/30m/1h/4h informative layers, FreqAI baselines trained per TF, backtested with fee realism, paper trading. Days 1–7 run exactly as the original plan — they are CPU-bound and need no GPU. Summary of the week (full task tables preserved from your original):

- Day 1: Freqtrade + TimescaleDB + Redis + Grafana via Docker; bulk-download 12+ months across all 7 TFs; multi-TF hypertable schema; write TF-aware RiskConfig with unit tests. (Only change: install Ollama on the 4080 and pull qwen3:8b instead of the dated qwen2.5:7b .)
- Day 2: populate_indicators() for 5m primary; @informative layers for 1m/15m/30m/1h/4h; custom 1m microstructure features from Cryptofeed; multi-horizon label generator with verified zero lookahead; feature-matrix validation (~2,000–3,500 features).
- Day 3: FreqAI LightGBM baselines for 5m/15m/1h horizons (separate identifiers); MLflow nested runs; SHAP feature-importance check per TF. Go/No-Go gate unchanged: 5m model must beat random walk (R² > 0.05).
- Day 4: Realistic per-TF fee model; backtests (LGBM vs XGBoost vs buy-and-hold vs EMA cross); Hyperopt on 5m; 3-fold walk-forward; fee-erosion analysis per TF.
- Day 5: Regime classifier (KMeans/RF on 1h/4h); TF-aware strategy selector; Nautilus pre-trade risk integration; 1m microstructure entry filter; circuit breakers.
- Day 6: Dry-run paper trading on Binance testnet; regime output wired in; full audit logging with per-TF context; 1m bar-builder validation vs exchange; 4h session.
- Day 7: 12h burn-in; dashboard review; paper-vs-backtest variance per TF; code freeze v0.1-quant ; decision log on best TF.

> Week 1 is the whole ballgame
> If Week 1's quant system trades profitably on 15m/1h in paper, you have a business. The LLM (Week 2) is an
> enhancement layer
> on top of a working quant spine, not a substitute for it. Do not touch the LLM until the Day-7 decision log shows a positive after-fee Sharpe.

## 5. Week 2 — LLM + Hardening (Re-scoped to 16 GB)

Goal by Day 14: a locally fine-tuned 8B/14B LLM specialist integrated with full TF context, running on the single 4080, in shadow mode against the quant system. The 70B is NOT launched here — it moves to Phase 3 (rental).

> Day 8 — LLM Dataset + 8B/14B Fine-Tuning (on the 4080)
> Time
> Task
> Tool
> Success Criteria
> 0–3h
> Generate 2,000 structured financial-reasoning examples with TF context → JSON output
> {regime, primary_tf, bias, confidence, risk, entry_precision}
> GPT-4/Claude API + curation
> JSONL saved; 80/10/10 split
> 3–5h
> Install Unsloth on the 4080; load
> unsloth/Qwen3-8B
> (4-bit). Verify VRAM
> Unsloth
> import unsloth
> works; 8B 4-bit < 8 GB
> 5–8h
> QLoRA fine-tune (rank 64, alpha 16, NF4, lr 2e-4, 3 epochs) on your 2k examples
> Unsloth + TRL
> Loss < 0.5; checkpoints every 500 steps
> 8–10h
> Export GGUF (Q4_K_M), load in Ollama, test 50 held-out examples
> Unsloth → Ollama
> Valid JSON; inference < 2s on 4080
> 10–12h
> Build TF-context prompt template + JSON parser with regex fallback
> Python/Jinja2
> Parser handles all 50 cases
> Model choice (updated):
> use
> Qwen3-8B
> as the dev workhorse, and if you want more reasoning headroom,
> Qwen3-14B
> at 4-bit (~10 GB) also QLoRA-fits on 16 GB. Do NOT use the old
> qwen2.5:7b
> — Qwen3 is materially stronger at the same size and supports a hybrid "thinking" mode.

> Day 9 — Evaluation + vLLM Serving (single card)
> Time
> Task
> Tool
> Success Criteria
> 0–3h
> Build 200-scenario frozen benchmark covering all TF conflict combinations
> JSON
> Benchmark runs automatically
> 3–6h
> Score the 8B/14B: direction accuracy, regime accuracy, JSON validity, hallucination rate
> Python
> Fine-tuned beats base by >15% on regime accuracy
> 6–8h
> Install vLLM on the 4080; load the fine-tuned model (AWQ/GPTQ-Int4)
> vLLM
> /v1/chat/completions
> responds; fits in 16 GB with KV cache
> 8–10h
> FastAPI
> /llm/analyze
> : TF context JSON → vLLM → parsed structured output + confidence
> FastAPI
> p99 < 2s; 3s timeout fallback
> 10–12h
> Integration test: Freqtrade calls
> /llm/analyze
> at each 5m close
> Python
> End-to-end latency < 3s (fine for 5m cadence)
> Single-GPU scheduling:
> vLLM serving and any QLoRA training never run simultaneously. Training runs in off-hours or between serving windows; Freqtrade (CPU) never blocks. If the LLM service is down >5 min, the circuit breaker falls back to pure-quant signals — trading continues regardless of GPU state.

> Day 10 — Macro/On-Chain/Options Features (replaces "3090 rig + 70B prep")
> The original Day 10 set up the 3090 rig. Without it, spend this day deepening the
> feature
> stack instead — this is pure CPU and adds real edge:
> Macro features:
> DXY, VIX, US10Y, CPI dates via FRED API → merged into 4h/1d informative dataframes.
> On-chain (free tiers):
> Glassnode MVRV-Z, SOPR, exchange inflow → 1d informative.
> Options (free):
> Deribit BTC 25-delta skew, ATM IV → 1h/4h informative.
> Full feature matrix
> now spans 1m micro → 5m primary → 15m/30m context → 1h/4h regime → 1d macro + on-chain + options.

> Day 11 — LLM↔Quant Fusion + Shadow Mode (no 70B launch)
> Full integration:
> Freqtrade sends full TF context to
> /llm/analyze
> at each 5m close; receives regime, bias, confidence, primary_tf, risk, size_adjustment.
> Fusion logic:
> confidence > 0.7 and aligned with quant → full size; 0.5–0.7 → half size; < 0.5 or conflict → hold. LLM
> primary_tf="15m"
> overrides 5m cadence (fewer, better trades).
> Shadow mode:
> LLM logs recommendations without overriding quant; agreement logged to
> shadow_log
> (target >70% in normal regimes).
> Grafana multi-TF dashboard
> +
> MLflow registry
> with promotion gates (shadow agreement > 75%, Sharpe improvement > 0.2 before going live).
> Critical reframe:
> the original plan launched 70B training here. On one 4080, Day 11 instead
> hardens the 14B integration
> . The 70B is a Phase-3 upgrade you trigger only after the 14B proves value in shadow mode.

> Days 12–14 — Adversarial Testing, 24h Burn-In, Go-Live Review
> Day 12:
> adversarial tests per TF (kill 1m/5m/1h feeds, API failure, flash-crash injection, LLM hallucination, fake news, latency spike, memory leak, high funding). One test changes: "70B training OOM" → "LLM serving OOM on 4080" (vLLM should degrade gracefully to quant fallback, not crash trading).
> Day 13:
> 24h paper burn-in, Telegram alerts per TF, paper-vs-backtest variance, 1m bar audit.
> Day 14:
> code freeze
> v0.1-paper
> ; final docs;
> timeframe decision log
> (is 5m viable after fees, or is 15m primary?); Phase-3 backlog.

## 6. Phase 3 — The 70B Path via Burst Rental

You do not buy 3090s to prove a model. You prove the pipeline on the 4080, then rent burst compute for the heavy run. This is both cheaper and a better story.

**Phase 3 — 70B via rental**

| Step | Action | Cost / Hardware |
|---|---|---|
| 1. Prove on 4080 | 14B shadow mode beats pure-quant on the frozen benchmark and in paper | $0 — your 4080 |
| 2. Rent training | Single A6000 48GB (~$0.5–0.8/hr) on RunPod/Vast; QLoRA DeepSeek-R1-Distill-Llama-70B on your 15–20k dataset | ~$40–80 for a weekend run |
| 3. Rent serving / A-B | Serve the 70B adapter on the same rental; A/B against the 14B on the frozen benchmark | Included in rental window |
| 4. Promote or discard | Only if 70B beats 14B by the promotion gate do you consider owning 24GB cards | Decision, not purchase |

The cascade you'll actually run: the 14B (4080) handles the high-frequency one-minute screening loop cheaply and always-on; the 70B (rental) handles only slow, high-stakes thesis generation. ~90% of the intelligence at ~15% of the inference cost.

## 7. Configs: Freqtrade, Unsloth, Model Choices

#### Freqtrade / FreqAI (multi-TF, unchanged)

```
"freqai": {
  "enabled": true,
  "identifier": "gtquant-v0.1-5m-primary",
  "feature_parameters": {
    "include_timeframes": ["1m","5m","15m","30m","1h","4h"],
    "include_corr_pairlist": ["BTC/USDT:USDT","ETH/USDT:USDT"],
    "include_shifted_candles": 3,
    "indicator_periods_candles": [8,14,20,50],
    "label_period_candles": 100,
    "DI_threshold": 20,
    "weight_factor": 0.9,
    "buffer_train_data_candles": 100
  },
  "data_split_parameters": { "test_size": 0.25 },
  "model_training_parameters": { "n_estimators": 800 }
}
```

#### Unsloth QLoRA (fits 16 GB — 8B/14B only)

```
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-14B",            # or unsloth/Qwen3-8B for faster iteration
    load_in_4bit=True, max_seq_length=2048)
model = FastLanguageModel.get_peft_model(
    model, r=64, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
# Trainer: lr 2e-4, 3 epochs, bf16, gradient_checkpointing="unsloth"
# Export: model.save_pretrained_gguf("gtquant-14b", quantization_method="q4_k_m")
```

#### Model choices on one 4080 (quick reference)

**What runs where**

| Model | Size / Quant | QLoRA on 16GB? | Inference on 16GB? | Role |
|---|---|---|---|---|
| Qwen3-8B | 8B, 4-bit | Yes (comfortable) | Fast | Dev loop, fast iteration |
| Qwen3-14B | 14B, 4-bit | Yes (tight, ~12–14 GB) | Good | Recommended production specialist |
| Qwen3-32B (Thinking) | 32B, 4-bit (~19 GB) | No | Slow (light CPU offload) | Optional research only |
| DeepSeek-R1-Distill-Llama-70B | 70B, 4-bit (~40 GB) | No | No | Phase 3 rental only |

## 8. VRAM Budget & The One-GPU Constraint

- 16 GB is the whole budget. Serving a 14B 4-bit model plus KV cache fits; training 8B/14B QLoRA fits; but never both at once, and 32B/70B training is off the table.
- Schedule GPU work: QLoRA training in off-hours; vLLM serving during live windows; Freqtrade/FreqAI always CPU. Trading never waits on the GPU.
- Cap the 1m tick buffer in Redis (e.g., 10k ticks) — on one shared-RAM machine this is the most common memory leak source.
- Fallback discipline: any LLM/GPU failure must degrade to pure-quant (FreqAI) signals within seconds. The circuit breaker that does this is the single most important line of code in the system, precisely because you have one GPU and no redundancy.

## 9. Built vs. Imported (Unchanged)

You still import the solved problems — Freqtrade, FreqAI, Cryptofeed, pandas-ta/TA-Lib, Nautilus-risk, TimescaleDB, Unsloth, vLLM/Ollama, MLflow, Prefect — and build only your edge: the per-TF feature definitions, the no-lookahead label generators, the regime classifier, the TF-hierarchy strategy logic, TF-aware risk limits, the 1m microstructure filter, the LLM dataset/prompt/parsers, the fusion logic, the circuit breakers, the audit logging, and the Grafana multi-TF dashboard. The single-machine change affects where things run , not what you build .

## 10. Honest Limitations & Disclaimer

Limitations of the single-4080 edition: (1) No 70B training or serving locally — Phase 3 requires paid burst rental. (2) The 14B specialist is a meaningful step below a reasoning-distilled 70B on the hardest multi-step theses; treat it as a screening/enhancement layer, not an oracle. (3) One GPU = one point of failure; the quant-CPU fallback is mandatory, not optional. (4) VRAM headroom is tight; large-context prompts and training must be scheduled, not concurrent. This document is a technical build plan for research/paper-trading infrastructure. It is not financial advice, not an offer of securities, and no profitability is promised or implied. Algorithmic trading involves substantial risk of loss; the benchmark figures quoted from the original plan are illustrative targets, not results. Live trading should only follow the original plan's own gate (30 days of profitable paper trading with backtest correlation > 0.8 per timeframe). © 2026 GT-Quant.
