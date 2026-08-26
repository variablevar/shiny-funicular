# GT-Quant Repository Layout (production)

Multi-timeframe crypto trading system per `GT-QUANT-MASTER-PLAN.md`
(Multi-Timeframe Edition). Trading engine is Freqtrade+FreqAI in Docker;
this repo holds the custom layers on top.

```
gt-quant/
├── GT-QUANT-MASTER-PLAN.md     # The plan (14-day, multi-TF edition)
├── docker-compose.yml          # Infra: TimescaleDB + Redis + Grafana (RUNNING)
├── requirements.txt            # venv deps (scripts/collectors/tests only)
├── setup.sh                    # Node bootstrap
│
├── core/                       # Reusable custom components (kept from legacy)
│   ├── risk_config.py          # TF-aware RiskConfig + TFRiskEngine (Day 1/5)
│   ├── regime_classifier.py    # KMeans regime model (Day 5 input: 1h/4h feats)
│   └── binance_client.py       # Binance futures REST client (custom feeders)
│
├── db/
│   └── schema.sql              # TimescaleDB multi-TF schema (13 hypertables)
│
├── ft_userdata/                # Freqtrade stack (Docker)
│   ├── docker-compose.yml      # Freqtrade service (stable_freqai image)
│   └── user_data/
│       ├── config.json         # 5m primary, futures, FreqAI multi-TF config
│       ├── strategies/
│       │   └── GTQuantMultiTF.py   # Multi-TF FreqAI strategy (969 features)
│       ├── data/               # Downloaded OHLCV (gitignored, regenerable)
│       └── models/             # FreqAI models (gitignored)
│
├── collectors/                 # (Day 2) Cryptofeed live 1m tick collectors
├── tests/
│   └── test_risk_config.py     # 15 risk unit tests
├── docs/
│   ├── repo-layout.md          # This file
│   └── freqai-setup-notes.md   # FreqAI gotchas + smoke-test recipe
│
├── generate_llm_dataset.py     # (Week 2) LLM dataset curation
├── train_llm.py                # (Week 2) Unsloth QLoRA fine-tuning
└── data/
    └── gtquant_llm_*.jsonl     # (Week 2) LLM datasets (regenerate w/ TF ctx)
```

## Legacy cleanup (2026-08-26)

Removed the pre-Freqtrade custom research stack (superseded by the plan's
Freqtrade/FreqAI architecture):

- `src/` runners (`main.py`, `run_milestone2.py`, `run_trend_following.py`)
- `src/backtest/` (Freqtrade backtester replaces it)
- `src/strategy/` (GTQuantMultiTF replaces them)
- `src/models/` (FreqAI replaces the custom ensemble)
- `src/features/`, `src/validation/`, `src/utils/`
- `scripts/research_harness.py`, `scripts/feature_diagnostics.py`
- `config.yaml` (replaced by `ft_userdata/user_data/config.json`)
- legacy `results/`, `logs/`, 1h parquet caches, `tests/test_pipeline.py`

Kept because the plan still calls for them: regime classifier (Day 5),
risk config (Day 1/5), Binance client (custom funding/OI feeders),
LLM dataset tooling (Week 2).

Legacy headline result for reference: custom 24h-horizon ensemble reached
OOS Sharpe 1.80 (26 trades, 61.5% win) before the pivot — evidence the
regime + funding feature family carries signal; those features are now in
the FreqAI feature set.
