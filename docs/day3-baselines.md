# Day 3 — FreqAI Baseline Models (Per Timeframe)

Three FreqAI identifiers trained on the multi-TF feature set (969 features,
5 TFs × periods × corr pairs × shifts), one prediction horizon each.

## Identifiers

| Identifier | Strategy | Target | Config |
|---|---|---|---|
| `gtquant-v0.1-5m-primary` | `GTQuantMultiTF` | `&-s-future_return_5m` (20 candles) | `config.json` |
| `gtquant-v0.1-15m-target` | `GTQuantMultiTF15m` | `&-s-future_return_15m` (3 candles) | `config_15m.json` |
| `gtquant-v0.1-1h-target` | `GTQuantMultiTF1h` | `&-s-future_return_1h` (12 candles) | `config_1h.json` |

Secondary identifiers are thin subclasses: they override `prediction_col`
and `set_freqai_targets`; all feature engineering is inherited.

## Baseline backtests (2026-04-01 → 2026-07-01, LightGBMRegressor)

Pre-tuning baselines with the smoke-test threshold (0.02%). All runs trained
with **zero exceptions** and produced predictions end-to-end.

| Identifier | Trades (BTC+ETH) | Win rate | Net |
|---|---|---|---|
| 5m-primary | 473 | ~57% | ≈ -1% |
| 15m-target | 1990 | ~46% | ≈ -8% |
| 1h-target | 1591 | ~47% | ≈ -6% |

Negative net is expected at this stage: the 0.02% entry threshold fires
constantly and fees dominate. Threshold/stoploss/ROI tuning is Day 4
(hyperopt + fee erosion analysis). Day 3's gate is pipeline correctness,
not profitability.

## Feature importance per timeframe (the key Day 3 result)

Gain-importance share by timeframe suffix, averaged over training windows:

| Model | 1h | 4h | 30m | 15m | 5m |
|---|---|---|---|---|---|
| 5m-primary (BTC) | ~33% | ~27% | ~18% | ~11% | ~10% |

- **Higher-TF context dominates** (~60% from 1h+4h) even for 5m predictions.
- Top feature families: `roc`, `atr`, `rsi`, `ema_dist` — momentum and
  volatility structure carry the signal.
- Per-window CSVs: `user_data/models/<identifier>/feature_importance_<pair>.csv`
- FreqAI also writes per-model HTML importance plots in each `sub-train-*` dir.

Caveat vs plan: 1m microstructure features are NOT yet in the FreqAI set
(the collector only started on Day 2; 1m features join after the burn-in
accumulates history). The plan's "1m features in top 20" check is deferred
to Day 6 retraining.

## MLflow tracking

- Server: `gtquant-mlflow` container, UI at http://localhost:5000
  (sqlite backend store, artifact volume `mlflow_data`).
- Logging: `analysis/analyze_models.py` — parses each identifier's latest
  trained model, computes TF importance shares, and logs params + metrics
  per run via the MLflow **REST API**.
- Why REST: `mlflow` 3.x pins `cryptography<50` while `ccxt` requires
  `>=50` — irreconcilable in one venv. The REST client (requests) avoids
  the conflict entirely. 29 runs logged on first pass.

## Reproduce

```bash
# secondary identifier backtests
cd ft_userdata
docker-compose run --rm --no-deps freqtrade backtesting \
  --config user_data/config_15m.json --strategy GTQuantMultiTF15m \
  --freqaimodel LightGBMRegressor --timerange 20260401-20260701 --cache none

# analysis + MLflow logging
source venv/bin/activate
python analysis/analyze_models.py
```
