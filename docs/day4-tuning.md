# Day 4 — Backtesting + Fee Realism + Hyperopt

## Fee model

Backtests use Binance futures lowest-tier fees pulled live by Freqtrade:
**0.05% taker / 0.02% maker**, plus perp funding at the 8h marks. Spread
estimates (BTC 0.01%, ETH 0.02%) come from the 1m collector's top-of-book
sampling once it has history.

## Fee erosion (the reason tuning matters)

Pre-tuning baselines (0.02% entry threshold, fires constantly):

| Identifier | Trades | Fees (USDT) | Gross | Net | Erosion |
|---|---|---|---|---|---|
| 5m-primary | 473 | 38.90 | 17.50 | -21.40 | **222%** |
| 15m-target | 1990 | 170.86 | -5.18 | -176.03 | n/a (gross < 0) |
| 1h-target | 1591 | 135.83 | 13.76 | -122.07 | **987%** |

The models' gross edge per trade (~0.04%) is smaller than the ~0.1% round-trip
fee. The fix is exactly what hyperopt optimizes: **higher conviction
thresholds → fewer, larger trades → fees shrink relative to gross.**
Plan targets after tuning: 5m ~10-20% erosion, 1h < 5%.

Reproduce: `python analysis/fee_erosion.py` (reads `backtest_results/*.zip`,
writes `results/fee_erosion/fee_erosion.json`).

## Hyperopt

Strategy thresholds are now hyperopt parameters
(`DecimalParameter`, spaces `buy`/`sell`); ROI tiers and stoploss use the
built-in `roi`/`stoploss` spaces.

```bash
cd ft_userdata
docker-compose run --rm --no-deps freqtrade hyperopt \
  --strategy GTQuantMultiTF --freqaimodel LightGBMRegressor \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces buy sell roi stoploss \
  --epochs 100 --timerange 20260401-20260701
```

FreqAI trains its sliding windows once (13 per pair over 90 days), then the
100 epochs reuse the cached models — epochs are fast after the initial
training pass.

## Walk-forward validation

`analysis/walk_forward.py` runs the backtest over sequential monthly folds
(FreqAI retrains on trailing 30d every 7d inside each fold, so every fold is
OOS) and reports per-fold Sharpe + the stability gate (std/|mean| < 20%):

```bash
python analysis/walk_forward.py --strategy GTQuantMultiTF \
  --folds 20260501-20260601 20260601-20260701 20260701-20260801
```
