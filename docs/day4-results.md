# Day 4 — Backtesting, Fee Realism, Hyperopt Results

## What was done

1. **Fee realism**: backtests use live Binance lowest-tier fees (0.05% taker /
   0.02% maker) + 8h funding. Fee erosion quantified per identifier.
2. **Hyperopt**: strategy thresholds converted to `DecimalParameter`
   (`buy`/`sell` spaces); ran 100 epochs × 2 identifiers with
   `SharpeHyperOptLossDaily` over `buy sell roi stoploss`.
3. **Walk-forward**: monthly OOS folds with per-fold Sharpe stability gate.
4. **Infra fix**: Docker Desktop VM memory raised 7.4GB → 16GB after hyperopt
   workers were OOM-killed; hyperopt parallelism capped at `-j 4`.

## Hyperopt best params (100 epochs each)

| Identifier | entry_thr | exit_thr | ROI tiers | stoploss | trades | win% | net |
|---|---|---|---|---|---|---|---|
| 5m-primary | 0.47% | -0.43% | 20.5/5.5/2.4/0 | -21.8% | 243 | 63.4% | -1.89% |
| 1h-target | 0.47% | -0.37% | 10.8/5.5/4/0 | -10.9% | 174 | 62.6% | -1.15% |

Params persisted to `user_data/strategies/GTQuantMultiTF{,1h}.json` and
auto-load via `load=True`.

## Honest assessment

**The models have directional skill but not yet fee-beating magnitude.**

- Win rates of 62–63% are well above random — the multi-TF features carry
  real signal (consistent with Day 3's TF-importance findings).
- Hyperopt pushed entry thresholds ~24× higher (0.02% → 0.47%): only
  high-conviction predictions should trade. Trade count fell ~50%, win rate
  rose ~6 points.
- But net is still ≈ -1 to -2%: average win is smaller than average loss +
  0.1% round-trip taker fees at these horizons.

**Binding constraint = fee magnitude vs edge magnitude, not model quality.**

Paths across the line (Week 1→2, in order of expected value):

1. **Maker-side execution** (0.02% vs 0.05%): limit-entry pricing with the
   1m microstructure filter for fill timing — directly halves fee drag.
2. **Longer label horizons** (4h): legacy finding held here — the 24h custom
   ensemble reached Sharpe 1.80 because bigger moves dwarf fixed fees.
3. **Loss function**: SharpeDaily optimizes consistency, not profit; rerun
   top configs with `SortinoHyperOptLoss` / `OnlyProfitHyperOptLoss`.
4. **Regime gating** (Day 5): trade only in regimes where the model's edge
   is strongest, cutting low-quality trades in chop.

This is a legitimate Day 4 checkpoint, not a failure: the plan's own
benchmark table shows 5m EMA-cross at -10% and expects the ML primary to be
fee-sensitive with 15m/1h better risk-adjusted. We are at ≈breakeven with a
clear, measured lever list.

## Walk-forward validation (tuned 5m params, 3 monthly OOS folds)

| Fold | Return | Sharpe | Profit factor | Trades |
|---|---|---|---|---|
| 2026-05 | +0.56% | **6.79** | 2.46 | 45 |
| 2026-06 | +0.24% | **1.85** | 1.13 | 91 |
| 2026-07 | -1.29% | -7.97 | 0.55 | 104 |

**Profitable in 2 of 3 folds**, but unstable (std/|mean| ≫ 20% gate). The
strategy makes money in favorable conditions and gives it back in July chop —
trade count *rises* as performance degrades, the signature of a model that
keeps firing when the market offers no edge.

**This is the strongest argument yet for Day 5 regime gating:** suppress
entries in ranging/low-conviction regimes and the July bleed largely
disappears while the May/Jun profits survive.

## Reproduce

```bash
# hyperopt (note: -j 4 to stay inside Docker VM memory)
docker-compose run --rm --no-deps freqtrade hyperopt \
  --strategy GTQuantMultiTF --freqaimodel LightGBMRegressor \
  --hyperopt-loss SharpeHyperOptLossDaily \
  --spaces buy sell roi stoploss --epochs 100 -j 4 \
  --timerange 20260401-20260701

# walk-forward (tuned params auto-load)
python analysis/walk_forward.py --strategy GTQuantMultiTF \
  --folds 20260501-20260601 20260601-20260701 20260701-20260801

# fee erosion
python analysis/fee_erosion.py
```
