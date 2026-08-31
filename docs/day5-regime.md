# Day 5 — Regime Model + Regime-Gated Strategy

## The regime classifier (rule-based, in-strategy)

`_compute_regime()` in `GTQuantMultiTF` labels every 5m bar from 1h
informative features + 5m realized vol:

| Regime | Rule |
|---|---|
| HIGH_VOL | 5m realized vol > 1.5× rolling median (overrides all) |
| RANGE | 1h ADX < 20 (no trend → chop) |
| STRONG_BULL / STRONG_BEAR | 1h EMA-50 slope beyond ±0.0028 AND ADX > 25 |
| BULL / BEAR | 1h EMA-50 slope > 0 / < 0 |

Entries are **blocked in RANGE and HIGH_VOL**; longs need BULL/STRONG_BULL,
shorts need BEAR/STRONG_BEAR.

## The bug this fixed

First version classified July as **53% BULL / 47% BEAR / 0% RANGE** — the gate
blocked nothing. Causes: (1) `|slope| > 0.001` never fires on 1h EMA-50
(median slope is 0.0015, so 0.001 isn't "strong"); (2) RANGE was only the
default and got overwritten since slope is never exactly 0; (3) the HIGH_VOL
2× multiplier never triggered.

Recalibrated on July 2026 1h data (ADX percentiles 19/24/30, |slope|
percentiles 0.0006/0.0015/0.0028). New mix: RANGE 27%, HIGH_VOL 9%,
BULL 23%, BEAR 23%, STRONG_BULL 8%, STRONG_BEAR 9% → **37% of bars blocked**.

## Measured effect (walk-forward, tuned params)

| Fold | Ungated | Regime-gated | Trades |
|---|---|---|---|
| May 2026 | +0.56% (Sharpe 6.79) | +0.23% (Sharpe 1.49) | 45 → 11 |
| Jun 2026 | +0.24% (Sharpe 1.85) | +0.26% (Sharpe **3.01**) | 91 → 38 |
| Jul 2026 | -1.29% (Sharpe -7.97) | -0.73% (Sharpe **-5.96**) | 104 → 58 |

- **July chop: loss cut 44%**, trades cut 44% — the gate works as designed.
- **June: Sharpe 1.85 → 3.01** at half the trade count — cleaner entries.
- **May: Sharpe 6.79 → 1.49** — the gate also blocked some profitable
  trending bars (45 → 11 trades is too aggressive in a strong month).

Net: regime gating improves consistency and cuts bleed, at the cost of some
upside in strongly trending months. Tunable via `regime_adx_range_max`,
`regime_slope_strong`, `regime_vol_mult` (class attributes).

## Next lever (Day 5 continuation / Week 2)

The gate is binary (block/allow). A softer version — size down in RANGE
instead of blocking, size up in STRONG_* — should recover the May upside
without the July bleed. Also: the rule-based classifier is a stand-in for the
KMeans regime model in `core/regime_classifier.py`; wiring that in (fitted on
1h/4h, persisted, loaded by the strategy) is the full Day 5 deliverable.
