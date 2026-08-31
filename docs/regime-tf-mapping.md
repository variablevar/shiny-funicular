# Regime → Timeframe → Size Mapping (Day 5)

How the current regime gates which timeframe trades and at what size.
This is the single source of truth for the regime-aware strategy selector.

## Regime detection (rule-based, in `GTQuantMultiTF._compute_regime`)

| Regime | Detection rule (1h informative + 5m vol) |
|---|---|
| STRONG_BULL | EMA-50 slope > +0.0028 AND ADX > 25 |
| BULL | EMA-50 slope > 0 (not strong) |
| RANGE | ADX < 20 (chop) |
| BEAR | EMA-50 slope < 0 (not strong) |
| STRONG_BEAR | EMA-50 slope < -0.0028 AND ADX > 25 |
| HIGH_VOL | 5m realized vol > 1.5× rolling median (overrides all) |

## Mapping table

| Regime | 5m primary | 15m target | 1h target | Size multiplier |
|---|---|---|---|---|
| STRONG_BULL | long only | long | long | **1.5×** |
| BULL | long only | long | long | 1.0× |
| RANGE | **halt** | mean-reversion (later) | **halt** | — |
| BEAR | short only | short | short | 1.0× |
| STRONG_BEAR | short only | short | short | **1.5×** |
| HIGH_VOL | **halt** | **halt** | **halt** | — |
| CRASH | **halt all** | **halt all** | **halt all** | — |

Size multipliers are applied in `custom_stake_amount()` (regime-aware sizing).
The per-TF base caps come from `core/risk_config.py`
(`tf_size_multipliers`: 5m 1.0×, 15m 1.5×, 1h 2.0× of base stake) — the two
multiply together: effective size = base_stake × tf_mult × regime_mult.

## Circuit breakers (Freqtrade protections, `config.json`)

| Protection | Rule | Effect |
|---|---|---|
| CooldownPeriod | 6 candles after any close | no immediate re-entry churn |
| StoplossGuard | 4 stoplosses in 48 candles | halt 24 candles |
| MaxDrawdown | 15% drawdown over 48 candles, ≥10 trades | halt 48 candles |

Harder breakers (API failure, stale-data, LLM-down) live in the collector /
service layer and the `TFRiskEngine` in `core/risk_config.py`, tested in
`tests/test_risk_config.py`.

## Implementation status

- [x] Regime detection + entry gating (RANGE/HIGH_VOL blocked)
- [x] Regime-aware sizing (STRONG_* 1.5×)
- [x] Freqtrade protections (cooldown / stoploss guard / max drawdown)
- [ ] RANGE mean-reversion sub-strategy (Week 2)
- [ ] KMeans regime model (`core/regime_classifier.py`) replacing rules
