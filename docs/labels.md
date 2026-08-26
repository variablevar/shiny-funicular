# GT-Quant Label Generator (Day 2)

`core/labels.py` builds forward-looking labels with a strict zero-lookahead
contract: the label at row `i` uses only rows after `i`, and the final `H`
rows of every `H`-horizon label are NaN by construction.

## Label set (per horizon)

| Column | Definition |
|---|---|
| `future_return` | `close[i+H] / close[i] - 1` |
| `future_volatility` | std of 1-candle log returns over the next H candles |
| `max_favorable_excursion` | `max(high[i+1..i+H]) / close[i] - 1` |
| `max_adverse_excursion` | `min(low[i+1..i+H]) / close[i] - 1` |

Horizon naming is caller-defined: `build_labels_multi(df, {"15m": 3, "1h": 12})`
on a 5m frame produces `15m_future_return`, `1h_future_return`, etc.
(Horizons are in CANDLES of the input frame, not minutes.)

## Usage

```python
from core.labels import build_labels, build_labels_multi

labeled = build_labels(df_5m, horizon=12, prefix="1h_")
labeled = build_labels_multi(df_5m, {"15m": 3, "1h": 12, "4h": 48})
```

## Validation (2026-08-26, BTC 5m futures data)

- Tail-NaN contract holds for all tested horizons (3/12/48 candles).
- Spot-check row 0: label `-0.00015624` == manual `close[i+3]/close[i]-1`.
- 12 unit tests in `tests/test_labels.py` cover alignment, tail NaN,
  future-only data dependence, MFE/MAE semantics, and error cases.

## Where labels are consumed

- **FreqAI identifiers** (Day 3): one `&-s-` label per identifier — the
  strategy's `set_freqai_targets` computes its target inline; this module is
  for offline analysis and dataset curation.
- **LLM dataset curation** (Week 2): MFE/MAE per horizon feeds reasoning
  examples ("what was the best/worst path over the next hour").
- **Regime analysis** (Day 5): forward-return distributions per regime.
