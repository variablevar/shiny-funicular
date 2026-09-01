# Day 11 — LLM ↔ Quant Fusion

`core/fusion.py` combines a FreqAI quant signal with the LLM's structured
decision into one trade decision. Pure functions, 12 unit tests.

## Fusion rules

| Condition | Action | Size |
|---|---|---|
| No quant signal | hold | 0 |
| LLM unavailable / `_fallback` | execute (pure quant) | 1.0× |
| LLM bias conflicts quant direction | hold | 0 |
| LLM flat | reduce | 0.5× |
| Aligned, confidence > 0.7 | execute | 1.0× |
| Aligned, confidence 0.5–0.7 | reduce | 0.5× |
| Aligned, confidence < 0.5 | hold | 0 |
| Aligned + LLM primary_tf slower than quant TF | defer to slower TF | per above |

## Design notes

- **The LLM never increases size above the quant base** — it can only confirm
  (full size), damp (0.5×), veto (hold), or slow the trade down (defer to a
  higher timeframe). The quant layer is the floor; the LLM is a filter, not an
  amplifier. This is the safe direction for a model that can hallucinate.
- **Fail-open to quant.** Any LLM error (timeout, unparseable JSON, the
  `_fallback` decision from the service) returns the quant signal unchanged.
  Day 12 adversarial spec: LLM down → trade on pure quant.
- **TF deferral** implements the plan's "if LLM says 15m but quant is 5m,
  defer to 15m" — fewer, higher-quality trades.

## Usage

```python
from core.fusion import QuantSignal, fuse

quant = QuantSignal(direction=1, timeframe="5m", confidence=0.8, regime="BULL")
llm = service_decision  # parsed /llm/analyze output (or None)
decision = fuse(quant, llm)
# decision.action, decision.size_multiplier, decision.effective_tf
```

## Wiring (Day 11 integration)

The Freqtrade strategy calls `/llm/analyze` at each 5m close with the current
TF snapshot, passes the parsed decision + the quant signal into `fuse()`, and
applies `size_multiplier` in `custom_stake_amount` and `effective_tf`/`action`
in entry gating. Shadow mode (log-only) first: recommendations go to the
`shadow_log` hypertable without overriding quant, so we measure agreement
before trusting it.
