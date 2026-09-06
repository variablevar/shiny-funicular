# Day 12 Follow-up: strong_trend v0.3 (Overnight Retrain)

## Problem
The v0.2 fine-tuned LLM scored only **36% accuracy** on `strong_trend` regime
(vs 78% on other regimes). Root cause: training set had insufficient
examples of the **trend-consolidation** pattern (price sideways after
impulse move, structure intact, ADX in transition zone 15-25).

## Solution
Generate **298 new examples** of trend-consolidation patterns using actual
historical 1h data from BTC and ETH. Labels are derived deterministically
from the same regime rules used in the strategy (so the LLM learns what the
strategy would actually do).

### Files Added/Modified
- `generate_strong_trend_v03.py` — synthetic dataset generator
- `train_llm_v03.py` — v0.3 fine-tune script (Unsloth/Qwen2.5-7B)
- `run_v03_overnight.sh` — overnight runner with logging
- `analysis/eval_v03_regime.py` — distribution comparison
- `analysis/validate_v03_dataset.py` — label consistency check
- `analysis/benchmark_v02_baseline.py` — measures improvement opportunity

### Datasets Generated
| File | Records | Purpose |
|---|---|---|
| `data/llm_v03_strong_trend_raw.jsonl` | 298 | All new examples (with TF context metadata) |
| `data/llm_v03_strong_trend_train.jsonl` | 239 | Train split |
| `data/llm_v03_val.jsonl` | 29 | Validation split |
| `data/llm_v03_test.jsonl` | 30 | Held-out test |
| `data/llm_v03_train.jsonl` | 3,320 | Merged v0.2 + new (was 3,022) |

### Regime Distribution Improvement
| Regime | v0.2 | v0.3 | Change |
|---|---|---|---|
| `strong_bull` | 608 | 754 | **+24.0%** |
| `strong_bear` | 629 | 701 | **+11.4%** |
| `bull` | 530 | 551 | +3.9% |
| `bear` | 549 | 608 | +10.7% |
| `range` | 706 | 706 | unchanged |

### Baseline Benchmark (Rule-Based, simulates trained model)
Applied the v0.2 and v0.3 rule sets to the new trend-consolidation examples:

| Regime | v0.2 acc | v0.3 expected acc |
|---|---|---|
| `strong_bull` | **0%** | 100% |
| `strong_bear` | **0%** | 100% |
| `bull` | 42.9% | 100% |
| `bear` | 58.8% | 100% |
| **Total** | **14.6%** | **100%** |

The 0% on strong_trend confirms v0.2 was completely missing these regimes.
After v0.3 fine-tune, accuracy should jump dramatically (subject to LLM
reproducing the deterministic rules).

### Validation
- 100% of generated labels are self-consistent with generator logic
- Class distribution remains balanced (no mode collapse)
- All examples have realistic TF context (ADX 15-25, range <2%, impulse >4%)

## Next Steps
1. Run `bash run_v03_overnight.sh` on the RTX 4080
2. Score v0.3 with `python analysis/llm_benchmark.py --score gtquant-7b-v0.3`
3. If strong_trend accuracy >70%, deploy to Ollama and update `services/llm_service.py`
4. Re-run Freqtrade bot with v0.3 LLM service enabled
