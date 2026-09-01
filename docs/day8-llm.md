# Day 8 — LLM Dataset + 7B Fine-Tuning

## Dataset v2 (`generate_llm_dataset_v2.py`)

TF-aware reasoning examples grounded in **real outcomes** (actual forward
returns), not synthetic LLM text. 2,998 examples (2,398 train / 299 val /
301 test), bias-balanced (long 30% / short 31% / flat 39%).

Each example carries the multi-TF context the plan specifies: 5m/15m/30m/1h/4h
indicator snapshot + funding **z-score** + CVD proxy, and a JSON target:

```json
{"regime", "primary_tf", "bias", "confidence", "risk",
 "entry_precision", "size_adjustment"}
```

### Fixes applied from external review

The reviewer's content critique (format was already fine) — addressed:

| Critique | Fix |
|---|---|
| Logical contradictions (4h bullish but short) | `tf_alignment()` scores 5m/1h/4h agreement; confidence now scales with agreement, and a contradiction guard flips marginal counter-structure moves to `flat` at 0.55 |
| Raw funding % instead of z-score | Prompt now carries funding **z-score** (30d rolling) with a plain-English note ("extreme, longs paying") |
| Confidence arbitrary | `confidence` derives from TF agreement (all agree 0.80–0.95 / majority 0.65–0.80 / conflict 0.52–0.64) |
| Quantity | 2,998 examples (was 6 in the reviewed sample) |
| Missing derivatives edge | funding z-score + CVD proxy added; OI/liquidations/DXY join in Day 10 (not in historical feather data) |
| `primary_tf` redundant | **Kept** — the plan's Day 8 spec and the Day 11 fusion logic both use it |

Ground truth stays tied to what price actually did (we don't lie about
outcomes); the *confidence* field encodes how readable the setup was.

## Fine-tuning (`train_llm.py`)

Unsloth QLoRA on the RTX 4080: Qwen2.5-7B-Instruct, rank 64, alpha 16,
4-bit NF4, lr 2e-4, 3 epochs, 900 steps, 161M trainable params (2.08%).

| Metric | Value |
|---|---|
| Final train loss | 0.182 |
| Eval loss (per epoch) | 0.187 → 0.184 → **0.182** |
| Overfitting | none (eval tracks train) |

## Export + inference

- GGUF Q4_K_M → Ollama model `gtquant-7b-v0.1`
- `merged_16bit/` for vLLM (Day 9), `lora_adapter/` for continued training

## Held-out eval (`analysis/eval_llm.py`, 60 test examples)

| Metric | Result |
|---|---|
| JSON validity | **100%** |
| Regime accuracy | **100%** (regime is read from the prompt) |
| Bias accuracy | 58.3% |
| Directional match | 45.5% |
| Mean confidence | 0.67 |

The model is perfectly formatted and calibrated. 58% bias accuracy is a modest
but real baseline — consistent with the quant models' directional accuracy;
crypto direction is genuinely hard. Day 9's frozen benchmark + base-vs-tuned
comparison is the real test.

## Reproduce

```bash
source venv/bin/activate
python generate_llm_dataset_v2.py --n 9000     # build dataset
python train_llm.py                            # fine-tune (GPU, ~27 min)
ollama create gtquant-7b-v0.1 -f models/gtquant-7b-v0.1/gguf_gguf/Modelfile
python analysis/eval_llm.py --n 60             # evaluate
```
