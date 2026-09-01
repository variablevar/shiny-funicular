# Day 9 — LLM Evaluation + Inference Service

## Frozen benchmark (`analysis/llm_benchmark.py`)

198 scenarios across 7 archetypes with rule-derived expected outputs:
`strong_trend`, `range_chop`, `tf_conflict`, `high_funding_crowded`, `crash`,
`low_liquidity`, `funding_reversal`. Frozen at `data/llm_benchmark_200.jsonl`
so base vs tuned comparisons are stable.

## Base vs fine-tuned (the Day 9 gate: tuned beats base by >15%)

| Model | JSON validity | Overall pass |
|---|---|---|
| qwen2.5:7b (base) | 100% | **11.1%** |
| gtquant-7b-v0.1 (tuned) | 100% | **73.7%** |

**+62.6 points** — far over the >15% gate. The base model does not understand
the risk/sizing schema (it even emitted `size_adjustment: "ADJUSTED"`, a type
hallucination, on the first run).

### Tuned per-archetype

| Archetype | Pass |
|---|---|
| crash | 100% |
| high_funding_crowded | 100% |
| range_chop | 100% |
| low_liquidity | 100% |
| tf_conflict | 100% |
| funding_reversal | 95% |
| strong_trend | **23%** ⚠️ |

**Known weakness — strong_trend:** the model says `flat` in strong trends.
Cause: the Day 8 ground truth ties bias to the *realized* 4h move
(`|ret_4h| > 1%`), and strong trends often consolidate over the next 4h, so
many strong-trend training examples are labeled flat. The model learned
"strong trend ≠ guaranteed immediate move" — honest, but over-conservative.
**Fix path:** lower the directional threshold when regime is `STRONG_*`
(trend-following intent) and retrain. Tracked for the v0.2 dataset.

## Inference service (`services/llm_service.py`)

FastAPI wrapper: `POST /llm/analyze {pair, tf_context}` → schema-validated
JSON decision. Tolerant parser (strips code fences, extracts first JSON
object, coerces numerics). On timeout (>3s) or unparseable output it returns
a safe `flat` fallback — the quant system never blocks on the LLM (Day 12
adversarial spec).

### Latency (100 calls, Ollama backend)

| metric | value | budget |
|---|---|---|
| mean | 0.44s | — |
| p99 | 0.44s | < 2s |
| max | 1.90s | < 3s |
| fallbacks | 0 | — |

## vLLM note

vLLM is the plan's *production* serving layer for the **4080 SUPER**, not this
research node. Two constraints found here:

1. Docker GPU access needs `nvidia-container-toolkit` + CDI, which requires
   interactive `sudo` (can't run non-interactively).
2. `vllm` pins `torch==2.13` while `unsloth` pins `torch<2.13` — they cannot
   share a venv. Serve and train environments must be split.

On this node Ollama already serves `gtquant-7b-v0.1` within the latency
budget, so vLLM setup is deferred to the production node. To run vLLM here:
create `python -m venv venv-vllm && venv-vllm/bin/pip install vllm` and serve
`models/gtquant-7b-v0.1/merged_16bit` — do NOT install vllm into the training
venv.

## Run it

```bash
source venv/bin/activate
uvicorn services.llm_service:app --host 0.0.0.0 --port 8000
curl -X POST localhost:8000/llm/analyze -H 'Content-Type: application/json' \
  -d '{"pair":"BTC/USDT","tf_context":"5m: EMA9>EMA21 ..."}'
```
