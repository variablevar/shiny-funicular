#!/usr/bin/env python3
"""
Day 9: LLM reasoning service.

FastAPI wrapper around the local model server (Ollama on this research node;
vLLM on the production 4080 SUPER). Exposes:

    POST /llm/analyze   {pair, tf_context} -> structured JSON decision
    GET  /health

The service calls the model, parses the JSON output (with a regex fallback),
enforces the schema, and returns a structured decision. On model timeout or
unparseable output it returns a safe fallback (flat / low confidence) so the
quant system never blocks on the LLM — per plan Day 12 adversarial spec.

Run:
    source venv/bin/activate
    uvicorn services.llm_service:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

# Model server: Ollama here, vLLM on production (same OpenAI-style generate).
MODEL_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gtquant-7b-v0.1"
TIMEOUT_S = 3.0           # plan: timeout handling, fallback after 3s
SYSTEM = ("You are GT-Quant, a crypto trading analyst. Analyze the multi-timeframe "
          "context and output a JSON decision. Consider: 1m for execution timing, 5m "
          "for primary signals, 15m/30m for trend confirmation, 1h/4h for regime.")

REQUIRED_KEYS = {"regime", "primary_tf", "bias", "confidence", "risk",
                 "entry_precision", "size_adjustment"}

FALLBACK = {
    "regime": "unknown", "primary_tf": "5m", "bias": "flat", "confidence": 0.3,
    "risk": "unknown", "entry_precision": "no_entry", "size_adjustment": 0.5,
    "_fallback": True,
}

app = FastAPI(title="GT-Quant LLM Service", version="0.1.0")


class AnalyzeRequest(BaseModel):
    pair: str
    tf_context: str          # the multi-TF snapshot text
    timeout_s: Optional[float] = TIMEOUT_S


def parse_decision(text: str) -> Optional[dict]:
    """Extract the first valid JSON object; tolerate markdown fencing/extra text."""
    # Strip code fences.
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not REQUIRED_KEYS.issubset(obj.keys()):
        return None
    # Coerce numeric fields; a hallucinated string -> None triggers fallback.
    try:
        obj["confidence"] = float(obj["confidence"])
        obj["size_adjustment"] = float(obj["size_adjustment"])
    except (TypeError, ValueError):
        return None
    return obj


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "backend": MODEL_URL}


@app.post("/llm/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    t0 = time.time()
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\nPair: {req.pair}\n{req.tf_context}\n\n"
              f"Output JSON with keys: regime, primary_tf, bias, confidence, risk, "
              f"entry_precision, size_adjustment<|im_end|>\n<|im_start|>assistant\n")
    try:
        resp = requests.post(MODEL_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
        }, timeout=req.timeout_s)
        raw = resp.json().get("response", "")
    except requests.exceptions.Timeout:
        logger.warning(f"LLM timeout after {req.timeout_s}s for {req.pair}")
        return {**FALLBACK, "latency_s": round(time.time() - t0, 3)}
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {**FALLBACK, "latency_s": round(time.time() - t0, 3)}

    decision = parse_decision(raw)
    latency = round(time.time() - t0, 3)
    if decision is None:
        logger.warning(f"LLM unparseable output for {req.pair}: {raw[:120]}")
        return {**FALLBACK, "latency_s": latency, "raw": raw[:300]}

    decision["latency_s"] = latency
    decision["_fallback"] = False
    return decision


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
