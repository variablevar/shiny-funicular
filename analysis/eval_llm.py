#!/usr/bin/env python3
"""
Day 9 prep: evaluate the fine-tuned 7B model on the held-out test set.

Metrics: JSON validity, regime accuracy, bias accuracy, directional agreement
(model long/short vs actual), mean confidence calibration.

Usage:
    source venv/bin/activate
    python analysis/eval_llm.py --n 100
"""
from __future__ import annotations

import argparse
import json

import requests

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gtquant-7b-v0.1"


def query(system: str, user: str) -> str:
    resp = requests.post(OLLAMA, json={
        "model": MODEL,
        "prompt": f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n",
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200},
    }, timeout=120)
    return resp.json()["response"]


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from model output (tolerant)."""
    try:
        start = text.index("{")
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    test = [json.loads(l) for l in open("data/llm_v2_test.jsonl")][: args.n]
    n = len(test)
    valid = regime_ok = bias_ok = 0
    dir_match = 0
    confs = []

    for i, ex in enumerate(test, 1):
        out = query(ex["instruction"], ex["input"])
        pred = parse_json(out)
        truth = json.loads(ex["output"])

        if pred is None:
            continue
        valid += 1
        if pred.get("regime") == truth["regime"]:
            regime_ok += 1
        if pred.get("bias") == truth["bias"]:
            bias_ok += 1
        # Directional agreement (flat counts as no-direction, not a mismatch).
        if truth["bias"] != "flat" and pred.get("bias") == truth["bias"]:
            dir_match += 1
        if "confidence" in pred:
            try:
                confs.append(float(pred["confidence"]))
            except (TypeError, ValueError):
                pass

        if i % 20 == 0:
            print(f"  [{i}/{n}] json={valid / i:.0%} regime={regime_ok / i:.0%} bias={bias_ok / i:.0%}")

    n_dir = sum(1 for ex in test if json.loads(ex["output"])["bias"] != "flat")
    print("\n=== EVAL RESULTS ===")
    print(f"examples:            {n}")
    print(f"JSON validity:       {valid / n:.1%}")
    print(f"regime accuracy:     {regime_ok / n:.1%}")
    print(f"bias accuracy:       {bias_ok / n:.1%}")
    print(f"directional match:   {dir_match}/{n_dir} ({dir_match / max(n_dir, 1):.1%} of directional)")
    print(f"mean confidence:     {sum(confs) / len(confs):.2f}" if confs else "n/a")


if __name__ == "__main__":
    main()
