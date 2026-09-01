#!/usr/bin/env python3
"""
Day 9: frozen 200-scenario LLM evaluation benchmark.

Crafts scenarios across the plan's edge cases, each with a rule-derived
expected output. Frozen to disk so base vs fine-tuned comparisons are stable.

Archetypes (per plan Day 9):
  strong_trend        all TFs aligned          -> directional, high conf
  range_chop          ADX weak, no structure   -> flat, low conf
  tf_conflict         5m vs 1h/4h disagree     -> flat/low conf
  high_funding_crowded funding z extreme       -> risk=crowded, size 0.5
  crash               1h slope very negative   -> short/flat, risk high
  low_liquidity       wide spread proxy (ATR)  -> entry_precision no_entry
  funding_reversal    funding z very negative  -> contrarian long lean

Usage:
    python analysis/llm_benchmark.py --build          # write benchmark
    python analysis/llm_benchmark.py --score MODEL    # score a model
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests

BENCH_PATH = Path("data/llm_benchmark_200.jsonl")
OLLAMA = "http://localhost:11434/api/generate"

SYSTEM = ("You are GT-Quant, a crypto trading analyst. Analyze the multi-timeframe "
          "context and output a JSON decision. Consider: 1m for execution timing, 5m "
          "for primary signals, 15m/30m for trend confirmation, 1h/4h for regime.")

KEYS = {"regime", "primary_tf", "bias", "confidence", "risk", "entry_precision", "size_adjustment"}


def _prompt(body: str) -> str:
    return body + "\n\nOutput JSON with keys: regime, primary_tf, bias, confidence, risk, entry_precision, size_adjustment"


def _mk(archetype: str, body: str, expect: dict) -> dict:
    return {"archetype": archetype, "input": _prompt(body), "expect": expect}


def build(n_per: int = 28, seed: int = 7) -> list[dict]:
    """Build the frozen benchmark. 7 archetypes x ~28 = ~200 scenarios."""
    rnd = random.Random(seed)
    out: list[dict] = []

    def trend(dirn: str, strong: bool):
        s = ">" if dirn == "bull" else "<"
        slope = 0.35 if strong else 0.08
        if dirn == "bear":
            slope = -slope
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9{s}EMA21, RSI={rnd.randint(55, 75) if dirn == 'bull' else rnd.randint(25, 45)}, ATR=0.25%\n"
                f"15m: price{s}EMA50, EMA50_slope={slope / 3:+.3f}%\n"
                f"30m: EMA50{s}EMA200, ADX={rnd.randint(28, 40)}\n"
                f"1h: EMA50_slope={slope:+.3f}%, ADX={rnd.randint(28, 42)}(trending), regime={'STRONG_' + dirn.upper() if strong else dirn.upper()}\n"
                f"4h: EMA50{s}EMA200, structure={dirn}ish\n"
                f"Funding z-score: z={rnd.uniform(-0.5, 0.5):+.1f} (neutral)\n"
                f"CVD 1h (signed volume): {rnd.uniform(5e3, 2e4) * (1 if dirn == 'bull' else -1):+.0f}")
        return _mk("strong_trend", body, {"bias": "long" if dirn == "bull" else "short",
                                          "risk_not": "crowded"})

    def range_chop():
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9~EMA21, RSI={rnd.randint(45, 55)}, ATR=0.12%\n"
                f"15m: price~EMA50, EMA50_slope=+0.001%\n"
                f"30m: EMA50~EMA200, ADX={rnd.randint(10, 17)}\n"
                f"1h: EMA50_slope={rnd.uniform(-0.02, 0.02):+.3f}%, ADX={rnd.randint(10, 16)}(weak), regime=RANGE\n"
                f"4h: EMA50~EMA200, structure=neutral\n"
                f"Funding z-score: z={rnd.uniform(-0.3, 0.3):+.1f} (neutral)\n"
                f"CVD 1h (signed volume): {rnd.uniform(-1e3, 1e3):+.0f}")
        return _mk("range_chop", body, {"bias": "flat", "entry_precision": "no_entry"})

    def tf_conflict():
        # 5m bullish but 1h bearish + 4h bearish (lower-TF bounce in downtrend).
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9>EMA21, RSI={rnd.randint(58, 68)}, ATR=0.30%\n"
                f"15m: price>EMA50, EMA50_slope=+0.05%\n"
                f"30m: EMA50<EMA200, ADX={rnd.randint(22, 30)}\n"
                f"1h: EMA50_slope={rnd.uniform(-0.25, -0.1):+.3f}%, ADX={rnd.randint(26, 35)}(trending), regime=BEAR\n"
                f"4h: EMA50<EMA200, structure=bearish\n"
                f"Funding z-score: z={rnd.uniform(-0.5, 0.5):+.1f} (neutral)\n"
                f"CVD 1h (signed volume): {rnd.uniform(1e3, 6e3):+.0f}")
        # Correct: counter-structure bounce -> not a confident long.
        return _mk("tf_conflict", body, {"bias_not": "long_confident"})

    def high_funding_crowded():
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9>EMA21, RSI={rnd.randint(60, 72)}, ATR=0.28%\n"
                f"15m: price>EMA50, EMA50_slope=+0.08%\n"
                f"30m: EMA50>EMA200, ADX={rnd.randint(25, 35)}\n"
                f"1h: EMA50_slope=+{rnd.uniform(0.1, 0.3):.3f}%, ADX={rnd.randint(28, 40)}(trending), regime=BULL\n"
                f"4h: EMA50>EMA200, structure=bullish\n"
                f"Funding z-score: z={rnd.uniform(2.2, 3.5):+.1f} (extreme, longs paying)\n"
                f"CVD 1h (signed volume): {rnd.uniform(8e3, 2e4):+.0f}")
        return _mk("high_funding_crowded", body, {"risk": "crowded", "size_adjustment_max": 0.5})

    def crash():
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9<EMA21, RSI={rnd.randint(12, 22)}, ATR={rnd.uniform(1.5, 2.8):.2f}%\n"
                f"15m: price<EMA50, EMA50_slope={rnd.uniform(-0.6, -0.3):+.3f}%\n"
                f"30m: EMA50<EMA200, ADX={rnd.randint(35, 50)}\n"
                f"1h: EMA50_slope={rnd.uniform(-1.2, -0.6):+.3f}%, ADX={rnd.randint(35, 50)}(trending), regime=STRONG_BEAR\n"
                f"4h: EMA50<EMA200, structure=bearish\n"
                f"Funding z-score: z={rnd.uniform(-3.5, -2.2):+.1f} (extreme, shorts paying)\n"
                f"CVD 1h (signed volume): {rnd.uniform(-5e4, -2e4):+.0f}")
        return _mk("crash", body, {"bias_in": ("short", "flat"), "risk_in": ("high", "crowded")})

    def low_liquidity():
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9>EMA21, RSI={rnd.randint(55, 65)}, ATR={rnd.uniform(0.05, 0.09):.2f}% (thin)\n"
                f"15m: price>EMA50, EMA50_slope=+0.03%\n"
                f"30m: EMA50>EMA200, ADX={rnd.randint(15, 22)}\n"
                f"1h: EMA50_slope=+{rnd.uniform(0.02, 0.08):.3f}%, ADX={rnd.randint(15, 22)}(weak), regime=RANGE\n"
                f"4h: EMA50>EMA200, structure=bullish\n"
                f"Funding z-score: z={rnd.uniform(-0.4, 0.4):+.1f} (neutral)\n"
                f"CVD 1h (signed volume): {rnd.uniform(-300, 300):+.0f}")
        return _mk("low_liquidity", body, {"bias": "flat", "entry_precision": "no_entry"})

    def funding_reversal():
        body = (f"Pair: BTC/USDT\n"
                f"5m: EMA9<EMA21, RSI={rnd.randint(28, 40)}, ATR=0.35%\n"
                f"15m: price<EMA50, EMA50_slope=-0.06%\n"
                f"30m: EMA50<EMA200, ADX={rnd.randint(20, 28)}\n"
                f"1h: EMA50_slope={rnd.uniform(-0.15, -0.05):+.3f}%, ADX={rnd.randint(20, 28)}(weak), regime=BEAR\n"
                f"4h: EMA50>EMA200, structure=bullish\n"
                f"Funding z-score: z={rnd.uniform(-3.0, -2.2):+.1f} (extreme, shorts paying)\n"
                f"CVD 1h (signed volume): {rnd.uniform(-1.5e4, -5e3):+.0f}")
        # Shorts paying heavily into a bull structure -> contrarian long lean,
        # but never high confidence (1h still bearish).
        return _mk("funding_reversal", body, {"bias_in": ("long", "flat"), "risk": "crowded"})

    archetypes = [lambda: trend("bull", True), lambda: trend("bear", True),
                  lambda: trend("bull", False), range_chop, tf_conflict,
                  high_funding_crowded, crash, low_liquidity, funding_reversal]
    # ~200 total: spread across archetypes.
    per = max(1, 200 // len(archetypes))
    for fn in archetypes:
        for _ in range(per):
            out.append(fn())
    rnd.shuffle(out)
    return out


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #

def parse_json(text: str) -> dict | None:
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


def _as_float(v, default: float = 0.0) -> float:
    """Tolerant float parse; non-numeric (a hallucinated string) -> default."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def score_one(pred: dict, expect: dict) -> tuple[bool, list[str]]:
    """Check pred against the scenario's expectations. Returns (pass, failures)."""
    fails = []
    if "bias" in expect and pred.get("bias") != expect["bias"]:
        fails.append(f"bias={pred.get('bias')}!={expect['bias']}")
    if "bias_in" in expect and pred.get("bias") not in expect["bias_in"]:
        fails.append(f"bias={pred.get('bias')} not in {expect['bias_in']}")
    if "bias_not" in expect and expect["bias_not"] == "long_confident":
        if pred.get("bias") == "long" and _as_float(pred.get("confidence", 0)) > 0.7:
            fails.append("confident long on counter-structure bounce")
    if "risk" in expect and pred.get("risk") != expect["risk"]:
        fails.append(f"risk={pred.get('risk')}!={expect['risk']}")
    if "risk_in" in expect and pred.get("risk") not in expect["risk_in"]:
        fails.append(f"risk={pred.get('risk')} not in {expect['risk_in']}")
    if "risk_not" in expect and pred.get("risk") == expect["risk_not"]:
        fails.append(f"risk should not be {expect['risk_not']}")
    if "entry_precision" in expect and pred.get("entry_precision") != expect["entry_precision"]:
        fails.append(f"entry_precision={pred.get('entry_precision')}")
    if "size_adjustment_max" in expect:
        sa = pred.get("size_adjustment", 1)
        try:
            sa_f = float(sa)
        except (TypeError, ValueError):
            fails.append(f"size_adjustment not numeric: {sa!r}")
        else:
            if sa_f > expect["size_adjustment_max"]:
                fails.append(f"size_adjustment={sa_f} > {expect['size_adjustment_max']}")
    return (len(fails) == 0), fails


def score_model(model: str, bench: list[dict]) -> None:
    total = valid = passed = 0
    by_arch: dict[str, list[int]] = {}
    for i, sc in enumerate(bench, 1):
        resp = requests.post(OLLAMA, json={
            "model": model,
            "prompt": f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{sc['input']}<|im_end|>\n<|im_start|>assistant\n",
            "stream": False, "options": {"temperature": 0.1, "num_predict": 200},
        }, timeout=120)
        pred = parse_json(resp.json()["response"])
        total += 1
        if pred is None or not KEYS.issubset(pred.keys()):
            by_arch.setdefault(sc["archetype"], []).append(0)
            continue
        valid += 1
        ok, fails = score_one(pred, sc["expect"])
        passed += 1 if ok else 0
        by_arch.setdefault(sc["archetype"], []).append(1 if ok else 0)
        if i % 40 == 0:
            print(f"  [{i}/{total}] valid={valid / total:.0%} pass={passed / total:.0%}")

    print(f"\n=== {model} ===")
    print(f"JSON validity:  {valid / total:.1%}")
    print(f"overall pass:   {passed / total:.1%}")
    print("per archetype:")
    for arch, res in sorted(by_arch.items()):
        print(f"  {arch:22s} {sum(res)}/{len(res)} ({sum(res) / len(res):.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", type=str, default=None)
    args = ap.parse_args()

    if args.build:
        bench = build()
        BENCH_PATH.write_text("\n".join(json.dumps(b) for b in bench))
        print(f"wrote {BENCH_PATH} ({len(bench)} scenarios)")
        from collections import Counter
        print(Counter(b["archetype"] for b in bench))
        return

    bench = [json.loads(l) for l in open(BENCH_PATH)]
    score_model(args.score, bench)


if __name__ == "__main__":
    main()
