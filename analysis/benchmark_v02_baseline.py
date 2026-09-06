#!/usr/bin/env python3
"""
Baseline benchmark: measure regime classification accuracy on v0.2 trained model
applied to the v0.3 strong_trend validation set.

This simulates what the v0.2 fine-tuned model would predict on the new examples,
using a heuristic based on the rule set the strategy uses (which v0.2 was
trained against). Shows whether v0.3 specifically targets the cases v0.2
misclassifies.

Usage:
    python analysis/benchmark_v02_baseline.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

VAL_PATH = Path("data/llm_v03_val.jsonl")


def v02_rule_predictor(ctx):
    """Simulate v0.2's regime classification rules."""
    h = ctx.get("1h", {})
    adx = h.get("adx", 0)
    slope = h.get("ema_50_slope", 0)
    bull_struct = h.get("structure_bull", False)

    # v0.2 ORIGINAL rules (strict)
    if adx < 20:
        return "range"
    if slope > 0.0028 and adx > 25:
        return "strong_bull"
    if slope < -0.0028 and adx > 25:
        return "strong_bear"
    return "bull" if bull_struct else "bear"


def v03_rule_predictor(ctx):
    """v0.3 updated rules (relaxed, matches strategy)."""
    h = ctx.get("1h", {})
    adx = h.get("adx", 0)
    slope = h.get("ema_50_slope", 0)
    bull_struct = h.get("structure_bull", False)

    if adx < 15:
        return "range"
    if bull_struct and slope > 0.0005:
        return "strong_bull"
    if not bull_struct and slope < -0.0005:
        return "strong_bear"
    return "bull" if bull_struct else "bear"


def main():
    print("[*] Running v0.2 vs v0.3 rule comparison on v0.3 val set...")

    # Load val examples with their context (we need the raw data, not the train version)
    raw_path = Path("data/llm_v03_strong_trend_raw.jsonl")
    records = []
    with open(raw_path) as f:
        for line in f:
            rec = json.loads(line)
            try:
                obj = json.loads(rec["output"])
                rec["label"] = obj["regime"]  # normalize
            except Exception:
                pass
            records.append(rec)

    # Take 30% as test (matching original 70/10/20 split)
    import random
    random.seed(42)
    random.shuffle(records)
    n_val = int(len(records) * 0.3) if records else 0
    val = records[:n_val]
    print(f"[*] Test set: {len(val)} examples")

    v02_correct = 0
    v03_correct = 0
    by_regime = {}
    for rec in val:
        ctx = rec.get("tf_context", {})
        true_label = rec.get("label", "")
        v02_pred = v02_rule_predictor(ctx)
        v03_pred = v03_rule_predictor(ctx)

        if true_label not in by_regime:
            by_regime[true_label] = {"v02": 0, "v03": 0, "total": 0}
        by_regime[true_label]["total"] += 1
        if v02_pred == true_label:
            by_regime[true_label]["v02"] += 1
            v02_correct += 1
        if v03_pred == true_label:
            by_regime[true_label]["v03"] += 1
            v03_correct += 1

    print(f"\n{'='*60}")
    print(f"v0.2 RULE BASELINE accuracy: {100*v02_correct/len(val):.1f}%")
    print(f"v0.3 RULE BASELINE accuracy: {100*v03_correct/len(val):.1f}%")
    print(f"{'='*60}")
    print(f"\nPer-regime accuracy:")
    print(f"{'Regime':<15} {'Total':<8} {'v0.2':<10} {'v0.3':<10} {'Delta':<10}")
    for r, stats in sorted(by_regime.items()):
        v02_pct = 100 * stats["v02"] / stats["total"] if stats["total"] else 0
        v03_pct = 100 * stats["v03"] / stats["total"] if stats["total"] else 0
        delta = v03_pct - v02_pct
        print(f"{r:<15} {stats['total']:<8} {v02_pct:<10.1f} {v03_pct:<10.1f} {delta:+<10.1f}")


if __name__ == "__main__":
    main()
