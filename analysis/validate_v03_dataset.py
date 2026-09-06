#!/usr/bin/env python3
"""
Validate v0.3 dataset label consistency.

For each new example in llm_v03_strong_trend_raw.jsonl, verify:
  1. The label is consistent with the input (ADX + structure + slope)
  2. The reasoning text mentions the correct regime
  3. The bias aligns with the structure

This catches ground-truth label errors before training.

Note: Validator matches the GENERATOR's logic exactly (post-fix relaxed
ADX thresholds for current market conditions).
"""
from __future__ import annotations

import json
from pathlib import Path

RAW_PATH = Path("data/llm_v03_strong_trend_raw.jsonl")


def expected_regime(adx, slope, bull_struct):
    """Mirror of generator's label_consolidation logic."""
    if adx < 15:
        return "range"
    slope_mag = abs(slope)
    if bull_struct and slope > 0.0005:
        return "strong_bull"
    if not bull_struct and slope < -0.0005:
        return "strong_bear"
    return "bull" if bull_struct else "bear"


def validate_example(rec):
    """Check label consistency for one v0.3 example."""
    issues = []
    ctx = rec.get("tf_context", {})
    output = rec.get("output", "")
    try:
        obj = json.loads(output)
    except Exception:
        return ["output not parseable JSON"]

    label = obj.get("regime", "")
    bias = obj.get("bias", "")
    reasoning = obj.get("reasoning", "")

    h = ctx.get("1h", {})
    adx = h.get("adx", 0)
    slope = h.get("ema_50_slope", 0)
    bull_struct = h.get("structure_bull", False)

    expected = expected_regime(adx, slope, bull_struct)
    if label != expected:
        issues.append(
            f"regime mismatch: got {label}, expected {expected} "
            f"(ADX={adx:.1f}, slope={slope:.4f}, bull_struct={bull_struct})"
        )

    expected_bias = "long" if bull_struct else "short"
    if bias != expected_bias:
        issues.append(f"bias mismatch: got {bias}, expected {expected_bias}")

    if label.lower() not in reasoning.lower():
        issues.append(f"reasoning does not mention regime '{label}'")

    return issues


def main():
    print("[*] Validating v0.3 dataset label consistency...")
    records = []
    with open(RAW_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"[*] Loaded {len(records)} records")

    issues_total = 0
    clean = 0
    issues_by_type = {}
    for rec in records:
        issues = validate_example(rec)
        if not issues:
            clean += 1
        for issue in issues:
            issues_total += 1
            key = issue.split(":")[0].split(" mismatch")[0]
            issues_by_type[key] = issues_by_type.get(key, 0) + 1

    print(f"\n[*] Clean records: {clean}/{len(records)} ({100*clean/len(records):.1f}%)")
    print(f"[*] Total issues: {issues_total}")
    if issues_by_type:
        print("\nIssues by type:")
        for k, v in sorted(issues_by_type.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    if issues_total == 0:
        print("\n[OK] All labels are self-consistent with generator logic.")
    else:
        print("\n[WARNING] Some labels are inconsistent. Review before training.")


if __name__ == "__main__":
    main()
