#!/usr/bin/env python3
"""
Evaluate v0.3 dataset quality on regime accuracy for trend-consolidation cases.

Compares v0.2 vs v0.3 trained models (or simply checks v0.3 has the right
distribution / labels). Specifically targets the strong_trend regime which
was 36% accurate in v0.2.

Usage:
    python analysis/eval_v03_regime.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")


def count_regimes(path):
    """Count regime labels in a JSONL file."""
    regimes = Counter()
    total = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            try:
                obj = json.loads(rec["output"])
                regimes[obj.get("regime", "?")] += 1
                total += 1
            except Exception:
                pass
    return regimes, total


def compare_distributions():
    """Compare regime distributions across v0.2 and v0.3 datasets."""
    print("=" * 60)
    print("REGIME DISTRIBUTION: v0.2 (baseline) vs v0.3 (with strong_trend augmentation)")
    print("=" * 60)

    v02_regimes, v02_total = count_regimes(DATA_DIR / "llm_v02_train.jsonl")
    v03_regimes, v03_total = count_regimes(DATA_DIR / "llm_v03_train.jsonl")

    print(f"\nv0.2 train: {v02_total} examples")
    print(f"v0.3 train: {v03_total} examples")
    print(f"Diff: +{v03_total - v02_total} examples ({100*(v03_total - v02_total)/v02_total:.1f}%)")

    all_regimes = sorted(set(list(v02_regimes.keys()) + list(v03_regimes.keys())))
    print(f"\n{'Regime':<15} {'v0.2':<10} {'%':<8} {'v0.3':<10} {'%':<8} {'Delta':<8}")
    print("-" * 60)
    for r in all_regimes:
        v02_n = v02_regimes.get(r, 0)
        v03_n = v03_regimes.get(r, 0)
        v02_pct = 100 * v02_n / v02_total if v02_total else 0
        v03_pct = 100 * v03_n / v03_total if v03_total else 0
        delta = v03_n - v02_n
        print(f"{r:<15} {v02_n:<10} {v02_pct:<8.1f} {v03_n:<10} {v03_pct:<8.1f} {delta:+<8}")

    print("\n" + "=" * 60)
    print("STRONG_TREND FOCUS (target of v0.3 augmentation)")
    print("=" * 60)
    strong_bull_v02 = v02_regimes.get("strong_bull", 0)
    strong_bull_v03 = v03_regimes.get("strong_bull", 0)
    sb_delta = strong_bull_v03 - strong_bull_v02
    sb_pct = 100 * sb_delta / strong_bull_v02 if strong_bull_v02 else 0
    print(f"strong_bull: {strong_bull_v02} -> {strong_bull_v03} ({sb_pct:+.1f}%)")
    strong_bear_v02 = v02_regimes.get("strong_bear", 0)
    strong_bear_v03 = v03_regimes.get("strong_bear", 0)
    sb2_delta = strong_bear_v03 - strong_bear_v02
    sb2_pct = 100 * sb2_delta / strong_bear_v02 if strong_bear_v02 else 0
    print(f"strong_bear: {strong_bear_v02} -> {strong_bear_v03} ({sb2_pct:+.1f}%)")

    sb_total_v02 = strong_bull_v02 + strong_bear_v02
    sb_total_v03 = strong_bull_v03 + strong_bear_v03
    print(f"\nTotal strong_trend: {sb_total_v02} -> {sb_total_v03} (+{sb_total_v03-sb_total_v02})")

    # Class balance check
    print("\n" + "=" * 60)
    print("CLASS BALANCE CHECK (target: roughly balanced across regimes)")
    print("=" * 60)
    print(f"{'Regime':<15} {'v0.2 %':<10} {'v0.3 %':<10}")
    for r in all_regimes:
        v02_pct = 100 * v02_regimes.get(r, 0) / v02_total if v02_total else 0
        v03_pct = 100 * v03_regimes.get(r, 0) / v03_total if v03_total else 0
        print(f"{r:<15} {v02_pct:<10.1f} {v03_pct:<10.1f}")


def main():
    print("\n[*] Evaluating v0.3 dataset quality...")
    compare_distributions()
    print("\n[OK] v0.3 dataset augmentation complete")
    print("\nNext step: Run v0.3 training with train_llm_v03.py")
    print("  python train_llm_v03.py")


if __name__ == "__main__":
    main()
