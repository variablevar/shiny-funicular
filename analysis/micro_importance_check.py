#!/usr/bin/env python3
"""
Day 14 hypothesis check: do 1m microstructure features rank in the top ~20
by LightGBM gain importance for the 5m-horizon model?

Reads the feature_importance_{pair}.csv files written by
analysis/analyze_models.py for the micro identifier and ranks all features.

Usage:
    source venv/bin/activate
    python analysis/analyze_models.py gtquant-v0.2-5m-micro --tag day14-micro
    python analysis/micro_importance_check.py gtquant-v0.2-5m-micro
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MODELS_ROOT = Path("ft_userdata/user_data/models")


def check(identifier: str) -> None:
    id_dir = MODELS_ROOT / identifier
    csvs = sorted(id_dir.glob("feature_importance_*.csv"))
    if not csvs:
        print(f"no feature_importance CSVs under {id_dir} — run analyze_models.py first")
        return

    any_top20 = False
    for csv in csvs:
        pair = csv.stem.replace("feature_importance_", "")
        imp = pd.read_csv(csv).sort_values("gain_norm", ascending=False).reset_index(drop=True)
        imp["rank"] = imp.index + 1
        micro = imp[imp["feature"].str.contains("micro_", regex=False)]

        print(f"\n=== {identifier} / {pair} ({len(imp)} features) ===")
        if micro.empty:
            print("  NO micro features in the trained model!")
            continue
        micro_share = micro["gain_norm"].sum()
        in_top20 = micro[micro["rank"] <= 20]
        any_top20 |= not in_top20.empty
        print(f"  micro features: {len(micro)}, gain share: {micro_share:.2%}, "
              f"in top 20: {len(in_top20)}")
        for _, row in micro.nsmallest(10, "rank").iterrows():
            marker = " <-- TOP 20" if row["rank"] <= 20 else ""
            print(f"  rank {int(row['rank']):4d}  {row['feature'][:55]:55s} "
                  f"gain_norm={row['gain_norm']:.4f}{marker}")

    print("\n=== Day-3 hypothesis (1m micro in top ~20 for 5m horizon): "
          + ("CONFIRMED" if any_top20 else "NOT confirmed") + " ===")


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "gtquant-v0.2-5m-micro")
