#!/usr/bin/env python3
"""
Day 4: walk-forward validation per TF (3 folds, no leakage).

Runs the FreqAI backtest over sequential monthly test windows. Within each
window, FreqAI retrains on a trailing 30-day window every 7 days, so each
fold is genuinely out-of-sample. We then report per-fold metrics and the
Sharpe variance across folds (plan target: variance < 20% of mean Sharpe).

Usage:
    source venv/bin/activate
    python analysis/walk_forward.py \
        --strategy GTQuantMultiTF --config user_data/config.json \
        --folds 20260501-20260601 20260601-20260701 20260701-20260801
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

FT_DIR = Path(__file__).parent.parent / "ft_userdata"


def run_backtest_fold(strategy: str, config: str, timerange: str, freqaimodel: str) -> dict:
    """Run one backtest fold inside the FreqAI container; parse key metrics."""
    log_path = FT_DIR / "user_data/logs" / f"wf_{strategy}_{timerange}.log"
    cmd = [
        "docker-compose", "run", "--rm", "--no-deps", "freqtrade", "backtesting",
        "--config", config,
        "--strategy", strategy,
        "--freqaimodel", freqaimodel,
        "--timerange", timerange,
        "--cache", "none",
    ]
    with open(log_path, "w") as lf:
        subprocess.run(cmd, cwd=FT_DIR, stdout=lf, stderr=subprocess.STDOUT, check=False)

    text = log_path.read_text()
    metrics: dict = {"timerange": timerange}

    # Parse the SUMMARY metrics table from the backtest output.
    def grab(pattern: str, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    metrics["total_profit_pct"] = grab(r"Total profit %\s+│\s*([\-\d.]+)%", float)
    metrics["sharpe"] = grab(r"Sharpe \(closed trades\)\s+│\s*([\-\d.]+)", float)
    metrics["sortino"] = grab(r"Sortino \(closed trades\)\s+│\s*([\-\d.]+)", float)
    metrics["profit_factor"] = grab(r"Profit factor\s+│\s*([\-\d.]+)", float)
    metrics["n_trades"] = grab(r"Total/Daily Avg Trades\s+│\s*(\d+)", int)
    metrics["exceptions"] = len(re.findall(r"raised exception", text))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--config", default="user_data/config.json")
    ap.add_argument("--freqaimodel", default="LightGBMRegressor")
    ap.add_argument("--folds", nargs="+", required=True,
                    help="timeranges like 20260501-20260601")
    args = ap.parse_args()

    results = []
    for fold in args.folds:
        print(f"=== fold {fold} ===")
        m = run_backtest_fold(args.strategy, args.config, fold, args.freqaimodel)
        results.append(m)
        print(json.dumps(m, indent=2))

    valid = [r for r in results if r.get("sharpe") is not None]
    if len(valid) >= 2:
        sharpes = [r["sharpe"] for r in valid]
        mean_sharpe = sum(sharpes) / len(sharpes)
        variance = sum((s - mean_sharpe) ** 2 for s in sharpes) / len(sharpes)
        print("\n=== WALK-FORWARD STABILITY ===")
        print(f"folds: {len(valid)} | mean Sharpe: {mean_sharpe:.3f} | "
              f"variance: {variance:.3f} | std: {variance ** 0.5:.3f}")
        if abs(mean_sharpe) > 1e-9:
            print(f"std/|mean| = {variance ** 0.5 / abs(mean_sharpe):.2%} "
                  f"(plan gate: < 20%)")

    out = Path("results") / "walk_forward"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"wf_{args.strategy}.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved to {out / f'wf_{args.strategy}.json'}")


if __name__ == "__main__":
    main()
