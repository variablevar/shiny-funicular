#!/usr/bin/env python3
"""
Day 14 A/B comparison: extract headline metrics from two backtest zips.

Usage:
    source venv/bin/activate
    python analysis/ab_compare.py <baseline.zip> <micro.zip>
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path


def extract(zip_path: str) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        main = [n for n in z.namelist() if n.endswith(".json") and "config" not in n][0]
        cfg_name = [n for n in z.namelist() if n.endswith("_config.json")]
        data = json.loads(z.read(main))
        cfg = json.loads(z.read(cfg_name[0])) if cfg_name else {}

    identifier = cfg.get("freqai", {}).get("identifier", "unknown")
    strat = next(iter(data["strategy"].values()))
    trades = strat.get("trades", [])

    total_fees = 0.0
    for t in trades:
        notional = float(t["amount"]) * float(t["open_rate"])
        total_fees += notional * (float(t["fee_open"]) + float(t["fee_close"]))
        total_fees += float(t.get("funding_fees", 0.0) or 0.0)

    net = float(strat["profit_total_abs"])
    gross = net + total_fees
    wins = sum(1 for t in trades if float(t["profit_abs"]) > 0)
    return {
        "identifier": identifier,
        "strategy": strat["strategy_name"],
        "timerange": strat["timerange"],
        "trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "net_profit_abs": net,
        "net_profit_pct": 100.0 * float(strat["profit_total"]),
        "gross_profit_abs": gross,
        "total_fees": total_fees,
        "fee_erosion_pct": (100.0 * total_fees / gross) if gross > 0 else None,
        "sharpe": float(strat.get("sharpe") or 0.0),
        "sortino": float(strat.get("sortino") or 0.0),
        "max_drawdown_pct": 100.0 * max(
            (float(w.get("max_drawdown", 0.0)) for w in strat.get("wallet_stats", {}).values()),
            default=0.0,
        ) if isinstance(strat.get("wallet_stats"), dict) else None,
    }


def main() -> None:
    rows = [extract(p) for p in sys.argv[1:]]
    keys = ["identifier", "timerange", "trades", "win_rate", "net_profit_abs",
            "net_profit_pct", "gross_profit_abs", "total_fees",
            "fee_erosion_pct", "sharpe", "sortino"]
    header = f"{'metric':18s}" + "".join(f"{r['identifier'][:24]:>26s}" for r in rows)
    print(header)
    for k in keys:
        line = f"{k:18s}"
        for r in rows:
            v = r[k]
            if isinstance(v, float):
                line += f"{v:26.4f}"
            else:
                line += f"{str(v):>26s}"
        print(line)
    out = Path("results") / "day14_ab_micro.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
