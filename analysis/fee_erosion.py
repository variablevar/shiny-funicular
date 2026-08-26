#!/usr/bin/env python3
"""
Day 4: fee erosion analysis per TF/identifier.

For each backtest result in user_data/backtest_results, computes:
    gross_pnl   = net PnL + total fees (what the strategy made before costs)
    total_fees  = taker/maker fees + funding fees across all trades
    fee_erosion = total_fees / gross_pnl   (only meaningful when gross > 0)

Plan benchmarks (0.05% taker/side): 5m ~10-20% erosion, 1h < 5%.

Usage:
    source venv/bin/activate
    python analysis/fee_erosion.py
"""
from __future__ import annotations

import glob
import json
import zipfile
from pathlib import Path

RESULTS_DIR = Path("ft_userdata/user_data/backtest_results")


def analyze_result(zip_path: Path) -> dict | None:
    with zipfile.ZipFile(zip_path) as z:
        main = [n for n in z.namelist() if n.endswith(".json") and "config" not in n]
        cfg_name = [n for n in z.namelist() if n.endswith("_config.json")]
        if not main:
            return None
        data = json.loads(z.read(main[0]))
        cfg = json.loads(z.read(cfg_name[0])) if cfg_name else {}

    identifier = cfg.get("freqai", {}).get("identifier", "unknown")
    strategies = data.get("strategy", {})
    rows = []
    for strat_name, strat in strategies.items():
        trades = strat.get("trades", [])
        if not trades:
            continue

        total_fees = 0.0
        net_pnl = 0.0
        for t in trades:
            notional = float(t["amount"]) * float(t["open_rate"])
            fees = notional * (float(t["fee_open"]) + float(t["fee_close"]))
            fees += float(t.get("funding_fees", 0.0) or 0.0)
            total_fees += fees
            net_pnl += float(t["profit_abs"])

        gross_pnl = net_pnl + total_fees
        erosion = (total_fees / gross_pnl) if gross_pnl > 0 else None
        rows.append({
            "identifier": identifier,
            "strategy": strat_name,
            "n_trades": len(trades),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "gross_pnl": round(gross_pnl, 2),
            "fee_erosion_pct": round(erosion * 100, 1) if erosion is not None else None,
        })
    return rows


def main() -> None:
    zips = sorted(glob.glob(str(RESULTS_DIR / "*.zip")))
    if not zips:
        print("no backtest results found")
        return

    print(f"{'identifier':28s} {'strategy':18s} {'trades':>7s} {'fees':>9s} "
          f"{'gross':>9s} {'net':>9s} {'erosion':>8s}")
    all_rows = []
    for zp in zips:
        rows = analyze_result(Path(zp)) or []
        for r in rows:
            all_rows.append(r)
            erosion = f"{r['fee_erosion_pct']}%" if r["fee_erosion_pct"] is not None else "n/a"
            print(f"{r['identifier']:28s} {r['strategy']:18s} {r['n_trades']:7d} "
                  f"{r['total_fees']:9.2f} {r['gross_pnl']:9.2f} {r['net_pnl']:9.2f} {erosion:>8s}")

    out = Path("results") / "fee_erosion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fee_erosion.json").write_text(json.dumps(all_rows, indent=2))
    print(f"\nsaved to {out / 'fee_erosion.json'}")


if __name__ == "__main__":
    main()
