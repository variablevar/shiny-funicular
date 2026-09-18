"""
Decompose backtest trade P&L by direction / exit reason / regime.

Reads trade lists out of freqtrade backtest result zips and prints:
  - P&L by direction (long/short)
  - P&L by exit reason, split by direction
  - P&L by month and by pair (regime proxy for the window)
  - for shorts: P&L grouped by the sign of price change over the trade
    (did shorts lose in up-moves?)

Usage: venv/bin/python analysis/decompose_trades.py <zip> [<zip> ...]
"""
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

BT_DIR = Path(__file__).resolve().parent.parent / "ft_userdata" / "user_data" / "backtest_results"


def load_trades(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        json_names = [n for n in z.namelist()
                      if n.endswith(".json") and "config" not in n]
        data = json.loads(z.read(json_names[0]))
    strat = list(data["strategy"].keys())[0]
    s = data["strategy"][strat]
    trades = s.get("trades") or []
    return strat, s, trades


def summarize(name, trades):
    n = len(trades)
    pnl = sum(t["profit_abs"] for t in trades)
    wins = sum(1 for t in trades if t["profit_abs"] > 0)
    wr = wins / n * 100 if n else 0.0
    return f"{name:<38} n={n:<4} pnl={pnl:>9.2f}  WR={wr:5.1f}%"


def report(zip_path: Path):
    strat, s, trades = load_trades(zip_path)
    print("=" * 100)
    print(f"{zip_path.name}  strategy={strat}")
    comp = s.get("strategy_comparison") or [{}]
    print(f"  window: {s.get('backtest_start')} -> {s.get('backtest_end')}  "
          f"timerange={s.get('timerange')}")
    if not trades:
        print("  (no trade list in zip)")
        return
    print(f"  total trades: {len(trades)}  pnl: {sum(t['profit_abs'] for t in trades):.2f}")

    by_dir = defaultdict(list)
    for t in trades:
        by_dir["short" if t.get("is_short") else "long"].append(t)
    print("\n-- by direction --")
    for d in ("long", "short"):
        if by_dir[d]:
            print("   " + summarize(d, by_dir[d]))

    print("\n-- by exit reason x direction --")
    combos = defaultdict(list)
    for t in trades:
        d = "S" if t.get("is_short") else "L"
        combos[(t.get("exit_reason", "?"), d)].append(t)
    for (reason, d), ts in sorted(combos.items(),
                                  key=lambda kv: sum(t["profit_abs"] for t in kv[1])):
        print("   " + summarize(f"{reason} [{d}]", ts))

    print("\n-- by month x direction --")
    by_m = defaultdict(list)
    for t in trades:
        month = str(t.get("open_date", ""))[:7]
        d = "S" if t.get("is_short") else "L"
        by_m[(month, d)].append(t)
    for (m, d), ts in sorted(by_m.items()):
        print("   " + summarize(f"{m} [{d}]", ts))

    print("\n-- by pair x direction --")
    by_p = defaultdict(list)
    for t in trades:
        d = "S" if t.get("is_short") else "L"
        by_p[(t["pair"].split("/")[0], d)].append(t)
    for (p, d), ts in sorted(by_p.items()):
        print("   " + summarize(f"{p} [{d}]", ts))

    print("\n-- shorts: outcome vs underlying move while trade open --")
    up, down = [], []
    for t in trades:
        if not t.get("is_short"):
            continue
        move = t["close_rate"] / t["open_rate"] - 1
        (up if move > 0 else down).append(t)
    if up:
        print("   " + summarize("shorts closed higher (adverse)", up))
    if down:
        print("   " + summarize("shorts closed lower (favorable)", down))
    print()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    for a in args:
        p = Path(a)
        if not p.exists():
            p = BT_DIR / a
            if not p.suffix:
                p = p.with_suffix(".zip")
        report(p)


if __name__ == "__main__":
    main()
