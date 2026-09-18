"""
Short-side asymmetry investigation (Phase 2b follow-up).

Uses the saved FreqAI backtest predictions for identifier
`gtquant-day14-1h-tune` (user_data/models/<id>/backtesting_predictions/) —
these are strategy-independent, shared by GTQuantMultiTF1h and GTQuantV02.

Answers:
  1. Prediction asymmetry: sign-agreement / mean realized 1h return for
     confident long signals vs confident short signals (|pred| > 0.00474).
  2. Regime mix of the 2026-06-14 -> 2026-09-11 window (rule-based regime
     from GTQuantMultiTF._compute_regime + KMeans regime from the artifact).
  3. Per-regime prediction quality by sign (is the model worse at down-moves
     in bull regimes?).
  4. Per-regime trade P&L of the hybrid GTQuantV02 run vs the plain-1h
     reference (trades joined to the regime at entry time).
"""
import json
import pickle
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UD = ROOT / "ft_userdata" / "user_data"
PRED_DIR = UD / "models" / "gtquant-day14-1h-tune" / "backtesting_predictions"
DATA_DIR = UD / "data" / "binance" / "futures"
BT_DIR = UD / "backtest_results"
REGIME_ART = UD / "models" / "regime"

T_START = pd.Timestamp("2026-06-14", tz="UTC")
T_END = pd.Timestamp("2026-09-11", tz="UTC")
ENTRY_THR = 0.00474

sys.path.insert(0, str(UD / "strategies"))
import regime_features as rf  # noqa: E402


def load_predictions():
    frames = []
    for f in sorted(PRED_DIR.glob("cb_*_prediction.feather")):
        pair = "BTC" if "_btc_" in f.name else "ETH"
        df = pd.read_feather(f)
        df["pair"] = pair
        frames.append(df)
    preds = pd.concat(frames, ignore_index=True)
    preds = preds.sort_values("date").drop_duplicates(["pair", "date"], keep="last")
    return preds


def load_candles(pair, tf):
    f = DATA_DIR / f"{pair}_USDT_USDT-{tf}-futures.feather"
    df = pd.read_feather(f)
    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")
    return df


def rule_based_regime(df5):
    """Replicates GTQuantMultiTF._compute_regime (1h informative shifted 1 bar)."""
    df1h = df5.set_index("date").resample("1h").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna().reset_index()
    close = df1h["close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    df1h["ema_50_slope"] = ema50.pct_change(5)
    # Wilder ADX(14) via talib-free approximation is risky; use talib if present.
    try:
        import talib.abstract as ta
        df1h["adx"] = ta.ADX(df1h, timeperiod=14)
    except Exception:
        tr = pd.concat([(df1h["high"] - df1h["low"]),
                        (df1h["high"] - df1h["close"].shift()).abs(),
                        (df1h["low"] - df1h["close"].shift()).abs()], axis=1).max(axis=1)
        plus_dm = (df1h["high"].diff()).clip(lower=0)
        minus_dm = (-df1h["low"].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df1h["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # informative merge: 1h bar closing at t is available from t onwards ->
    # shift the 1h frame one hour before merging onto 5m bar open times.
    df1h["date_m"] = (df1h["date"] + pd.Timedelta("1h")).astype("datetime64[ms, UTC]")
    m = pd.merge_asof(df5[["date", "close"]].sort_values("date"),
                      df1h[["date_m", "ema_50_slope", "adx"]],
                      left_on="date", right_on="date_m")

    slope, adx = m["ema_50_slope"], m["adx"]
    rv = df5["close"].pct_change().rolling(12).std()
    rv_med = rv.rolling(288).median()

    regime = pd.Series("RANGE", index=df5.index, dtype=object)
    trending = adx >= 20.0
    strong_bull = (slope > 0.0018) & trending
    strong_bear = (slope < -0.0018) & trending
    regime[(slope > 0) & ~strong_bull] = "BULL"
    regime[(slope < 0) & ~strong_bear] = "BEAR"
    regime[strong_bull] = "STRONG_BULL"
    regime[strong_bear] = "STRONG_BEAR"
    regime[adx < 15.0] = "RANGE"
    regime[rv.values > (rv_med * 1.5).values] = "HIGH_VOL"
    return regime, df1h


def kmeans_regime(df1h):
    """KMeans regime + action per 1h bar from the saved artifact."""
    summary = json.loads((REGIME_ART / "regime_summary.json").read_text())
    with open(REGIME_ART / "regime_runtime.pkl", "rb") as f:
        art = pickle.load(f)
    feats = rf.build_regime_features(df1h)
    valid = feats.notna().all(axis=1)
    action = pd.Series("UNKNOWN", index=df1h.index, dtype=object)
    name = pd.Series("UNKNOWN", index=df1h.index, dtype=object)
    if valid.any():
        X = art["scaler"].transform(feats.loc[valid, rf.FEATURE_COLS].astype(float).to_numpy())
        clusters = art["model"].predict(X)
        amap = {int(k): v for k, v in summary["cluster_action_map"].items()}
        rmap = {int(k): v for k, v in summary["cluster_regime_map"].items()}
        action.loc[valid] = [amap.get(int(c), "UNKNOWN") for c in clusters]
        name.loc[valid] = [rmap.get(int(c), "UNKNOWN") for c in clusters]
    out = df1h[["date"]].copy()
    out["kmeans_regime"] = name.values
    out["kmeans_action"] = action.values
    return out


def load_trades(zip_name):
    with zipfile.ZipFile(BT_DIR / zip_name) as z:
        data = json.loads(z.read([n for n in z.namelist()
                                  if n.endswith(".json") and "config" not in n][0]))
    strat = list(data["strategy"].keys())[0]
    trades = pd.DataFrame(data["strategy"][strat]["trades"])
    return strat, trades


def main():
    preds = load_predictions()
    print(f"predictions: {len(preds)} rows, "
          f"{preds['date'].min()} -> {preds['date'].max()}")

    # --- 5m candles, realized forward returns, regimes, per pair ---
    frames = []
    km_1h = {}
    for pair in ("BTC", "ETH"):
        df5 = load_candles(pair, "5m")
        df5["realized_1h"] = df5["close"].shift(-12) / df5["close"] - 1
        regime, df1h = rule_based_regime(df5)
        df5["regime"] = regime
        km = kmeans_regime(df1h)
        km["date_m"] = (km["date"] + pd.Timedelta("1h")).astype("datetime64[ms, UTC]")
        km_1h[pair] = km
        df5 = pd.merge_asof(df5.sort_values("date"),
                            km[["date_m", "kmeans_regime", "kmeans_action"]],
                            left_on="date", right_on="date_m")
        df5["pair"] = pair
        frames.append(df5[["date", "pair", "close", "realized_1h", "regime",
                           "kmeans_regime", "kmeans_action"]])
    px = pd.concat(frames, ignore_index=True)

    df = preds.merge(px, on=["pair", "date"], how="left")
    df = df[(df["date"] >= T_START) & (df["date"] < T_END)]
    df = df[df["do_predict"] == 1]
    print(f"confident rows in window: {len(df)}")

    # ---------- 1. prediction asymmetry ----------
    print("\n== 1. Prediction asymmetry (confident rows, window) ==")
    for label, mask in [
        ("all", df["&-s-future_return_1h"].notna()),
        ("pred > +thr (long signal)", df["&-s-future_return_1h"] > ENTRY_THR),
        ("pred < -thr (short signal)", df["&-s-future_return_1h"] < -ENTRY_THR),
        ("|pred| <= thr", df["&-s-future_return_1h"].abs() <= ENTRY_THR),
    ]:
        sub = df[mask & df["realized_1h"].notna()]
        if not len(sub):
            continue
        p, r = sub["&-s-future_return_1h"], sub["realized_1h"]
        agree = (np.sign(p) == np.sign(r)).mean()
        print(f"  {label:<28} n={len(sub):<6} corr={p.corr(r):+.3f} "
              f"sign_agree={agree:.3f} mean_pred={p.mean():+.5f} "
              f"mean_real={r.mean():+.5f} med_real={r.median():+.5f}")

    # ---------- 2. regime mix ----------
    print("\n== 2. Regime mix in window (5m bars, both pairs pooled) ==")
    win = px[(px["date"] >= T_START) & (px["date"] < T_END)]
    for col in ("regime", "kmeans_regime", "kmeans_action"):
        vc = win[col].value_counts(normalize=True).sort_index()
        print(f"  {col}: " + ", ".join(f"{k}={v:.1%}" for k, v in vc.items()))

    # ---------- 3. per-regime signal quality ----------
    print("\n== 3. Per-regime: confident signals, mean realized 1h by direction ==")
    for reg in ("STRONG_BULL", "BULL", "RANGE", "BEAR", "STRONG_BEAR", "HIGH_VOL"):
        sub = df[df["regime"] == reg]
        longs = sub[sub["&-s-future_return_1h"] > ENTRY_THR]
        shorts = sub[sub["&-s-future_return_1h"] < -ENTRY_THR]
        if not len(sub):
            continue
        ls = f"long_sig n={len(longs):<5} mean_real={longs['realized_1h'].mean():+.5f}" \
            if len(longs) else "long_sig n=0"
        ss = f"short_sig n={len(shorts):<5} mean_real={shorts['realized_1h'].mean():+.5f}" \
            if len(shorts) else "short_sig n=0"
        print(f"  {reg:<12} {ls}   {ss}")

    # ---------- 4. per-regime trade P&L ----------
    for zip_name in ("backtest-result-2026-09-17_18-37-24.zip",   # hybrid V02 final
                     "backtest-result-2026-09-16_22-50-53.zip"):  # plain 1h tuned ref
        strat, trades = load_trades(zip_name)
        print(f"\n== 4. Per-regime trade P&L: {strat} ({zip_name}) ==")
        trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True).astype("datetime64[ms, UTC]")
        trades["pair2"] = trades["pair"].str.split("/").str[0]
        joined = pd.merge_asof(
            trades.sort_values("open_date"),
            px[["date", "pair", "regime", "kmeans_regime", "kmeans_action"]]
                .rename(columns={"date": "open_date", "pair": "pair2"})
                .sort_values("open_date"),
            on="open_date", by="pair2", direction="backward")
        for col in ("regime", "kmeans_regime"):
            print(f"  -- by {col} --")
            agg = defaultdict(lambda: [0, 0.0])
            for _, t in joined.iterrows():
                d = "S" if t["is_short"] else "L"
                key = (t[col], d)
                agg[key][0] += 1
                agg[key][1] += t["profit_abs"]
            for (reg, d), (n, pnl) in sorted(agg.items()):
                print(f"     {str(reg):<14} [{d}] n={n:<4} pnl={pnl:>9.2f}")


if __name__ == "__main__":
    main()
