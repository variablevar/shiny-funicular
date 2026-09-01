#!/usr/bin/env python3
"""
Day 8: TF-aware LLM training dataset generator.

Builds structured multi-timeframe reasoning examples from the downloaded
BTC/ETH futures data (5m/15m/30m/1h/4h + funding). Ground truth is derived
from ACTUAL forward returns — not synthetic LLM output — so the model learns
real outcome-conditioned reasoning.

Output JSON per plan Day 8:
    {"regime", "primary_tf", "bias", "confidence", "risk",
     "entry_precision", "size_adjustment"}

Split 80/10/10 train/val/test -> data/llm_v2_*.jsonl

Usage:
    source venv/bin/activate
    python generate_llm_dataset_v2.py --n 2000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

DATA_DIR = Path("ft_userdata/user_data/data/binance/futures")
OUT_DIR = Path("data")

# --------------------------------------------------------------------- #
# Minimal indicator helpers (pandas; no talib in the venv)
# --------------------------------------------------------------------- #

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_v = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_v
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_v
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def regime_label(slope: float, adx_v: float) -> str:
    """Same rule set as the strategy's _compute_regime (minus HIGH_VOL)."""
    if adx_v < 20:
        return "RANGE"
    if slope > 0.0028 and adx_v > 25:
        return "STRONG_BULL"
    if slope < -0.0028 and adx_v > 25:
        return "STRONG_BEAR"
    return "BULL" if slope > 0 else "BEAR"


# --------------------------------------------------------------------- #
# TF snapshot assembly
# --------------------------------------------------------------------- #

class TFContext:
    """Precomputed per-TF frames with indicators; snapshot by timestamp."""

    def __init__(self, pair_file: str):
        self.frames: dict[str, pd.DataFrame] = {}
        for tf in ["5m", "15m", "30m", "1h", "4h"]:
            df = pd.read_feather(DATA_DIR / f"{pair_file}-{tf}-futures.feather")
            df = df.set_index("date")
            self.frames[tf] = self._add_indicators(df, tf)

    @staticmethod
    def _add_indicators(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        df = df.copy()
        df["ema_9"] = ema(df["close"], 9)
        df["ema_21"] = ema(df["close"], 21)
        df["ema_50"] = ema(df["close"], 50)
        df["ema_200"] = ema(df["close"], 200)
        df["rsi"] = rsi(df["close"])
        macd = ema(df["close"], 12) - ema(df["close"], 26)
        df["macd_hist"] = macd - ema(macd, 9)
        df["atr"] = atr(df)
        df["adx"] = adx(df)
        df["ema_50_slope"] = df["ema_50"].pct_change(5)
        df["atr_pct"] = df["atr"] / df["close"] * 100
        return df

    def at(self, tf: str, ts: pd.Timestamp) -> pd.Series | None:
        """Latest row at or before ts (no lookahead)."""
        df = self.frames[tf]
        idx = df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return None
        return df.iloc[idx]


def fmt_snapshot(pair: str, ctx: TFContext, ts: pd.Timestamp,
                 funding: float | None, funding_z: float | None,
                 cvd_1h: float | None) -> str | None:
    """Build the plan's TF-context user prompt. None if any TF is missing."""
    r5, r15, r30, r1h, r4h = (ctx.at(tf, ts) for tf in ["5m", "15m", "30m", "1h", "4h"])
    if any(r is None for r in [r5, r15, r30, r1h, r4h]):
        return None

    reg1h = regime_label(r1h["ema_50_slope"], r1h["adx"])

    # Funding shown as z-score (the actual edge) + raw for reference.
    if funding_z is not None and not pd.isna(funding_z):
        fz = float(funding_z)
        fz_note = ("extreme, longs paying" if fz > 2 else
                   "extreme, shorts paying" if fz < -2 else
                   "elevated" if abs(fz) > 1 else "neutral")
        funding_str = f"z={fz:+.1f} ({fz_note})"
    else:
        funding_str = "n/a"

    cvd_str = f"{cvd_1h:+.2f}" if cvd_1h is not None and not pd.isna(cvd_1h) else "n/a"

    return f"""Pair: {pair}
5m: EMA9{'>' if r5['ema_9'] > r5['ema_21'] else '<'}EMA21, RSI={r5['rsi']:.0f}, MACD_hist={r5['macd_hist']:+.1f}, ATR={r5['atr_pct']:.2f}%
15m: price{'>' if r15['close'] > r15['ema_50'] else '<'}EMA50, RSI={r15['rsi']:.0f}, EMA50_slope={r15['ema_50_slope']*100:+.3f}%
30m: EMA50{'>' if r30['ema_50'] > r30['ema_200'] else '<'}EMA200, ADX={r30['adx']:.0f}
1h: EMA50_slope={r1h['ema_50_slope']*100:+.3f}%, ADX={r1h['adx']:.0f}({'trending' if r1h['adx'] > 25 else 'weak'}), regime={reg1h}
4h: EMA50{'>' if r4h['ema_50'] > r4h['ema_200'] else '<'}EMA200, structure={'bullish' if r4h['ema_50'] > r4h['ema_200'] else 'bearish'}
Funding z-score: {funding_str}
CVD 1h (signed volume): {cvd_str}

Output JSON with keys: regime, primary_tf, bias, confidence, risk, entry_precision, size_adjustment"""


def tf_alignment(r5: pd.Series, r1h: pd.Series, r4h: pd.Series) -> int:
    """
    Directional agreement across TFs, -3..+3.
    5m: EMA9 vs EMA21; 1h: EMA-50 slope sign; 4h: structure (EMA50 vs 200).
    """
    score = 0
    score += 1 if r5["ema_9"] > r5["ema_21"] else -1
    score += 1 if r1h["ema_50_slope"] > 0 else -1
    score += 1 if r4h["ema_50"] > r4h["ema_200"] else -1
    return score


def ground_truth(ctx: TFContext, ts: pd.Timestamp, funding: float | None,
                 funding_z: float | None) -> dict | None:
    """Derive the correct output from ACTUAL forward returns (1h/4h on 5m frame)."""
    df5 = ctx.frames["5m"]
    i = df5.index.searchsorted(ts, side="right") - 1
    if i < 0 or i + 48 >= len(df5):
        return None

    close_now = df5["close"].iloc[i]
    ret_1h = df5["close"].iloc[i + 12] / close_now - 1
    ret_4h = df5["close"].iloc[i + 48] / close_now - 1
    atr_pct = df5["atr_pct"].iloc[i]

    r5 = ctx.at("5m", ts)
    r1h = ctx.at("1h", ts)
    r4h = ctx.at("4h", ts)
    regime = regime_label(r1h["ema_50_slope"], r1h["adx"])

    # Bias from the dominant horizon move (ground truth = what happened).
    # Regime-aware threshold: in STRONG_* regimes trends consolidate, so a
    # smaller realized move still counts as directional continuation. In
    # weaker regimes we demand a decisive move. (Fixes the strong_trend
    # labeler that over-labeled flat in Day 8 v1.)
    if regime in ("STRONG_BULL", "STRONG_BEAR"):
        dir_thresh_4h, dir_thresh_1h = 0.004, 0.002
    else:
        dir_thresh_4h, dir_thresh_1h = 0.01, 0.005

    if ret_4h > dir_thresh_4h or (ret_1h > dir_thresh_1h and ret_4h > 0):
        bias = "long"
    elif ret_4h < -dir_thresh_4h or (ret_1h < -dir_thresh_1h and ret_4h < 0):
        bias = "short"
    else:
        bias = "flat"

    # Confidence from TF AGREEMENT with the realized direction (not just
    # magnitude). This is the contradiction guard: 4h bullish + price fell
    # -> the short label carries LOW confidence, teaching the model that
    # conflicting structure means weak conviction.
    align = tf_alignment(r5, r1h, r4h)
    if bias == "long":
        agree = align
    elif bias == "short":
        agree = -align
    else:
        agree = 0

    magnitude = abs(ret_4h)
    if bias == "flat":
        confidence = 0.6
    elif agree >= 3:      # all TFs agree with outcome
        confidence = float(np.clip(0.80 + magnitude * 4, 0.80, 0.95))
    elif agree >= 1:      # majority agree
        confidence = float(np.clip(0.65 + magnitude * 4, 0.65, 0.80))
    else:                 # conflicting structure
        confidence = float(np.clip(0.52 + magnitude * 3, 0.52, 0.64))

    # Contradiction guard: if the realized direction fights the 4h structure
    # AND the move wasn't decisive, the honest label is flat (conflict = no trade).
    structure_bull = r4h["ema_50"] > r4h["ema_200"]
    if bias == "short" and structure_bull and magnitude < 0.02:
        bias, confidence = "flat", 0.55
    elif bias == "long" and not structure_bull and magnitude < 0.02:
        bias, confidence = "flat", 0.55

    # Risk from funding crowding (z-score) + volatility.
    fz = float(funding_z) if funding_z is not None and not pd.isna(funding_z) else 0.0
    if abs(fz) > 2:
        risk = "crowded"
    elif atr_pct > 1.5:
        risk = "high"
    elif bias == "flat":
        risk = "low"
    else:
        risk = "moderate"

    primary_tf = "1h" if regime in ("STRONG_BULL", "STRONG_BEAR") else ("15m" if regime in ("BULL", "BEAR") else "5m")
    entry_precision = "wait_for_1m_pullback" if bias != "flat" else "no_entry"
    size_adjustment = 0.5 if risk in ("crowded", "high") else (1.5 if regime.startswith("STRONG") else 1.0)

    return {
        "regime": regime.lower(),
        "primary_tf": primary_tf,
        "bias": bias,
        "confidence": round(confidence, 2),
        "risk": risk,
        "entry_precision": entry_precision,
        "size_adjustment": size_adjustment,
    }


SYSTEM = ("You are GT-Quant, a crypto trading analyst. Analyze the multi-timeframe "
          "context and output a JSON decision. Consider: 1m for execution timing, 5m "
          "for primary signals, 15m/30m for trend confirmation, 1h/4h for regime.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="examples to generate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    funding_df = pd.read_feather(DATA_DIR / "BTC_USDT_USDT-1h-funding_rate.feather").set_index("date")
    # Freqtrade stores the funding rate series in the `open` column.
    funding_df = funding_df.rename(columns={"open": "funding_rate"})
    # Funding z-score over the trailing 30 days (720 1h marks) — the real edge
    # signal (is this funding extreme vs recent history?).
    fr = funding_df["funding_rate"]
    funding_df["funding_z"] = (fr - fr.rolling(720).mean()) / fr.rolling(720).std()
    logger.info(f"funding rows: {len(funding_df)}")

    examples = []
    for pair_file, pair in [("BTC_USDT_USDT", "BTC/USDT"), ("ETH_USDT_USDT", "ETH/USDT")]:
        ctx = TFContext(pair_file)
        df5 = ctx.frames["5m"]

        # CVD proxy on the 5m frame: signed volume (up/down candle) summed
        # over the trailing hour (12 x 5m). True taker CVD needs the 1m
        # collector feed; this is the historical approximation.
        signed = np.where(df5["close"] >= df5["open"], df5["volume"], -df5["volume"])
        ctx.frames["5m"]["cvd_1h"] = pd.Series(signed, index=df5.index).rolling(12).sum()

        # Sample timestamps with enough history + future room.
        valid = df5.index[300:-60]
        sample_ts = random.sample(list(valid), min(args.n // 2, len(valid)))

        for ts in sample_ts:
            if len(funding_df):
                f_row = funding_df["funding_rate"].asof(ts)
                fz_row = funding_df["funding_z"].asof(ts)
            else:
                f_row = fz_row = None
            # CVD proxy value at this timestamp (no lookahead).
            cvd_series = ctx.frames["5m"]["cvd_1h"]
            cvd_idx = cvd_series.index.searchsorted(ts, side="right") - 1
            cvd_row = float(cvd_series.iloc[cvd_idx]) if cvd_idx >= 0 else None
            prompt = fmt_snapshot(pair, ctx, ts, f_row, fz_row, cvd_row)
            truth = ground_truth(ctx, ts, f_row, fz_row)
            if prompt is None or truth is None:
                continue
            examples.append({
                "instruction": SYSTEM,
                "input": prompt,
                "output": json.dumps(truth),
            })
        logger.info(f"{pair}: {len(examples)} examples so far")

    random.shuffle(examples)

    # Balance bias classes: downsample the majority (flat) so the model
    # learns directional reasoning, not just "always flat".
    by_bias: dict[str, list] = {"long": [], "short": [], "flat": []}
    for ex in examples:
        by_bias[json.loads(ex["output"])["bias"]].append(ex)
    target = min(len(v) for v in by_bias.values()) if all(by_bias.values()) else 0
    if target > 0:
        # cap majority classes at 1.3x the smallest class (keep some realism)
        cap = int(target * 1.3)
        balanced = []
        for bias, exs in by_bias.items():
            random.shuffle(exs)
            balanced.extend(exs[:cap] if len(exs) > cap else exs)
        random.shuffle(balanced)
        logger.info(f"balanced: {dict((b, len(e)) for b, e in by_bias.items())} -> {len(balanced)} examples")
        examples = balanced

    n = len(examples)
    n_train, n_val = int(n * 0.8), int(n * 0.1)

    OUT_DIR.mkdir(exist_ok=True)
    for name, subset in [("train", examples[:n_train]),
                         ("val", examples[n_train:n_train + n_val]),
                         ("test", examples[n_train + n_val:])]:
        path = OUT_DIR / f"llm_v02_{name}.jsonl"
        with open(path, "w") as f:
            for ex in subset:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"wrote {path} ({len(subset)})")

    logger.info(f"\ntotal: {n} examples | train {n_train} | val {n_val} | test {n - n_train - n_val}")
    logger.info("\n=== SAMPLE ===")
    logger.info(examples[0]["input"])
    logger.info(examples[0]["output"])


if __name__ == "__main__":
    main()
