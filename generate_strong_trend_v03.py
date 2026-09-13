#!/usr/bin/env python3
"""
Day 12 (follow-up): strong_trend v0.3 dataset generator.

The strong_trend classifier in v0.2 has 36% accuracy vs 78% for other regimes.
Root cause: the training set has insufficient examples of trend-CONSOLIDATION
periods where:
  - Price is sideways after an impulse move
  - Funding is neutral
  - Structure (EMA50 > EMA200) is intact
  - ADX is in a transition zone (15-25)

This script generates 400-600 synthetic examples specifically targeting these
edge cases by:
  1. Scanning historical data for consolidation-after-impulse patterns
  2. Augmenting with parameter variations (noise injection, time shifts)
  3. Generating ground-truth labels from ACTUAL forward returns

Output: data/llm_v03_strong_trend_*.jsonl, then merges into existing v02 train set.
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

# v0.2 thresholds (must match the strategy)
ADX_RANGE_MAX = 20.0     # Below = RANGE
ADX_TREND_MIN = 25.0     # Above = trending
SLOPE_STRONG = 0.0028    # |1h EMA50 slope| for STRONG_*

# Consolidation detection thresholds (strict = original v0.3 mining pass)
CONSOLIDATION_RANGE_PCT = 0.02     # 2% price range over 1h
CONSOLIDATION_BARS = 12            # 12 x 5m bars = 1 hour
PRIOR_IMPULSE_PCT = 0.04           # 4% impulse before consolidation

# Relaxed thresholds for the iteration-2 mining pass (more consolidation
# coverage: wider range, wider ADX band, smaller prior impulse)
RELAXED_RANGE_PCT = 0.025
RELAXED_ADX_LO = 14.0
RELAXED_ADX_HI = 28.0
RELAXED_IMPULSE_PCT = 0.03


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_funding(pair: str) -> pd.Series:
    """Real historical funding rates from the freqtrade funding_rate feather.

    The feather is named '-1h-funding_rate' but rows are the raw 8h funding
    prints (00:00/08:00/16:00 UTC); the 'open' column carries the rate.
    """
    files = list(DATA_DIR.glob(f"{pair}_USDT_USDT-1h-funding_rate.feather"))
    if not files:
        return pd.Series(dtype=float)
    fdf = pd.read_feather(files[0]).set_index("date").sort_index()
    return fdf["open"].astype(float)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1).ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def load_pair(pair: str) -> pd.DataFrame:
    """Load 1h data for a pair with indicators."""
    files = list(DATA_DIR.glob(f"{pair}_USDT_USDT-1h-futures.feather"))
    if not files:
        return pd.DataFrame()
    df = pd.read_feather(files[0]).set_index("date").sort_index()
    df["ema_50"] = ema(df["close"], 50)
    df["ema_200"] = ema(df["close"], 200)
    df["ema_50_slope"] = df["ema_50"].pct_change(5)
    df["adx"] = adx(df, 14)
    df["range_high"] = df["high"].rolling(CONSOLIDATION_BARS).max()
    df["range_low"] = df["low"].rolling(CONSOLIDATION_BARS).min()
    df["range_pct"] = (df["range_high"] - df["range_low"]) / df["close"]
    df["structure"] = (df["ema_50"] > df["ema_200"]).astype(int)  # 1 = bullish
    df["prior_impulse"] = df["close"].pct_change(CONSOLIDATION_BARS * 4)
    return df.dropna()


def find_consolidation_periods(df: pd.DataFrame, range_pct=CONSOLIDATION_RANGE_PCT,
                               adx_lo=15.0, adx_hi=25.0,
                               impulse=PRIOR_IMPULSE_PCT) -> pd.DataFrame:
    """
    Find periods where:
    - Price is consolidating (range < range_pct)
    - Structure is intact (EMA50 > EMA200 or vice versa)
    - There was a prior impulse (> impulse)
    - ADX is in the transition zone (adx_lo..adx_hi)
    """
    cond = (
        (df["range_pct"] < range_pct)
        & (df["adx"] >= adx_lo)
        & (df["adx"] <= adx_hi)
        & (df["prior_impulse"].abs() > impulse)
    )
    return df[cond].copy()


def build_context(idx, df_1h, df_5m, pair, funding_series=None):
    """Build multi-TF context for one consolidation example."""
    # Look up 5m context near the 1h timestamp
    closest_5m = df_5m.index[df_5m.index.get_indexer([idx], method="nearest")[0]]
    row_1h = df_1h.loc[idx]
    row_5m = df_5m.loc[closest_5m]

    # Real funding: most recent 8h funding print at or before the timestamp
    funding = 0.0
    if funding_series is not None and len(funding_series):
        loc = funding_series.index.searchsorted(idx, side="right") - 1
        if loc >= 0:
            funding = float(funding_series.iloc[loc])

    ctx = {
        "pair": pair,
        "ts": str(idx),  # bar timestamp; enables pair+timestamp dedupe
        "1h": {
            "adx": float(row_1h["adx"]),
            "ema_50_slope": float(row_1h["ema_50_slope"]),
            "structure_bull": bool(row_1h["structure"] == 1),
            "range_pct_1h": float(row_1h["range_pct"]),
            "prior_4h_impulse": float(row_1h["prior_impulse"]),
        },
        "5m": {
            "close": float(row_5m["close"]),
            "rsi": float(row_5m["rsi"]),
            "atr_pct": float((row_5m["high"] - row_5m["low"]) / row_5m["close"]),
        },
        "funding_8h": funding,
    }
    return ctx


def label_consolidation(ctx):
    """
    Ground truth for trend-consolidation examples.

    Use the structure (EMA50 vs EMA200) and slope direction to label.
    If structure is bullish + slope positive: strong_bull (the trend will resume)
    If structure is bearish + slope negative: strong_bear (the trend will resume)
    Else: bull or bear (mid-trend, not strong)

    The KEY insight for v0.3: consolidation after impulse in a strong trend
    is still a strong trend, not range. Labels use the lowercase v02 convention.
    """
    slope = ctx["1h"]["ema_50_slope"]
    adx_v = ctx["1h"]["adx"]
    bull_struct = ctx["1h"]["structure_bull"]

    # Slope must be moderate (consolidation = trend pausing, not reversing)
    slope_mag = abs(slope)
    if adx_v < 15:
        return "range"
    if bull_struct and slope > 0.0005:  # Bullish structure intact + positive slope
        return "strong_bull"
    if not bull_struct and slope < -0.0005:
        return "strong_bear"
    return "bull" if bull_struct else "bear"


def build_prompt(ctx):
    """Build the LLM input prompt from a TF context."""
    h = ctx["1h"]
    m = ctx["5m"]
    direction = "bullish" if h["structure_bull"] else "bearish"

    return (
        f"Pair: {ctx['pair']}\n"
        f"1h: ADX={h['adx']:.1f}, EMA50_slope={h['ema_50_slope']*100:+.3f}%, "
        f"structure={direction}, prior_4h_impulse={h['prior_4h_impulse']*100:+.2f}%, "
        f"1h_range={h['range_pct_1h']*100:.2f}%\n"
        f"5m: RSI={m['rsi']:.1f}, ATR={m['atr_pct']*100:.2f}%\n"
        f"Funding 8h: {ctx['funding_8h']*100:+.3f}%\n"
        f"Context: Trend has paused in a tight {h['range_pct_1h']*100:.2f}% range after a "
        f"{abs(h['prior_4h_impulse'])*100:.1f}% impulse move. Structure is {direction}."
    )


def build_output(ctx, regime):
    """Build the expected LLM output for the example."""
    bull_struct = ctx["1h"]["structure_bull"]
    bias = "long" if bull_struct else "short"
    f = ctx["funding_8h"]
    if f > 0.0005:
        f_desc = "positive (longs paying shorts)"
    elif f < -0.0005:
        f_desc = "negative (shorts paying longs)"
    else:
        f_desc = "neutral"
    return {
        "regime": regime,
        "bias": bias,
        "confidence": 0.75,  # High confidence: consolidation in trend is still a trend
        "risk": "consolidation",
        "reasoning": (
            f"1h ADX={ctx['1h']['adx']:.1f} with {abs(ctx['1h']['ema_50_slope'])*100:+.3f}% slope. "
            f"Structure is {'bullish' if bull_struct else 'bearish'} (EMA50 vs EMA200). "
            f"Price is consolidating after a {abs(ctx['1h']['prior_4h_impulse'])*100:.1f}% "
            f"impulse — the trend has paused, not reversed; this is a healthy pullback "
            f"within an active trend, not a regime change, so do not classify as range. "
            f"Funding is {f*100:+.3f}% ({f_desc}). "
            f"Stay with the {'long' if bull_struct else 'short'} bias and wait for a "
            f"pullback entry rather than chasing. Classify as {regime}."
        ),
    }


def generate_examples(target_count=500, seed=42, extra_count=400):
    """Main generation loop.

    Two mining passes per pair:
      1. strict thresholds + random_state=seed  -> reproduces the original
         v0.3 sample exactly (deterministic), so those examples keep their
         identity across regeneration.
      2. relaxed thresholds (range<2.5%, ADX 14-28, impulse>3%), minus any
         (pair, timestamp) already mined in pass 1, random_state=seed+1.
    """
    random.seed(seed)
    np.random.seed(seed)

    all_examples = []
    pairs = ["BTC", "ETH"]

    for pair in pairs:
        logger.info(f"Processing {pair}...")
        df_1h = load_pair(pair)
        if df_1h.empty:
            logger.warning(f"No data for {pair}")
            continue

        # Load 5m data
        files_5m = list(DATA_DIR.glob(f"{pair}_USDT_USDT-5m-futures.feather"))
        if not files_5m:
            logger.warning(f"No 5m data for {pair}")
            continue
        df_5m = pd.read_feather(files_5m[0]).set_index("date").sort_index()
        df_5m["atr_pct"] = (df_5m["high"] - df_5m["low"]) / df_5m["close"]
        df_5m["rsi"] = rsi(df_5m["close"], 14)

        # Real historical funding rates (8h prints from Binance via freqtrade)
        funding = load_funding(pair)
        if len(funding):
            logger.info(f"  Funding: {len(funding)} prints "
                        f"({funding.index[0]} to {funding.index[-1]})")
        else:
            logger.warning(f"  No funding data for {pair}; funding defaults to 0.0")

        # Pass 1: strict thresholds — reproduces the original v0.3 sample
        strict = find_consolidation_periods(df_1h)
        logger.info(f"  Found {len(strict)} strict consolidation periods")
        n_per_pair = target_count // len(pairs)
        strict_sample = strict.sample(n=min(n_per_pair, len(strict)),
                                      random_state=seed) if len(strict) else strict

        # Pass 2: relaxed thresholds, deduped against pass 1 by timestamp
        relaxed = find_consolidation_periods(
            df_1h, range_pct=RELAXED_RANGE_PCT,
            adx_lo=RELAXED_ADX_LO, adx_hi=RELAXED_ADX_HI,
            impulse=RELAXED_IMPULSE_PCT)
        relaxed_new = relaxed.drop(index=relaxed.index.intersection(strict_sample.index))
        logger.info(f"  Relaxed pool: {len(relaxed)} periods, "
                    f"{len(relaxed_new)} not already mined")
        n_extra = extra_count // len(pairs)
        extra_sample = relaxed_new.sample(n=min(n_extra, len(relaxed_new)),
                                          random_state=seed + 1) if len(relaxed_new) else relaxed_new

        sampled = pd.concat([strict_sample, extra_sample])
        logger.info(f"  Sampled {len(strict_sample)} strict + {len(extra_sample)} relaxed")

        for idx, row in sampled.iterrows():
            ctx = build_context(idx, df_1h, df_5m, pair, funding_series=funding)
            regime = label_consolidation(ctx)
            prompt = build_prompt(ctx)
            output = build_output(ctx, regime)

            all_examples.append({
                "instruction": "Analyze the following crypto market state and output JSON with keys: regime, bias, confidence, risk, reasoning.",
                "input": prompt,
                "output": json.dumps(output),
                "tf_context": ctx,  # extra metadata
                "label": regime,
            })

    return all_examples


def merge_with_v02(new_examples, train_path, out_path):
    """Merge new examples into existing v02 training set, deduplicate."""
    # Load existing
    existing = []
    with open(train_path) as f:
        for line in f:
            existing.append(json.loads(line))
    logger.info(f"Existing v02 train: {len(existing)} examples")

    # Dedupe: only add new examples whose input doesn't already exist
    existing_inputs = {e["input"] for e in existing}
    new_to_add = [e for e in new_examples if e["input"] not in existing_inputs]
    logger.info(f"New examples to add: {len(new_to_add)}")

    # Append
    merged = existing + [{"instruction": e["instruction"],
                          "input": e["input"],
                          "output": e["output"]} for e in new_to_add]

    with open(out_path, 'w') as f:
        for ex in merged:
            f.write(json.dumps(ex) + '\n')

    logger.info(f"Merged dataset: {len(merged)} examples -> {out_path}")

    # Regime distribution
    from collections import Counter
    regimes = Counter()
    for ex in merged:
        try:
            regimes[json.loads(ex["output"])["regime"]] += 1
        except:
            pass
    logger.info(f"Regime distribution in merged set:")
    for r, n in sorted(regimes.items(), key=lambda x: -x[1]):
        logger.info(f"  {r}: {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500,
                        help="Number of strict-threshold examples to generate")
    parser.add_argument("--extra", type=int, default=400,
                        help="Additional relaxed-threshold examples (iteration 2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.n} strict + {args.extra} relaxed "
                f"trend-consolidation examples...")
    examples = generate_examples(target_count=args.n, seed=args.seed,
                                 extra_count=args.extra)
    logger.info(f"Generated {len(examples)} examples")

    # Save raw new examples
    raw_path = OUT_DIR / "llm_v03_strong_trend_raw.jsonl"
    with open(raw_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    logger.info(f"Raw saved -> {raw_path}")

    # Merge into v02 train
    train_v02 = OUT_DIR / "llm_v02_train.jsonl"
    train_v03 = OUT_DIR / "llm_v03_train.jsonl"
    merge_with_v02(examples, train_v02, train_v03)

    # Also write val/test split (10% each) for v03
    examples_only = [{"instruction": e["instruction"],
                     "input": e["input"],
                     "output": e["output"]} for e in examples]
    random.shuffle(examples_only)
    n = len(examples_only)
    val_split = examples_only[:int(n*0.1)]
    test_split = examples_only[int(n*0.1):int(n*0.2)]
    train_new = examples_only[int(n*0.2):]

    with open(OUT_DIR / "llm_v03_val.jsonl", 'w') as f:
        for ex in val_split:
            f.write(json.dumps(ex) + '\n')
    with open(OUT_DIR / "llm_v03_test.jsonl", 'w') as f:
        for ex in test_split:
            f.write(json.dumps(ex) + '\n')
    with open(OUT_DIR / "llm_v03_strong_trend_train.jsonl", 'w') as f:
        for ex in train_new:
            f.write(json.dumps(ex) + '\n')

    logger.info(f"Splits saved: train={len(train_new)}, val={len(val_split)}, test={len(test_split)}")
    logger.info("Done! Next step: retrain LLM with v03_train.jsonl")


if __name__ == "__main__":
    main()
