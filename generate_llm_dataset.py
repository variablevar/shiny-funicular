#!/usr/bin/env python3
"""
Generate GT-Quant LLM Training Dataset from Historical Data
Run after Milestone 1 has downloaded data.
"""
import json
import random
import pandas as pd
from pathlib import Path
from loguru import logger

def generate_example(row: pd.Series) -> dict:
    """Convert a historical bar + outcome into a training example."""
    
    # Build market context
    context = f"""BTC Price: {row['close']:.2f}
4h Return: {row.get('return_4h', 0)*100:+.2f}%
24h Return: {row.get('return_24h', 0)*100:+.2f}%
Funding Z-Score: {row.get('funding_zscore', 0):.2f}
OI Change 1h: {row.get('oi_change_1h', 0)*100:+.1f}%
RSI 14: {row.get('rsi_14', 50):.1f}
ATR 14: {row.get('atr_14', 0):.2f}
Volume Z-Score: {row.get('volume_zscore', 0):.2f}
CVD: {row.get('cvd', 0):.0f}
EMA 20 Distance: {row.get('ema_20_dist', 0)*100:+.2f}%
EMA 50 Distance: {row.get('ema_50_dist', 0)*100:+.2f}%"""

    # Determine label from future 4h return
    fut = row.get('future_return_4h', 0)
    regime = row.get('regime', 'UNKNOWN')
    
    if pd.isna(fut):
        return None
        
    if fut > 0.008:  # Strong up
        bias = "long"
        conf = min(0.55 + fut * 20, 0.95)
        risk = "low" if row.get('funding_zscore', 0) < 1.0 else "crowded"
        reasoning = f"Strong bullish momentum with {fut*100:+.2f}% expected 4h return. "
        if row.get('funding_zscore', 0) > 1.5:
            reasoning += "However, funding is elevated indicating crowded longs. Caution warranted."
        else:
            reasoning += "Funding is reasonable. Favorable risk/reward."
            
    elif fut > 0.003:  # Moderate up
        bias = "long"
        conf = min(0.55 + fut * 15, 0.85)
        risk = "moderate"
        reasoning = f"Moderate bullish outlook with {fut*100:+.2f}% expected return. Range-bound but upward bias."
        
    elif fut < -0.008:  # Strong down
        bias = "short"
        conf = min(0.55 + abs(fut) * 20, 0.95)
        risk = "high" if row.get('funding_zscore', 0) < -1.5 else "moderate"
        reasoning = f"Strong bearish momentum with {fut*100:+.2f}% expected 4h return. Consider short exposure."
        
    elif fut < -0.003:  # Moderate down
        bias = "short"
        conf = min(0.55 + abs(fut) * 15, 0.85)
        risk = "moderate"
        reasoning = f"Moderate bearish outlook. Downward pressure expected."
        
    else:  # Flat
        bias = "flat"
        conf = 0.65
        risk = "low"
        reasoning = "No clear directional edge. Market likely to consolidate. Best to wait."

    output = {
        "regime": regime,
        "bias": bias,
        "confidence": round(float(conf), 2),
        "risk": risk,
        "reasoning": reasoning
    }

    return {
        "instruction": "Analyze the following BTC perpetual futures market state and output structured reasoning as JSON with keys: regime, bias, confidence, risk, reasoning.",
        "input": context,
        "output": json.dumps(output)
    }


def main():
    logger.info("=== GT-Quant LLM Dataset Generator ===")
    
    # Load processed features from Milestone 1
    feature_path = Path("data/processed/BTCUSDT_features.parquet")
    if not feature_path.exists():
        logger.error(f"Features not found at {feature_path}")
        logger.info("Run Milestone 1 first: python src/main.py")
        return
        
    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded {len(df)} rows of features")
    
    # Generate examples
    examples = []
    for _, row in df.iterrows():
        ex = generate_example(row)
        if ex:
            examples.append(ex)
    
    logger.info(f"Generated {len(examples)} raw examples")
    
    # Deduplicate similar inputs
    seen_inputs = set()
    unique_examples = []
    for ex in examples:
        if ex['input'] not in seen_inputs:
            seen_inputs.add(ex['input'])
            unique_examples.append(ex)
    
    logger.info(f"After dedup: {len(unique_examples)} unique examples")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(unique_examples)
    
    # Save full dataset
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "gtquant_llm_dataset.jsonl", "w") as f:
        for ex in unique_examples:
            f.write(json.dumps(ex) + "\n")
    
    # Save train/val split (90/10)
    split_idx = int(len(unique_examples) * 0.9)
    train = unique_examples[:split_idx]
    val = unique_examples[split_idx:]
    
    with open(output_dir / "gtquant_llm_train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
            
    with open(output_dir / "gtquant_llm_val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")
    
    logger.info(f"Saved:")
    logger.info(f"  - data/gtquant_llm_dataset.jsonl ({len(unique_examples)} total)")
    logger.info(f"  - data/gtquant_llm_train.jsonl ({len(train)} train)")
    logger.info(f"  - data/gtquant_llm_val.jsonl ({len(val)} val)")
    
    # Show sample
    logger.info("\n=== Sample Example ===")
    sample = unique_examples[0]
    logger.info(f"Input:\n{sample['input']}")
    logger.info(f"Output:\n{sample['output']}")


if __name__ == "__main__":
    main()