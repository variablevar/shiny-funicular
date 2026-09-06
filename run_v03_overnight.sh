#!/usr/bin/env bash
# Day 12 follow-up: overnight v0.3 fine-tune runner.
#
# Runs the v0.3 fine-tune in the background, logs to file, and
# verifies artifacts on completion.
#
# Usage:
#     bash run_v03_overnight.sh
#
# Pre-requisites:
#     - Unsloth + CUDA on RTX 4080
#     - venv activated
#     - Generated v0.3 dataset in data/

set -e

cd /home/gt/Desktop/gt-quant

echo "[$(date)] Starting v0.3 fine-tune..."
LOG="logs/v03_train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Pre-flight checks
if [ ! -f "data/llm_v03_train.jsonl" ]; then
    echo "[ERROR] data/llm_v03_train.jsonl missing - run generate_strong_trend_v03.py first"
    exit 1
fi

if [ ! -f "data/llm_v03_val.jsonl" ]; then
    echo "[ERROR] data/llm_v03_val.jsonl missing"
    exit 1
fi

# Run training
python3 train_llm_v03.py 2>&1 | tee "$LOG"

echo "[$(date)] Training complete. Verifying artifacts..."
ls -la models/gtquant-7b-v0.3/lora_adapter 2>&1 | head -5
ls -la models/gtquant-7b-v0.3/merged_16bit 2>&1 | head -5
ls -la models/gtquant-7b-v0.3/gguf 2>&1 | head -5

echo "[$(date)] Done."
