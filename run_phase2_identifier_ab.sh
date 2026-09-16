#!/usr/bin/env bash
# Phase 2 Task 1: 1h identifier A/B on the Phase 1 window (2026-06-14..09-11).
# gtquant-v0.1-1h-target vs gtquant-day14-1h-tune, host venv-ft (docker down).
set -uo pipefail
cd "$(dirname "$0")/ft_userdata"
FT=../venv-ft/bin/freqtrade
TIMERANGE=20260614-20260911
LOGDIR=../logs
mkdir -p "$LOGDIR"

for pair in "config_1h.json:v0_1_1h_target" "config_tune_1h.json:day14_1h_tune"; do
    cfg="${pair%%:*}"; name="${pair##*:}"
    echo "=== $(date -Is) $name ==="
    $FT backtesting --userdir user_data --config "user_data/$cfg" \
        --strategy GTQuantMultiTF1h --freqaimodel LightGBMRegressor \
        --timerange "$TIMERANGE" --cache none \
        > "$LOGDIR/phase2_ab_${name}.log" 2>&1
    echo "exit=$? $name"
done
echo "=== A/B done ==="
