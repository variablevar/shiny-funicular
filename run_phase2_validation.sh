#!/usr/bin/env bash
# Phase 2 Task 5: GTQuantV02 vs GTQuantRegimeGated vs GTQuantMultiTF (base),
# Phase 1 window 2026-06-14..2026-09-11, host venv-ft (docker unavailable).
set -uo pipefail
cd "$(dirname "$0")/ft_userdata"
FT=../venv-ft/bin/freqtrade
TIMERANGE=20260614-20260911
LOGDIR=../logs
mkdir -p "$LOGDIR"

run() {
    local name="$1" cfg="$2" strat="$3"
    echo "=== $(date -Is) $name ==="
    $FT backtesting --userdir user_data --config "user_data/$cfg" \
        --strategy "$strat" --freqaimodel LightGBMRegressor \
        --timerange "$TIMERANGE" --cache none \
        > "$LOGDIR/phase2_${name}.log" 2>&1
    echo "exit=$? $name"
}

run base  config_regime_base.json  GTQuantMultiTF
run gated config_regime_gated.json GTQuantRegimeGated
run v02   config_v02.json          GTQuantV02
echo "=== phase2 comparison done ==="
