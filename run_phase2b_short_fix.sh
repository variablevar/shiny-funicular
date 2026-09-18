#!/usr/bin/env bash
# Phase 2b asymmetry fix A/B: GTQuantV02 short-side variants on the Phase 2
# window 2026-06-14..2026-09-11, fee 0.05% (exchange worst-case), host venv-ft.
#
# Historical record of the 2026-09-17 A/B (results in logs/phase2b_*.log):
#   baseline (GTQuantV02Base, gate off + flip exits):  110 trades, -11.52 USDT
#   gate     (shorts KMeans-BEAR only):                 61 trades,  +2.28 USDT
#   noflip   (no short flip exit, no gate):             98 trades,  -7.68 USDT
#   combo    (gate + no short flip):                    61 trades,  +3.05 USDT
# The combo won (and won the 3-fold walk-forward) and is now the GTQuantV02
# default, so arm "gate" below IS the shipped strategy minus the flip flag.
# The combo arm's subclass file was removed after the flag became the default.
#
# NOTE: config_v02_ab.json is config_v02.json with a dedicated FreqAI
# identifier (gtquant-phase2b-ab). The live gtquant-freqtrade-1h container
# shares identifier gtquant-day14-1h-tune via the user_data bind mount and
# purged its backtest sub-train models on live retrain — never backtest
# against a live identifier.
set -uo pipefail
cd "$(dirname "$0")/ft_userdata"
FT=../venv-ft/bin/freqtrade
TIMERANGE=20260614-20260911
LOGDIR=../logs
mkdir -p "$LOGDIR"

run() {
    local name="$1" strat="$2"
    echo "=== $(date -Is) $name ($strat) ==="
    $FT backtesting --userdir user_data --config user_data/config_v02_ab.json \
        --strategy "$strat" --freqaimodel LightGBMRegressor \
        --timerange "$TIMERANGE" --cache none \
        > "$LOGDIR/phase2b_${name}.log" 2>&1
    echo "exit=$? $name"
}

run final    GTQuantV02            # shipped: gate + no short flip exit
run baseline GTQuantV02Base        # control: pre-fix behavior
run noflip   GTQuantV02NoGateNoFlip
echo "=== phase2b A/B done ==="
