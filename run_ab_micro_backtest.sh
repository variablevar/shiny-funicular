#!/usr/bin/env bash
# Day 14 A/B backtest: baseline 5m-primary vs +1m-micro features.
# Same 90-day fee-realistic window as the other Day-14 tuning runs.
#
# Guards from docs/freqai-setup-notes.md:
#   - fresh identifiers (no stale backtesting_predictions reuse)
#   - --cache none (populate_indicators must re-run)
#   - freqai image via docker-compose service
#
# Usage: bash run_ab_micro_backtest.sh
set -euo pipefail
cd "$(dirname "$0")/ft_userdata"

TIMERANGE=20260614-20260911
RESULTS=user_data/backtest_results
LOGDIR=../logs
mkdir -p "$LOGDIR"

snapshot() { ls "$RESULTS"/*.zip 2>/dev/null | sort; }

run_backtest() {
    local name="$1"; shift
    local before after
    before=$(snapshot)
    echo "=== $(date -Is) starting $name ==="
    docker-compose run --rm --no-deps freqtrade backtesting \
        "$@" --timerange "$TIMERANGE" --cache none \
        2>&1 | tee "$LOGDIR/ab_${name}.log" | grep -E 'Training|exception|Total/Daily|TOTAL|Sharpe|Win' || true
    after=$(snapshot)
    comm -13 <(echo "$before") <(echo "$after") > "$LOGDIR/ab_${name}.newzips"
    echo "=== $name results: $(cat "$LOGDIR/ab_${name}.newzips") ==="
}

# Identifiers are fresh dirs -> nothing to purge, but purge defensively
# (gotcha #6: stale backtesting_predictions are silently reused).
rm -rf user_data/models/gtquant-v0.1-5m-primary-ab \
       user_data/models/gtquant-v0.2-5m-micro

run_backtest baseline \
    --config user_data/config_ab_baseline.json \
    --strategy GTQuantMultiTF \
    --freqaimodel LightGBMRegressor

run_backtest micro \
    --config user_data/config_micro.json \
    --strategy GTQuantMultiTFMicro \
    --freqaimodel LightGBMRegressor

echo "=== A/B done ==="
