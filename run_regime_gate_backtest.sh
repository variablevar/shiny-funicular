#!/usr/bin/env bash
# v0.2 Phase 1 validation backtest: GTQuantMultiTF (baseline, rule-based gate)
# vs GTQuantRegimeGated (KMeans regime gate) over the same ~90-day window.
#
# Guards from docs/freqai-setup-notes.md:
#   - fresh identifiers (config_regime_base.json / config_regime_gated.json)
#     -> no stale backtesting_predictions reuse
#   - --cache none (populate_indicators must re-run)
#   - freqai image via docker-compose service
#
# Usage: bash run_regime_gate_backtest.sh
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
        2>&1 | tee "$LOGDIR/regime_${name}.log" | grep -E 'Training|exception|Total/Daily|TOTAL|Sharpe|Win|regime gate' || true
    after=$(snapshot)
    comm -13 <(echo "$before") <(echo "$after") > "$LOGDIR/regime_${name}.newzips"
    echo "=== $name results: $(cat "$LOGDIR/regime_${name}.newzips") ==="
}

# Fresh identifiers -> nothing to reuse, but purge defensively (gotcha #6).
rm -rf user_data/models/gtquant-v0.2-regime-base \
       user_data/models/gtquant-v0.2-regime-gated

run_backtest base \
    --config user_data/config_regime_base.json \
    --strategy GTQuantMultiTF \
    --freqaimodel LightGBMRegressor

run_backtest gated \
    --config user_data/config_regime_gated.json \
    --strategy GTQuantRegimeGated \
    --freqaimodel LightGBMRegressor

echo "=== regime gate A/B done ==="
