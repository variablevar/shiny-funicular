"""
GTQuantMultiTF4h — FreqAI identifier for the 4h-forward return (Day 4 lever).

Rationale: fee erosion dominates short horizons. A 4h target captures larger
moves per trade, so the ~0.1% round-trip taker fee is a smaller fraction of
gross. Legacy custom ensemble held Sharpe 1.80 at a 24h horizon; 4h is the
FreqAI sweet spot between fee-heavy 5m and slow 24h.

Target: 48 candles on the 5m frame = 4h forward return.
Config identifier: gtquant-v0.1-4h-target.
"""
from pandas import DataFrame

from GTQuantMultiTF import GTQuantMultiTF


class GTQuantMultiTF4h(GTQuantMultiTF):
    """4h-forward return target (48 x 5m candles)."""

    prediction_col = "&-s-future_return_4h"

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["&-s-future_return_4h"] = dataframe["close"].shift(-48) / dataframe["close"] - 1
        return dataframe
