"""
GTQuantMultiTF1h — FreqAI secondary identifier (Day 3).

Same multi-TF feature set as GTQuantMultiTF; the prediction target is the
1-hour-forward return (12 candles on the 5m frame), used with config
identifier `gtquant-v0.1-1h-target`.
"""
from pandas import DataFrame

from GTQuantMultiTF import GTQuantMultiTF


class GTQuantMultiTF1h(GTQuantMultiTF):
    """Secondary identifier: 1h-forward return target (12 x 5m candles)."""

    prediction_col = "&-s-future_return_1h"

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["&-s-future_return_1h"] = dataframe["close"].shift(-12) / dataframe["close"] - 1
        return dataframe
