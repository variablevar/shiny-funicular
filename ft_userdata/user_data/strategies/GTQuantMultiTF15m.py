"""
GTQuantMultiTF15m — FreqAI secondary identifier (Day 3).

Same multi-TF feature set as GTQuantMultiTF; the prediction target is the
15-minute-forward return (3 candles on the 5m frame), used with config
identifier `gtquant-v0.1-15m-target`.

FreqAI identifiers are selected via the CONFIG (`freqai.identifier`), not the
strategy. Each identifier trains its own model with its own label.
"""
from pandas import DataFrame

from GTQuantMultiTF import GTQuantMultiTF


class GTQuantMultiTF15m(GTQuantMultiTF):
    """Secondary identifier: 15m-forward return target (3 x 5m candles)."""

    prediction_col = "&-s-future_return_15m"

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["&-s-future_return_15m"] = dataframe["close"].shift(-3) / dataframe["close"] - 1
        return dataframe
