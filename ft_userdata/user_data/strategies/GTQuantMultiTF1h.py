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

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        # Skip the parent's 9 %-features: they depend on strategy indicators
        # (ema_9, ...) which exist only on the live prediction dataframe, not on
        # FreqAI's raw training frames — so live produced 969 columns vs the 960
        # the pipeline was trained on. The 5m bot masks this with a patched
        # BaseRegressionModel inside its container; this identifier's models were
        # trained without these features, so omitting them matches the pipeline.
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["&-s-future_return_1h"] = dataframe["close"].shift(-12) / dataframe["close"] - 1
        return dataframe
