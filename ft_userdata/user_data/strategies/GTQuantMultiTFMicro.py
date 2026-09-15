"""
GTQuantMultiTFMicro — GTQuantMultiTF + historical 1m microstructure features
in the FreqAI feature set (Day 14 A/B experiment).

Differences vs the parent strategy (which stays untouched for live):

1. populate_indicators merges aggTrades-derived 1m microstructure bars
   (volume_delta, tick_momentum, realized_variance, trade_count —
   see collectors/historical_micro.py) aggregated into 5m buckets.
   Lookahead contract: features on the 5m row with open time t use only
   1m bars with close time <= t (one 5m bucket of deliberate lag).
   See strategies/micro_features.py.

2. feature_engineering_standard exposes them as %-micro_* FreqAI features.

Runs under a separate FreqAI identifier (config_micro.json:
gtquant-v0.2-5m-micro) so backtest predictions never mix with the
baseline gtquant-v0.1-5m-primary models.

Note: `include_timeframes` still cannot contain 1m (FreqAI rejects TFs
smaller than the main 5m) — the micro data is merged from parquet/DB,
not from FreqAI's own multi-TF machinery.
"""
import sys
from pathlib import Path

from pandas import DataFrame

_STRAT_DIR = str(Path(__file__).resolve().parent)
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)

import micro_features as mf  # noqa: E402
from GTQuantMultiTF import GTQuantMultiTF  # noqa: E402

import logging
logger = logging.getLogger(__name__)


class GTQuantMultiTFMicro(GTQuantMultiTF):
    use_micro_features = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.use_micro_features:
            runmode = getattr(self.dp, "runmode", None)
            runmode = getattr(runmode, "value", str(runmode or "backtest"))
            try:
                dataframe = mf.attach_micro_features(
                    dataframe,
                    metadata["pair"],
                    Path(self.config.get("user_data_dir", "user_data")),
                    str(runmode),
                )
            except Exception as e:
                # Fail open: micro features are an enhancement, never a reason
                # to drop the whole indicator pipeline.
                logger.warning(f"micro features unavailable ({e}); continuing without")
        return super().populate_indicators(dataframe, metadata)

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe = super().feature_engineering_standard(dataframe, metadata, **kwargs)
        # Only the base 5m frame carries the merged micro columns (corr-pair
        # and higher-TF frames never saw attach_micro_features).
        if "micro_vol_delta_ratio" in dataframe.columns:
            for col in mf.MICRO_FEATURE_COLS:
                dataframe[f"%-{col}"] = dataframe[col]
        return dataframe
