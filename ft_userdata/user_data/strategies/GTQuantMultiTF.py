"""
GTQuantMultiTF — Multi-timeframe FreqAI strategy (Day 2 skeleton).

Timeframe hierarchy (per master plan):
    1m  microstructure  -> feeds 5m entry precision (Cryptofeed, Day 2 live)
    5m  PRIMARY trading TF (this strategy's dataframe)
    15m swing confirmation / trend filter   (@informative)
    30m short-term regime context           (@informative)
    1h  regime classification input         (@informative)
    4h  macro trend anchor                  (@informative)

FreqAI handles the *model* feature expansion across the timeframes listed in
config `freqai.feature_parameters.include_timeframes` (5m/15m/30m/1h/4h) via
`feature_engineering_expand_all` — one call per (timeframe, period). The
@informative decorators below serve the *strategy's own* entry/exit logic.
"""
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from technical import qtpylib
from freqtrade.strategy import DecimalParameter, IStrategy, informative


class GTQuantMultiTF(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "5m"

    # FreqAI needs startup for its own buffering on top of indicator warmup.
    startup_candle_count: int = 240
    process_only_new_candles = True

    # Wide default ROI/stoploss; the model drives exits. Hyperopt tunes later.
    minimal_roi = {"0": 0.10, "120": 0.05, "240": 0.02}
    stoploss = -0.05
    trailing_stop = False

    can_short = True

    # Entry/exit thresholds on the model's predicted return (hyperopt-tuned).
    entry_threshold = DecimalParameter(0.0001, 0.005, default=0.0005, decimals=6,
                                       space="buy", optimize=True, load=True)
    exit_threshold = DecimalParameter(-0.005, -0.0001, default=-0.0005, decimals=6,
                                      space="sell", optimize=True, load=True)

    # FreqAI prediction column this strategy trades on. Secondary
    # identifiers (15m/1h) override this + set_freqai_targets.
    prediction_col = "&-s-future_return_5m"

    # ------------------------------------------------------------------ #
    # Informative timeframes for strategy logic
    # ------------------------------------------------------------------ #

    @informative("15m")
    def populate_informative_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Swing confirmation: EMA trend, RSI, VWAP distance."""
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        # VWAP distance (rolling 20 candles as approximation).
        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        vwap = (typical * dataframe["volume"]).rolling(20).sum() / dataframe["volume"].rolling(20).sum()
        dataframe["vwap_dist"] = (dataframe["close"] - vwap) / vwap
        return dataframe

    @informative("30m")
    def populate_informative_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Short-term regime context: EMA50/200 structure, ATR."""
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["ema_50_200_dist"] = (dataframe["ema_50"] - dataframe["ema_200"]) / dataframe["ema_200"]
        return dataframe

    @informative("1h")
    def populate_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Regime classification input: EMAs, MACD, ADX."""
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macdhist"] = macd["macdhist"]
        dataframe["ema_50_slope"] = dataframe["ema_50"].pct_change(5)
        return dataframe

    @informative("4h")
    def populate_informative_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Macro trend anchor: EMA structure, Supertrend proxy."""
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["bull_structure"] = (dataframe["ema_50"] > dataframe["ema_200"]).astype(int)
        return dataframe

    # ------------------------------------------------------------------ #
    # Primary 5m indicators
    # ------------------------------------------------------------------ #

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """5m PRIMARY feature set (per plan Day 2)."""
        # EMAs
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # RSI / MACD / ATR
        dataframe["rsi_14"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]
        dataframe["atr_14"] = ta.ATR(dataframe, timeperiod=14)

        # Bollinger Bands (20, 2)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_upper"] = bb["upper"]
        dataframe["bb_pct"] = (dataframe["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])

        # Volume z-score (20)
        vol_mean = dataframe["volume"].rolling(20).mean()
        vol_std = dataframe["volume"].rolling(20).std()
        dataframe["volume_zscore"] = (dataframe["volume"] - vol_mean) / vol_std

        # CVD proxy from candle direction × volume (true taker delta needs
        # the Cryptofeed collector — Day 2 live task).
        signed_vol = DataFrame(
            np.where(dataframe["close"] >= dataframe["open"], dataframe["volume"], -dataframe["volume"]),
            index=dataframe.index,
        )[0]
        dataframe["cvd_50"] = signed_vol.rolling(50).sum()

        # Returns over multiple candle spans
        for span in (1, 5, 10, 20):
            dataframe[f"ret_{span}"] = dataframe["close"].pct_change(span)

        # FreqAI hook (REQUIRED): runs feature engineering across all
        # include_timeframes, trains per sliding window in backtests, and
        # appends prediction columns (&-*, do_predict, DI_values).
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    # ------------------------------------------------------------------ #
    # Entry / exit logic (model-driven with regime filter)
    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enter when FreqAI is confident (do_predict == 1) and the predicted
        return clears the threshold. 1h trend gates shorts/longs.
        """
        prediction = self.prediction_col
        if prediction not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        long_cond = [
            dataframe["do_predict"] == 1,
            dataframe[prediction] > self.entry_threshold.value,
            dataframe["ema_50_slope_1h"] > 0,       # 1h trend up
            dataframe["volume_zscore"] > -1.0,      # avoid dead bars
        ]
        short_cond = [
            dataframe["do_predict"] == 1,
            dataframe[prediction] < -self.entry_threshold.value,
            dataframe["ema_50_slope_1h"] < 0,       # 1h trend down
            dataframe["volume_zscore"] > -1.0,
        ]

        dataframe.loc[reduce(lambda x, y: x & y, long_cond), ["enter_long", "enter_tag"]] = (1, "freqai_long")
        dataframe.loc[reduce(lambda x, y: x & y, short_cond), ["enter_short", "enter_tag"]] = (1, "freqai_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit when the model flips against the position."""
        prediction = self.prediction_col
        if prediction not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe[prediction] < self.exit_threshold.value),
            ["exit_long", "exit_tag"],
        ] = (1, "model_flip_down")
        dataframe.loc[
            (dataframe["do_predict"] == 1) & (dataframe[prediction] > -self.exit_threshold.value),
            ["exit_short", "exit_tag"],
        ] = (1, "model_flip_up")
        return dataframe

    # ------------------------------------------------------------------ #
    # FreqAI feature engineering
    # ------------------------------------------------------------------ #

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        """
        Expanded features (% prefix): called once per (timeframe, period) from
        include_timeframes x indicator_periods_candles.
        """
        dataframe[f"%-ema_dist_{period}"] = (
            dataframe["close"] - ta.EMA(dataframe, timeperiod=period)
        ) / ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-rsi_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-atr_{period}"] = ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        dataframe[f"%-roc_{period}"] = ta.ROC(dataframe, timeperiod=period)
        dataframe[f"%-vol_rel_{period}"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """Basic features (% prefix): computed once per timeframe, not per period."""
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-candle_body_pct"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]
        dataframe["%-candle_range_pct"] = (dataframe["high"] - dataframe["low"]) / dataframe["open"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """Standard features (% prefix): computed once on the base 5m frame."""
        dataframe["%-ema_9_21_cross"] = (dataframe["ema_9"] - dataframe["ema_21"]) / dataframe["close"]
        dataframe["%-ema_9_50_cross"] = (dataframe["ema_9"] - dataframe["ema_50"]) / dataframe["close"]
        dataframe["%-macd_hist_norm"] = dataframe["macd_hist"] / dataframe["close"]
        dataframe["%-bb_pct"] = dataframe["bb_pct"]
        dataframe["%-volume_zscore"] = dataframe["volume_zscore"]
        dataframe["%-ema_50_slope_1h"] = dataframe["ema_50_slope_1h"]
        dataframe["%-adx_1h"] = dataframe["adx_1h"]
        dataframe["%-ema_50_200_dist_30m"] = dataframe["ema_50_200_dist_30m"]
        dataframe["%-bull_structure_4h"] = dataframe["bull_structure_4h"]
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Label (& prefix). Primary target for identifier gtquant-v0.1-5m-primary:
        5m forward return over 20 candles (= 100 minutes).

        Secondary horizons (15m/1h) run as SEPARATE FreqAI identifiers on Day 3
        (a standard single-target regressor cannot train on multiple labels).
        """
        dataframe["&-s-future_return_5m"] = dataframe["close"].shift(-20) / dataframe["close"] - 1
        return dataframe
