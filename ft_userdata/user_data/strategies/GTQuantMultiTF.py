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

    # Circuit breakers (Day 5). In modern Freqtrade these live on the
    # strategy, not in config.
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 6},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 4,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 10,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.15,
            },
        ]

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

        # Regime classification from 1h informative features (Day 5).
        dataframe["regime"] = self._compute_regime(dataframe)

        # FreqAI hook (REQUIRED): runs feature engineering across all
        # include_timeframes, trains per sliding window in backtests, and
        # appends prediction columns (&-*, do_predict, DI_values).
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    # ------------------------------------------------------------------ #
    # Regime classification (Day 5)
    # ------------------------------------------------------------------ #

    # Regimes in which entries are blocked entirely.
    BLOCKED_REGIMES = ("RANGE", "HIGH_VOL")
    use_regime_gating = True

    # Calibrated on BTC 1h (July 2026): ADX<20 = chop (28% of bars);
    # |slope| 75th pct = 0.0028. ADX is the chop detector, slope the compass.
    regime_adx_range_max = 20.0     # ADX below this = RANGE (no trend)
    regime_adx_trend_min = 25.0     # ADX above this = trending
    regime_slope_strong = 0.0028    # |1h EMA-50 slope| for STRONG_*
    regime_vol_mult = 1.5           # HIGH_VOL when rv > mult * rolling median

    def _compute_regime(self, dataframe: DataFrame) -> "pd.Series":
        """
        Rule-based regime from 1h informative features + 5m realized vol.

            HIGH_VOL    : 5m realized vol > 1.5x rolling median (overrides all)
            RANGE       : 1h ADX < 20 (no trend -> chop)
            STRONG_BULL : 1h slope > +0.0028 AND ADX > 25
            STRONG_BEAR : 1h slope < -0.0028 AND ADX > 25
            BULL        : 1h slope > 0
            BEAR        : 1h slope < 0
        """
        import pandas as pd

        slope = dataframe["ema_50_slope_1h"]
        adx = dataframe["adx_1h"]
        # 1h of 5m bars realized vol, vs its ~1-day rolling median.
        rv = dataframe["close"].pct_change().rolling(12).std()
        rv_med = rv.rolling(288).median()

        regime = pd.Series("RANGE", index=dataframe.index, dtype=object)
        trending = adx >= self.regime_adx_trend_min
        strong_bull = (slope > self.regime_slope_strong) & trending
        strong_bear = (slope < -self.regime_slope_strong) & trending
        bull = (slope > 0) & ~strong_bull
        bear = (slope < 0) & ~strong_bear

        # Order matters: direction first, then strong-trend upgrade,
        # then chop override, then high-vol override.
        regime[bull] = "BULL"
        regime[bear] = "BEAR"
        regime[strong_bull] = "STRONG_BULL"
        regime[strong_bear] = "STRONG_BEAR"
        regime[adx < self.regime_adx_range_max] = "RANGE"
        regime[rv > rv_med * self.regime_vol_mult] = "HIGH_VOL"
        return regime

    # ------------------------------------------------------------------ #
    # Entry / exit logic (model-driven with regime filter)
    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enter when FreqAI is confident (do_predict == 1), the predicted return
        clears the threshold, and the regime allows the direction (Day 5):
        RANGE/HIGH_VOL are blocked entirely, longs need a bull regime, shorts
        need a bear regime.
        """
        prediction = self.prediction_col
        if prediction not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        if self.use_regime_gating and "regime" in dataframe.columns:
            blocked = dataframe["regime"].isin(self.BLOCKED_REGIMES)
            long_regime_ok = dataframe["regime"].isin(["STRONG_BULL", "BULL"])
            short_regime_ok = dataframe["regime"] == "BEAR"
        else:
            blocked = dataframe["close"] != dataframe["close"]  # all False
            long_regime_ok = ~blocked
            short_regime_ok = ~blocked

        long_cond = [
            dataframe["do_predict"] == 1,
            dataframe[prediction] > self.entry_threshold.value,
            dataframe["ema_50_slope_1h"] > 0,       # 1h trend up
            dataframe["volume_zscore"] > -1.0,      # avoid dead bars
            long_regime_ok,
        ]
        short_cond = [
            dataframe["do_predict"] == 1,
            dataframe[prediction] < -self.entry_threshold.value,
            dataframe["ema_50_slope_1h"] < 0,       # 1h trend down
            dataframe["volume_zscore"] > -1.0,
            short_regime_ok,
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
    # Regime-aware position sizing (Day 5)
    # ------------------------------------------------------------------ #

    # Size multiplier per regime. RANGE/HIGH_VOL are blocked at entry, so
    # these only affect the regimes that trade.
    regime_size_mult = {
        "STRONG_BULL": 1.5,
        "STRONG_BEAR": 1.5,
        "BULL": 1.0,
        "BEAR": 1.0,
    }

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake, max_stake,
                            leverage: float, entry_tag, side: str, **kwargs) -> float:
        """
        Scale stake by the current regime: size up in strong trends, base size
        otherwise. Falls back to proposed_stake if the regime can't be read.
        """
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            regime = df["regime"].iloc[-1]
            mult = self.regime_size_mult.get(str(regime), 1.0)
            stake = proposed_stake * mult
            return max(min_stake, min(stake, max_stake))
        except Exception:
            return proposed_stake

    # ------------------------------------------------------------------ #
    # 1m microstructure entry filter (Day 6)
    # ------------------------------------------------------------------ #

    use_micro_filter = True
    micro_max_spread = 0.0005      # 0.05% top-of-book spread
    micro_vol_delta_sigma = 2.0    # reject if latest 1m volume delta < -2σ

    # Strategy pair -> TimescaleDB symbol
    _PAIR_TO_DB = {"BTC/USDT:USDT": "BTCUSDT", "ETH/USDT:USDT": "ETHUSDT"}

    def _latest_1m_micro(self, pair: str, lookback_bars: int = 60):
        """
        Fetch recent 1m microstructure bars from TimescaleDB.
        Returns (latest_spread, latest_volume_delta, vol_delta_std) or None.
        """
        symbol = self._PAIR_TO_DB.get(pair)
        if symbol is None:
            return None
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname="gtquant", user="gtquant", password="gtquant_local",
                host="host.docker.internal", port=5432, connect_timeout=3,
            )
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT spread, volume_delta FROM ohlcv_1m
                        WHERE symbol = %s AND spread IS NOT NULL
                        ORDER BY time DESC LIMIT %s
                        """,
                        (symbol, lookback_bars),
                    )
                    rows = cur.fetchall()
            finally:
                conn.close()
        except Exception:
            return None
        if not rows:
            return None
        spreads = [r[0] for r in rows]
        deltas = [r[1] for r in rows if r[1] is not None]
        import numpy as np
        vd_std = float(np.std(deltas)) if len(deltas) > 5 else None
        return rows[0][0], rows[0][1], vd_std

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time,
                            entry_tag, side: str, **kwargs) -> bool:
        """
        1m microstructure gate (live/dry-run only — in backtests the DB holds
        *live* data, so applying it there would be lookahead).

        Reject entry when:
          - latest 1m spread > 0.05% (poor execution), or
          - latest 1m volume delta < -2σ of recent deltas (sell-pressure spike)
        Freqtrade retries on the next candle, which gives the 1-2 candle
        delay the plan calls for. Fails open when no micro data is available.
        """
        if not self.use_micro_filter:
            return True
        # Only gate live/dry-run; never backtest/hyperopt.
        if getattr(self.dp, "runmode", None) not in ("live", "dry_run"):
            return True

        micro = self._latest_1m_micro(pair)
        if micro is None:
            return True  # fail open = safe (plan Day 12)
        spread, volume_delta, vd_std = micro

        if spread is not None and spread > self.micro_max_spread:
            logger.info(f"[micro-filter] {pair} rejected: spread {spread:.5f} > {self.micro_max_spread}")
            self._audit(pair, current_time, side, "rejected_spread", rate)
            return False
        if volume_delta is not None and vd_std and volume_delta < -self.micro_vol_delta_sigma * vd_std:
            logger.info(f"[micro-filter] {pair} rejected: volume_delta {volume_delta:.1f} < -{self.micro_vol_delta_sigma}σ")
            self._audit(pair, current_time, side, "rejected_volume_delta", rate, spread, volume_delta)
            return False
        self._audit(pair, current_time, side, "approved", rate, spread, volume_delta)
        return True

    # ------------------------------------------------------------------ #
    # Audit logging to TimescaleDB (Day 6)
    # ------------------------------------------------------------------ #

    def _audit(self, pair: str, ts, side: str, decision: str, price: float,
               spread=None, volume_delta: float = None) -> None:
        """Write one decision row to audit_log. Live/dry-run only."""
        if getattr(self.dp, "runmode", None) not in ("live", "dry_run"):
            return
        symbol = self._PAIR_TO_DB.get(pair)
        if symbol is None:
            return
        try:
            import psycopg2
            conn = psycopg2.connect(
                dbname="gtquant", user="gtquant", password="gtquant_local",
                host="host.docker.internal", port=5432, connect_timeout=3,
            )
            try:
                regime = None
                try:
                    df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    regime = str(df["regime"].iloc[-1])
                except Exception:
                    pass
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO audit_log
                            (time, pair, timeframe, signal_type, regime,
                             spread_1m, volume_delta_1m, risk_decision, fill_price)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (ts, symbol, self.timeframe, f"entry_{side}", regime,
                         spread, volume_delta, decision, price),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"audit write failed (non-fatal): {e}")

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
