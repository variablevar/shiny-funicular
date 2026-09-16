"""
GTQuantV02 — v0.2 Phase 2 (docs/v0.2-strategy.md §Layer 2, §Layer 3,
§Execution Rules, §Phase 2): 1h signal + 5m pullback timing + maker-only
limit entries, on top of the Phase 1 KMeans regime gate.

Chain (in order, gate first):
    1. Regime gate (inherited from GTQuantRegimeGated): KMeans action on the
       latest closed 1h bar must be TRENDING/MEAN_REVERTING, else HOLD before
       the ML prediction is consulted. Artifact missing/stale -> parent's
       rule-based gate fallback, exactly as Phase 1.
    2. Direction ONLY from the 1h FreqAI prediction (&-s-future_return_1h,
       identifier gtquant-day14-1h-tune). The bias is sampled on the 5m bar
       whose close lands on the hour (minute == 55) and forward-filled across
       the following hour — the bias for the whole hour is fixed at the 1h
       bar close, using only closed candles (no lookahead).
    3. Entry timing on 5m: the signal fires on a pullback AGAINST the 1h bias
       (long: close below 20-bar VWAP and a negative 3-candle return; short:
       mirrored), not on the 1h flip alone.
    4. Maker execution: custom_entry_price places the limit at the pullback
       extreme (min low of the last 3 candles for longs, max high for shorts,
       always strictly beyond the current price so the order rests on the
       book). unfilledtimeout entry = 15 minutes (3 x 5m candles) — unfilled
       means no trade. Exits stay market.

Risk (doc §Execution Rules):
    - hard stoploss -1.5%
    - time stop 4h (custom_exit)
    - target 1.5 x ATR(14) of the 1h frame at entry (custom_exit; ATR stored
      per pair at signal time — one open position per pair makes the pair key
      unique)
    - trailing stop once +1.1% in profit, trail 0.5% (trailing_stop_positive)
    - model-flip exit (populate_exit_trend, inherited) as secondary signal
    - 5% of NAV per trade (custom_stake_amount)
    - one position per pair, max_open_trades = 2 (config)
    - daily -5% halt (MaxDrawdown protection over 288 x 5m candles)
    - directional inventory cap 15% of NAV (confirm_trade_entry)

Backtesting deviation (verified in freqtrade 2026.8
optimize/backtesting.py: Backtesting.set_fee): the backtester charges ONE
worst-case fee (max(taker, maker) = 0.05% taker) for every order — the maker
fee discount (0.02%) is NOT applied to limit fills in backtesting, only the
limit price improvement is. Maker-fee savings therefore only show up in
dry-run/live; backtest numbers here are conservative by ~0.03% per side.
"""
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, informative

from GTQuantRegimeGated import GTQuantRegimeGated

import logging
logger = logging.getLogger(__name__)


class GTQuantV02(GTQuantRegimeGated):
    """1h direction + 5m pullback-timed maker entries (v0.2 Phase 2)."""

    # Layer 2: direction comes only from the 1h-forward-return prediction.
    prediction_col = "&-s-future_return_1h"

    # Tuned entry threshold (Day 14 grid, GTQuantMultiTFTune.json). The exit
    # threshold stays at the parent default; model-flip is a secondary exit.
    entry_threshold = DecimalParameter(0.0001, 0.005, default=0.001444, decimals=6,
                                       space="buy", optimize=True, load=True)

    # Exit rules (doc §Execution Rules). ROI table disabled — custom_exit and
    # trailing drive the exits.
    minimal_roi = {"0": 100.0}
    stoploss = -0.015
    trailing_stop = False
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True

    # Time stop / ATR target.
    time_stop_hours = 4.0
    atr_target_mult = 1.5

    # Pullback definition (Layer 3).
    pullback_candles = 3        # return lookback for the dip
    vwap_candles = 20           # short VWAP window (100 minutes)

    # Maker limit placement (Layer 3).
    entry_limit_lookback = 3    # pullback extreme = min low / max high of N candles
    min_maker_offset = 0.0002   # keep the limit strictly beyond current price

    # Risk (§Risk Rules).
    stake_nav_fraction = 0.05
    max_directional_nav_fraction = 0.15

    # ATR(14)/close at signal time, per pair. One open position per pair makes
    # the pair key unique; lost on bot restart (documented — the fallback in
    # custom_exit then uses the current ATR). Lazily an instance attribute so
    # instances never share state.
    @property
    def _entry_atr(self) -> dict:
        if "_entry_atr_store" not in self.__dict__:
            self.__dict__["_entry_atr_store"] = {}
        return self.__dict__["_entry_atr_store"]

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 4,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
            {
                # Daily -5% halt: 288 x 5m candles = 24h lookback/halt.
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 20,
                "stop_duration_candles": 288,
                "max_allowed_drawdown": 0.05,
            },
        ]

    # ------------------------------------------------------------------ #
    # FreqAI label: 1h forward return (12 x 5m candles)
    # ------------------------------------------------------------------ #

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["&-s-future_return_1h"] = dataframe["close"].shift(-12) / dataframe["close"] - 1
        return dataframe

    # ------------------------------------------------------------------ #
    # 1h informative: ATR(14) for the exit target
    # ------------------------------------------------------------------ #

    @informative("1h")
    def populate_informative_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Add ATR(14) on the 1h frame (merged as atr_1h, shifted one 1h candle
        by the informative merge — no lookahead). The exit target is
        1.5 x ATR(14) on the 1h frame: direction comes from the 1h model, so
        the volatility reference is the 1h frame too. (First cut used the 5m
        ATR; the payoff asymmetry vs the -1.5% stop made wins ~4x smaller
        than losses.)
        """
        dataframe = super().populate_informative_1h(dataframe, metadata)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    # ------------------------------------------------------------------ #
    # Layer 3: 1h bias + 5m pullback masks (pure pandas, unit-testable)
    # ------------------------------------------------------------------ #

    def _bias_1h(self, dataframe: DataFrame, prediction: str):
        """
        Long/short bias boolean series, set at each 1h close (the 5m bar with
        minute == 55) from do_predict + |pred| > threshold, forward-filled
        across the hour. Causal: bar t only carries predictions from bars
        that closed at or before t.
        """
        thr = self.entry_threshold.value
        confident = dataframe["do_predict"] == 1
        at_1h_close = dataframe["date"].dt.minute == 55
        long_raw = confident & (dataframe[prediction] > thr)
        short_raw = confident & (dataframe[prediction] < -thr)
        # Sample at the 1h close (NaN elsewhere), forward-fill across the hour.
        long_state = pd.Series(np.nan, index=dataframe.index)
        short_state = pd.Series(np.nan, index=dataframe.index)
        long_state[at_1h_close] = long_raw[at_1h_close].astype(float)
        short_state[at_1h_close] = short_raw[at_1h_close].astype(float)
        long_bias = long_state.ffill().fillna(0.0).astype(bool)
        short_bias = short_state.ffill().fillna(0.0).astype(bool)
        return long_bias, short_bias

    def _pullback_masks(self, dataframe: DataFrame):
        """
        Pullback AGAINST the 1h bias, on closed 5m bars:
          long : close below the 20-bar VWAP AND negative 3-candle return
          short: close above the 20-bar VWAP AND positive 3-candle return
        """
        typical = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        vwap = ((typical * dataframe["volume"]).rolling(self.vwap_candles).sum()
                / dataframe["volume"].rolling(self.vwap_candles).sum())
        ret_n = dataframe["close"].pct_change(self.pullback_candles)
        long_pullback = (dataframe["close"] < vwap) & (ret_n < 0)
        short_pullback = (dataframe["close"] > vwap) & (ret_n > 0)
        return long_pullback.fillna(False), short_pullback.fillna(False)

    # ------------------------------------------------------------------ #
    # Entry signal: regime gate FIRST, then 1h bias, then 5m pullback
    # ------------------------------------------------------------------ #

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self._kmeans_gate_active(dataframe):
            return super().populate_entry_trend(dataframe, metadata)

        prediction = self.prediction_col
        if prediction not in dataframe.columns or "do_predict" not in dataframe.columns:
            return dataframe

        # Gate first: VOLATILE/QUIET/UNKNOWN bars never reach the prediction.
        tradable = dataframe["regime_action_1h"].isin(self.KMEANS_TRADABLE_ACTIONS)

        long_bias, short_bias = self._bias_1h(dataframe, prediction)
        long_pullback, short_pullback = self._pullback_masks(dataframe)

        long_cond = [tradable, long_bias, long_pullback,
                     dataframe["volume_zscore"] > -1.0]
        short_cond = [tradable, short_bias, short_pullback,
                      dataframe["volume_zscore"] > -1.0]

        dataframe.loc[reduce(lambda x, y: x & y, long_cond),
                      ["enter_long", "enter_tag"]] = (1, "v02_pullback_long")
        dataframe.loc[reduce(lambda x, y: x & y, short_cond),
                      ["enter_short", "enter_tag"]] = (1, "v02_pullback_short")
        return dataframe

    # ------------------------------------------------------------------ #
    # Maker execution (Layer 3)
    # ------------------------------------------------------------------ #

    def _limit_price(self, recent: DataFrame, side: str, proposed_rate: float) -> float:
        """
        Limit at the pullback extreme, always strictly beyond the current
        price so the order rests on the book (maker): below for longs, above
        for shorts. `recent` = last N closed candles ending at the signal.
        """
        if side == "long":
            return float(min(recent["low"].min(),
                             proposed_rate * (1 - self.min_maker_offset)))
        return float(max(recent["high"].max(),
                         proposed_rate * (1 + self.min_maker_offset)))

    def _atr_pct(self, df: DataFrame) -> float | None:
        """ATR(14)/close, preferring the 1h frame (atr_1h) over 5m (atr_14)."""
        for col in ("atr_1h", "atr_14"):
            if col in df.columns:
                atr_pct = float(df[col].iloc[-1] / df["close"].iloc[-1])
                if np.isfinite(atr_pct) and atr_pct > 0:
                    return atr_pct
        return None

    def custom_entry_price(self, pair: str, trade, current_time,
                           proposed_rate: float, entry_tag, side: str,
                           **kwargs) -> float:
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or len(df) < self.entry_limit_lookback:
                return proposed_rate
            # ATR reference for the 1.5xATR target, taken at signal time.
            atr_pct = self._atr_pct(df)
            if atr_pct is not None:
                self._entry_atr[pair] = atr_pct
            recent = df.tail(self.entry_limit_lookback)
            price = self._limit_price(recent, side, proposed_rate)
            logger.info(f"[v02] {side} limit {pair}: {price:.2f} "
                        f"(proposed {proposed_rate:.2f})")
            return price
        except Exception as e:
            logger.warning(f"[v02] custom_entry_price fallback for {pair}: {e}")
            return proposed_rate

    # ------------------------------------------------------------------ #
    # Exits (§Exit Rules): time stop 4h, ATR target; model flip inherited
    # ------------------------------------------------------------------ #

    def custom_exit(self, pair: str, trade, current_time,
                    current_rate: float, current_profit: float,
                    **kwargs):
        held_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if held_hours >= self.time_stop_hours:
            return "time_stop_4h"

        atr_pct = self._entry_atr.get(pair)
        if atr_pct is None:
            # Restart fallback: current ATR(14)/close from the analyzed frame.
            try:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                atr_pct = self._atr_pct(df)
            except Exception:
                atr_pct = None
        if atr_pct and atr_pct > 0 and current_profit >= self.atr_target_mult * atr_pct:
            return "atr_target"
        return None

    # ------------------------------------------------------------------ #
    # Risk (§Risk Rules): 5% NAV stake, 15% directional inventory cap
    # ------------------------------------------------------------------ #

    def _nav(self) -> float | None:
        try:
            return float(self.wallets.get_total_stake_amount())
        except Exception:
            return None

    def _directional_exposure(self, side: str) -> float:
        """Open notional (stake x leverage) in one direction, in stake currency."""
        try:
            from freqtrade.persistence import Trade
            total = 0.0
            for t in Trade.get_open_trades():
                if str(t.trade_direction) == side:
                    total += float(t.stake_amount) * float(t.leverage or 1.0)
            return total
        except Exception:
            return 0.0

    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                            proposed_stake: float, min_stake, max_stake,
                            leverage: float, entry_tag, side: str, **kwargs) -> float:
        """5% of NAV per trade (doc §Risk Rules); no regime multiplier in v0.2."""
        nav = self._nav()
        if nav is None or nav <= 0:
            return proposed_stake
        stake = self.stake_nav_fraction * nav
        if min_stake is not None:
            stake = max(stake, min_stake)
        return min(stake, max_stake)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                            rate: float, time_in_force: str, current_time,
                            entry_tag, side: str, **kwargs) -> bool:
        # Parent first: 1m microstructure filter (live/dry-run only).
        if not super().confirm_trade_entry(pair, order_type, amount, rate,
                                           time_in_force, current_time,
                                           entry_tag, side, **kwargs):
            return False
        # Directional inventory cap: block new entries on a side once open
        # notional in that direction exceeds 15% of NAV (all runmodes — this
        # is a pure wallet check, no lookahead).
        nav = self._nav()
        if nav is not None and nav > 0:
            exposure = self._directional_exposure(side) + amount * rate
            if exposure > self.max_directional_nav_fraction * nav:
                logger.info(
                    f"[v02] {pair} {side} blocked: directional exposure "
                    f"{exposure:.2f} > {self.max_directional_nav_fraction:.0%} of NAV {nav:.2f}")
                return False
        return True
