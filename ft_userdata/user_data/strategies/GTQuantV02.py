"""
GTQuantV02 — v0.2 Phase 2 entries (docs/v0.2-strategy.md §Layer 2, §Layer 3)
with HYBRID exits: the doc-literal exit stack was replaced by the tuned-1h
economics from the positive Phase 2 reference (GTQuantMultiTF1h +
strategies/GTQuantMultiTF1h.json on the identical window/identifier:
98 trades, +10.77 USDT, Sharpe 3.27, WR 64.3%).

Phase 2b short-side asymmetry fix (2026-09-17): decomposition of the hybrid
run (110 trades, -11.52 USDT) showed longs +0.83 / shorts -12.35, with
-15.67 of the short bleed from model_flip_up exits opened in KMeans
BULL/STRONG_BULL regimes — shorts structurally fading an up-trend. Shorts
in KMeans BEAR were +1.45. Fix: shorts enter ONLY when the KMeans regime on
the latest closed 1h bar is BEAR (`kmeans_short_regimes`), longs ungated
(BEAR-regime pullback longs were the best long bucket, +5.28). A/B on the
same window (dedicated identifier gtquant-phase2b-ab; control arm reproduced
the baseline to the cent):
    baseline  (no gate, flip exits on):     110 trades, -11.52 USDT
    gate      (shorts BEAR-only):            61 trades,  +2.28 USDT
    noflip    (no short flip, no gate):      98 trades,  -7.68 USDT
    combo     (gate + no short flip):        61 trades,  +3.05 USDT
No-flip alone stays negative (shorts ride into the -10.9% catastrophe stop:
-17.91 USDT over 3 stops) — the gate is the fix; dropping the short flip
exit only stacks on top of it (the 2 gated BEAR flip-up exits were -0.76).
The combo (= current defaults) passed the 3-fold walk-forward (folds
20260614-20260714 / 20260714-20260814 / 20260814-20260911): every fold
positive (+0.48% / +0.40% / +0.20%), mean fold Sharpe 6.60 vs the pre-fix
hybrid's -1.37% fold-3 blow-up (mean Sharpe 0.84).

Why the deviation (Phase 2 data, logs/phase2_v02_atr1h.log, same window and
identifier, entry mechanics identical): the doc-literal exits lost -31.92
USDT over 595 trades. Exit-reason anatomy:
    stop_loss (-1.5% hard stop):  76 exits, -68.08 USDT (100% losers)
    time_stop_4h:                131 exits, -18.49 USDT (WR 32.8%)
    model_flip_*:                368 exits, +44.46 USDT (WR ~69%)
    atr_target (1.5x 1h ATR):     18 exits, +10.66 USDT (100% wins)
The tight stop was a trading stop, not a catastrophe stop, and the 4h time
stop clipped trades the model would have exited profitably. The hybrid keeps
the proven entry chain untouched and swaps the exits for the tuned-1h stack.

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

Exits (HYBRID — tuned-1h economics, NOT the doc §Exit Rules):
    - model-flip exit (populate_exit_trend, inherited, exit_threshold
      -0.0037 from GTQuantMultiTF1h.json) is the PRIMARY exit signal for
      LONGS; shorts skip model_flip_up (Phase 2b, `short_flip_exit = False`)
    - ROI ladder (tuned): 10.8% immediate, 5.5% after 39m, 4% after 91m,
      any profit after 205m
    - wide stoploss -10.9% (tuned): a catastrophe stop, not a trading stop
    - 1.5 x ATR(14) of the 1h frame at entry (custom_exit). Kept: it lost the
      single-window A/B by 0.58 USDT but WON the 3-fold walk-forward (mean
      fold Sharpe 0.84 vs 0.02 without), and WF is the selection criterion.
    - REMOVED vs the doc: the 4h time stop and the +1.1%/0.5% trailing stop
      (both cost money in the Phase 2 backtest)

Risk (doc §Execution Rules, unchanged):
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
    """1h direction + 5m pullback-timed maker entries, tuned-1h exits."""

    # Layer 2: direction comes only from the 1h-forward-return prediction.
    prediction_col = "&-s-future_return_1h"

    # Bias threshold: tuned-1h 0.00474 (GTQuantMultiTF1h.json) won the Day 15
    # A/B over the Day 14 5m-tuned 0.001444 on the Phase 2 window
    # (-10.94 USDT/112 trades vs -17.86/329, Sharpe -2.31 vs -4.63).
    entry_threshold = DecimalParameter(0.0001, 0.005, default=0.00474, decimals=6,
                                       space="buy", optimize=True, load=True)

    # Model-flip threshold (tuned 1h, GTQuantMultiTF1h.json) — now the PRIMARY
    # exit, not a secondary signal.
    exit_threshold = DecimalParameter(-0.005, -0.0001, default=-0.0037, decimals=6,
                                      space="sell", optimize=True, load=True)

    # Hybrid exit economics (tuned 1h, GTQuantMultiTF1h.json): ROI ladder +
    # wide catastrophe stop. The -10.9% stop is NOT a trading stop — the
    # doc-literal -1.5% stop was the single biggest loser in Phase 2
    # (-68.08 USDT over 76 forced exits).
    minimal_roi = {"0": 0.108, "39": 0.055, "91": 0.04, "205": 0}
    stoploss = -0.109
    # Trailing fully disabled (doc's +1.1%/0.5% trailing cost money in
    # backtest; tuned-1h reference has it off).
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False

    # ATR target (doc §Exit Rules): 1.5 x ATR(14) of the 1h frame at entry.
    # Day 15 A/B: lost the single window by 0.58 USDT but won the 3-fold
    # walk-forward (mean fold Sharpe 0.84 vs 0.02 without) -> KEPT (WF is
    # the selection criterion). The 4h time stop is REMOVED (Phase 2:
    # -18.49 USDT, WR 32.8%).
    use_atr_target = True
    atr_target_mult = 1.5

    # Phase 2b short-side asymmetry fix: SHORTS are only allowed when the
    # KMeans regime on the latest closed 1h bar is BEAR. Decomposition of
    # the Phase 2b hybrid run (110 trades, -11.52 USDT): longs +0.83,
    # shorts -12.35, and -15.67 of the short bleed came from model_flip_up
    # exits in KMeans BULL/STRONG_BULL regimes (shorting into an up-trend);
    # shorts in KMeans BEAR were +1.45. The KMeans cluster map has no
    # STRONG_BEAR cluster, so ("BEAR",) is the full bear set. Longs are NOT
    # gated: BEAR-regime pullback longs were the best long bucket (+5.28).
    # None disables the gate (A/B arm).
    kmeans_short_regimes: tuple | None = ("BEAR",)

    # A/B arm flag kept for reproducibility: when True, shorts exit on
    # model_flip_up (pre-Phase-2b behavior). Phase 2b settled on False —
    # gated (BEAR-only) shorts that skip the flip exit won the window
    # (+3.05 vs +2.28 USDT) and every WF fold (mean Sharpe 6.60 vs 6.24).
    # Without the gate, False is dangerous (shorts ride into the -10.9%
    # catastrophe stop) — the two changes ship together.
    short_flip_exit = False

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

        # Short-side regime gate (Phase 2b): shorts only in a KMeans bear
        # regime. Fail closed if the regime-name column is missing while the
        # gate is active (both columns are produced together; a missing one
        # means the informative frame is broken, not "regime unknown").
        if self.kmeans_short_regimes is None:
            short_regime_ok = tradable
        elif "regime_kmeans_1h" in dataframe.columns:
            short_regime_ok = dataframe["regime_kmeans_1h"].isin(self.kmeans_short_regimes)
        else:
            short_regime_ok = pd.Series(False, index=dataframe.index)

        long_bias, short_bias = self._bias_1h(dataframe, prediction)
        long_pullback, short_pullback = self._pullback_masks(dataframe)

        long_cond = [tradable, long_bias, long_pullback,
                     dataframe["volume_zscore"] > -1.0]
        short_cond = [tradable, short_regime_ok, short_bias, short_pullback,
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
    # Exits (hybrid): ROI ladder + wide stop + model flip (primary) are
    # declarative; custom_exit only implements the optional ATR target.
    # ------------------------------------------------------------------ #

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_exit_trend(dataframe, metadata)
        if not self.short_flip_exit and "exit_tag" in dataframe.columns:
            # A/B arm: shorts ignore model_flip_up (ROI/ATR/stop only).
            mask = dataframe["exit_tag"] == "model_flip_up"
            dataframe.loc[mask, ["exit_short", "exit_tag"]] = (0, None)
        return dataframe

    def custom_exit(self, pair: str, trade, current_time,
                    current_rate: float, current_profit: float,
                    **kwargs):
        if not self.use_atr_target:
            return None

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
