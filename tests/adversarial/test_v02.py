"""
tests/adversarial/test_v02.py — v0.2 Phase 2 tests for GTQuantV02.

Exercises the REAL GTQuantV02 strategy (loaded by path, same stub approach as
the rest of this suite — see conftest.py). Covered contracts:

  * 1h bias: sampled at the 1h close (minute == 55), forward-filled across
    the hour; mid-hour predictions do NOT set the bias; no lookahead.
  * Pullback: long fires below short VWAP with a negative 3-candle return
    (short mirrored); bias without pullback yields no entry.
  * Regime gate stays FIRST: VOLATILE/QUIET block even with bias + pullback;
    missing artifact falls back to the parent rule-based gate.
  * Maker limit placement: long limit strictly below current price, short
    strictly above; pullback-extreme anchoring; fallback to proposed rate.
  * Exits (hybrid, Day 15): tuned-1h ROI ladder + wide -10.9% catastrophe
    stop + model-flip primary exit (threshold -0.0037); optional 1.5x ATR
    target; NO time stop and NO trailing stop.
  * Risk: stake = 5% of NAV; directional inventory cap 15% of NAV blocks.
"""
import sys
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import PAIR, MockDataProvider  # noqa: E401  (stubs installed there)

REPO_ROOT = Path(__file__).resolve().parents[2]
STRAT_DIR = REPO_ROOT / "ft_userdata" / "user_data" / "strategies"
sys.path.insert(0, str(STRAT_DIR))

PRED = "&-s-future_return_1h"
STRATEGY_PATH = STRAT_DIR / "GTQuantV02.py"


def load_v02():
    import importlib.util
    # Load the parent chain explicitly so `from GTQuantRegimeGated import ...`
    # resolves inside GTQuantV02.
    for name, path in (("GTQuantMultiTF", STRAT_DIR / "GTQuantMultiTF.py"),
                       ("GTQuantRegimeGated", STRAT_DIR / "GTQuantRegimeGated.py"),
                       ("GTQuantV02", STRATEGY_PATH)):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["GTQuantV02"].GTQuantV02


@pytest.fixture
def strat():
    cls = load_v02()
    s = cls()
    # Arm the KMeans gate without the artifact pipeline: the gate-order tests
    # only need `regime_action_1h` values to be honored.
    s._load_kmeans = lambda: {"dummy": True}
    s.dp = MockDataProvider()
    return s


# --------------------------------------------------------------------------- #
# Frame builders
# --------------------------------------------------------------------------- #
def make_5m(closes, start="2026-09-01 00:00", vol=1000.0) -> pd.DataFrame:
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": vol,
    })


def entry_df(action="TRENDING", n=48, tail="dip", pred_at_55=0.005) -> pd.DataFrame:
    """
    48 x 5m bars (4 hours). Flat price, then a tail 'dip' (long pullback) or
    'rip' (short pullback). The 1h-close bars (minute == 55) carry `pred_at_55`;
    mid-hour bars carry a HUGE counter-prediction that must be ignored by the
    bias sampler.
    """
    closes = [100.0] * (n - 4)
    closes += [99.6, 99.3, 99.1, 99.0] if tail == "dip" else [100.4, 100.7, 100.9, 101.0]
    df = make_5m(closes)
    at_55 = df["date"].dt.minute == 55
    df["do_predict"] = 1
    df[PRED] = np.where(at_55, pred_at_55, -10 * pred_at_55)  # mid-hour = counter
    df["volume_zscore"] = 0.0
    df["atr_14"] = 0.30  # 0.3% of price
    if action is not None:
        df["regime_action_1h"] = action
    return df


def n_entries(df) -> int:
    return sum(int(df[c].fillna(0).sum()) for c in ("enter_long", "enter_short")
               if c in df.columns)


# --------------------------------------------------------------------------- #
# 1h bias sampling (Layer 2)
# --------------------------------------------------------------------------- #
class TestBias1h:
    def test_bias_set_at_1h_close_and_ffilled(self, strat):
        df = entry_df(pred_at_55=0.005)
        long_bias, short_bias = strat._bias_1h(df, PRED)
        at_55 = df["date"].dt.minute == 55
        # Bias goes True at each :55 bar and stays True until the next :55 bar.
        assert long_bias[at_55].all()
        assert not short_bias.any()
        first_55 = int(np.argmax(at_55.to_numpy()))
        assert not long_bias.iloc[:first_55].any()          # before 1st close: no bias
        assert long_bias.iloc[first_55:first_55 + 12].all()  # ffilled across the hour

    def test_mid_hour_prediction_ignored(self, strat):
        """A huge prediction on a non-:55 bar must NOT create a bias."""
        df = entry_df(pred_at_55=0.0)      # neutral at 1h closes
        df.loc[~df.index.isin(df.index[df["date"].dt.minute == 55]), PRED] = 0.5
        long_bias, short_bias = strat._bias_1h(df, PRED)
        assert not long_bias.any() and not short_bias.any()

    def test_bias_is_causal(self, strat):
        """Perturbing predictions after T must not change the bias at/before T."""
        df = entry_df(pred_at_55=0.005)
        ref_long, ref_short = strat._bias_1h(df, PRED)
        cut = 30
        shocked = df.copy()
        shocked.loc[shocked.index > cut, PRED] = -0.9
        got_long, got_short = strat._bias_1h(shocked, PRED)
        pd.testing.assert_series_equal(ref_long.iloc[: cut + 1], got_long.iloc[: cut + 1])
        pd.testing.assert_series_equal(ref_short.iloc[: cut + 1], got_short.iloc[: cut + 1])


# --------------------------------------------------------------------------- #
# Pullback detection (Layer 3)
# --------------------------------------------------------------------------- #
class TestPullback:
    def test_long_pullback_on_dip(self, strat):
        df = entry_df(tail="dip")
        long_pb, short_pb = strat._pullback_masks(df)
        assert long_pb.iloc[-1] and not short_pb.iloc[-1]

    def test_short_pullback_on_rip(self, strat):
        df = entry_df(tail="rip")
        long_pb, short_pb = strat._pullback_masks(df)
        assert short_pb.iloc[-1] and not long_pb.iloc[-1]

    def test_flat_tail_is_no_pullback(self, strat):
        df = make_5m([100.0] * 48)
        long_pb, short_pb = strat._pullback_masks(df)
        assert not long_pb.iloc[-1] and not short_pb.iloc[-1]


# --------------------------------------------------------------------------- #
# Entry chain: regime gate first, then bias, then pullback
# --------------------------------------------------------------------------- #
class TestEntryChain:
    def test_long_entry_on_bias_plus_pullback(self, strat):
        df = strat.populate_entry_trend(entry_df(tail="dip", pred_at_55=0.005), {"pair": PAIR})
        assert df["enter_long"].fillna(0).sum() >= 1
        assert (df["enter_tag"].dropna() == "v02_pullback_long").all()

    def test_short_entry_on_bias_plus_pullback(self, strat):
        df = strat.populate_entry_trend(entry_df(tail="rip", pred_at_55=-0.005), {"pair": PAIR})
        assert df["enter_short"].fillna(0).sum() >= 1
        assert (df["enter_tag"].dropna() == "v02_pullback_short").all()

    @pytest.mark.parametrize("action", ["VOLATILE", "QUIET", "UNKNOWN"])
    def test_gate_blocks_even_with_bias_and_pullback(self, strat, action):
        df = strat.populate_entry_trend(
            entry_df(action=action, tail="dip", pred_at_55=0.005), {"pair": PAIR})
        assert n_entries(df) == 0

    def test_bias_without_pullback_yields_no_entry(self, strat):
        df = make_5m([100.0] * 48)
        df["do_predict"] = 1
        df[PRED] = np.where(df["date"].dt.minute == 55, 0.005, 0.0)
        df["volume_zscore"] = 0.0
        df["regime_action_1h"] = "TRENDING"
        out = strat.populate_entry_trend(df, {"pair": PAIR})
        assert n_entries(out) == 0

    def test_prediction_columns_missing_is_noop(self, strat):
        df = entry_df().drop(columns=[PRED, "do_predict"])
        out = strat.populate_entry_trend(df, {"pair": PAIR})
        assert n_entries(out) == 0


# --------------------------------------------------------------------------- #
# Fallback when the KMeans artifact is gone
# --------------------------------------------------------------------------- #
class TestFallback:
    def test_missing_artifact_uses_parent_gate(self):
        cls = load_v02()
        s = cls()
        s.regime_artifact_dir = "/nonexistent/regime"
        s.dp = MockDataProvider()
        df = entry_df(action=None)
        df["regime"] = "RANGE"
        df["ema_50_slope_1h"] = 0.001
        blocked = s.populate_entry_trend(df.copy(), {"pair": PAIR})
        assert n_entries(blocked) == 0  # parent rule-based gate: RANGE blocked
        df["regime"] = "BULL"
        allowed = s.populate_entry_trend(df.copy(), {"pair": PAIR})
        assert int(allowed["enter_long"].fillna(0).sum()) >= 1


# --------------------------------------------------------------------------- #
# Maker limit placement
# --------------------------------------------------------------------------- #
class TestMakerEntryPrice:
    def _df_for_pricing(self):
        df = entry_df(tail="dip")
        return df

    def test_long_limit_below_current_price(self, strat):
        df = self._df_for_pricing()
        strat.dp = MockDataProvider({(PAIR, "5m"): df})
        proposed = 99.05
        price = strat.custom_entry_price(PAIR, None, df["date"].iloc[-1],
                                         proposed, None, "long")
        assert price < proposed
        assert price == pytest.approx(min(df["low"].tail(3).min(),
                                          proposed * (1 - strat.min_maker_offset)))

    def test_short_limit_above_current_price(self, strat):
        df = entry_df(tail="rip")
        strat.dp = MockDataProvider({(PAIR, "5m"): df})
        proposed = 100.95
        price = strat.custom_entry_price(PAIR, None, df["date"].iloc[-1],
                                         proposed, None, "short")
        assert price > proposed
        assert price == pytest.approx(max(df["high"].tail(3).max(),
                                          proposed * (1 + strat.min_maker_offset)))

    def test_fallback_when_dataframe_too_short(self, strat):
        strat.dp = MockDataProvider({(PAIR, "5m"): make_5m([100.0, 100.1])})
        price = strat.custom_entry_price(PAIR, None, pd.Timestamp.now(tz="UTC"),
                                         99.0, None, "long")
        assert price == 99.0

    def test_atr_reference_stored_at_signal(self, strat):
        df = self._df_for_pricing()
        strat.dp = MockDataProvider({(PAIR, "5m"): df})
        strat.custom_entry_price(PAIR, None, df["date"].iloc[-1], 99.05, None, "long")
        assert strat._entry_atr[PAIR] == pytest.approx(0.30 / 99.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Exits (hybrid): ROI ladder + wide stop + model flip; optional ATR target
# --------------------------------------------------------------------------- #
class TestExits:
    def _trade(self, open_ts):
        return types.SimpleNamespace(open_date_utc=open_ts)

    def test_roi_ladder_matches_tuned_1h(self, strat):
        """Tuned-1h ROI ladder (GTQuantMultiTF1h.json), NOT the doc's
        disabled table ({"0": 100.0})."""
        assert strat.minimal_roi == {"0": 0.108, "39": 0.055, "91": 0.04, "205": 0}

    def test_stoploss_is_wide_catastrophe_stop(self, strat):
        """-10.9% tuned catastrophe stop, NOT the doc-literal -1.5% trading
        stop that lost -68.08 USDT in Phase 2."""
        assert strat.stoploss == pytest.approx(-0.109)

    def test_trailing_fully_disabled(self, strat):
        """The doc's +1.1%/0.5% trailing cost money in Phase 2 — removed."""
        assert strat.trailing_stop is False
        assert not strat.trailing_stop_positive
        assert strat.trailing_stop_positive_offset == 0.0
        assert strat.trailing_only_offset_is_reached is False

    def test_exit_threshold_is_tuned_1h(self, strat):
        """Model-flip exit threshold from GTQuantMultiTF1h.json (-0.0037),
        not the parent default -0.0005."""
        assert strat.exit_threshold.value == pytest.approx(-0.0037)

    def test_model_flip_is_primary_exit(self, strat):
        """Inherited populate_exit_trend fires model_flip_* on the tuned
        threshold."""
        df = make_5m([100.0] * 48)
        df["do_predict"] = 1
        df[PRED] = -0.004  # below exit_threshold -0.0037 -> exit longs
        out = strat.populate_exit_trend(df.copy(), {"pair": PAIR})
        assert out["exit_long"].fillna(0).sum() == 48
        assert (out["exit_tag"].dropna() == "model_flip_down").all()
        df[PRED] = -0.003  # inside the band -> no exit
        out = strat.populate_exit_trend(df.copy(), {"pair": PAIR})
        assert "exit_long" not in out.columns or out["exit_long"].fillna(0).sum() == 0

    def test_no_time_stop(self, strat):
        """The 4h time stop is REMOVED: a 10h-old trade below target must
        not be custom-exited."""
        t0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")
        strat._entry_atr[PAIR] = 0.002  # target = 0.3%
        assert strat.custom_exit(PAIR, self._trade(t0), t0 + timedelta(hours=10),
                                 100.0, 0.001) is None

    def test_atr_target(self, strat):
        t0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")
        strat._entry_atr[PAIR] = 0.002  # target = 1.5 * 0.2% = 0.3%
        trade = self._trade(t0)
        assert strat.custom_exit(PAIR, trade, t0 + timedelta(hours=1),
                                 100.0, 0.0029) is None
        assert strat.custom_exit(PAIR, trade, t0 + timedelta(hours=1),
                                 100.0, 0.003) == "atr_target"

    def test_atr_target_never_fires_on_loss(self, strat):
        t0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")
        strat._entry_atr[PAIR] = 0.002
        assert strat.custom_exit(PAIR, self._trade(t0), t0 + timedelta(hours=1),
                                 100.0, -0.01) is None

    def test_atr_target_disabled_by_flag(self, strat):
        """use_atr_target=False (A/B arm) makes custom_exit inert."""
        strat.use_atr_target = False
        t0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")
        strat._entry_atr[PAIR] = 0.002
        assert strat.custom_exit(PAIR, self._trade(t0), t0 + timedelta(hours=1),
                                 100.0, 0.05) is None


# --------------------------------------------------------------------------- #
# Risk: 5% NAV stake, 15% directional inventory cap
# --------------------------------------------------------------------------- #
class TestRisk:
    def _stub_wallets(self, strat, nav):
        strat.wallets = types.SimpleNamespace(get_total_stake_amount=lambda: nav)

    def test_stake_is_5pct_of_nav(self, strat):
        self._stub_wallets(strat, 1000.0)
        stake = strat.custom_stake_amount(PAIR, None, 100.0, 100.0,
                                          10.0, 1000.0, 1.0, None, "long")
        assert stake == pytest.approx(50.0)

    def test_stake_clamped_to_min(self, strat):
        self._stub_wallets(strat, 100.0)  # 5% = 5 < min_stake 10
        stake = strat.custom_stake_amount(PAIR, None, 100.0, 100.0,
                                          10.0, 1000.0, 1.0, None, "long")
        assert stake == 10.0

    def test_inventory_cap_blocks_beyond_15pct_nav(self, strat):
        self._stub_wallets(strat, 1000.0)
        # 140 already open on the long side + 50 new = 190 > 150 cap.
        strat._directional_exposure = lambda side: 140.0
        ok = strat.confirm_trade_entry(PAIR, "limit", 0.5, 100.0, "GTC",
                                       pd.Timestamp.now(tz="UTC"), None, "long")
        assert ok is False

    def test_inventory_cap_allows_below_15pct_nav(self, strat):
        self._stub_wallets(strat, 1000.0)
        strat._directional_exposure = lambda side: 50.0  # 50 + 50 = 100 < 150
        ok = strat.confirm_trade_entry(PAIR, "limit", 0.5, 100.0, "GTC",
                                       pd.Timestamp.now(tz="UTC"), None, "long")
        assert ok is True

    def test_inventory_cap_is_per_direction(self, strat):
        self._stub_wallets(strat, 1000.0)
        # Heavy LONG exposure must not block a SHORT entry.
        strat._directional_exposure = lambda side: 140.0 if side == "long" else 0.0
        ok = strat.confirm_trade_entry(PAIR, "limit", 0.5, 100.0, "GTC",
                                       pd.Timestamp.now(tz="UTC"), None, "short")
        assert ok is True
