"""
Unit tests for the 1m-micro -> 5m FreqAI feature plumbing.

The critical property is the lookahead contract: features attached to the
5m row with open time `t` may depend only on 1m bars whose close time is
<= t (i.e. 1m bars in [t-5m, t)). These tests build synthetic data where
every 1m bar carries a unique marker and verify the contract exactly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ft_userdata" / "user_data" / "strategies"))

import micro_features as mf

T0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")


def _micro_1m(n_minutes: int, start: pd.Timestamp = T0) -> pd.DataFrame:
    """Synthetic 1m bars; volume_delta of the bar at minute i is exactly i."""
    dates = [start + pd.Timedelta(minutes=i) for i in range(n_minutes)]
    return pd.DataFrame({
        "date": dates,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 10.0,
        "volume_delta": np.arange(n_minutes, dtype=float),
        "tick_momentum": 1.0,
        "realized_variance": 1e-6,
        "trade_count": 100,
    })


def _df5m(n_bars: int, start: pd.Timestamp = T0) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [start + pd.Timedelta(minutes=5 * i) for i in range(n_bars)],
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 50.0,
    })


class TestAggregation:
    def test_bucket_sums_and_shift_key(self):
        """5 1m bars of bucket [t-5m, t) land on the 5m row keyed t."""
        micro = _micro_1m(30)
        buckets = mf.aggregate_to_5m(micro)
        assert len(buckets) == 6
        # 1m bars 0..4 (deltas 0+1+2+3+4=10) key to row T0+5m.
        row = buckets.loc[buckets["date"] == T0 + pd.Timedelta(minutes=5)].iloc[0]
        assert row["m_vol_delta"] == 0 + 1 + 2 + 3 + 4
        assert row["m_volume"] == 50.0
        assert row["m_trade_count"] == 500
        assert row["m_bars"] == 5
        assert row["m_rv"] == pytest.approx(5e-6)

    def test_empty_micro(self):
        buckets = mf.aggregate_to_5m(pd.DataFrame())
        assert list(buckets.columns) == ["date"] + mf.BUCKET_COLS
        assert buckets.empty


class TestNoLookahead:
    def test_row_uses_previous_bucket_only(self):
        """5m row t must contain exactly the 1m bars closing at or before t."""
        micro = _micro_1m(60)
        df5m = _df5m(12)
        # Merge directly (bypass the file-loading entry point).
        buckets = mf.aggregate_to_5m(micro)
        out = df5m.merge(buckets, on="date", how="left", sort=False)
        out[mf.BUCKET_COLS] = out[mf.BUCKET_COLS].fillna(0.0)

        # Row at T0+10m must hold 1m deltas 5..9 (bars [T0+5m, T0+10m)).
        row = out.loc[out["date"] == T0 + pd.Timedelta(minutes=10)].iloc[0]
        assert row["m_vol_delta"] == 5 + 6 + 7 + 8 + 9

    def test_future_perturbation_never_leaks_backward(self):
        """Changing a 1m bar must not alter features of any earlier 5m row."""
        micro = _micro_1m(60)
        buckets_a = mf.aggregate_to_5m(micro)
        out_a = mf.compute_micro_features(
            _df5m(12).merge(buckets_a, on="date", how="left").fillna(0.0))

        micro_b = micro.copy()
        hit = 37  # perturb minute 37
        micro_b.loc[micro_b["date"] == T0 + pd.Timedelta(minutes=hit),
                    "volume_delta"] = 10_000.0
        buckets_b = mf.aggregate_to_5m(micro_b)
        out_b = mf.compute_micro_features(
            _df5m(12).merge(buckets_b, on="date", how="left").fillna(0.0))

        # The perturbed 1m bar closes at T0+38m, so the earliest 5m row
        # allowed to change has open time T0+40m.
        boundary = T0 + pd.Timedelta(minutes=40)
        cols = mf.MICRO_FEATURE_COLS
        before_a = out_a.loc[out_a["date"] < boundary, cols]
        before_b = out_b.loc[out_b["date"] < boundary, cols]
        pd.testing.assert_frame_equal(before_a, before_b)
        # And the change must be visible at/after the boundary (signal flows).
        after_a = out_a.loc[out_a["date"] >= boundary, cols]
        after_b = out_b.loc[out_b["date"] >= boundary, cols]
        assert not after_a.equals(after_b)

    def test_first_row_has_zero_micro(self):
        """The first 5m row of a window has no earlier bucket -> zeros."""
        micro = _micro_1m(30)
        buckets = mf.aggregate_to_5m(micro)
        out = _df5m(6).merge(buckets, on="date", how="left", sort=False)
        out[mf.BUCKET_COLS] = out[mf.BUCKET_COLS].fillna(0.0)
        first = out.iloc[0]
        assert first["m_bars"] == 0
        assert first["m_vol_delta"] == 0.0


class TestMissingData:
    def test_gap_bucket_zero_filled_features(self):
        """A 5m bucket with no 1m bars yields m_bars=0 and 0-valued features."""
        micro = _micro_1m(30)
        # Remove minutes 10..14 -> bucket keyed T0+15m has no bars.
        gap = micro[(micro["date"] < T0 + pd.Timedelta(minutes=10))
                    | (micro["date"] >= T0 + pd.Timedelta(minutes=15))]
        buckets = mf.aggregate_to_5m(gap)
        out = _df5m(6).merge(buckets, on="date", how="left", sort=False)
        out[mf.BUCKET_COLS] = out[mf.BUCKET_COLS].fillna(0.0)
        out = mf.compute_micro_features(out)
        gap_row = out.loc[out["date"] == T0 + pd.Timedelta(minutes=15)].iloc[0]
        assert gap_row["m_bars"] == 0
        for col in mf.MICRO_FEATURE_COLS:
            assert np.isfinite(gap_row[col])

    def test_features_finite_everywhere(self):
        micro = _micro_1m(240)
        buckets = mf.aggregate_to_5m(micro)
        out = _df5m(48).merge(buckets, on="date", how="left", sort=False)
        out[mf.BUCKET_COLS] = out[mf.BUCKET_COLS].fillna(0.0)
        out = mf.compute_micro_features(out)
        assert np.isfinite(out[mf.MICRO_FEATURE_COLS].to_numpy()).all()


class TestLoad:
    def test_missing_parquet_returns_empty(self, tmp_path):
        df = mf.load_micro_1m("BTC/USDT:USDT", T0, T0 + pd.Timedelta(hours=1),
                              tmp_path, "backtest")
        assert df.empty

    def test_unknown_pair_returns_empty(self, tmp_path):
        df = mf.load_micro_1m("DOGE/USDT:USDT", T0, T0 + pd.Timedelta(hours=1),
                              tmp_path, "backtest")
        assert df.empty
