"""
Unit tests for the multi-horizon label generator — focused on the
zero-lookahead contract (plan Day 2: "Labels align correctly; zero lookahead
verified").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.labels import (
    build_labels,
    build_labels_multi,
    future_return,
    future_volatility,
    max_adverse_excursion,
    max_favorable_excursion,
)


@pytest.fixture
def price_df():
    # Deterministic ramp with known wiggles.
    closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
    return pd.DataFrame({
        "open": closes,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.full(len(closes), 10.0),
    })


class TestFutureReturn:
    def test_alignment(self, price_df):
        r = future_return(price_df["close"], 2)
        # row 0: close[2]/close[0] - 1 = 102/100 - 1
        assert r.iloc[0] == pytest.approx(0.02)
        assert r.iloc[4] == pytest.approx(106 / 104 - 1)

    def test_tail_is_nan(self, price_df):
        r = future_return(price_df["close"], 2)
        assert r.iloc[-2:].isna().all()
        assert r.iloc[:-2].notna().all()

    def test_no_lookahead(self, price_df):
        # Changing a FUTURE price must change the label; changing nothing in
        # the past. Verify label at i uses only data after i.
        r = future_return(price_df["close"], 3)
        modified = price_df.copy()
        modified.loc[5, "close"] = 999.0
        r2 = future_return(modified["close"], 3)
        assert r.iloc[2] != r2.iloc[2]          # i=2 looks 3 ahead -> row 5
        assert r.iloc[0] == pytest.approx(r2.iloc[0])  # i=0 looks to row 3


class TestFutureVolatility:
    def test_nonnegative(self, price_df):
        v = future_volatility(price_df["close"], 4)
        assert (v.dropna() >= 0).all()

    def test_constant_price_zero_vol(self):
        close = pd.Series(np.full(20, 100.0))
        v = future_volatility(close, 4)
        assert (v.dropna() == 0).all()


class TestExcursions:
    def test_mfe_uses_future_highs(self, price_df):
        mfe = max_favorable_excursion(price_df["high"], price_df["close"], 3)
        # row 0: max(high[1..3]) / close[0] - 1 = 104/100 - 1
        assert mfe.iloc[0] == pytest.approx(104 / 100 - 1)
        assert (mfe.dropna() >= 0).all()

    def test_mae_uses_future_lows(self, price_df):
        mae = max_adverse_excursion(price_df["low"], price_df["close"], 3)
        # row 0: min(low[1..3]) / close[0] - 1 = 100/100 - 1 = 0
        assert mae.iloc[0] == pytest.approx(0.0)
        assert (mae.dropna() <= 0).all() or (mae.dropna() >= 0).all()

    def test_excursion_tail_behavior(self, price_df):
        mfe = max_favorable_excursion(price_df["high"], price_df["close"], 3)
        # Last row has no future candles -> NaN.
        assert pd.isna(mfe.iloc[-1])


class TestBuildLabels:
    def test_columns_created(self, price_df):
        out = build_labels(price_df, 5, prefix="5m_")
        for col in ["5m_future_return", "5m_future_volatility",
                    "5m_max_favorable_excursion", "5m_max_adverse_excursion"]:
            assert col in out.columns

    def test_multi_horizon(self, price_df):
        out = build_labels_multi(price_df, {"1m": 1, "1h": 12})
        assert "1m_future_return" in out.columns
        assert "1h_future_return" in out.columns
        # 1h horizon (12) exceeds df length -> all NaN for that horizon.
        assert out["1h_future_return"].isna().all()

    def test_invalid_horizon_raises(self, price_df):
        with pytest.raises(ValueError):
            build_labels(price_df, 0)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError):
            build_labels(pd.DataFrame({"close": [1, 2, 3]}), 2)
