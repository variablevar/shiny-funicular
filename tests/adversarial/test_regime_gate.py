"""
tests/adversarial/test_regime_gate.py — v0.2 Phase 1 KMeans regime gate tests.

Exercises the REAL GTQuantRegimeGated strategy (loaded by path, same stub
approach as the rest of this suite) against a REAL fitted RegimeClassifier
artifact written to tmp_path. No reimplemented gate logic: entries come from
strategy.populate_entry_trend, regimes from strategy.populate_informative_1h,
features from the shared strategies/regime_features.py builder.

Covered contracts:
  * VOLATILE / QUIET  -> zero entries (pre-ML HOLD)
  * TRENDING / MEAN_REVERTING -> entries allowed (direction from model+slope)
  * UNKNOWN (warmup)  -> blocked (fail closed)
  * missing / stale artifact -> rule-based gate fallback (parent behavior)
  * no lookahead: perturbing future bars never changes past features
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import PAIR, MockDataProvider  # noqa: E401  (stubs installed there)

REPO_ROOT = Path(__file__).resolve().parents[2]
STRAT_DIR = STRAT = REPO_ROOT / "ft_userdata" / "user_data" / "strategies"
sys.path.insert(0, str(STRAT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import regime_features as rf  # noqa: E402
from core.regime_classifier import RegimeClassifier  # noqa: E402
from analysis.train_regime_model import assign_actions  # noqa: E402

PRED_COL = "&-s-future_return_5m"
METADATA = {"pair": PAIR}
STRATEGY_PATH = STRAT_DIR / "GTQuantRegimeGated.py"


def load_gated_strategy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("GTQuantRegimeGated", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["GTQuantRegimeGated"] = mod
    spec.loader.exec_module(mod)
    return mod.GTQuantRegimeGated


# --------------------------------------------------------------------------- #
# Synthetic 1h market with four clearly separable regimes
# --------------------------------------------------------------------------- #
def make_1h_ohlcv(n_per_regime: int = 300, seed: int = 7) -> pd.DataFrame:
    """Blocks: uptrend, downtrend, high-vol chop, dead-quiet flat (x2)."""
    rng = np.random.default_rng(seed)
    closes, volumes = [50_000.0], [1_000.0]
    blocks = [
        (0.003, 0.001, 1.0),    # uptrend: +0.3%/bar, low noise, avg volume
        (-0.003, 0.001, 1.0),   # downtrend
        (0.0, 0.020, 3.0),      # volatile chop: +/-2% swings, high volume
        (0.0, 0.0002, 0.2),     # quiet: flat, thin volume
    ] * (n_per_regime // 300 * 2 if n_per_regime > 300 else 2)
    for drift, vol, vmult in blocks:
        for _ in range(min(300, n_per_regime)):
            closes.append(closes[-1] * (1 + drift + rng.standard_normal() * vol))
            volumes.append(1_000.0 * vmult * (1 + 0.1 * rng.standard_normal()))
    closes = np.array(closes[1:])
    volumes = np.array(volumes[1:])
    n = len(closes)
    dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "open": closes * (1 - 0.0005),
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": volumes,
    })


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory) -> Path:
    """Fit a real RegimeClassifier on the synthetic market and persist it in
    the exact artifact format analysis/train_regime_model.py produces."""
    df = make_1h_ohlcv()
    feats = rf.build_regime_features(df)
    frame = pd.concat([df[["date", "close"]], feats], axis=1).dropna(
        subset=rf.FEATURE_COLS).reset_index(drop=True)

    clf = RegimeClassifier(feature_cols=rf.FEATURE_COLS, n_clusters=4, random_state=0)
    labeled = clf.fit_transform(frame)

    # Action map via the real mapping code (highest vol -> VOLATILE etc.).
    stats = {}
    for c, sub in labeled.groupby("cluster"):
        stats[c] = {
            "volatility": sub["realized_vol_24h"].mean(),
            "volume_zscore": sub["volume_zscore"].mean(),
            "trend_score": RegimeClassifier._trend_score(sub),
        }
    cluster_action = assign_actions(pd.DataFrame(stats).T)
    assert set(cluster_action.values()) == {"VOLATILE", "QUIET", "TRENDING", "MEAN_REVERTING"}

    out = tmp_path_factory.mktemp("regime_artifact")
    clf.save(out)
    import pickle
    with open(out / "regime_runtime.pkl", "wb") as f:
        pickle.dump({"scaler": clf.scaler, "model": clf.model,
                     "feature_cols": clf.feature_cols,
                     "n_clusters": clf.n_clusters}, f)
    summary = json.loads((out / "regime_summary.json").read_text())
    summary["trained_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    summary["cluster_action_map"] = {str(c): a for c, a in cluster_action.items()}
    (out / "regime_summary.json").write_text(json.dumps(summary, indent=2))
    return out


@pytest.fixture
def gated(artifact_dir):
    cls = load_gated_strategy()
    s = cls()
    s.regime_artifact_dir = str(artifact_dir)
    s.dp = MockDataProvider()
    return s


def entry_df(action=None, regime="BULL", pred=0.005, slope=0.001, vol_z=0.0,
             n=5) -> pd.DataFrame:
    """5m frame holding the columns populate_entry_trend reads."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-09-01", periods=n, freq="5min", tz="UTC"),
        "close": [50_000.0] * n,
        "regime": [regime] * n,
        "do_predict": [1] * n,
        PRED_COL: [pred] * n,
        "ema_50_slope_1h": [slope] * n,
        "volume_zscore": [vol_z] * n,
    })
    if action is not None:
        df["regime_action_1h"] = action
    return df


def n_entries(df: pd.DataFrame) -> int:
    total = 0
    for col in ("enter_long", "enter_short"):
        if col in df.columns:
            total += int(df[col].fillna(0).sum())
    return total


# --------------------------------------------------------------------------- #
# Gate behavior
# --------------------------------------------------------------------------- #
class TestKMeansGate:
    @pytest.mark.parametrize("action", ["VOLATILE", "QUIET"])
    def test_blocked_actions_yield_zero_entries(self, gated, action):
        df = gated.populate_entry_trend(entry_df(action=action).copy(), METADATA)
        assert n_entries(df) == 0

    @pytest.mark.parametrize("action", ["TRENDING", "MEAN_REVERTING"])
    def test_tradable_actions_allow_long(self, gated, action):
        df = gated.populate_entry_trend(entry_df(action=action).copy(), METADATA)
        assert int(df["enter_long"].fillna(0).sum()) == 5
        assert (df["enter_tag"].dropna() == "freqai_long_regime").all()

    def test_tradable_allows_short(self, gated):
        df = gated.populate_entry_trend(
            entry_df(action="TRENDING", pred=-0.005, slope=-0.001).copy(), METADATA)
        assert int(df["enter_short"].fillna(0).sum()) == 5

    def test_unknown_action_is_blocked(self, gated):
        """Warmup bars (UNKNOWN) fail closed."""
        df = gated.populate_entry_trend(entry_df(action="UNKNOWN").copy(), METADATA)
        assert n_entries(df) == 0

    def test_blocked_even_with_strong_prediction(self, gated):
        """The gate runs before the ML threshold: a huge prediction in a
        VOLATILE regime must still produce no entry."""
        df = gated.populate_entry_trend(
            entry_df(action="VOLATILE", pred=0.05, slope=0.01).copy(), METADATA)
        assert n_entries(df) == 0


# --------------------------------------------------------------------------- #
# Fallback behavior
# --------------------------------------------------------------------------- #
class TestFallback:
    def test_missing_artifact_falls_back_to_rule_gate(self):
        cls = load_gated_strategy()
        s = cls()
        s.regime_artifact_dir = "/nonexistent/regime"
        s.dp = MockDataProvider()
        # Parent rule-based gate: RANGE blocks ...
        blocked = s.populate_entry_trend(entry_df(regime="RANGE"), METADATA)
        assert n_entries(blocked) == 0
        # ... BULL allows (pred/slope are bullish in the fixture).
        allowed = s.populate_entry_trend(entry_df(regime="BULL"), METADATA)
        assert int(allowed["enter_long"].fillna(0).sum()) == 5

    def test_stale_artifact_falls_back(self, artifact_dir, tmp_path):
        stale = tmp_path / "stale"
        stale.mkdir()
        for f in artifact_dir.iterdir():
            (stale / f.name).write_bytes(f.read_bytes())
        summary = json.loads((stale / "regime_summary.json").read_text())
        summary["trained_at"] = (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=120)).isoformat()
        (stale / "regime_summary.json").write_text(json.dumps(summary))

        cls = load_gated_strategy()
        s = cls()
        s.regime_artifact_dir = str(stale)
        s.dp = MockDataProvider()
        df = s.populate_entry_trend(entry_df(regime="RANGE"), METADATA)
        assert n_entries(df) == 0  # rule-based gate engaged, not the kmeans one

    def test_gate_disabled_uses_parent(self, gated):
        gated.use_kmeans_gate = False
        df = gated.populate_entry_trend(entry_df(regime="BULL"), METADATA)
        assert int(df["enter_long"].fillna(0).sum()) == 5


# --------------------------------------------------------------------------- #
# Informative column + no-lookahead contracts
# --------------------------------------------------------------------------- #
class TestInformativeAndLookahead:
    def test_informative_produces_action_column(self, gated):
        df1h = make_1h_ohlcv()
        out = gated.populate_informative_1h(df1h.copy(), METADATA)
        assert "regime_action" in out.columns
        assert "regime_kmeans" in out.columns
        tail = out["regime_action"].iloc[-50:]
        assert tail.isin(gated.KMEANS_TRADABLE_ACTIONS + gated.KMEANS_BLOCKED_ACTIONS).all()
        # Warmup rows (NaN features) are UNKNOWN, never a silent guess.
        feats = rf.build_regime_features(df1h)
        nan_rows = ~feats.notna().all(axis=1)
        assert nan_rows.iloc[0]  # first bar always warm-up
        assert (out.loc[nan_rows, "regime_action"] == "UNKNOWN").all()
        assert (out.loc[~nan_rows, "regime_action"] != "UNKNOWN").all()

    def test_informative_without_artifact_adds_no_columns(self):
        cls = load_gated_strategy()
        s = cls()
        s.regime_artifact_dir = "/nonexistent/regime"
        s.dp = MockDataProvider()
        out = s.populate_informative_1h(make_1h_ohlcv(300).copy(), METADATA)
        assert "regime_action" not in out.columns

    def test_features_are_causal(self):
        """Perturbing bars after T must not change any feature at/before T."""
        df = make_1h_ohlcv()
        base = rf.build_regime_features(df)
        cut = 700
        shocked = df.copy()
        shocked.loc[shocked.index > cut, "close"] *= 1.5
        shocked.loc[shocked.index > cut, "volume"] *= 10.0
        perturbed = rf.build_regime_features(shocked)
        pd.testing.assert_frame_equal(
            base.iloc[: cut + 1], perturbed.iloc[: cut + 1])

    def test_regime_at_t_ignores_future(self, gated):
        """End-to-end: regime_action at bar T is identical when the future
        is violently repriced."""
        df = make_1h_ohlcv()
        cut = 800
        ref = gated.populate_informative_1h(df.copy(), METADATA)["regime_action"]
        shocked = df.copy()
        shocked.loc[shocked.index > cut, "close"] *= 2.0
        shocked.loc[shocked.index > cut, "volume"] *= 20.0
        got = gated.populate_informative_1h(shocked, METADATA)["regime_action"]
        pd.testing.assert_series_equal(ref.iloc[: cut + 1], got.iloc[: cut + 1])


# --------------------------------------------------------------------------- #
# Action mapping (real code from analysis/train_regime_model.py)
# --------------------------------------------------------------------------- #
class TestAssignActions:
    def test_mapping_rules(self):
        stats = pd.DataFrame({
            0: {"volatility": 0.010, "volume_zscore": 1.0, "trend_score": 0.05},
            1: {"volatility": 0.002, "volume_zscore": -0.5, "trend_score": 0.001},
            2: {"volatility": 0.005, "volume_zscore": 0.3, "trend_score": 0.002},
            3: {"volatility": 0.004, "volume_zscore": 0.2, "trend_score": -0.04},
            4: {"volatility": 0.006, "volume_zscore": 0.1, "trend_score": 0.03},
        }).T
        mapping = assign_actions(stats)
        assert mapping[0] == "VOLATILE"          # highest realized vol
        assert mapping[1] == "QUIET"             # lowest realized vol
        assert mapping[2] == "MEAN_REVERTING"    # weakest drift of the rest
        assert mapping[3] == "TRENDING"
        assert mapping[4] == "TRENDING"
        assert len(set(mapping)) == 5            # every cluster used exactly once
