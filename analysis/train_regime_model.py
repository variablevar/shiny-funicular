#!/usr/bin/env python3
"""
Train the KMeans regime classifier (Phase 1 of docs/v0.2-strategy.md).

Pipeline:
  1. Load 1h OHLCV feathers for BTC/ETH from ft_userdata/user_data/data/binance.
  2. Build the causal feature frame via strategies/regime_features.py (the SAME
     code the gated strategy uses at runtime — fit/runtime consistency).
  3. Fit core.regime_classifier.RegimeClassifier (StandardScaler + KMeans, k=5)
     on the last FIT_DAYS of bars; transform the full history for statistics.
  4. Map the 5 clusters to the plan's 4 action classes by measured stats:
       VOLATILE        : highest mean realized_vol_24h
       QUIET           : lowest mean realized_vol_24h (tie-break: lowest volume z)
       MEAN_REVERTING  : lowest |trend score| among the rest
       TRENDING        : the remaining directional clusters
  5. Validate: per-regime forward 24h return table + regime-colored price
     chart (results/regime_price_chart.png).
  6. Persist models/regime/ (pickle + regime_summary.json augmented with the
     action map) and copy to ft_userdata/user_data/models/regime/ so the
     Docker container (which mounts only user_data) can load it.

Usage:
    venv/bin/python analysis/train_regime_model.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRAT_DIR = ROOT / "ft_userdata" / "user_data" / "strategies"
sys.path.insert(0, str(STRAT_DIR))
sys.path.insert(0, str(ROOT))

from regime_features import FEATURE_COLS, build_regime_features  # noqa: E402
from core.regime_classifier import RegimeClassifier  # noqa: E402

DATA_DIR = ROOT / "ft_userdata" / "user_data" / "data" / "binance" / "futures"
MODELS_DIR = ROOT / "models" / "regime"
CONTAINER_MODELS_DIR = ROOT / "ft_userdata" / "user_data" / "models" / "regime"
RESULTS_DIR = ROOT / "results"

PAIRS = {
    "BTC/USDT:USDT": "BTC_USDT_USDT-1h-futures.feather",
    "ETH/USDT:USDT": "ETH_USDT_USDT-1h-futures.feather",
}

FIT_DAYS = 90            # fit window (trailing)
CHART_DAYS = 120         # chart window
ACTIONS = ("TRENDING", "MEAN_REVERTING", "VOLATILE", "QUIET")


# --------------------------------------------------------------------------- #
# Data + features
# --------------------------------------------------------------------------- #
def load_pair_frame(pair: str) -> pd.DataFrame:
    path = DATA_DIR / PAIRS[pair]
    df = pd.read_feather(path)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    feats = build_regime_features(df)
    out = pd.concat([df[["date", "close"]], feats], axis=1)
    out["pair"] = pair
    return out.dropna(subset=FEATURE_COLS).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cluster -> action mapping
# --------------------------------------------------------------------------- #
def assign_actions(cluster_stats: pd.DataFrame) -> dict[int, str]:
    """
    Map cluster ids to the plan's 4 action classes from measured stats.

    cluster_stats: DataFrame indexed by cluster id with columns
    ``volatility``, ``volume_zscore``, ``trend_score``.

    Order of assignment (each cluster used exactly once):
      1. highest mean realized vol            -> VOLATILE
      2. lowest mean realized vol (remaining) -> QUIET
      3. lowest |trend score| (remaining)     -> MEAN_REVERTING
      4. everything else                      -> TRENDING
    """
    remaining = dict.fromkeys(cluster_stats.index)
    mapping: dict[int, str] = {}

    volatile = cluster_stats["volatility"].idxmax()
    mapping[volatile] = "VOLATILE"
    remaining.pop(volatile)

    quiet = cluster_stats.loc[list(remaining)].sort_values(
        ["volatility", "volume_zscore"]).index[0]
    mapping[quiet] = "QUIET"
    remaining.pop(quiet)

    rest = cluster_stats.loc[list(remaining)]
    mean_reverting = rest["trend_score"].abs().idxmin()
    mapping[mean_reverting] = "MEAN_REVERTING"
    remaining.pop(mean_reverting)

    for c in remaining:
        mapping[c] = "TRENDING"
    return mapping


# --------------------------------------------------------------------------- #
# Validation stats
# --------------------------------------------------------------------------- #
def forward_return_stats(labeled: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Per-regime forward 24h return stats over the full (labeled) history."""
    df = labeled.sort_values(["pair", "date"]).copy()
    df["fwd_ret"] = df.groupby("pair")["close"].shift(-horizon) / df["close"] - 1.0
    df = df.dropna(subset=["fwd_ret"])
    stats = df.groupby("regime")["fwd_ret"].agg(
        n="count", mean="mean", median="median", std="std",
        pct_positive=lambda s: float((s > 0).mean()),
    )
    stats["mean_abs"] = df.groupby("regime")["fwd_ret"].apply(lambda s: s.abs().mean())
    return stats


def cluster_stat_frame(clf: RegimeClassifier, fit_labeled: pd.DataFrame) -> pd.DataFrame:
    """Measured per-cluster stats used for the action mapping."""
    rows = {}
    for c, sub in fit_labeled.groupby("cluster"):
        rows[c] = {
            "volatility": sub["realized_vol_24h"].mean(),
            "volume_zscore": sub["volume_zscore"].mean(),
            "trend_score": RegimeClassifier._trend_score(sub),
            "return_24h": sub["return_24h"].mean(),
            "rsi_14": sub["rsi_14"].mean(),
            "n_bars": len(sub),
        }
    return pd.DataFrame(rows).T


def save_chart(labeled: pd.DataFrame, cluster_action: dict[int, str],
               out_path: Path, days: int = CHART_DAYS) -> None:
    """Regime-colored 1h close chart for BTC and ETH (visual sanity check)."""
    colors = {"TRENDING": "#2ca02c", "MEAN_REVERTING": "#1f77b4",
              "VOLATILE": "#d62728", "QUIET": "#7f7f7f"}
    cutoff = labeled["date"].max() - pd.Timedelta(days=days)
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(16, 9), sharex=True)
    for ax, pair in zip(axes, PAIRS):
        sub = labeled[(labeled["pair"] == pair) & (labeled["date"] >= cutoff)]
        ax.plot(sub["date"], sub["close"], color="black", lw=0.6, alpha=0.6)
        for action, color in colors.items():
            mask = sub["cluster"].map(cluster_action) == action
            ax.scatter(sub.loc[mask, "date"], sub.loc[mask, "close"],
                       s=4, color=color, label=action, rasterized=True)
        ax.set_title(f"{pair} 1h close colored by action class (last {days}d)")
        ax.legend(loc="upper left", markerscale=3, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("date")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"Loading 1h feathers from {DATA_DIR} ...")
    frames = {pair: load_pair_frame(pair) for pair in PAIRS}
    full = pd.concat(frames.values(), ignore_index=True)
    print(f"Full feature frame: {len(full)} rows, "
          f"{full['date'].min()} -> {full['date'].max()}")

    fit_cutoff = full["date"].max() - pd.Timedelta(days=FIT_DAYS)
    fit_df = full[full["date"] >= fit_cutoff].reset_index(drop=True)
    print(f"Fit window: last {FIT_DAYS}d -> {len(fit_df)} rows "
          f"({fit_df['date'].min()} -> {fit_df['date'].max()})")

    clf = RegimeClassifier(feature_cols=FEATURE_COLS, n_clusters=5, random_state=42)
    fit_labeled = clf.fit_transform(fit_df)

    # Label the full history for stats/chart (transform only — no refit).
    full_labeled = clf.transform(full)

    # --- cluster stats + action mapping ----------------------------------- #
    cstats = cluster_stat_frame(clf, fit_labeled)
    cluster_action = assign_actions(cstats)
    regime_action = {clf.cluster_regime_map[c].value: a for c, a in cluster_action.items()}

    print("\nPer-cluster measured stats (fit window):")
    cstats_show = cstats.copy()
    cstats_show["regime"] = [clf.cluster_regime_map[c].value for c in cstats_show.index]
    cstats_show["action"] = [cluster_action[c] for c in cstats_show.index]
    print(cstats_show.to_string(float_format=lambda x: f"{x:+.5f}"))

    # --- forward-return validation ---------------------------------------- #
    fwd = forward_return_stats(full_labeled)
    fwd["action"] = [regime_action[r] for r in fwd.index]
    print("\nPer-regime forward 24h return stats (full history):")
    print(fwd.to_string(float_format=lambda x: f"{x:+.5f}"))

    occ = full_labeled.groupby(["pair", "regime"]).size().unstack(fill_value=0)
    occ_pct = occ.div(occ.sum(axis=1), axis=0)
    print("\nRegime occupancy (full history, fraction of bars):")
    print(occ_pct.to_string(float_format=lambda x: f"{x:.2%}"))

    # --- persist ----------------------------------------------------------- #
    clf.save(MODELS_DIR)

    # Container-portable runtime artifact: the full regime_artifacts.pkl
    # embeds core.regime_classifier.Regime enums, which do not exist inside
    # the Freqtrade Docker image. The runtime pickle holds pure sklearn
    # objects only; cluster maps travel in regime_summary.json instead.
    import pickle
    with open(MODELS_DIR / "regime_runtime.pkl", "wb") as f:
        pickle.dump({
            "scaler": clf.scaler,
            "model": clf.model,
            "feature_cols": clf.feature_cols,
            "n_clusters": clf.n_clusters,
        }, f)

    summary_path = MODELS_DIR / "regime_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "fit_days": FIT_DAYS,
        "fit_window": [str(fit_df["date"].min()), str(fit_df["date"].max())],
        "feature_source": "strategies/regime_features.py (pure pandas, causal)",
        "funding_zscore": "dropped from DEFAULT_FEATURES (not available in-strategy)",
        "cluster_action_map": {str(c): a for c, a in sorted(cluster_action.items())},
        "regime_action_map": regime_action,
        "cluster_stats": {
            str(c): {k: float(v) for k, v in row.items()}
            for c, row in cstats.iterrows()
        },
        "forward_return_stats_24h": {
            regime: {
                **{k: float(v) for k, v in row.items() if k != "action"},
                "action": row["action"],
            }
            for regime, row in fwd.iterrows()
        },
    })
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nArtifacts written to {MODELS_DIR}")

    CONTAINER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODELS_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, CONTAINER_MODELS_DIR / f.name)
    print(f"Copied to {CONTAINER_MODELS_DIR} (Docker-visible)")

    chart_path = RESULTS_DIR / "regime_price_chart.png"
    save_chart(full_labeled, cluster_action, chart_path)
    print(f"Chart saved to {chart_path}")

    # --- degenerate-output guard ------------------------------------------- #
    largest = occ_pct.max(axis=1).max()
    if largest > 0.90:
        print(f"\nWARNING: dominant regime covers {largest:.1%} of bars — "
              "clustering looks degenerate; keep the rule-based gate as default.")
    elif "VOLATILE" not in regime_action.values() or "QUIET" not in regime_action.values():
        print("\nWARNING: VOLATILE/QUIET actions not assigned — mapping incomplete.")
    else:
        print("\nSanity: no dominant-cluster degeneracy detected.")


if __name__ == "__main__":
    main()
