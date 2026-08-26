"""
GT-Quant TF-Aware Risk Configuration and Pre-Trade Validation (Day 1 / Day 5).

Risk limits are code, not comments. Per-timeframe position size multipliers
follow the master plan:

    1m : 0.5x  (microstructure — not traded directly, feeds 5m precision)
    5m : 1.0x  (primary trading TF)
    15m: 1.5x  (swing confirmation)
    1h : 2.0x  (regime-driven, slowest trades, lowest fee erosion)

Circuit breakers halt trading on daily loss, weekly drawdown, excessive
funding carry, or too many open positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# Supported timeframes in the hierarchy (1m is not traded directly).
TRADED_TIMEFRAMES: Tuple[str, ...] = ("5m", "15m", "1h")


@dataclass
class RiskConfig:
    """TF-aware risk limits for the GT-Quant system."""

    # --- Per-timeframe position size multipliers (fraction of base stake) ---
    tf_size_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.5, "5m": 1.0, "15m": 1.5, "1h": 2.0}
    )

    # --- Hard caps ---
    max_open_positions: int = 3
    max_leverage: float = 5.0
    base_stake: float = 1000.0          # reference equity for size multipliers

    # --- Loss limits ---
    max_risk_per_trade: float = 0.0025  # 0.25% of equity at stop
    max_daily_loss: float = 0.03        # 3% daily loss halt
    max_weekly_drawdown: float = 0.05   # 5% drawdown halt

    # --- Funding carry filter ---
    # Reject new longs when funding > +threshold (longs pay too much) and
    # reject new shorts when funding < -threshold.
    funding_rate_threshold: float = 0.0001  # 0.01% per 8h

    # --- Stale data thresholds (seconds), per TF ---
    stale_data_thresholds: Dict[str, int] = field(
        default_factory=lambda: {"1m": 120, "5m": 600, "15m": 1800, "1h": 3600}
    )

    def size_multiplier(self, timeframe: str) -> float:
        """Return the position size multiplier for a timeframe."""
        if timeframe not in self.tf_size_multipliers:
            raise ValueError(f"Unknown timeframe for sizing: {timeframe!r}")
        return self.tf_size_multipliers[timeframe]

    def max_notional(self, timeframe: str) -> float:
        """Max notional exposure allowed for a timeframe."""
        return self.base_stake * self.size_multiplier(timeframe)

    def stale_threshold(self, timeframe: str) -> int:
        """Max allowed data age in seconds for a timeframe."""
        if timeframe not in self.stale_data_thresholds:
            raise ValueError(f"Unknown timeframe for staleness: {timeframe!r}")
        return self.stale_data_thresholds[timeframe]


class TFRiskEngine:
    """
    Pre-trade validation with TF-aware limits and circuit breakers.

    Tracks running state (daily PnL, peak equity, open positions) and
    approves/rejects candidate trades before execution.
    """

    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.open_positions: int = 0
        self.halted: bool = False
        self.halt_reason: str = ""

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #

    def reset_day(self) -> None:
        """Reset daily counters (call at 00:00 UTC)."""
        self.daily_pnl = 0.0

    def update_state(self, pnl: float, equity: float) -> None:
        """Record a closed-trade PnL and current equity."""
        self.daily_pnl += pnl
        self.peak_equity = max(self.peak_equity, equity)

    def on_position_opened(self) -> None:
        self.open_positions += 1

    def on_position_closed(self) -> None:
        self.open_positions = max(0, self.open_positions - 1)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate_trade(
        self,
        timeframe: str,
        direction: int,
        notional: float,
        equity: float,
        funding_rate: float | None = None,
    ) -> Tuple[bool, str]:
        """
        Approve or reject a candidate trade.

        Parameters
        ----------
        timeframe : str
            One of the traded timeframes ("5m", "15m", "1h").
        direction : int
            +1 long, -1 short.
        notional : float
            Proposed position notional in quote currency.
        equity : float
            Current account equity.
        funding_rate : float | None
            Current funding rate (per 8h), if known.

        Returns
        -------
        (allowed, reason)
        """
        if self.halted:
            return False, f"trading halted: {self.halt_reason}"

        if timeframe not in TRADED_TIMEFRAMES:
            return False, f"timeframe {timeframe!r} not tradeable"

        if direction not in (1, -1):
            return False, f"invalid direction {direction}"

        # TF size cap.
        if notional > self.cfg.max_notional(timeframe):
            return (
                False,
                f"notional {notional:.2f} exceeds {timeframe} cap "
                f"{self.cfg.max_notional(timeframe):.2f}",
            )

        # Leverage cap.
        if equity > 0 and notional / equity > self.cfg.max_leverage:
            return False, "leverage cap exceeded"

        # Open positions cap.
        if self.open_positions >= self.cfg.max_open_positions:
            return False, "max open positions reached"

        # Daily loss halt.
        if equity > 0 and self.daily_pnl < -equity * self.cfg.max_daily_loss:
            return False, "daily loss limit reached"

        # Weekly drawdown halt.
        if not self.check_drawdown(equity):
            return False, "weekly drawdown limit reached"

        # Funding carry filter.
        if funding_rate is not None:
            thr = self.cfg.funding_rate_threshold
            if direction == 1 and funding_rate > thr:
                return False, f"funding {funding_rate:+.4%} too costly for longs"
            if direction == -1 and funding_rate < -thr:
                return False, f"funding {funding_rate:+.4%} too costly for shorts"

        return True, "approved"

    def check_drawdown(self, equity: float) -> bool:
        """Return False if peak-to-current drawdown exceeds the weekly limit."""
        if self.peak_equity <= 0:
            return True
        dd = (self.peak_equity - equity) / self.peak_equity
        return dd <= self.cfg.max_weekly_drawdown

    # ------------------------------------------------------------------ #
    # Circuit breakers
    # ------------------------------------------------------------------ #

    def trigger_halt(self, reason: str) -> None:
        """Halt all trading until manually resumed."""
        self.halted = True
        self.halt_reason = reason

    def resume(self) -> None:
        """Resume trading after a halt."""
        self.halted = False
        self.halt_reason = ""

    def check_stale_data(self, timeframe: str, data_age_seconds: float) -> bool:
        """Return False if the data feed for a timeframe is stale."""
        return data_age_seconds <= self.cfg.stale_threshold(timeframe)
