"""
LLM → Quant fusion logic (Day 11).

Combines the quant signal (FreqAI prediction + regime) with the LLM's
structured decision into a single trade decision. Pure functions — unit-tested
in tests/test_fusion.py.

Fusion rules (per plan Day 11):
  - LLM confidence > 0.7 AND bias aligns with quant signal  -> full size
  - LLM confidence 0.5–0.7                                 -> 50% size
  - LLM confidence < 0.5 OR bias conflicts                 -> hold / reduce
  - LLM primary_tf slower than quant TF                    -> defer to slower TF
  - LLM _fallback True (timeout/unparseable)               -> pure quant, size 1.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Timeframe speed ordering (lower = faster).
TF_ORDER = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6}


@dataclass
class QuantSignal:
    """The quant system's candidate signal."""

    direction: int          # +1 long, -1 short, 0 flat
    timeframe: str          # TF the quant model fired on
    confidence: float       # quant model confidence / prob spread
    regime: str


@dataclass
class FusionDecision:
    """The fused trade decision."""

    direction: int          # +1 long, -1 short, 0 = hold
    size_multiplier: float  # applied to the base stake
    effective_tf: str       # TF to actually trade on (after deferral)
    action: str             # "execute" | "reduce" | "hold" | "defer"
    reason: str
    llm_bias: str
    llm_confidence: float
    quant_direction: int


def _bias_dir(bias: str) -> int:
    return {"long": 1, "short": -1, "flat": 0}.get(str(bias).lower(), 0)


def fuse(quant: QuantSignal, llm: Optional[dict],
         conf_full: float = 0.7, conf_reduce: float = 0.5) -> FusionDecision:
    """
    Fuse a quant signal with an LLM decision.

    llm is the parsed decision dict (or None / {"_fallback": True} when the
    LLM is unavailable -> pure quant passthrough).
    """
    # No quant signal -> nothing to fuse.
    if quant.direction == 0:
        return FusionDecision(0, 0.0, quant.timeframe, "hold",
                              "no quant signal", "n/a", 0.0, 0)

    # LLM unavailable / fallback -> trust the quant layer as-is.
    if llm is None or llm.get("_fallback", False):
        return FusionDecision(quant.direction, 1.0, quant.timeframe, "execute",
                              "llm unavailable, pure quant", "n/a", 0.0, quant.direction)

    llm_bias = str(llm.get("bias", "flat")).lower()
    llm_conf = float(llm.get("confidence", 0.0))
    llm_dir = _bias_dir(llm_bias)
    llm_tf = str(llm.get("primary_tf", quant.timeframe))

    # Bias conflict -> hold.
    if llm_dir != 0 and llm_dir != quant.direction:
        return FusionDecision(0, 0.0, quant.timeframe, "hold",
                              f"llm {llm_bias} conflicts quant {quant.direction:+d}",
                              llm_bias, llm_conf, quant.direction)

    # LLM flat (no directional view) -> reduce.
    if llm_dir == 0:
        return FusionDecision(quant.direction, 0.5, quant.timeframe, "reduce",
                              "llm flat", llm_bias, llm_conf, quant.direction)

    # Alignment: size by LLM confidence.
    if llm_conf > conf_full:
        size, action = 1.0, "execute"
        reason = f"aligned, high conf {llm_conf:.2f}"
    elif llm_conf >= conf_reduce:
        size, action = 0.5, "reduce"
        reason = f"aligned, mid conf {llm_conf:.2f}"
    else:
        return FusionDecision(0, 0.0, quant.timeframe, "hold",
                              f"aligned but low conf {llm_conf:.2f}",
                              llm_bias, llm_conf, quant.direction)

    # TF deferral: LLM recommending a slower TF -> defer to it (lower
    # frequency, higher quality per the plan).
    effective_tf = quant.timeframe
    if TF_ORDER.get(llm_tf, 99) > TF_ORDER.get(quant.timeframe, 99):
        effective_tf = llm_tf
        action = "defer"
        reason += f"; defer to {llm_tf}"

    return FusionDecision(quant.direction, size, effective_tf, action,
                          reason, llm_bias, llm_conf, quant.direction)
