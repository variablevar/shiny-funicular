"""
Unit tests for the LLM-quant fusion logic (Day 11).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.fusion import QuantSignal, fuse


def q(direction=1, tf="5m", conf=0.8, regime="BULL"):
    return QuantSignal(direction=direction, timeframe=tf, confidence=conf, regime=regime)


def llm(bias="long", confidence=0.8, primary_tf="5m", fallback=False):
    d = {"bias": bias, "confidence": confidence, "primary_tf": primary_tf,
         "regime": "bull", "risk": "low", "entry_precision": "x", "size_adjustment": 1.0}
    if fallback:
        d["_fallback"] = True
    return d


class TestFusion:
    def test_no_quant_signal_holds(self):
        d = fuse(q(direction=0), llm())
        assert d.action == "hold" and d.direction == 0

    def test_llm_unavailable_is_passthrough(self):
        d = fuse(q(direction=1), None)
        assert d.action == "execute" and d.size_multiplier == 1.0 and d.direction == 1

    def test_llm_fallback_is_passthrough(self):
        d = fuse(q(direction=1), llm(fallback=True))
        assert d.action == "execute" and d.size_multiplier == 1.0

    def test_aligned_high_confidence_full_size(self):
        d = fuse(q(direction=1), llm(bias="long", confidence=0.85))
        assert d.action == "execute" and d.size_multiplier == 1.0 and d.direction == 1

    def test_aligned_mid_confidence_reduces(self):
        d = fuse(q(direction=1), llm(bias="long", confidence=0.6))
        assert d.action == "reduce" and d.size_multiplier == 0.5

    def test_aligned_low_confidence_holds(self):
        d = fuse(q(direction=1), llm(bias="long", confidence=0.4))
        assert d.action == "hold" and d.direction == 0

    def test_conflict_holds(self):
        d = fuse(q(direction=1), llm(bias="short", confidence=0.9))
        assert d.action == "hold" and d.direction == 0
        assert "conflict" in d.reason

    def test_llm_flat_reduces(self):
        d = fuse(q(direction=1), llm(bias="flat", confidence=0.9))
        assert d.action == "reduce" and d.size_multiplier == 0.5

    def test_tf_deferral_to_slower(self):
        d = fuse(q(direction=1, tf="5m"), llm(bias="long", confidence=0.85, primary_tf="1h"))
        assert d.action == "defer" and d.effective_tf == "1h"

    def test_no_deferral_when_llm_faster(self):
        d = fuse(q(direction=1, tf="1h"), llm(bias="long", confidence=0.85, primary_tf="5m"))
        assert d.action == "execute" and d.effective_tf == "1h"

    def test_short_alignment(self):
        d = fuse(q(direction=-1, regime="BEAR"), llm(bias="short", confidence=0.8))
        assert d.action == "execute" and d.direction == -1

    def test_confidence_boundary(self):
        # exactly at conf_full -> full size
        assert fuse(q(1), llm("long", 0.7)).size_multiplier == 0.5  # 0.7 not > 0.7
        assert fuse(q(1), llm("long", 0.71)).size_multiplier == 1.0
