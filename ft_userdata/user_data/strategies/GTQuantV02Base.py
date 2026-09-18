"""
GTQuantV02Base — Phase 2b A/B CONTROL arm.

Reproduces the pre-fix GTQuantV02 exactly (no short regime gate, shorts exit
on model_flip_up) under the dedicated backtest identifier
gtquant-phase2b-ab. Expected to reproduce the Phase 2b baseline
(110 trades, -11.52 USDT, Sharpe -2.43); a large deviation would mean the
retrained models diverged from the day14-1h-tune identifier and all arm
comparisons are void.
"""
from GTQuantV02 import GTQuantV02


class GTQuantV02Base(GTQuantV02):
    kmeans_short_regimes = None
    short_flip_exit = True
