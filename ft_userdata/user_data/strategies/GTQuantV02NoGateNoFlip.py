"""
GTQuantV02NoGateNoFlip — Phase 2b A/B arm B (exit fix in ISOLATION).

Identical to the pre-fix GTQuantV02 (no short regime gate) except shorts
ignore the model_flip_up exit: they ride ROI / ATR target / the -10.9%
catastrophe stop. Tests whether the -16.42 USDT of model_flip_up short
exits were salvageable without touching entries.
"""
from GTQuantV02 import GTQuantV02


class GTQuantV02NoGateNoFlip(GTQuantV02):
    kmeans_short_regimes = None
    short_flip_exit = False
