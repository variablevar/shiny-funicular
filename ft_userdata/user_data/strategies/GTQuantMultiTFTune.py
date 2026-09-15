"""
GTQuantMultiTFTune — Day 14 tuning copy of GTQuantMultiTF.

Identical logic to the live strategy; only the entry/exit threshold
DecimalParameter ranges are widened (0 is allowed) and their defaults can be
overridden via environment variables GTQ_ENTRY_THR / GTQ_EXIT_THR so a
threshold grid can run without editing files per point:

    docker-compose run --rm --no-deps -e GTQ_ENTRY_THR=0.0002 freqtrade \
        backtesting --strategy GTQuantMultiTFTune --config user_data/config_tune.json ...

Use with config_tune.json (separate FreqAI identifier) so the live model dir
is never touched. Live strategy GTQuantMultiTF.py stays untouched.
"""
import os

from freqtrade.strategy import DecimalParameter

from GTQuantMultiTF import GTQuantMultiTF

_ENTRY_THR = float(os.environ.get("GTQ_ENTRY_THR", "0.0005"))
_EXIT_THR = float(os.environ.get("GTQ_EXIT_THR", "-0.0005"))


class GTQuantMultiTFTune(GTQuantMultiTF):
    """Day 14: threshold-tunable clone of GTQuantMultiTF."""

    entry_threshold = DecimalParameter(0.0, 0.005, default=_ENTRY_THR, decimals=6,
                                       space="buy", optimize=True, load=True)
    exit_threshold = DecimalParameter(-0.005, 0.0, default=_EXIT_THR, decimals=6,
                                      space="sell", optimize=True, load=True)
