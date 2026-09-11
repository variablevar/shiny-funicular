"""Tests for the Deribit options collector."""
import json
import re
import pytest
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, '.')

from collectors.deribit_options import (
    _filter_chain,
    _parse_instrument,
    compute_metrics,
)


class TestParseInstrument:
    def test_parses_valid_call(self):
        info = _parse_instrument('BTC-25SEP26-105000-C')
        assert info == {
            'ccy': 'BTC', 'expiry_str': '25SEP26',
            'strike': 105000.0, 'option_type': 'call',
        }

    def test_parses_valid_put(self):
        info = _parse_instrument('ETH-12MAR27-4500-P')
        assert info == {
            'ccy': 'ETH', 'expiry_str': '12MAR27',
            'strike': 4500.0, 'option_type': 'put',
        }

    def test_rejects_malformed(self):
        assert _parse_instrument('invalid') == {}
        assert _parse_instrument('') == {}
        assert _parse_instrument('BTC-25SEP26-105000-X') == {}


def _make_summary(strikes_ivs):
    """Build a synthetic book summary with given (strike, iv) pairs.
    Returns a list of dicts matching Deribit's get_book_summary_by_currency shape.
    """
    out = []
    for i, (strike, iv, opt_type) in enumerate(strikes_ivs):
        out.append({
            'instrument_name': f'BTC-30SEP26-{int(strike)}-{opt_type}',
            'strike': strike,
            'mark_iv': iv,
            'mark_price': 0.01,
            'bid_price': 0.001,
            'ask_price': 0.002,
            'underlying_price': 77700.0,
            'open_interest': 1.0,
            'volume': 0.0,
        })
    return out


class TestFilterChain:
    def test_filters_extreme_iv(self):
        """IV < 10% or > 200% should be filtered out (junk strikes)."""
        synth = _make_summary([
            (75000, 5, 'C'),   # too low (deep ITM)
            (77700, 50, 'C'),  # OK
            (77700, 250, 'C'), # too high
        ])
        chain = _filter_chain(synth, 77700)
        assert len(chain) == 1
        assert chain[0]['iv'] == 50

    def test_filters_zero_bid_ask(self):
        """No bid/ask = illiquid, should be filtered."""
        synth = _make_summary([(77700, 30, 'C')])
        # Make illiquid
        synth[0]['bid_price'] = 0
        synth[0]['ask_price'] = 0
        chain = _filter_chain(synth, 77700)
        assert len(chain) == 0

    def test_synthesizes_summary_with_parse(self):
        """End-to-end: synthetic summary -> parse name -> filter -> sanity check."""
        # Build summary where strike is in name but not in dict (Deribit style)
        synth = []
        for strike, iv in [(75000, 25), (77700, 30), (80000, 28)]:
            synth.append({
                'instrument_name': f'BTC-30SEP26-{strike}-C',
                'mark_iv': iv,
                'bid_price': 0.001, 'ask_price': 0.002,
                'underlying_price': 77700.0,
            })
        chain = _filter_chain(synth, 77700)
        strikes = [c['strike'] for c in chain]
        assert strikes == [75000.0, 77700.0, 80000.0]
        assert all(c['iv'] in (25, 30, 28) for c in chain)


class TestComputeMetrics:
    def test_metrics_summary_format(self):
        """compute_metrics returns a dict with expected keys."""
        # Mock the network calls
        with patch('collectors.deribit_options.get_book_summary') as mock_bs, \
             patch('collectors.deribit_options.get_instruments') as mock_inst, \
             patch('collectors.deribit_options.get_volatility_index') as mock_vol:
            mock_bs.return_value = _make_summary([
                (77700, 30, 'C'),
                (75000, 35, 'P'),
                (80000, 25, 'C'),
            ])
            mock_inst.return_value = [{
                'instrument_name': 'BTC-30SEP26-77700-C',
                'expiration_timestamp': int(1789000000 * 1000),
            }]
            mock_vol.return_value = [[int(1789063685000), 35, 35, 36, 37]]
            m = compute_metrics('BTC')
        assert 'atm_iv_pct' in m
        assert 'dvol_pct' in m
        assert 'skew_25d' in m
        assert 'term_ratio' in m
        assert m['currency'] == 'BTC'
        assert m['n_instruments'] == 3

    def test_empty_summary_returns_empty(self):
        with patch('collectors.deribit_options.get_book_summary') as mock_bs:
            mock_bs.return_value = []
            m = compute_metrics('BTC')
        assert m == {}
