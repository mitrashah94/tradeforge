"""Tests for equity-scaled per-trade sizing."""

import pytest

from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk


def test_sizing_scales_linearly_with_equity():
    limits = load_limits()
    # RI 5 -> 1.0% per trade.
    r1k = per_trade_dollar_risk(1000, 5, limits)
    r2k = per_trade_dollar_risk(2000, 5, limits)
    assert r1k == pytest.approx(10.0)
    assert r2k == pytest.approx(20.0)
    # Sizing recomputes off current equity, so doubling equity doubles $ risk.
    assert r2k == pytest.approx(2 * r1k)


def test_sizing_second_ri():
    limits = load_limits()
    # RI 8 -> 2.0% per trade; 2.0% of 10000 == 200.0.
    assert per_trade_dollar_risk(10000, 8, limits) == pytest.approx(200.0)
