"""tests/test_fast_loop_sizing.py — volatility-target sizing keeps $-risk constant.

MASTER_PLAN §1.B: position size scales inversely to volatility (ATR-derived stop
distance) so the $-risk per trade is CONSTANT across vol regimes. A low-vol
regime (tight stop) buys MORE shares; a high-vol regime (wide stop) buys FEWER —
but ``qty * stop_distance == $risk`` in both. Degenerate inputs skip the trade.
"""

import pytest

from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk
from orchestrator.fast_loop.sizing import (
    SizingResult,
    atr_stop_distance,
    vol_target_qty,
)


def test_vol_target_keeps_dollar_risk_constant_across_atr_regimes():
    limits = load_limits()
    equity = 10_000.0
    ri = 5  # 1.0% per trade -> $100 risk budget
    expected_risk = per_trade_dollar_risk(equity, ri, limits)
    assert expected_risk == pytest.approx(100.0)

    entry = 100.0
    # Low-vol regime: ATR small -> tight stop (distance 1.0/share).
    low_vol = vol_target_qty(
        equity=equity, ri=ri, limits=limits,
        entry_price=entry, stop_price=entry - atr_stop_distance(atr=0.5, atr_mult=2.0),
    )
    # High-vol regime: ATR large -> wide stop (distance 4.0/share).
    high_vol = vol_target_qty(
        equity=equity, ri=ri, limits=limits,
        entry_price=entry, stop_price=entry - atr_stop_distance(atr=2.0, atr_mult=2.0),
    )

    assert low_vol.stop_distance == pytest.approx(1.0)
    assert high_vol.stop_distance == pytest.approx(4.0)

    # Low vol -> larger qty; high vol -> smaller qty.
    assert low_vol.qty > high_vol.qty
    assert low_vol.qty == pytest.approx(100.0)
    assert high_vol.qty == pytest.approx(25.0)

    # The whole point: $-risk per trade is EQUAL across regimes.
    assert low_vol.qty * low_vol.stop_distance == pytest.approx(expected_risk)
    assert high_vol.qty * high_vol.stop_distance == pytest.approx(expected_risk)
    assert (low_vol.qty * low_vol.stop_distance) == pytest.approx(
        high_vol.qty * high_vol.stop_distance
    )


def test_vol_target_inverse_in_atr():
    """Doubling ATR (vol) should roughly halve the size at the same $-risk."""
    limits = load_limits()
    entry = 50.0
    a = vol_target_qty(
        equity=5000, ri=8, limits=limits,
        entry_price=entry, stop_price=entry - 1.0,
    )
    b = vol_target_qty(
        equity=5000, ri=8, limits=limits,
        entry_price=entry, stop_price=entry - 2.0,
    )
    assert b.qty == pytest.approx(a.qty / 2.0)
    assert a.qty * a.stop_distance == pytest.approx(b.qty * b.stop_distance)


def test_zero_stop_distance_skips():
    limits = load_limits()
    res = vol_target_qty(
        equity=10_000, ri=5, limits=limits, entry_price=100.0, stop_price=100.0
    )
    assert isinstance(res, SizingResult)
    assert res.skipped is True
    assert res.qty == 0.0
    assert res.reason == "zero_stop_distance"


def test_non_positive_equity_skips():
    limits = load_limits()
    res = vol_target_qty(
        equity=0.0, ri=5, limits=limits, entry_price=100.0, stop_price=99.0
    )
    assert res.skipped is True
    assert res.qty == 0.0


def test_whole_share_flooring_and_min_qty():
    limits = load_limits()
    # $10 risk budget (RI 5 on $1000) with a $7 stop distance -> 1.42 shares.
    res_frac = vol_target_qty(
        equity=1000, ri=5, limits=limits,
        entry_price=100.0, stop_price=93.0, allow_fractional=True,
    )
    assert res_frac.qty == pytest.approx(10.0 / 7.0)

    # Whole-share mode floors to 1 share.
    res_whole = vol_target_qty(
        equity=1000, ri=5, limits=limits,
        entry_price=100.0, stop_price=93.0, allow_fractional=False,
    )
    assert res_whole.qty == pytest.approx(1.0)

    # If a wide stop drops whole-share qty below 1, with min_qty=1 it skips.
    res_skip = vol_target_qty(
        equity=1000, ri=5, limits=limits,
        entry_price=100.0, stop_price=80.0, allow_fractional=False, min_qty=1.0,
    )
    assert res_skip.skipped is True
    assert res_skip.qty == 0.0


def test_atr_stop_distance_helper():
    assert atr_stop_distance(atr=2.0, atr_mult=3.0) == pytest.approx(6.0)
    assert atr_stop_distance(atr=0.0, atr_mult=3.0) == 0.0
    assert atr_stop_distance(atr=2.0, atr_mult=0.0) == 0.0
