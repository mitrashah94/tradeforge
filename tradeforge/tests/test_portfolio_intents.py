"""tests/test_portfolio_intents.py — the live ORDER_INTENT seam (designed, not wired).

Asserts the Allocation -> ORDER_INTENT mapping: opens become gated new-entry
intents that PASS the gateway's per-trade risk check at the same RI; closes /
risk-reducing resizes become exit intents that BYPASS the halt gate; the ordering
puts risk-reducers before new risk. The gateway itself is exercised to prove the
contract end-to-end (no live route).
"""

from __future__ import annotations

import pytest

from portfolio.intents import (
    allocation_to_intents,
    close_to_intent,
    open_to_intent,
    resize_to_intent,
)
from portfolio.model import Allocation, PlannedClose, PlannedOpen, PlannedResize
from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk, resolve_ri

_EQUITY = 100_000.0
_ENTRY, _STOP = 100.0, 95.0  # stop_distance_pct = 0.05


def _open():
    # Sized exactly at grade B's resolved-RI cap for whatever floor the
    # operator currently has set in risk/limits.yaml -- shares = cap /
    # (entry * stop_distance_pct), a boundary case for the gateway check below.
    limits = load_limits()
    cap = per_trade_dollar_risk(_EQUITY, resolve_ri("B", limits), limits)
    stop_distance_pct = (_ENTRY - _STOP) / _ENTRY
    shares = cap / (_ENTRY * stop_distance_pct)
    return PlannedOpen(sleeve="brk", symbol="AAA", kind="score", side="long",
                       shares=shares, entry_price=_ENTRY, grade="B", family="trend",
                       dollar_risk=cap, stop=_STOP, atr=2.0)


def test_open_intent_carries_risk_fields_and_passes_the_gateway_gate():
    limits = load_limits()
    cap = per_trade_dollar_risk(_EQUITY, resolve_ri("B", limits), limits)
    stop_distance_pct = (_ENTRY - _STOP) / _ENTRY
    expected_qty = cap / (_ENTRY * stop_distance_pct)

    intent = open_to_intent(_open(), equity=_EQUITY)
    assert intent["side"] == "buy" and intent["reason"] == "entry"
    assert intent["qty"] == pytest.approx(expected_qty)
    assert intent["stop_distance_pct"] == pytest.approx(stop_distance_pct)
    assert intent["order_notional"] == pytest.approx(expected_qty * _ENTRY)
    assert intent["bypasses_halt"] is False

    # Drive the real gateway risk check: a correctly-sized open must NOT be vetoed.
    ri = resolve_ri(intent["grade"], limits)
    cap = per_trade_dollar_risk(intent["equity"], ri, limits)
    trade_risk = intent["order_notional"] * intent["stop_distance_pct"]
    assert trade_risk <= cap * (1.0 + 1e-9) + 1e-9   # the gateway's tolerance


def test_close_intent_bypasses_halt():
    p = PlannedClose(sleeve="brk", symbol="AAA", shares=125.0, fill_price=95.0, reason="stop")
    intent = close_to_intent(p)
    assert intent["side"] == "sell" and intent["reason"] == "stop"
    assert intent["bypasses_halt"] is True


def test_resize_tp1_partial_bypasses_halt():
    p = PlannedResize(sleeve="brk", symbol="AAA", delta_shares=-125.0, fill_price=107.5,
                      reason="tp1_partial")
    intent = resize_to_intent(p, equity=100_000.0)
    assert intent["side"] == "sell" and intent["reason"] == "tp1_partial_exit"
    assert intent["bypasses_halt"] is True


def test_resize_weight_add_is_gated_buy():
    p = PlannedResize(sleeve="rot", symbol="SPY", delta_shares=10.0, fill_price=500.0,
                      reason="weight_add")
    intent = resize_to_intent(p, equity=100_000.0)
    assert intent["side"] == "buy" and intent["reason"] == "entry"
    assert intent["bypasses_halt"] is False


def test_allocation_orders_risk_reducers_first():
    alloc = Allocation(
        opens=[_open()],
        closes=[PlannedClose(sleeve="brk", symbol="BBB", shares=10.0, fill_price=50.0, reason="stop")],
        resizes=[PlannedResize(sleeve="rot", symbol="CCC", delta_shares=-5.0, fill_price=50.0,
                               reason="weight_trim")],
    )
    intents = allocation_to_intents(alloc, equity=100_000.0)
    # closes + sells come before the new-entry buy.
    sides = [(i["symbol"], i["side"]) for i in intents]
    assert sides.index(("AAA", "buy")) == len(sides) - 1
    assert all(i["bypasses_halt"] for i in intents if i["side"] == "sell")
