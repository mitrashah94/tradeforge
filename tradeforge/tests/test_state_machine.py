"""tests/test_state_machine.py — order FSM, persistence, OCO, slippage.

Deterministic, offline. Each test uses an in-memory OrderBook (db_path=":memory:")
so nothing touches disk and runs are repeatable.
"""

from __future__ import annotations

import pytest

from orderbook.state_machine import (
    IllegalTransition,
    OrderBook,
    OrderState,
    can_transition,
    validate_transition,
)


@pytest.fixture
def ob():
    book = OrderBook(db_path=":memory:")
    yield book
    book.close()


# --------------------------------------------------------------------------- #
# transition legality                                                         #
# --------------------------------------------------------------------------- #
def test_legal_transitions_succeed(ob):
    oid = ob.create_order("SPY", "buy", 10, intended_price=100.0)
    assert ob.get_state(oid) == OrderState.STAGED
    assert ob.transition(oid, OrderState.APPROVED) == OrderState.APPROVED
    assert ob.transition(oid, OrderState.SUBMITTED) == OrderState.SUBMITTED
    assert ob.transition(oid, OrderState.WORKING) == OrderState.WORKING
    assert ob.transition(oid, OrderState.FILLED) == OrderState.FILLED


def test_illegal_transition_raises(ob):
    oid = ob.create_order("SPY", "buy", 10)
    # STAGED -> WORKING is illegal (must pass through APPROVED/SUBMITTED).
    with pytest.raises(IllegalTransition):
        ob.transition(oid, OrderState.WORKING)


def test_terminal_state_has_no_exit(ob):
    oid = ob.create_order("SPY", "buy", 10)
    ob.transition(oid, OrderState.APPROVED)
    ob.transition(oid, OrderState.SUBMITTED)
    ob.transition(oid, OrderState.FILLED)
    with pytest.raises(IllegalTransition):
        ob.transition(oid, OrderState.CANCELLED)


def test_can_transition_table():
    assert can_transition(OrderState.STAGED, OrderState.APPROVED)
    assert can_transition(OrderState.WORKING, OrderState.FILLED)
    assert not can_transition(OrderState.STAGED, OrderState.WORKING)
    assert not can_transition(OrderState.FILLED, OrderState.WORKING)
    # validate_transition raises where can_transition is False.
    with pytest.raises(IllegalTransition):
        validate_transition(OrderState.FILLED, OrderState.WORKING)


# --------------------------------------------------------------------------- #
# full lifecycle persists                                                     #
# --------------------------------------------------------------------------- #
def test_full_staged_to_filled_persists(ob):
    oid = ob.create_order("AAPL", "buy", 5, intended_price=200.0, strategy="brk")
    for dst in (OrderState.APPROVED, OrderState.SUBMITTED, OrderState.WORKING):
        ob.transition(oid, dst)
    fill = ob.record_fill(oid, price=200.05)
    ob.transition(oid, OrderState.FILLED)

    order = ob.get_order(oid)
    assert order.state == OrderState.FILLED
    assert order.symbol == "AAPL"
    assert order.strategy == "brk"

    fills = ob.fills_for(oid)
    assert len(fills) == 1
    assert fills[0]["price"] == 200.05
    # slippage = price - intended
    assert fills[0]["slippage"] == pytest.approx(0.05)


def test_slippage_recorded_fill_minus_intended(ob):
    oid = ob.create_order("SPY", "sell", 10, intended_price=450.00)
    ob.transition(oid, OrderState.APPROVED)
    ob.transition(oid, OrderState.SUBMITTED)
    ob.transition(oid, OrderState.WORKING)
    fill = ob.record_fill(oid, price=449.90)
    assert fill["slippage"] == pytest.approx(449.90 - 450.00)
    assert fill["slippage"] == pytest.approx(-0.10)


# --------------------------------------------------------------------------- #
# OCO bracket: filling one protective leg cancels the sibling                  #
# --------------------------------------------------------------------------- #
def _stage_active_bracket(ob, symbol="SPY"):
    """Build an entry + stop + target bracket and bring it to ACTIVE."""
    entry = ob.create_order(symbol, "buy", 10, intended_price=100.0)
    stop = ob.create_order(symbol, "sell", 10, order_type="stop_market",
                           stop_price=99.0, intended_price=99.0)
    target = ob.create_order(symbol, "sell", 10, order_type="limit",
                             limit_price=102.0, intended_price=102.0)
    bid = ob.create_bracket(symbol, entry, stop, target)
    # Drive entry to FILLED and protective legs to WORKING.
    for dst in (OrderState.APPROVED, OrderState.SUBMITTED, OrderState.WORKING,
                OrderState.FILLED):
        ob.transition(entry, dst)
    ob.activate_bracket(bid)
    for leg in (stop, target):
        for dst in (OrderState.APPROVED, OrderState.SUBMITTED, OrderState.WORKING):
            ob.transition(leg, dst)
    return bid, entry, stop, target


def test_oco_stop_fill_cancels_target(ob):
    bid, entry, stop, target = _stage_active_bracket(ob)
    # Stop leg fills -> target leg must be CANCELLED.
    ob.record_fill(stop, price=99.0)
    ob.transition(stop, OrderState.FILLED)
    cancelled = ob.on_leg_fill(stop)
    assert target in cancelled
    assert ob.get_state(target) == OrderState.CANCELLED
    assert ob.get_state(stop) == OrderState.FILLED
    assert ob.get_bracket(bid)["state"] == "CLOSED"


def test_oco_target_fill_cancels_stop(ob):
    bid, entry, stop, target = _stage_active_bracket(ob)
    # Target leg fills -> stop leg must be CANCELLED.
    ob.record_fill(target, price=102.0)
    ob.transition(target, OrderState.FILLED)
    cancelled = ob.on_leg_fill(target)
    assert stop in cancelled
    assert ob.get_state(stop) == OrderState.CANCELLED
    assert ob.get_state(target) == OrderState.FILLED


# --------------------------------------------------------------------------- #
# positions: MFE/MAE + realized PnL                                          #
# --------------------------------------------------------------------------- #
def test_position_mfe_mae_and_realized_pnl(ob):
    pid = ob.open_position("SPY", "buy", 10, avg_price=100.0)
    ob.update_position_excursion(pid, mark_price=103.0)  # +3 * 10 = +30 favorable
    ob.update_position_excursion(pid, mark_price=98.0)   # -2 * 10 = -20 adverse
    ob.update_position_excursion(pid, mark_price=101.0)  # in between
    pos = ob.get_position(pid)
    assert pos.mfe == pytest.approx(30.0)
    assert pos.mae == pytest.approx(-20.0)

    booked = ob.close_position(pid, exit_price=102.0)
    assert booked["realized_pnl"] == pytest.approx(20.0)  # (102-100)*10
    assert ob.get_position(pid).state == "CLOSED"


def test_short_position_pnl_sign(ob):
    pid = ob.open_position("QQQ", "sell", 4, avg_price=400.0)
    ob.update_position_excursion(pid, mark_price=395.0)  # short profits on down
    pos = ob.get_position(pid)
    assert pos.mfe == pytest.approx(20.0)  # (400-395)*4
    booked = ob.close_position(pid, exit_price=410.0)
    assert booked["realized_pnl"] == pytest.approx(-40.0)  # short loses up


# --------------------------------------------------------------------------- #
# dead-man's switch flag                                                      #
# --------------------------------------------------------------------------- #
def test_dead_mans_switch_required_with_live_bracket(ob):
    _stage_active_bracket(ob)
    assert ob.requires_dead_mans_switch is True
    with pytest.raises(RuntimeError):
        ob.assert_dead_mans_switch_armed(armed=False)
    # Armed -> no raise.
    ob.assert_dead_mans_switch_armed(armed=True)
