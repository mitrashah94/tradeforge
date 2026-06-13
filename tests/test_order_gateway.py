"""tests/test_order_gateway.py — the ONE event path: intent -> approve -> fill.

Deterministic, offline. Verifies the gateway gates (halt, risk, live hook),
emits the right lifecycle events, drives the FSM + OCO brackets, and that the
live broker stub is gated and never reached by default.
"""

from __future__ import annotations

import pytest

from orchestrator.bus import EventBus
from orchestrator.events import Event, EventType
from orchestrator.tools.brokers import LiveBroker, PaperBroker
from orchestrator.tools.order_gateway import OrderGateway
from orderbook.state_machine import OrderBook, OrderState


@pytest.fixture
def rig(tmp_path):
    bus = EventBus(db_path=str(tmp_path / "events.duckdb"))
    ob = OrderBook(db_path=":memory:")
    paper = PaperBroker(db_path=":memory:")
    gw = OrderGateway(bus, ob, paper_broker=paper)
    gw.register()
    yield bus, ob, paper, gw
    bus.close(); ob.close(); paper.close()


def _types(bus):
    return [e.type for e in bus.events()]


def test_paper_intent_flows_to_filled(rig):
    bus, ob, paper, gw = rig
    paper.set_price("SPY", 100.0)
    res = gw.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "order_type": "market",
        "intended_price": 100.0, "ref_price": 100.0,
    }))
    assert res["filled"] is True
    assert ob.get_state(res["order_id"]) == OrderState.FILLED
    # The identical event path emitted approve -> submitted -> working -> filled.
    seq = _types(bus)
    for et in (EventType.ORDER_APPROVED, EventType.ORDER_SUBMITTED,
               EventType.ORDER_WORKING, EventType.ORDER_FILLED):
        assert et in seq
    # A position was opened.
    assert ob.position_for_symbol("SPY") is not None


def test_halt_vetoes_new_intents(rig):
    bus, ob, paper, gw = rig
    # A breaker trips -> gateway latches halted and vetoes.
    bus.publish(Event(EventType.CIRCUIT_BREAKER_TRIPPED, {"reason": "daily loss"}))
    assert gw.halted is True
    paper.set_price("SPY", 100.0)
    res = gw.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "ref_price": 100.0,
    }))
    assert res["approved"] is False
    assert "halted" in res["reason"]
    assert EventType.ORDER_VETOED in _types(bus)


def test_risk_cap_vetoes_oversized_trade(rig):
    bus, ob, paper, gw = rig
    paper.set_price("SPY", 100.0)
    # equity 1000, RI floor 5 -> per-trade cap 1% = $10. A 50% stop on a
    # $1000 notional risks $500 -> must veto.
    res = gw.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "ref_price": 100.0,
        "intended_price": 100.0, "equity": 1000.0, "order_notional": 1000.0,
        "stop_distance_pct": 0.5, "grade": "B",
    }))
    assert res["approved"] is False
    assert "per-trade risk" in res["reason"]


def test_risk_cap_allows_within_limit(rig):
    bus, ob, paper, gw = rig
    paper.set_price("SPY", 100.0)
    # Risk $5 on a $1000 equity (cap $10 at RI5) -> allowed.
    res = gw.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "ref_price": 100.0,
        "intended_price": 100.0, "equity": 1000.0, "order_notional": 1000.0,
        "stop_distance_pct": 0.005, "grade": "B",
    }))
    assert res["approved"] is True


def test_oco_bracket_via_gateway_then_stop_cancels_target(rig):
    bus, ob, paper, gw = rig
    paper.set_price("SPY", 100.0)
    res = gw.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "order_type": "market",
        "intended_price": 100.0, "ref_price": 100.0,
        "bracket": {"stop_price": 99.0, "target_price": 102.0},
    }))
    bid = res["bracket_id"]
    assert bid is not None
    bracket = ob.get_bracket(bid)
    # Protective legs are WORKING after the entry filled + bracket activated.
    assert ob.get_state(bracket["stop_id"]) == OrderState.WORKING
    assert ob.get_state(bracket["target_id"]) == OrderState.WORKING

    # Stop triggers -> sibling target cancelled, position closed.
    out = gw.fill_bracket_leg(bracket["stop_id"], price=99.0)
    assert bracket["target_id"] in out["cancelled"]
    assert ob.get_state(bracket["target_id"]) == OrderState.CANCELLED
    assert EventType.ORDER_CANCELLED in _types(bus)
    assert EventType.POSITION_CLOSED in _types(bus)


def test_live_route_blocked_by_hook_without_gates(rig):
    bus, ob, paper, gw = rig
    gw.live_broker = LiveBroker(enabled=False)  # gated stub
    res = gw.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 1, "ref_price": 100.0,
        "route": "live",  # missing LIVE gates -> hook blocks
    }))
    assert res["approved"] is False
    assert "live hook blocked" in res["reason"]
    # The live stub was never invoked (no NotImplementedError surfaced).
    assert EventType.ORDER_VETOED in _types(bus)


def test_live_broker_stub_raises_when_reached(rig):
    # Even if reached, the stub refuses unless explicitly enabled.
    lb = LiveBroker(enabled=False)
    from orderbook.state_machine import Order
    o = Order(order_id="x", symbol="SPY", side="buy", qty=1, order_type="market",
              state="WORKING")
    with pytest.raises(NotImplementedError):
        lb.submit(o)
