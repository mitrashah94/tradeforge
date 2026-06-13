"""tests/test_reconcile_recovery.py — on-boot crash recovery (§7).

Simulates a crash: events for an open position + a working order are written to
the log, the in-memory engine objects are dropped, then ``reconcile`` runs FIRST
on boot. It must:
  - replay the durable event log;
  - query the (paper) venue for ACTUAL open orders + positions;
  - ADOPT a planted orphan position the ledger lost;
  - CANCEL a planted orphan order the ledger never intended;
  - end consistent and resume.
And, separately, HALT (no resume) on an unreconcilable position mismatch.

Deterministic, offline. Bus uses a temp-file DuckDB; ledger/venue use :memory:.
"""

from __future__ import annotations

import pytest

from orchestrator.bus import EventBus
from orchestrator.events import Event, EventType
from orchestrator.tools.brokers import PaperBroker
from orchestrator.tools.order_gateway import OrderGateway
from orderbook.reconcile import reconcile
from orderbook.state_machine import OrderBook


@pytest.fixture
def bus_path(tmp_path):
    return str(tmp_path / "events.duckdb")


def test_crash_recovery_adopts_position_and_cancels_orphan(bus_path):
    # ---- Pre-crash: a real intent flows through the gateway, leaving durable
    #      events in the bus log and a filled position in the paper venue. ----
    bus = EventBus(db_path=bus_path)
    ob = OrderBook(db_path=":memory:")
    paper = PaperBroker(db_path=":memory:")
    paper.set_price("SPY", 100.0)
    gw = OrderGateway(bus, ob, paper_broker=paper)
    gw.register()

    bus.publish(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "order_type": "market",
        "intended_price": 100.0, "ref_price": 100.0, "strategy": "brk",
    }))
    # The venue now holds a SPY position and the event log records the lifecycle.
    assert any(p["symbol"] == "SPY" for p in paper.positions())

    # Plant a venue-side ORPHAN POSITION the ledger doesn't know about (e.g. a
    # fill that landed during the crash window) and a venue-side ORPHAN ORDER
    # the program never intended (no matching client_order_id).
    paper._plant_position("QQQ", "buy", 4, 380.0)
    orphan_vid = paper._plant_order("IWM", "sell", 3, client_order_id=None)

    bus.close()

    # ---- CRASH: drop in-memory engine state. The bus log + paper ledger are
    #      durable; the OrderBook/gateway objects are gone. ----
    del bus, ob, gw

    # ---- BOOT: reconcile runs FIRST, before any new orders. ----
    fresh_bus = EventBus(db_path=bus_path)  # reopen durable log
    fresh_ob = OrderBook(db_path=":memory:")  # fresh ledger (lost the SPY pos)

    result = reconcile(fresh_bus, fresh_ob, paper)

    assert result.ok is True
    assert result.resume is True

    # Adopted both venue positions the fresh ledger didn't know (SPY + QQQ).
    adopted_syms = {a["symbol"] for a in result.adopted_positions}
    assert "QQQ" in adopted_syms
    assert "SPY" in adopted_syms  # lost on crash, re-adopted from the venue
    ledger_syms = {p.symbol for p in fresh_ob.open_positions()}
    assert {"SPY", "QQQ"}.issubset(ledger_syms)

    # Cancelled the orphan order on the venue.
    cancelled_vids = {c["venue_order_id"] for c in result.cancelled_orders}
    assert orphan_vid in cancelled_vids
    assert all(o["venue_order_id"] != orphan_vid for o in paper.open_orders())

    # RECONCILED events were emitted on the bus.
    recon_events = [e for e in fresh_bus.events() if e.type == EventType.RECONCILED]
    assert any(e.data.get("action") == "adopt_position" for e in recon_events)
    assert any(e.data.get("action") == "cancel_orphan_order" for e in recon_events)
    assert any(e.data.get("action") == "complete" for e in recon_events)

    # A clean recon snapshot was persisted with a resume resolution.
    snaps = fresh_ob.recon_snapshots()
    assert any(s["resolution"] == "resume" for s in snaps)

    fresh_bus.close()
    paper.close()
    fresh_ob.close()


def test_unreconcilable_mismatch_halts_and_does_not_resume(bus_path):
    bus = EventBus(db_path=bus_path)
    ob = OrderBook(db_path=":memory:")
    paper = PaperBroker(db_path=":memory:")

    # Ledger knows a SPY long of 10; the venue reports a SPY long of 25 ->
    # an unreconcilable qty mismatch.
    ob.open_position("SPY", "buy", 10, avg_price=100.0)
    paper._plant_position("SPY", "buy", 25, 100.0)

    result = reconcile(bus, ob, paper)
    assert result.ok is False
    assert result.resume is False
    assert result.mismatches
    assert "SPY" == result.mismatches[0]["symbol"]

    # A circuit breaker was tripped on the bus (halt, do not resume).
    cb = [e for e in bus.events() if e.type == EventType.CIRCUIT_BREAKER_TRIPPED]
    assert cb and cb[0].data.get("kind") == "reconciliation_mismatch"

    bus.close()
    paper.close()
    ob.close()


def test_reconcile_runs_clean_with_no_orphans(bus_path):
    bus = EventBus(db_path=bus_path)
    ob = OrderBook(db_path=":memory:")
    paper = PaperBroker(db_path=":memory:")
    result = reconcile(bus, ob, paper)
    assert result.ok is True and result.resume is True
    assert result.adopted_positions == [] and result.cancelled_orders == []
    bus.close()
    paper.close()
    ob.close()
