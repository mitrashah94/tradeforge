"""tests/test_crash_recovery.py — P3 GATE: simulated mid-trade crash -> clean
orphan recovery via the REAL boot path (MASTER_PLAN.md §7; §8 P3 gate).

Scenario (the catastrophe-insurance path):
  1. Pre-crash, a live position is opened and its OCO bracket left WORKING — the
     lifecycle is recorded in the durable event log and the fill lands in the
     durable paper venue ledger.
  2. A venue-side ORPHAN order is planted that the program never intended (no
     matching client_order_id) — e.g. a stray working order from a crash window.
  3. CRASH: the in-memory engine objects (OrderBook / gateway / fast loop) are
     dropped. Only the durable event log + paper ledger survive.
  4. BOOT: a FRESH system is assembled on the SAME durable stores and
     :func:`orchestrator.main.boot` runs — which calls ``reconcile`` FIRST,
     BEFORE any new order. We assert it:
       - rebuilds state from the venue (the open position is ADOPTED),
       - CANCELS the orphan venue order,
       - emits RECONCILED events,
       - ends consistent and RESUMES (the fast loop starts).
  5. Separately: an UNRECONCILABLE qty mismatch -> ``resume is False`` -> the
     system stays HALTED and does NOT start trading.

Deterministic + offline: temp-file DuckDB for the durable stores (so they
survive the simulated crash), :memory: for the throwaway throwaway pieces.
"""

from __future__ import annotations

import pytest

from orchestrator.events import Event, EventType
from orchestrator.main import build_system, boot


@pytest.fixture
def durable(tmp_path):
    """Durable store paths that survive the simulated crash (file-backed)."""
    return {
        "events_db": str(tmp_path / "events.duckdb"),
        "paper_ledger_db": str(tmp_path / "ledger.duckdb"),
    }


# --------------------------------------------------------------------------- #
# GATE 1: mid-trade crash -> reconcile adopts the position, cancels the orphan,
#         emits RECONCILED, and the system resumes.
# --------------------------------------------------------------------------- #
def test_mid_trade_crash_recovers_cleanly_and_resumes(durable):
    # ---- Pre-crash: open a real position + WORKING bracket through the wired
    #      system, leaving durable events + a durable venue fill. ----
    pre = build_system(
        armed=[], equity=10_000.0,
        events_db=durable["events_db"],
        orderbook_db=":memory:",                 # the in-memory ledger is LOST on crash
        paper_ledger_db=durable["paper_ledger_db"],
    )
    boot(pre)  # reconcile (clean, nothing yet) -> resume
    assert pre.started is True

    pre.mark_price("SPY", 100.0)
    res = pre.gateway.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 10, "order_type": "market",
        "intended_price": 100.0, "ref_price": 100.0, "strategy": "brk",
        "reason": "entry_long",
        "bracket": {"stop_price": 99.0, "target_price": 102.0},
    }))
    assert res["filled"] is True
    # The venue now durably holds the SPY position; the bracket legs are WORKING.
    assert any(p["symbol"] == "SPY" for p in pre.paper_broker.positions())

    # Plant a venue-side ORPHAN order the program never intended (no client id).
    orphan_vid = pre.paper_broker._plant_order("IWM", "sell", 3, client_order_id=None)
    # And a venue-side ORPHAN POSITION the in-memory ledger will never have seen
    # (a fill that landed during the crash window).
    pre.paper_broker._plant_position("QQQ", "buy", 4, 380.0)

    # Snapshot the durable venue state, then simulate the CRASH: drop in-memory
    # engine state (the OrderBook/gateway/fast loop). The bus log + paper ledger
    # are durable; we close their connections so a fresh process can reopen them.
    pre.bus.close()
    pre.orderbook.close()
    pre.paper_broker.close()
    del pre

    # ---- BOOT a FRESH system on the SAME durable stores. reconcile runs FIRST. ----
    fresh = build_system(
        armed=[], equity=10_000.0,
        events_db=durable["events_db"],
        orderbook_db=":memory:",                 # fresh ledger: lost the SPY position
        paper_ledger_db=durable["paper_ledger_db"],
    )
    boot(fresh)

    result = fresh.recon_result
    # Reconcile ran and decided to resume.
    assert result.ok is True
    assert result.resume is True
    assert fresh.started is True            # the fast loop started (resumed)
    assert fresh.halted_on_boot is False
    assert fresh.gateway.halted is False

    # State rebuilt from the venue: BOTH the crash-lost SPY position and the
    # planted QQQ orphan position are ADOPTED into the fresh ledger.
    adopted_syms = {a["symbol"] for a in result.adopted_positions}
    assert {"SPY", "QQQ"}.issubset(adopted_syms)
    ledger_syms = {p.symbol for p in fresh.orderbook.open_positions()}
    assert {"SPY", "QQQ"}.issubset(ledger_syms)

    # The orphan order was CANCELLED on the venue.
    cancelled_vids = {c["venue_order_id"] for c in result.cancelled_orders}
    assert orphan_vid in cancelled_vids
    assert all(o["venue_order_id"] != orphan_vid
               for o in fresh.paper_broker.open_orders())

    # RECONCILED events were emitted on the shared bus (adopt + cancel + complete).
    recon = [e for e in fresh.bus.events() if e.type == EventType.RECONCILED]
    actions = {e.data.get("action") for e in recon}
    assert "adopt_position" in actions
    assert "cancel_orphan_order" in actions
    assert "complete" in actions

    # A clean recon snapshot with a resume resolution was persisted.
    snaps = fresh.orderbook.recon_snapshots()
    assert any(s["resolution"] == "resume" for s in snaps)

    # ---- Able to resume: a NEW entry now flows through cleanly (system live). ----
    fresh.mark_price("MSFT", 50.0)
    res2 = fresh.gateway.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "MSFT", "side": "buy", "qty": 2, "order_type": "market",
        "intended_price": 50.0, "ref_price": 50.0, "strategy": "brk",
        "reason": "entry_long",
    }))
    assert res2["filled"] is True

    fresh.close()


# --------------------------------------------------------------------------- #
# GATE 2: an unreconcilable mismatch -> resume=False -> stay HALTED, no trading.
# --------------------------------------------------------------------------- #
def test_unreconcilable_mismatch_halts_and_does_not_resume(durable):
    system = build_system(
        armed=[], equity=10_000.0,
        events_db=durable["events_db"],
        orderbook_db=":memory:",
        paper_ledger_db=durable["paper_ledger_db"],
    )
    # The fresh ledger "knows" a SPY long of 10 (e.g. rebuilt from the log) but the
    # venue reports a SPY long of 25 -> a contradiction reconcile cannot auto-fix.
    system.orderbook.open_position("SPY", "buy", 10, avg_price=100.0)
    system.paper_broker._plant_position("SPY", "buy", 25, 100.0)

    boot(system)
    result = system.recon_result

    # reconcile refused to resume.
    assert result.ok is False
    assert result.resume is False
    assert result.mismatches
    assert result.mismatches[0]["symbol"] == "SPY"

    # The boot path did NOT start trading and the gateway is HALTED.
    assert system.started is False
    assert system.halted_on_boot is True
    assert system.gateway.halted is True

    # A reconciliation-mismatch circuit breaker was tripped on the bus.
    cb = [e for e in system.bus.events()
          if e.type == EventType.CIRCUIT_BREAKER_TRIPPED]
    assert cb and cb[0].data.get("kind") == "reconciliation_mismatch"

    # While halted, a new entry intent is VETOED (cannot trade after a bad boot).
    system.mark_price("SPY", 100.0)
    res = system.gateway.on_intent(Event(EventType.ORDER_INTENT, {
        "symbol": "SPY", "side": "buy", "qty": 1, "order_type": "market",
        "ref_price": 100.0, "reason": "entry_long",
    }))
    assert res["approved"] is False
    assert "halted" in res["reason"]
    assert EventType.ORDER_VETOED in [e.type for e in system.bus.events()]

    system.close()
