"""tests/test_integration_f1_f2.py — P3 GATE: F1 (signal->exec) + F2 (lifecycle)
end-to-end through the REAL bus + gateway + paper broker + fast loop, then a
REPLAY proof of determinism (MASTER_PLAN.md §4 flows F1/F2; §8 P3 gate).

Unlike the unit tests (which drive each piece against a fake bus), this exercises
the FULLY WIRED system assembled by :func:`orchestrator.main.build_system` /
:func:`orchestrator.main.boot`:

  F1  a synthetic bar triggers an armed entry -> ORDER_INTENT -> ORDER_APPROVED ->
      ORDER_SUBMITTED -> ORDER_WORKING -> paper ORDER_FILLED, with the OCO bracket
      armed and a position opened in the shared OrderBook.
  F2  price runs to +1R -> TP1 partial (ORDER_PARTIAL, position reduced) + stop
      ratcheted to BREAKEVEN; runner trails; the trail stop is hit -> exit ->
      POSITION_CLOSED, with the expected realized PnL and a final FLAT state.

  REPLAY  the persisted event log is replayed into a FRESH OrderBook (a pure
      projection over ORDER_FILLED / ORDER_PARTIAL / POSITION_CLOSED) and the
      rebuilt end state (realized PnL, flat) MATCHES the live run — proving the
      ONE event path is a complete, deterministic, replayable record.

Deterministic + offline: in-memory DuckDB, injected clock, synthetic bars. No
LLM, no MCP, no network anywhere in the path.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from backtest.engine.engine import PartialPlan, Strategy
from orchestrator.events import Event, EventType
from orchestrator.fast_loop import ArmedStrategy, LifecycleState
from orchestrator.main import build_system, boot


# --------------------------------------------------------------------------- #
# A deterministic synthetic strategy: arm a long once with a partial+runner plan.
# (Mirrors the lifecycle unit test's ArmOnceLong so the trigger is precise.)
# --------------------------------------------------------------------------- #
class ArmOnceLong(Strategy):
    def __init__(self, *, fire_at, stop, tp1, fraction=0.5):
        self.fire_at = fire_at
        self.stop = stop
        self.tp1 = tp1
        self.fraction = fraction
        self._fired = False

    def on_session_start(self, ctx):
        self._fired = False

    def on_bar(self, ctx):
        if ctx.position is not None or self._fired:
            return
        if ctx.bar_index >= self.fire_at:
            plan = PartialPlan(tp1=self.tp1, tp1_r=1.0, fraction=self.fraction,
                               trail_mode="prior_bar")
            ctx.enter_long(stop=self.stop, target=None, partial=plan)
            self._fired = True


def _bar(o, h, l, c, *, is_eod=False, ts=0, symbol="QQQ"):
    return {"symbol": symbol, "ts": ts, "open": o, "high": h, "low": l,
            "close": c, "volume": 0.0, "is_eod": is_eod}


def _build(events_db=":memory:"):
    """A fully wired system on a SINGLE bus + OrderBook, armed long.

    ``events_db`` defaults to in-memory; pass a temp FILE path when the test must
    reopen the durable log for the replay proof (a second ``:memory:`` connection
    is a distinct empty DB and would not see the persisted events).
    """
    strat = ArmOnceLong(fire_at=0, stop=99.0, tp1=101.0, fraction=0.5)
    armed = ArmedStrategy(name="be", strategy=strat, symbol="QQQ", tp1_fraction=0.5)
    system = build_system(
        armed=[armed],
        equity=10_000.0,
        ri=5,
        events_db=events_db,
        orderbook_db=":memory:",
        paper_ledger_db=":memory:",
        clock=lambda: datetime(2026, 6, 1, 14, 30),
        latency_budget_ms=50.0,
    )
    return system


def _types(bus):
    return [e.type for e in bus.events()]


# --------------------------------------------------------------------------- #
# F1 + F2 end-to-end through the real wired system.
# --------------------------------------------------------------------------- #
def test_f1_f2_end_to_end_through_real_bus(tmp_path):
    # File-backed event log so the REPLAY step can reopen the SAME durable log.
    system = _build(events_db=str(tmp_path / "events.duckdb"))
    boot(system)  # reconcile runs FIRST; clean resume -> fast loop started
    assert system.started is True
    assert system.recon_result.resume is True

    bus, ob = system.bus, system.orderbook

    # ---- F1: bar 0 triggers the armed entry through the full chain. ----
    system.feed_bar(_bar(100, 100, 100, 100, ts=0))

    # The ONE event path emitted the full F1 chain in order.
    seq = _types(bus)
    for et in (EventType.BAR, EventType.ORDER_INTENT, EventType.ORDER_APPROVED,
               EventType.ORDER_SUBMITTED, EventType.ORDER_WORKING,
               EventType.ORDER_FILLED):
        assert et in seq, f"missing {et} in F1 chain"

    # Position opened in the shared OrderBook with an ACTIVE OCO bracket.
    pos = ob.position_for_symbol("QQQ")
    assert pos is not None
    assert pos.qty == pytest.approx(100.0)        # $100 risk / $1 stop dist
    assert pos.avg_price == pytest.approx(100.0)
    bracket = ob.active_bracket_for_symbol("QQQ")
    assert bracket is not None and bracket["state"] == "ACTIVE"
    assert ob.get_state(bracket["stop_id"]) == "WORKING"
    assert ob.get_state(bracket["target_id"]) == "WORKING"
    # The fast loop's managed position is live (entry filled -> OPEN).
    assert system.fast_loop._managed.state is LifecycleState.OPEN

    # ---- F2: bar 1 runs to +1R (101) -> TP1 partial + stop->breakeven. ----
    system.feed_bar(_bar(100, 101, 100, 100.5, ts=1))

    assert any(e.type == EventType.ORDER_PARTIAL for e in bus.events())
    assert any(e.type == EventType.TP1_HIT for e in bus.events())
    managed = system.fast_loop._managed
    assert managed.state is LifecycleState.RUNNER
    assert managed.stop_price == pytest.approx(100.0)   # moved to breakeven (entry)
    assert managed.qty == pytest.approx(50.0)           # runner = 50% of 100
    # The position was REDUCED (TP1 partial booked +$50 on 50 shares @ +$1).
    pos = ob.position_for_symbol("QQQ")
    assert pos.qty == pytest.approx(50.0)
    assert pos.realized_pnl == pytest.approx(50.0)

    # ---- F2: runner trails up on the next bars (prior-bar low). ----
    system.feed_bar(_bar(101, 103, 101, 102.5, ts=2))
    system.feed_bar(_bar(102, 104, 102, 103.5, ts=3))
    assert system.fast_loop._managed.stop_price == pytest.approx(101.0)  # trailed up

    # ---- F2: bar 4 drops through the trailed stop (101) -> exit -> CLOSED. ----
    system.feed_bar(_bar(103, 103, 100, 100.5, ts=4))

    assert any(e.type == EventType.STOP_HIT for e in bus.events())
    assert any(e.type == EventType.POSITION_CLOSED for e in bus.events())
    # Slot freed; system is flat.
    assert system.fast_loop._managed is None
    assert ob.open_positions() == []

    # ---- Expected realized outcome: partial +$50 (50sh @ +$1) + runner +$50
    #      (50sh exits at the trailed stop 101 vs entry 100). Total = +$100. ----
    closed = [e for e in bus.events() if e.type == EventType.POSITION_CLOSED]
    assert closed[-1].data["realized_pnl"] == pytest.approx(100.0)

    # Capture the live end state for the replay comparison.
    live_realized = closed[-1].data["realized_pnl"]
    live_open_positions = len(ob.open_positions())

    # ---- REPLAY: rebuild end state from the persisted log into a FRESH book. ----
    rebuilt = _replay_into_fresh_orderbook(bus)
    assert rebuilt["realized_pnl"] == pytest.approx(live_realized)   # SAME pnl
    assert rebuilt["realized_pnl"] == pytest.approx(100.0)
    assert rebuilt["open_positions"] == live_open_positions == 0     # SAME flat state
    assert rebuilt["symbol_flat"]["QQQ"] is True

    # Hot-path latency budget honoured (no LLM/MCP/network -> trivially fast).
    system.fast_loop.assert_within_budget()
    system.close()


# --------------------------------------------------------------------------- #
# Replay projection: rebuild order/position end state PURELY from the event log.
# This is the deterministic "one event path" proof — the durable log alone
# reconstructs the same end state, with no access to the live OrderBook object.
# --------------------------------------------------------------------------- #
def _replay_into_fresh_orderbook(source_bus) -> dict:
    """Replay the persisted log through a fresh projection and return end state.

    Subscribes a pure projector to a FRESH bus opened on the SAME durable db, then
    calls ``bus.replay()`` to re-emit every persisted event in seq order. The
    projector folds the order-lifecycle events into per-symbol net position + a
    running realized PnL — exactly the state the live OrderBook ends in.
    """
    from orchestrator.bus import EventBus

    # Reopen the SAME durable event log in a fresh bus (fresh in-memory state).
    fresh_bus = EventBus(db_path=source_bus.db_path)

    state = {"net_qty": {}, "avg": {}, "realized": 0.0}

    def _project(ev: Event) -> None:
        t = ev.type
        d = ev.data or {}
        sym = d.get("symbol")
        if t == EventType.ORDER_FILLED:
            # Entry fill (opening) -> establish/extend the net position.
            # Management/exit fills are reflected via ORDER_PARTIAL/POSITION_CLOSED
            # which carry the realized component, so we only OPEN here when this
            # is an entry (reason starts with "entry").
            if str(d.get("reason", "")).startswith("entry") and sym is not None:
                qty = float(d.get("qty", 0.0) or 0.0)
                px = float(d.get("price", d.get("fill_price", 0.0)) or 0.0)
                state["net_qty"][sym] = state["net_qty"].get(sym, 0.0) + qty
                state["avg"][sym] = px
        elif t == EventType.ORDER_PARTIAL:
            if sym is not None:
                qty = float(d.get("qty", 0.0) or 0.0)
                state["net_qty"][sym] = max(0.0, state["net_qty"].get(sym, 0.0) - qty)
                if d.get("realized_pnl") is not None:
                    state["realized"] += float(d["realized_pnl"])
        elif t == EventType.POSITION_CLOSED:
            if sym is not None:
                state["net_qty"][sym] = 0.0
            # POSITION_CLOSED carries the TOTAL realized (partial + runner); use it
            # as the authoritative final realized for the position.
            if d.get("realized_pnl") is not None:
                state["realized"] = float(d["realized_pnl"])

    fresh_bus.subscribe("*", _project)
    fresh_bus.replay()  # re-emit the persisted log in seq order
    fresh_bus.close()

    open_positions = sum(1 for q in state["net_qty"].values() if abs(q) > 1e-9)
    symbol_flat = {s: abs(q) <= 1e-9 for s, q in state["net_qty"].items()}
    return {
        "realized_pnl": state["realized"],
        "open_positions": open_positions,
        "symbol_flat": symbol_flat,
    }


# --------------------------------------------------------------------------- #
# Halt honoured end-to-end: a tripped breaker suppresses NEW entries through the
# wired system (the gateway honours CIRCUIT_BREAKER_TRIPPED; the loop stands down).
# --------------------------------------------------------------------------- #
def test_breaker_halt_suppresses_new_entry_end_to_end():
    system = _build()
    boot(system)
    bus = system.bus

    # A breaker trips on the shared bus -> gateway latches + loop stands down.
    bus.publish(Event(EventType.CIRCUIT_BREAKER_TRIPPED,
                      {"reason": "daily loss", "scope": "daily"}))
    assert system.gateway.halted is True
    assert system.fast_loop.standing_down is True

    # The armed strategy WOULD fire on this bar, but the entry is suppressed.
    system.feed_bar(_bar(100, 101, 100, 100, ts=0))
    intents = [e for e in bus.events() if e.type == EventType.ORDER_INTENT]
    assert intents == []
    assert system.orderbook.position_for_symbol("QQQ") is None
    system.close()
