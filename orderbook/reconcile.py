"""orderbook/reconcile.py — ON-BOOT RECOVERY (MASTER_PLAN.md §4, §7).

Runs FIRST on every boot, BEFORE any new order is placed. Deterministic; NO LLM,
NO MCP. The catastrophe-insurance recovery sequence:

  (i)   replay the event log to rebuild internal order/position state;
  (ii)  query the VENUE (paper ledger; live broker stub) for ACTUAL open orders
        + positions;
  (iii) diff internal vs venue:
          - venue has a position the ledger doesn't know  -> ADOPT it
          - venue has a working order the ledger doesn't know -> CANCEL it
            (orphan order; the program did not intend it / lost track of it)
          - emit ``RECONCILED`` events describing each action;
  (iv)  if an UNRECONCILABLE mismatch is found, set a halt and DO NOT resume;
        otherwise resume.

"Reconcile with broker first" + "orphan-order recovery on every startup" + the
"broker-vs-ledger reconciliation halt on any mismatch" (§2 catastrophe
protections) all live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.events import Event, EventType


@dataclass
class ReconResult:
    """Outcome of a reconciliation pass."""

    ok: bool
    resume: bool
    adopted_positions: list = field(default_factory=list)
    cancelled_orders: list = field(default_factory=list)
    mismatches: list = field(default_factory=list)
    halt_reason: str = ""


def _norm_side(side: str) -> str:
    s = (side or "").lower()
    if s in ("buy", "long"):
        return "buy"
    if s in ("sell", "short"):
        return "sell"
    return s


def reconcile(
    bus,
    orderbook,
    venue,
    source: str = "reconcile",
    qty_tolerance: float = 1e-6,
) -> ReconResult:
    """Run the on-boot recovery sequence. Returns a :class:`ReconResult`.

    Args:
        bus:        event bus (publish/replay). Replayed first to rebuild state.
        orderbook:  the :class:`orderbook.state_machine.OrderBook` to reconcile.
        venue:      a broker adapter exposing ``open_orders()`` / ``positions()``
                    and ``cancel(venue_order_id)`` (paper broker or live stub).
        qty_tolerance: position-qty mismatch tolerance for "consistent".

    The bus event log is the source of truth for INTENDED state; the venue is the
    source of truth for ACTUAL state. We adopt actual positions the ledger lost,
    cancel actual orders the ledger never intended, and HALT (no resume) on a
    contradiction we cannot safely auto-resolve (e.g. a known position whose
    venue qty/side disagrees with the ledger).
    """
    # (i) Replay the event log to rebuild internal state. The OrderBook is the
    # durable projection; replay re-emits to any subscribers (e.g. a breaker
    # rebuilding its halt state). Reconcile does not need to mutate the ledger
    # from replay because the ledger DB is itself durable, but replaying keeps
    # downstream subscribers consistent and confirms the log is readable.
    if hasattr(bus, "replay"):
        bus.replay()

    adopted: list = []
    cancelled: list = []
    mismatches: list = []

    # (ii) Query the venue for ACTUAL open orders + positions.
    venue_positions = list(venue.positions())
    venue_orders = list(venue.open_orders())

    # Internal (ledger) views.
    ledger_positions = {p.symbol: p for p in orderbook.open_positions()}
    ledger_open_orders = orderbook.open_orders()
    # Map ledger orders by the venue id they were routed to is not stored here;
    # we treat orphan detection on the VENUE side by client_order_id linkage.
    ledger_order_ids = {o.order_id for o in ledger_open_orders}

    # (iii-a) Positions: adopt venue positions the ledger doesn't know; flag a
    # hard mismatch when both know a symbol but disagree on qty/side.
    for vp in venue_positions:
        sym = vp["symbol"]
        lp = ledger_positions.get(sym)
        if lp is None:
            # Orphan/adopted position: the venue holds it; the ledger lost it.
            pid = orderbook.open_position(
                symbol=sym, side=_norm_side(vp["side"]), qty=float(vp["qty"]),
                avg_price=float(vp.get("avg_price", 0.0)),
            )
            orderbook.record_recon(
                kind="adopt_position",
                detail=f"{sym} {vp['side']} {vp['qty']}@{vp.get('avg_price')}",
                resolution="adopted",
            )
            adopted.append({"symbol": sym, "position_id": pid, **vp})
            _emit(bus, source, EventType.RECONCILED,
                  {"action": "adopt_position", "symbol": sym, "position_id": pid,
                   "qty": vp["qty"], "side": vp["side"]})
        else:
            same_side = _norm_side(lp.side) == _norm_side(vp["side"])
            same_qty = abs(float(lp.qty) - float(vp["qty"])) <= qty_tolerance
            if not (same_side and same_qty):
                mismatches.append(
                    {"symbol": sym, "ledger": {"side": lp.side, "qty": lp.qty},
                     "venue": {"side": vp["side"], "qty": vp["qty"]}}
                )
                orderbook.record_recon(
                    kind="position_mismatch",
                    detail=f"{sym} ledger {lp.side}/{lp.qty} vs venue {vp['side']}/{vp['qty']}",
                    resolution="HALT",
                )

    # (iii-b) Orders: cancel venue working orders the ledger never intended.
    for vo in venue_orders:
        client_id = vo.get("client_order_id")
        known = client_id is not None and client_id in ledger_order_ids
        if not known:
            venue.cancel(vo["venue_order_id"])
            orderbook.record_recon(
                kind="cancel_orphan_order",
                detail=f"{vo.get('symbol')} {vo.get('side')} {vo.get('qty')} "
                       f"venue_id={vo['venue_order_id']}",
                resolution="cancelled",
            )
            cancelled.append(vo)
            _emit(bus, source, EventType.RECONCILED,
                  {"action": "cancel_orphan_order",
                   "venue_order_id": vo["venue_order_id"],
                   "symbol": vo.get("symbol")})

    # (iv) Decide resume vs halt.
    if mismatches:
        reason = f"unreconcilable position mismatch(es): {mismatches}"
        orderbook.record_recon(kind="halt", detail=reason, resolution="HALT")
        _emit(bus, source, EventType.CIRCUIT_BREAKER_TRIPPED,
              {"reason": reason, "kind": "reconciliation_mismatch", "source": source})
        return ReconResult(
            ok=False, resume=False, adopted_positions=adopted,
            cancelled_orders=cancelled, mismatches=mismatches, halt_reason=reason,
        )

    # Consistent: record the clean snapshot and signal resume.
    orderbook.record_recon(
        kind="reconciled",
        detail=f"adopted={len(adopted)} cancelled={len(cancelled)} "
               f"positions={len(venue_positions)} orders={len(venue_orders)}",
        resolution="resume",
    )
    _emit(bus, source, EventType.RECONCILED,
          {"action": "complete", "resume": True,
           "adopted": len(adopted), "cancelled": len(cancelled)})
    return ReconResult(
        ok=True, resume=True, adopted_positions=adopted, cancelled_orders=cancelled,
    )


def _emit(bus, source: str, etype: EventType, data: dict) -> None:
    if bus is not None and hasattr(bus, "publish"):
        bus.publish(Event(type=etype, data=data, source=source))
