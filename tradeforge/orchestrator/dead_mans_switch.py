"""orchestrator/dead_mans_switch.py — the MANDATORY dead-man's switch (MASTER_PLAN §3, §7).

Brackets in TradeForge are LOCAL (Robinhood exposes no native server-side OCO —
CLAUDE.md P0 decision #3). A local stop/target only protects while THIS engine is
alive AND connected to the broker/data feed. If the broker/data connection is lost
while a position is open, the local protective leg can no longer be enforced — an
adverse move could run unbounded. That is the exact scenario the dead-man's switch
exists for, and it is why §3 makes it MANDATORY whenever brackets are local.

Decision (deterministic, per the current RI policy):

  - broker REACHABLE  -> cancel working orders, then FLATTEN every open position
    via a market close (``cancel_fn`` then ``flatten_fn``). The local bracket can
    no longer be trusted, so we exit to flat rather than ride an unmanaged stop.
  - broker UNREACHABLE -> we CANNOT confirm a flatten, so we DO NOT claim one
    (no phantom flatten). Instead ALERT-AND-HALT: publish a hard
    CIRCUIT_BREAKER_TRIPPED so the gateway refuses everything, notify a human,
    and leave the position for manual recovery. Pretending to flatten when the
    broker is down would corrupt the ledger and hide real risk.

The decision + action MUST complete within ``dms_timeout`` (driven by the injected
clock in tests). Pure, deterministic Python — NO LLM, NO MCP. The watchdog
(orchestrator/watchdog.py) is the independent process that detects the loss and
calls this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from orchestrator.events import Event, EventType

# The two deterministic outcomes.
ACTION_FLATTEN = "flatten"
ACTION_ALERT_AND_HALT = "alert_and_halt"
ACTION_NOOP = "noop"


@dataclass
class DmsAction:
    """Outcome of a dead-man's-switch evaluation.

    Attributes:
        fired:            True if the switch triggered (loss >= timeout with an
                          open position). False on a no-op (healthy / no position
                          / loss not yet past the timeout).
        action:           one of ACTION_FLATTEN / ACTION_ALERT_AND_HALT / ACTION_NOOP.
        flattened:        symbols actually flattened (only on a CONFIRMED flatten).
        cancelled_orders: working-order ids cancelled before the flatten.
        halted:           True when we alert-and-halt (broker unreachable).
        reason:           human-readable explanation.
        within_timeout:   True if the decision+action completed within dms_timeout.
        elapsed_s:        seconds the decision+action took (clock-driven).
    """

    fired: bool
    action: str = ACTION_NOOP
    flattened: list = field(default_factory=list)
    cancelled_orders: list = field(default_factory=list)
    halted: bool = False
    reason: str = ""
    within_timeout: bool = True
    elapsed_s: float = 0.0


def _both_links_down(connectivity: dict) -> bool:
    """Broker OR data link down counts as "connection lost" for the switch.

    Either link going dark means we can no longer manage the local bracket
    reliably, so the switch arms on the loss of EITHER (conservative).
    """
    return not connectivity.get("broker", False) or not connectivity.get("data", False)


def _broker_reachable(connectivity: dict) -> bool:
    """Can we still reach the broker to confirm a flatten?"""
    return bool(connectivity.get("broker", False))


def dead_mans_switch(
    *,
    open_positions: list,
    working_orders: list,
    connectivity: dict,
    ri: int,
    limits,
    flatten_fn: Callable[[object], dict],
    cancel_fn: Callable[[object], dict],
    notify_fn: Callable[..., str],
    clock: Callable[[], datetime],
    lost_since: datetime | None,
    dms_timeout: float = 30.0,
    bus=None,
    source: str = "dead_mans_switch",
) -> DmsAction:
    """Evaluate (and, if armed, EXECUTE) the dead-man's switch.

    Fires only when ALL of: a connection (broker or data) is lost, a position is
    open, AND the loss has lasted ``>= dms_timeout`` (computed from ``lost_since``
    against the injected ``clock``). Otherwise it is a deterministic no-op.

    Args:
        open_positions:  current open positions (objects with ``.symbol`` etc).
        working_orders:  current working orders (objects with ``.order_id``); these
                         are the now-orphaned local protective legs to cancel.
        connectivity:    ``{"broker": bool, "data": bool}`` liveness snapshot.
        ri:              current risk index (the policy dial); flatten-vs-halt is
                         catastrophe insurance and is NOT on the dial, but ri/limits
                         are threaded through for policy-documentation + future use.
        limits:          the validated ``risk.config.Limits`` (single source of truth).
        flatten_fn:      ``flatten_fn(position) -> dict`` market-closes ONE position.
                         MUST raise on failure so an unconfirmed flatten is never
                         silently claimed.
        cancel_fn:       ``cancel_fn(order) -> dict`` cancels ONE working order.
        notify_fn:       the NOTIFY sink (``orchestrator.tools.notify.notify``).
        clock:           injected ``() -> datetime`` (UTC). The ONLY time source.
        lost_since:      when the connection loss began (None => not lost => no-op).
        dms_timeout:     seconds of sustained loss before the switch fires.
        bus:             optional event bus to publish DEAD_MANS_SWITCH_TRIPPED /
                         CIRCUIT_BREAKER_TRIPPED onto (publish/duck-typed).
        source:          producer tag for emitted events.

    Returns:
        A :class:`DmsAction` describing what happened.
    """
    start = clock()

    # ---- Guard rails: only fire on a sustained loss WITH an open position. ----
    if not open_positions:
        return DmsAction(fired=False, action=ACTION_NOOP,
                         reason="no open position — nothing to protect")
    if not _both_links_down(connectivity):
        return DmsAction(fired=False, action=ACTION_NOOP,
                         reason="connectivity healthy — no trigger")
    if lost_since is None:
        return DmsAction(fired=False, action=ACTION_NOOP,
                         reason="loss start unknown — not yet armed")

    elapsed_loss = (start - lost_since).total_seconds()
    if elapsed_loss < dms_timeout:
        return DmsAction(
            fired=False, action=ACTION_NOOP,
            reason=f"loss {elapsed_loss:.1f}s < dms_timeout {dms_timeout:.1f}s",
        )

    # ---- ARMED: connection lost past the timeout with an open position. ----
    if _broker_reachable(connectivity):
        result = _flatten(
            open_positions=open_positions, working_orders=working_orders,
            flatten_fn=flatten_fn, cancel_fn=cancel_fn, notify_fn=notify_fn,
            ri=ri, bus=bus, source=source,
        )
    else:
        result = _alert_and_halt(
            open_positions=open_positions, notify_fn=notify_fn, ri=ri,
            bus=bus, source=source,
        )

    end = clock()
    result.elapsed_s = (end - start).total_seconds()
    result.within_timeout = result.elapsed_s <= dms_timeout
    return result


def _flatten(
    *, open_positions, working_orders, flatten_fn, cancel_fn, notify_fn, ri,
    bus, source,
) -> DmsAction:
    """Broker reachable: cancel working orders then market-close every position.

    Cancels FIRST (so the now-orphaned local stop/target legs can't fire against
    the flatten), then flattens. A flatten is only recorded once ``flatten_fn``
    returns without raising — if it raises, we fall back to alert-and-halt (we
    will not claim a flatten we could not confirm).
    """
    cancelled: list = []
    for o in list(working_orders):
        try:
            cancel_fn(o)
            cancelled.append(getattr(o, "order_id", o))
        except Exception:  # noqa: BLE001 — best-effort cancel; flatten still proceeds
            pass

    flattened: list = []
    try:
        for p in list(open_positions):
            flatten_fn(p)
            flattened.append(getattr(p, "symbol", p))
    except Exception as exc:  # noqa: BLE001 — flatten not confirmed -> halt instead
        # We could not confirm the flatten — do NOT claim it. Degrade to the
        # alert-and-halt path so a human takes over, with the partial work noted.
        res = _alert_and_halt(
            open_positions=open_positions, notify_fn=notify_fn, ri=ri,
            bus=bus, source=source,
            extra_reason=f"flatten failed mid-way ({exc}); cancelled={cancelled}",
        )
        res.cancelled_orders = cancelled
        res.flattened = flattened  # whatever DID confirm before the failure
        return res

    reason = (
        f"DMS FLATTEN: broker reachable, connection lost with open position(s) "
        f"-> cancelled {len(cancelled)} order(s) + flattened {flattened} (RI={ri})"
    )
    notify_fn(reason, title="DEAD-MAN'S SWITCH", channel="file")
    _emit(bus, source, EventType.DEAD_MANS_SWITCH_TRIPPED, {
        "action": ACTION_FLATTEN, "flattened": flattened,
        "cancelled_orders": cancelled, "ri": ri, "reason": reason,
    })
    return DmsAction(
        fired=True, action=ACTION_FLATTEN, flattened=flattened,
        cancelled_orders=cancelled, halted=False, reason=reason,
    )


def _alert_and_halt(
    *, open_positions, notify_fn, ri, bus, source, extra_reason: str = "",
) -> DmsAction:
    """Broker unreachable: ALERT-AND-HALT. No phantom flatten.

    We cannot confirm a close while the broker is down, so we publish a HARD halt
    (CIRCUIT_BREAKER_TRIPPED — latches the gateway) plus DEAD_MANS_SWITCH_TRIPPED
    recording the alert_and_halt action, and notify a human. The open position is
    left intact in the ledger for manual recovery (truthful state).
    """
    symbols = [getattr(p, "symbol", p) for p in open_positions]
    reason = (
        f"DMS ALERT-AND-HALT: broker UNREACHABLE with open position(s) {symbols} "
        f"-> cannot confirm flatten; HARD HALT + manual recovery required (RI={ri})"
    )
    if extra_reason:
        reason = f"{reason} | {extra_reason}"
    notify_fn(reason, title="DEAD-MAN'S SWITCH", channel="file")
    # Hard halt the gateway (refuse all new orders) — same shape the gateway latches on.
    _emit(bus, source, EventType.CIRCUIT_BREAKER_TRIPPED, {
        "reason": reason, "kind": "dead_mans_switch_halt", "source": source,
    })
    _emit(bus, source, EventType.DEAD_MANS_SWITCH_TRIPPED, {
        "action": ACTION_ALERT_AND_HALT, "open_symbols": symbols, "ri": ri,
        "reason": reason,
    })
    return DmsAction(
        fired=True, action=ACTION_ALERT_AND_HALT, flattened=[], halted=True,
        reason=reason,
    )


def _emit(bus, source: str, etype: EventType, data: dict) -> None:
    if bus is not None and hasattr(bus, "publish"):
        bus.publish(Event(type=etype, data=data, source=source))


def make_flatten_fn(orderbook, venue, clock: Callable[[], datetime]):
    """Build a production ``flatten_fn(position) -> dict`` from an OrderBook + venue.

    Market-closes the position on the venue and books the close in the ledger. The
    venue submit MUST succeed (return a filled/working result) or this raises so
    the caller never records an unconfirmed flatten. Kept here so the watchdog and
    boot can wire a real closer without re-deriving the close mechanics.
    """
    from orderbook.state_machine import Order, OrderState  # local import

    def flatten_fn(position) -> dict:
        close_side = "sell" if str(position.side).lower() in ("buy", "long") else "buy"
        ref = None
        # Best-effort current mark for the close fill price (paper venue uses it).
        if hasattr(venue, "_prices"):
            ref = venue._prices.get(position.symbol)
        close = Order(
            order_id=f"dms_close_{position.position_id}",
            symbol=position.symbol, side=close_side, qty=position.qty,
            order_type="market", state=OrderState.SUBMITTED,
            intended_price=ref, route=getattr(venue, "route", "paper"),
        )
        res = venue.submit(close, ref_price=ref)
        if res.get("status") not in ("filled", "working", "partial"):
            raise RuntimeError(f"flatten rejected by venue: {res}")
        fill_price = res.get("fill_price")
        if fill_price is not None:
            orderbook.close_position(position.position_id, exit_price=fill_price)
        return res

    return flatten_fn


def make_cancel_fn(venue):
    """Build a production ``cancel_fn(order) -> dict`` that cancels on the venue.

    Maps a ledger order to its venue working order by client_order_id and cancels
    it. Best-effort (cancel is allowed to be a no-op if the order is already gone).
    """
    def cancel_fn(order) -> dict:
        oid = getattr(order, "order_id", order)
        for vo in venue.open_orders():
            if vo.get("client_order_id") == oid:
                return venue.cancel(vo["venue_order_id"])
        return {"status": "not_found", "client_order_id": oid}

    return cancel_fn
