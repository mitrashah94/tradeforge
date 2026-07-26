"""orchestrator/watchdog.py — INDEPENDENT heartbeat monitor + dead-man's switch driver.

The §7 resilience watchdog. It runs as a SEPARATE PROCESS from the trading engine
(a launchd job, like ``scripts/cron/com.tradeforge.nightly.plist``) so it survives
an engine hang/crash and can act when the engine itself is the thing that failed.

Two jobs, both driven off INJECTED callables (so the same class is an independent
process in prod and a pure unit under test):

  1. HEARTBEAT WATCH. The engine publishes ``HEARTBEAT`` events to the durable
     ``events.duckdb`` log. The watchdog reads the latest HEARTBEAT timestamp via
     ``heartbeat_source()``. If ``now - last_heartbeat > heartbeat_timeout`` the
     engine is HUNG -> enter SAFE-MODE: publish ``SAFE_MODE`` + a
     ``CIRCUIT_BREAKER_TRIPPED`` (so the gateway refuses NEW orders) and notify.

  2. CONNECTIVITY WATCH -> DEAD-MAN'S SWITCH. ``connectivity()`` returns
     ``{"broker": bool, "data": bool}``. If a link is lost and stays lost beyond
     ``dms_timeout`` WHILE a position is open, the local bracket can no longer be
     managed, so the watchdog drives :func:`orchestrator.dead_mans_switch.dead_mans_switch`
     (cancel+flatten if broker reachable, else alert-and-halt).

PROD vs TEST seams (all injected on the ctor):
  - ``heartbeat_source() -> datetime | None``  (prod: :func:`heartbeat_from_events_db`)
  - ``connectivity() -> {"broker": bool, "data": bool}`` (prod: a liveness ping;
    paper: :func:`simulated_connectivity`)
  - ``clock() -> datetime`` (the ONLY time source; tests inject a fake)
  - ``positions_source() -> list`` / ``working_orders_source() -> list``
    (prod: the OrderBook; tests: fakes)
  - ``flatten_fn`` / ``cancel_fn`` / ``notify_fn`` (the DMS effectors)

Determinism: ``check()`` is one tick and does no IO of its own beyond the injected
sources; ``run()`` is the independent-process loop. NO LLM, NO MCP.

Run it standalone:  ``PYTHONPATH=. .venv/bin/python -m orchestrator.watchdog``
(see the ``__main__`` block + scripts/cron/com.tradeforge.watchdog.plist).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from orchestrator.dead_mans_switch import DmsAction, dead_mans_switch
from orchestrator.events import Event, EventType
from orchestrator.tools.notify import notify as _default_notify


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# Prod source helpers (kept here so __main__ can wire a real watchdog).        #
# --------------------------------------------------------------------------- #
def heartbeat_from_events_db(events_db: str = "events.duckdb"):
    """Return a ``heartbeat_source()`` reading the latest HEARTBEAT ts from the log.

    PROD heartbeat: the engine persists HEARTBEAT events to ``events.duckdb`` (the
    same durable bus log). The watchdog — a SEPARATE process — opens that DB
    read-only and queries the max ts of type 'HEARTBEAT'. Returns None when there
    is no heartbeat yet (treated as stale by ``check`` once the engine should have
    started). Imports duckdb lazily so importing this module stays cheap. The
    connection is opened ONCE and reused across ticks.
    """
    state: dict = {"con": None}

    def source() -> datetime | None:
        import duckdb  # lazy

        if state["con"] is None:
            state["con"] = duckdb.connect(events_db, read_only=True)
        row = state["con"].execute(
            "SELECT MAX(ts_utc) FROM events WHERE type = 'HEARTBEAT'"
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    return source


def heartbeat_from_connection(con):
    """Return a ``heartbeat_source()`` querying HEARTBEAT off an EXISTING connection.

    Used when the watchdog shares a single DuckDB connection with its event bus in
    the SAME process (DuckDB refuses a second connection to the same file with a
    different read/write config). The standalone process uses this so its bus
    (read-write, for publishing SAFE_MODE / CIRCUIT_BREAKER_TRIPPED) and its
    heartbeat read use ONE connection.
    """
    def source() -> datetime | None:
        row = con.execute(
            "SELECT MAX(ts_utc) FROM events WHERE type = 'HEARTBEAT'"
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    return source


def simulated_connectivity() -> dict:
    """Paper-mode connectivity ping: always-up (paper has no real broker link)."""
    return {"broker": True, "data": True}


@dataclass
class WatchdogState:
    """Latched watchdog state across ticks (the independent process's memory)."""

    safe_mode: bool = False
    safe_mode_reason: str = ""
    # When the current connectivity loss began (None => currently connected).
    lost_since: datetime | None = None
    dms_fired: bool = False
    last_action: DmsAction | None = None
    ticks: int = 0


class Watchdog:
    """Independent-process heartbeat + connectivity monitor with dead-man's switch.

    Public API:
      - ``Watchdog(...)`` — inject sources/effectors + timeouts + clock.
      - ``check() -> WatchdogState`` — one monitoring tick (the testable unit).
      - ``run(max_ticks=None)`` — the standalone polling loop (the prod process).
      - ``arm()`` / ``armed`` — satisfies ``OrderBook.assert_dead_mans_switch_armed``.
    """

    def __init__(
        self,
        *,
        bus=None,
        heartbeat_source: Callable[[], datetime | None],
        connectivity: Callable[[], dict],
        positions_source: Callable[[], list],
        working_orders_source: Callable[[], list] | None = None,
        flatten_fn: Callable[[object], dict] | None = None,
        cancel_fn: Callable[[object], dict] | None = None,
        limits=None,
        ri: int = 5,
        heartbeat_timeout: float = 15.0,
        dms_timeout: float = 30.0,
        poll_interval: float = 1.0,
        clock: Callable[[], datetime] | None = None,
        notify_fn: Callable[..., str] | None = None,
        source: str = "watchdog",
    ):
        self.bus = bus
        self.heartbeat_source = heartbeat_source
        self.connectivity = connectivity
        self.positions_source = positions_source
        self.working_orders_source = working_orders_source or (lambda: [])
        self.flatten_fn = flatten_fn or (lambda p: {"status": "noop"})
        self.cancel_fn = cancel_fn or (lambda o: {"status": "noop"})
        self.limits = limits
        self.ri = ri
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.dms_timeout = float(dms_timeout)
        self.poll_interval = float(poll_interval)
        self.clock = clock or _utcnow
        self.notify_fn = notify_fn or _default_notify
        self.source = source

        self.state = WatchdogState()
        self._armed = False

    # ------------------------------------------------------------------ #
    # arming — satisfies OrderBook.assert_dead_mans_switch_armed         #
    # ------------------------------------------------------------------ #
    def arm(self) -> "Watchdog":
        """Mark the dead-man's switch ARMED.

        The state machine refuses to run live with LOCAL OCO brackets unless the
        switch is armed (``OrderBook.assert_dead_mans_switch_armed(armed)``). Once
        this watchdog is constructed + arming, that requirement is satisfied — pass
        ``watchdog.armed`` into the assertion / into ``boot(..., dead_mans_switch_armed=...)``.
        """
        self._armed = True
        return self

    @property
    def armed(self) -> bool:
        return self._armed

    def assert_armed(self, orderbook) -> None:
        """Convenience: assert the orderbook's DMS requirement against THIS watchdog."""
        orderbook.assert_dead_mans_switch_armed(self._armed)

    # ------------------------------------------------------------------ #
    # one tick                                                           #
    # ------------------------------------------------------------------ #
    def check(self) -> WatchdogState:
        """Run ONE monitoring tick. Deterministic given the injected sources.

        Order of operations:
          1. heartbeat staleness -> SAFE-MODE (publish SAFE_MODE +
             CIRCUIT_BREAKER_TRIPPED, notify) if hung.
          2. connectivity -> track ``lost_since``; on sustained loss past
             ``dms_timeout`` with an open position, drive the dead-man's switch.
        Returns the (mutated) latched :class:`WatchdogState`.
        """
        self.state.ticks += 1
        now = self.clock()

        # ---- 1) HEARTBEAT WATCH -> SAFE MODE ----
        last_hb = self.heartbeat_source()
        hb_stale = last_hb is None or (now - last_hb).total_seconds() > self.heartbeat_timeout
        if hb_stale:
            self._enter_safe_mode(now, last_hb)
        # (We do NOT auto-clear safe-mode here: a recovered engine + a human
        #  acknowledging the alert is the deliberate re-arm path. Safe-mode is a
        #  latch, like the gateway's halt.)

        # ---- 2) CONNECTIVITY WATCH -> DEAD-MAN'S SWITCH ----
        conn = self.connectivity()
        link_lost = not conn.get("broker", False) or not conn.get("data", False)
        if link_lost:
            if self.state.lost_since is None:
                self.state.lost_since = now  # mark the start of the loss window
        else:
            self.state.lost_since = None  # link healthy -> reset the window

        positions = list(self.positions_source())
        if link_lost and positions:
            action = dead_mans_switch(
                open_positions=positions,
                working_orders=list(self.working_orders_source()),
                connectivity=conn,
                ri=self.ri,
                limits=self.limits,
                flatten_fn=self.flatten_fn,
                cancel_fn=self.cancel_fn,
                notify_fn=self.notify_fn,
                clock=self.clock,
                lost_since=self.state.lost_since,
                dms_timeout=self.dms_timeout,
                bus=self.bus,
                source=self.source,
            )
            self.state.last_action = action
            if action.fired:
                self.state.dms_fired = True

        return self.state

    def _enter_safe_mode(self, now: datetime, last_hb: datetime | None) -> None:
        """Latch SAFE-MODE: publish SAFE_MODE + CIRCUIT_BREAKER_TRIPPED, notify once."""
        age = "never" if last_hb is None else f"{(now - last_hb).total_seconds():.1f}s"
        reason = (
            f"ENGINE HUNG: last heartbeat {age} ago > timeout "
            f"{self.heartbeat_timeout:.1f}s -> SAFE MODE (gateway refusing new orders)"
        )
        already = self.state.safe_mode
        self.state.safe_mode = True
        self.state.safe_mode_reason = reason
        if already:
            return  # alert + publish only on the EDGE into safe-mode (idempotent latch)

        self.notify_fn(reason, title="WATCHDOG SAFE-MODE", channel="file")
        self._emit(EventType.SAFE_MODE, {
            "reason": reason, "last_heartbeat": str(last_hb),
            "heartbeat_timeout_s": self.heartbeat_timeout,
        })
        # Trip the gateway's halt so NO new orders are accepted while the engine
        # is suspect (the gateway latches on CIRCUIT_BREAKER_TRIPPED.data.reason).
        self._emit(EventType.CIRCUIT_BREAKER_TRIPPED, {
            "reason": reason, "kind": "watchdog_safe_mode", "source": self.source,
        })

    def _emit(self, etype: EventType, data: dict) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            self.bus.publish(Event(type=etype, data=data, source=self.source))

    # ------------------------------------------------------------------ #
    # the independent-process loop                                       #
    # ------------------------------------------------------------------ #
    def run(self, max_ticks: int | None = None, sleep: Callable[[float], None] | None = None) -> WatchdogState:
        """Poll :meth:`check` every ``poll_interval`` seconds (the prod process loop).

        Runs forever in production (``max_ticks=None``). ``max_ticks`` bounds it for
        tests/manual runs; ``sleep`` is injectable so tests don't actually wait.
        """
        sleeper = sleep or time.sleep
        n = 0
        while max_ticks is None or n < max_ticks:
            self.check()
            n += 1
            if max_ticks is not None and n >= max_ticks:
                break
            sleeper(self.poll_interval)
        return self.state


# --------------------------------------------------------------------------- #
# standalone entry point — the INDEPENDENT process                            #
# --------------------------------------------------------------------------- #
def build_prod_watchdog(
    *,
    events_db: str = "events.duckdb",
    orderbook_db: str = "orderbook/orderbook.duckdb",
    paper_ledger_db: str = "paper/ledger.duckdb",
    heartbeat_timeout: float = 15.0,
    dms_timeout: float = 30.0,
    poll_interval: float = 1.0,
):
    """Wire a real, armed watchdog reading the durable DBs (used by ``__main__``).

    The watchdog publishes onto the SAME ``events.duckdb`` the engine uses (so its
    SAFE_MODE / CIRCUIT_BREAKER_TRIPPED reach the gateway via the durable log) and
    reads positions/orders from the OrderBook + paper venue for the DMS effectors.
    Paper-only: connectivity is simulated and the venue is the paper ledger.
    """
    from orchestrator.bus import EventBus
    from orchestrator.dead_mans_switch import make_cancel_fn, make_flatten_fn
    from orchestrator.tools.brokers import PaperBroker
    from orderbook.state_machine import OrderBook
    from risk.config import load_limits

    bus = EventBus(db_path=events_db)
    orderbook = OrderBook(db_path=orderbook_db, bus=bus)
    venue = PaperBroker(db_path=paper_ledger_db)
    limits = load_limits()

    # Reuse the bus's own (read-write) connection for the heartbeat read: DuckDB
    # forbids a second connection to the same file with a different config in one
    # process. (In prod the watchdog is a separate process from the engine, so the
    # files are shared across processes, not connections within one.)
    heartbeat_source = (
        heartbeat_from_connection(bus._con)
        if events_db != ":memory:" and hasattr(bus, "_con")
        else heartbeat_from_events_db(events_db)
    )

    wd = Watchdog(
        bus=bus,
        heartbeat_source=heartbeat_source,
        connectivity=simulated_connectivity,
        positions_source=orderbook.open_positions,
        working_orders_source=orderbook.open_orders,
        flatten_fn=make_flatten_fn(orderbook, venue, _utcnow),
        cancel_fn=make_cancel_fn(venue),
        limits=limits,
        ri=limits.default_ri,
        heartbeat_timeout=heartbeat_timeout,
        dms_timeout=dms_timeout,
        poll_interval=poll_interval,
    ).arm()
    return wd


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="TradeForge watchdog + dead-man's switch (independent process)",
    )
    parser.add_argument("--events-db", default="events.duckdb")
    parser.add_argument("--orderbook-db", default="orderbook/orderbook.duckdb")
    parser.add_argument("--paper-ledger-db", default="paper/ledger.duckdb")
    parser.add_argument("--heartbeat-timeout", type=float, default=15.0)
    parser.add_argument("--dms-timeout", type=float, default=30.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="bound the loop (default: run forever)")
    args = parser.parse_args(argv)

    wd = build_prod_watchdog(
        events_db=args.events_db,
        orderbook_db=args.orderbook_db,
        paper_ledger_db=args.paper_ledger_db,
        heartbeat_timeout=args.heartbeat_timeout,
        dms_timeout=args.dms_timeout,
        poll_interval=args.poll_interval,
    )
    print(
        f"[watchdog] independent process up — heartbeat_timeout="
        f"{args.heartbeat_timeout}s dms_timeout={args.dms_timeout}s "
        f"poll={args.poll_interval}s; reading {args.events_db}. "
        "DMS armed; SAFE_MODE/CIRCUIT_BREAKER_TRIPPED published to the durable log."
    )
    wd.run(max_ticks=args.max_ticks)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
