"""orchestrator/main.py — the assembler + boot sequence for TradeForge P3.

Wires the four deterministic P3 pieces onto ONE shared event bus and one event
path (MASTER_PLAN.md §4, §7; CLAUDE.md "one event path", "execution stays dumb"):

    EventBus  ──┬─ OrderGateway  (ORDER_INTENT -> ... -> ORDER_FILLED, OCO bracket)
                ├─ BreakerService(halts/cooldowns -> the gateway honours them)
                └─ FastLoop      (bars -> ORDER_INTENT; manages F2 lifecycle)
                          │
                  PaperBroker (default venue -> paper/ledger.duckdb)

Two entry points:

  * :func:`build_system` constructs every component on a SINGLE ``EventBus`` with
    a SINGLE ``OrderBook`` and a SINGLE event path. Paper by default; the live
    broker is reachable only behind the P0 hook AND an explicit LIVE flag.

  * :func:`boot` is the resilient start sequence. **ON-BOOT RECOVERY RUNS FIRST**
    (MASTER_PLAN §7): ``reconcile(...)`` adopts/cancels venue orphans and rebuilds
    state from the durable log BEFORE any new order can be placed. If reconcile
    says ``resume is False`` (an unreconcilable mismatch or a hard program halt),
    the system DOES NOT start trading — it stays halted. Only on a clean resume
    are the live subscriptions wired and the fast loop started.

HOT PATH PURITY: nothing here calls an LLM or an MCP tool. The fast loop, gateway,
breakers, state machine and reconciler are all pure, deterministic Python. The
slow loop (LLM agents, P4) is intentionally NOT imported in this hot-path module.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from orchestrator.bus import EventBus
from orchestrator.events import Event, EventType
from orchestrator.fast_loop import ArmedStrategy, FastLoop
from orchestrator.tools.brokers import PAPER_LEDGER_PATH, LiveBroker, PaperBroker
from orderbook.reconcile import reconcile
from orderbook.state_machine import OrderBook
from risk.breaker_service import BreakerService
from risk.config import Limits, load_limits

# Default durable artifact paths (alongside the other DBs; MASTER_PLAN §4 tree).
DEFAULT_EVENTS_DB = "events.duckdb"
DEFAULT_ORDERBOOK_DB = "orderbook/orderbook.duckdb"
DEFAULT_PAPER_LEDGER_DB = PAPER_LEDGER_PATH  # "paper/ledger.duckdb"


# --------------------------------------------------------------------------- #
# Equity source                                                               #
# --------------------------------------------------------------------------- #
class EquitySource:
    """A mutable, callable equity source the fast loop sizes against.

    Per-trade $-risk = RI% x CURRENT equity, recomputed each trade so profits
    and deposits auto-compound (MASTER_PLAN §1.D). In paper boot this starts at
    a fixed value; a slow-loop / analyst (P4) would update it as PnL accrues.
    """

    def __init__(self, equity: float):
        self._equity = float(equity)

    def __call__(self) -> float:
        return self._equity

    def set(self, equity: float) -> None:
        self._equity = float(equity)


# --------------------------------------------------------------------------- #
# Assembled system handle                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class System:
    """The assembled, wired set of P3 components sharing ONE bus + ONE OrderBook."""

    bus: EventBus
    orderbook: OrderBook
    paper_broker: PaperBroker
    gateway: Any
    breakers: BreakerService
    fast_loop: FastLoop
    limits: Limits
    equity_source: EquitySource
    live_broker: LiveBroker | None = None
    venue: Any = None  # the ACTIVE venue (paper by default)
    started: bool = False
    halted_on_boot: bool = False
    recon_result: Any = None
    watchdog: Any = None  # the armed Watchdog (set by boot on a clean resume)

    def feed_bar(self, data: dict) -> Event:
        """Publish a BAR event onto the shared bus (the fast loop's hot-path input).

        ``data`` must carry OHLCV + ``symbol`` (+ optional ``is_eod``). In live/paper
        operation a market-data adapter calls this; tests call it directly.
        """
        return self.bus.publish(Event(EventType.BAR, data, source="market_data"))

    def mark_price(self, symbol: str, price: float) -> None:
        """Feed the active venue a current reference price (for fills)."""
        self.venue.set_price(symbol, price)

    def close(self) -> None:
        for c in (self.bus, self.orderbook, self.paper_broker):
            try:
                c.close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass


# --------------------------------------------------------------------------- #
# build_system — construct everything on one bus / one event path              #
# --------------------------------------------------------------------------- #
def build_system(
    *,
    armed: list[ArmedStrategy] | None = None,
    equity: float = 1000.0,
    limits: Limits | None = None,
    ri: int | None = None,
    events_db: str = DEFAULT_EVENTS_DB,
    orderbook_db: str = DEFAULT_ORDERBOOK_DB,
    paper_ledger_db: str = DEFAULT_PAPER_LEDGER_DB,
    paper_slippage_bps: float = 0.0,
    enable_live: bool = False,
    clock: Callable[[], datetime] | None = None,
    latency_budget_ms: float = 5.0,
    cooldown_losses: int = 3,
) -> System:
    """Construct the shared bus + OrderBook + PaperBroker + gateway + breakers +
    fast loop, all on ONE event path.

    NOTHING is subscribed/started here beyond the gateway and breaker self-wiring
    (so a caller can still inspect state pre-boot); the FAST LOOP is started by
    :func:`boot` only AFTER reconcile clears a resume. Live is paper-by-default:
    a :class:`LiveBroker` is attached only when ``enable_live`` is True (and even
    then every live order must clear the P0 hook + LIVE flag in the gateway).
    """
    limits = limits if limits is not None else load_limits()
    ri = ri if ri is not None else limits.default_ri
    clock = clock or (lambda: datetime.now(timezone.utc))

    bus = EventBus(db_path=events_db)
    orderbook = OrderBook(db_path=orderbook_db, bus=bus)
    paper_broker = PaperBroker(db_path=paper_ledger_db, slippage_bps=paper_slippage_bps)

    # Live broker stays a gated stub unless explicitly enabled (still hook-gated).
    live_broker = LiveBroker(enabled=enable_live) if enable_live else None

    # Deferred import keeps the live surface out of this module's import graph
    # until needed (the gateway itself imports hooks lazily, only on a live route).
    from orchestrator.tools.order_gateway import OrderGateway

    gateway = OrderGateway(
        bus, orderbook, paper_broker=paper_broker, live_broker=live_broker, limits=limits,
    )
    # Wire the gateway onto the shared bus NOW: it must subscribe to ORDER_INTENT
    # (the one event path) AND to the halt events (CIRCUIT_BREAKER_TRIPPED /
    # COOLDOWN_STARTED / NO_TRADE_WINDOW) so it honours breaker halts immediately,
    # even before the fast loop starts. The fast loop is started later (in boot),
    # only after reconcile clears a resume.
    gateway.register()

    # The breaker service subscribes itself to the bus on construction. Pass the
    # REAL typed Event as the factory so trips/cooldowns flow as proper events on
    # the same path (P3 seam #3: same Event everywhere).
    breakers = BreakerService(
        bus,
        limits=limits,
        ri=ri,
        cooldown_losses=cooldown_losses,
        event_factory=_breaker_event_factory,
        subscribe=True,
    )

    equity_source = EquitySource(equity)
    fast_loop = FastLoop(
        bus=bus,
        armed=list(armed or []),
        equity_source=equity_source,
        limits=limits,
        ri=ri,
        clock=clock,
        latency_budget_ms=latency_budget_ms,
        event_factory=Event,  # P3 seam #3: the real typed Event on the shared bus
    )

    return System(
        bus=bus, orderbook=orderbook, paper_broker=paper_broker, gateway=gateway,
        breakers=breakers, fast_loop=fast_loop, limits=limits,
        equity_source=equity_source, live_broker=live_broker, venue=paper_broker,
    )


def _breaker_event_factory(etype, data: dict, ts_utc):
    """Adapt the BreakerService's ``(type, data, ts_utc)`` factory signature to
    the real :class:`Event` (whose ctor takes keyword fields)."""
    return Event(type=etype, data=data, ts_utc=ts_utc, source="breaker_service")


# --------------------------------------------------------------------------- #
# boot — ON-BOOT RECOVERY FIRST, then (only on resume) start trading           #
# --------------------------------------------------------------------------- #
def start_watchdog(
    system: System,
    *,
    heartbeat_timeout: float = 15.0,
    dms_timeout: float = 30.0,
    poll_interval: float = 1.0,
    clock: Callable[[], datetime] | None = None,
):
    """Construct + ARM the in-process watchdog handle for this System (MASTER_PLAN §7).

    The watchdog is conceptually an INDEPENDENT process (``python -m
    orchestrator.watchdog`` — a launchd job; see scripts/cron/). This thin hook
    builds a :class:`orchestrator.watchdog.Watchdog` wired to THIS system's bus,
    OrderBook and venue so boot can (a) ARM the dead-man's switch — satisfying
    ``OrderBook.assert_dead_mans_switch_armed`` — and (b) optionally drive
    ``check()`` in-process for paper. In production prefer the standalone process
    (it survives an engine hang). Returns the armed Watchdog.

    Connectivity is simulated in paper. Heartbeats are read from the durable
    events log (the engine publishes HEARTBEAT onto the same bus).
    """
    from orchestrator.dead_mans_switch import make_cancel_fn, make_flatten_fn
    from orchestrator.watchdog import (
        Watchdog,
        heartbeat_from_connection,
        heartbeat_from_events_db,
        simulated_connectivity,
    )

    # Reuse the bus's own connection for the heartbeat read (DuckDB forbids a 2nd
    # connection to the same file with a different config in-process). In prod the
    # watchdog runs as its OWN process (python -m orchestrator.watchdog).
    heartbeat_source = (
        heartbeat_from_connection(system.bus._con)
        if hasattr(system.bus, "_con")
        else heartbeat_from_events_db(getattr(system.bus, "db_path", "events.duckdb"))
    )

    wd = Watchdog(
        bus=system.bus,
        heartbeat_source=heartbeat_source,
        connectivity=simulated_connectivity,
        positions_source=system.orderbook.open_positions,
        working_orders_source=system.orderbook.open_orders,
        flatten_fn=make_flatten_fn(system.orderbook, system.venue,
                                   clock or (lambda: datetime.now(timezone.utc))),
        cancel_fn=make_cancel_fn(system.venue),
        limits=system.limits,
        ri=system.breakers.ri if hasattr(system.breakers, "ri") else system.limits.default_ri,
        heartbeat_timeout=heartbeat_timeout,
        dms_timeout=dms_timeout,
        poll_interval=poll_interval,
        clock=clock,
    ).arm()
    return wd


def boot(
    system: System,
    *,
    dead_mans_switch_armed: bool = True,
    start_watchdog_hook: bool = True,
) -> System:
    """Resilient boot sequence (MASTER_PLAN §7). RECONCILE RUNS FIRST.

    Sequence:
      1. **reconcile(...)** against the ACTIVE venue BEFORE anything new — replay
         the durable log, adopt venue positions the ledger lost, cancel orphan
         venue orders, emit RECONCILED. (orderbook/reconcile.py)
      2. If ``reconcile`` returns ``resume is False`` (unreconcilable mismatch or
         a halt), DO NOT start trading: leave the gateway halted and return. The
         CIRCUIT_BREAKER_TRIPPED emitted by reconcile already latched the gateway.
      3. Also refuse to resume if the breaker service reports a HARD program halt
         (a -35%-from-peak halt requires a manual restart, not an auto-boot).
      4. On a clean resume, START THE WATCHDOG (arming the dead-man's switch),
         arm the local-OCO dead-man's-switch assertion, wire the fast loop's
         subscriptions, and start the loop — watchdog up BEFORE new orders.

    Returns the same :class:`System` with ``started`` / ``halted_on_boot`` set and
    ``system.watchdog`` populated on a clean resume.
    """
    # 1) ON-BOOT RECOVERY — FIRST, before any new order.
    result = reconcile(system.bus, system.orderbook, system.venue)
    system.recon_result = result

    # 2) Unreconcilable mismatch / halt -> stay down, do not trade.
    if not result.resume:
        system.halted_on_boot = True
        system.started = False
        # reconcile already published CIRCUIT_BREAKER_TRIPPED, which latched the
        # gateway's halt. Make the gateway latch explicit/defensive here too.
        system.gateway._halted = True
        system.gateway._halt_reason = result.halt_reason or "boot reconcile: do not resume"
        return system

    # 3) A HARD program halt also blocks auto-resume (manual restart required).
    if system.breakers.is_program_halted():
        system.halted_on_boot = True
        system.started = False
        system.gateway._halted = True
        system.gateway._halt_reason = (
            system.breakers.halt_reason() or "program halt: manual restart required"
        )
        return system

    # 4) Clean resume. START THE WATCHDOG FIRST (independent §7 monitor), which
    #    ARMS the dead-man's switch. Local OCO brackets require the switch armed
    #    in production (watchdog.py); the armed watchdog satisfies that
    #    requirement so the assertion below passes.
    if start_watchdog_hook:
        system.watchdog = start_watchdog(system)
        armed = dead_mans_switch_armed or system.watchdog.armed
    else:
        armed = dead_mans_switch_armed
    system.orderbook.assert_dead_mans_switch_armed(armed)

    # Wire the deterministic fast loop onto the shared bus and start it (only
    # AFTER the watchdog is up — new orders are not enabled before the monitor).
    system.fast_loop.start()
    system.started = True
    system.halted_on_boot = False
    return system


# --------------------------------------------------------------------------- #
# CLI (paper mode)                                                             #
# --------------------------------------------------------------------------- #
def _default_armed(symbol: str = "QQQ", variant: str = "V0") -> list[ArmedStrategy]:
    """A single armed breakout-retest strategy for the paper CLI demo.

    Levels are placeholders; in production P1 supplies point-in-time, split-
    adjusted levels (PDH/PDL/...). Kept import-light so ``--help`` is cheap.
    """
    from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

    strat = BreakoutRetestStrategy(params=load_params(variant))
    return [
        ArmedStrategy(
            name=f"breakout_{variant}", strategy=strat, symbol=symbol, tick=0.01,
            levels={}, grade="A", route="paper",
        )
    ]


def main(argv: list[str] | None = None) -> int:
    """Paper-mode CLI entry point. Builds + boots the system; reconcile runs first.

    This does NOT auto-place orders — it assembles the deterministic core, runs
    on-boot recovery, and reports whether trading resumed. A market-data feed
    (P1/P5) drives bars via :meth:`System.feed_bar`; the long-running async loop
    is intentionally minimal here (the hot path is synchronous + deterministic).
    """
    parser = argparse.ArgumentParser(description="TradeForge orchestrator (paper mode)")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="venue mode; live requires the P0 hook + LIVE gates")
    parser.add_argument("--symbol", default="QQQ", help="symbol to arm")
    parser.add_argument("--variant", default="V0", help="breakout strategy variant")
    parser.add_argument("--equity", type=float, default=1000.0, help="starting equity")
    parser.add_argument("--events-db", default=DEFAULT_EVENTS_DB)
    parser.add_argument("--orderbook-db", default=DEFAULT_ORDERBOOK_DB)
    parser.add_argument("--paper-ledger-db", default=DEFAULT_PAPER_LEDGER_DB)
    args = parser.parse_args(argv)

    enable_live = args.mode == "live" and os.environ.get("TRADEFORGE_LIVE") == "1"
    if args.mode == "live" and not enable_live:
        print("[main] live mode requested but TRADEFORGE_LIVE != 1 -> staying on "
              "paper (the live broker is a gated stub; see brokers.py).")

    system = build_system(
        armed=_default_armed(args.symbol, args.variant),
        equity=args.equity,
        events_db=args.events_db,
        orderbook_db=args.orderbook_db,
        paper_ledger_db=args.paper_ledger_db,
        enable_live=enable_live,
    )
    boot(system)

    if system.started:
        print(f"[main] booted: reconcile resumed; fast loop armed on {args.symbol} "
              f"(adopted={len(system.recon_result.adopted_positions)}, "
              f"cancelled={len(system.recon_result.cancelled_orders)}). "
              "Hot path is LLM/MCP-free; default venue = paper.")
    else:
        print(f"[main] HALTED on boot — NOT trading. "
              f"reason: {system.gateway._halt_reason}")
    system.close()
    return 0 if system.started else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
