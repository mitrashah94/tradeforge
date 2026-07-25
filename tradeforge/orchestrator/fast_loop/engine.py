"""orchestrator/fast_loop/engine.py — the DETERMINISTIC fast loop (MASTER_PLAN §4, flows F1/F2).

The real-time trading engine with the **LLM/MCP OUT of the hot path**. Per
incoming bar/price event it:

  1. ENTRY (F1, pre-armed rules): drives armed ``Strategy`` instances through a
     live ``Context`` adapter whose ``enter_long/enter_short/close`` PUBLISH
     ``ORDER_INTENT`` events instead of simulating fills. Only armed strategies
     fire; a NO_TRADE_WINDOW or CIRCUIT_BREAKER_TRIPPED forces a stand-down.
  2. SIZING (§1.B vol-target): converts the strategy's stop into a share count
     via ``sizing.vol_target_qty`` so $-risk per trade is constant across vol.
  3. LIFECYCLE (F2): a deterministic ``PositionLifecycle`` state machine manages
     TP1 partial -> breakeven -> trail -> time-stop -> session-flatten, reacting
     to ORDER_FILLED / ORDER_PARTIAL / POSITION_CLOSED.
  4. LATENCY BUDGET: every event handler is timed with ``time.perf_counter`` and
     guarded against a budget (default a few ms); a summary is exposed.

CONSTRUCTION (the public API)
-----------------------------
    loop = FastLoop(
        bus=bus,                      # duck-typed: subscribe(types, handler), publish(event)
        armed=[ArmedStrategy(...)],   # pre-armed strategy instances + their context
        equity_source=lambda: 1000.0, # callable -> current equity (compounds)
        limits=load_limits(),         # risk.config.Limits (single source of truth)
        ri=5,                         # risk index (or per-ArmedStrategy override)
        clock=lambda: datetime.utcnow(),  # injected clock (deterministic in tests)
    )
    loop.start()                      # wires bus subscriptions

Bars are fed by PUBLISHING a bar/price event onto the bus; the loop's handler
runs synchronously in the hot path. There is **no LLM call, no MCP call, and no
network I/O** anywhere below — by construction (see test_fast_loop_latency).

EVENT CONTRACT (depended on; implemented by a sibling agent)
------------------------------------------------------------
SUBSCRIBES: bar/price input events (``EventType.BAR`` or ``PRICE_CROSS_LEVEL``),
``ORDER_FILLED``, ``ORDER_PARTIAL``, ``POSITION_CLOSED``,
``CIRCUIT_BREAKER_TRIPPED``, ``NO_TRADE_WINDOW``.
PUBLISHES: ``ORDER_INTENT`` (entry / partial-exit / stop-move / close) and the
lifecycle markers ``TP1_HIT`` / ``STOP_HIT`` / ``POSITION_CLOSED`` as the
deterministic record of what it did.

We depend ONLY on the duck-typed bus + the event-name strings, so we are robust
to whichever concrete ``Event``/``EventType`` the sibling agent ships.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from risk.config import Limits
from orchestrator.fast_loop.lifecycle import (
    LifecycleState,
    ManagedPosition,
    PositionLifecycle,
)
from orchestrator.fast_loop.sizing import vol_target_qty

# --------------------------------------------------------------------------- #
# Event-name constants (string values; match orchestrator.events.Event members)
# We reference event TYPES by their canonical string name so the loop does not
# hard-depend on a particular enum class the sibling agent ships. The bus is
# duck-typed; an EventType member compares equal to its str value (StrEnum/str
# mixin), and a plain string works too.
# --------------------------------------------------------------------------- #
EVT_BAR = "BAR"
EVT_PRICE_CROSS_LEVEL = "PRICE_CROSS_LEVEL"
EVT_ORDER_INTENT = "ORDER_INTENT"
EVT_ORDER_FILLED = "ORDER_FILLED"
EVT_ORDER_PARTIAL = "ORDER_PARTIAL"
EVT_ORDER_VETOED = "ORDER_VETOED"
EVT_ORDER_REJECTED = "ORDER_REJECTED"
EVT_POSITION_CLOSED = "POSITION_CLOSED"
EVT_TP1_HIT = "TP1_HIT"
EVT_STOP_HIT = "STOP_HIT"
EVT_NO_TRADE_WINDOW = "NO_TRADE_WINDOW"
EVT_CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"

# Default hot-path latency budget (milliseconds). A bar event must be processed
# faster than this; the loop records every measurement and can assert the budget.
DEFAULT_LATENCY_BUDGET_MS = 5.0


# --------------------------------------------------------------------------- #
# Live bar container (duck-typed to the engine's Bar; tests can use any object
# exposing ts/open/high/low/close + optional is_eod). We define a light dataclass
# so the loop can normalize whatever the bar event carries.
# --------------------------------------------------------------------------- #
@dataclass
class LiveBar:
    """One OHLCV bar delivered on a bar event (mirror of backtest Bar + is_eod)."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    is_eod: bool = False
    # optional precomputed chandelier band for atr trailing
    trail_stop: float | None = None

    @classmethod
    def from_data(cls, data: dict) -> "LiveBar":
        return cls(
            ts=data.get("ts") or data.get("ts_utc") or datetime.now(timezone.utc),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0.0) or 0.0),
            is_eod=bool(data.get("is_eod", False)),
            trail_stop=data.get("trail_stop"),
        )


# --------------------------------------------------------------------------- #
# Live Context adapter — drives the existing Strategy interface, but order
# methods PUBLISH ORDER_INTENT instead of simulating fills.
# --------------------------------------------------------------------------- #
class LiveContext:
    """A ``Context``-shaped adapter for the deterministic live loop.

    Re-implements the read surface the engine ``Context`` exposes (bar, prev_bars,
    bar_index, levels, position, in_session, is_eod, symbol/asset_class/tick) and
    the order methods (``enter_long/enter_short/close``). Order calls are routed
    to the owning :class:`FastLoop`, which sizes them (vol-target) and publishes
    an ``ORDER_INTENT`` event — NO fill simulation happens here.
    """

    def __init__(self, loop: "FastLoop", armed: "ArmedStrategy"):
        self._loop = loop
        self._armed = armed
        self.bar: LiveBar | None = None
        self.bar_index: int = -1
        self.prev_bars: list[LiveBar] = []
        self.levels: dict = dict(armed.levels or {})
        self.in_session: bool = True
        self.is_eod: bool = False
        self.symbol: str = armed.symbol
        self.asset_class: str = armed.asset_class
        self.tick: float = armed.tick

    @property
    def position(self):
        """The fast loop's managed position for this strategy, or None."""
        mp = self._loop._managed
        if mp is None or mp.strategy != self._armed.name:
            return None
        # Present a minimal position view: strategies only read `.side`.
        return mp

    def enter_long(self, stop, target=None, partial=None) -> None:
        self._loop._request_entry(self._armed, "long", stop, target, partial)

    def enter_short(self, stop, target=None, partial=None) -> None:
        self._loop._request_entry(self._armed, "short", stop, target, partial)

    def close(self) -> None:
        self._loop._request_strategy_close(self._armed)


# --------------------------------------------------------------------------- #
# Armed strategy descriptor — a strategy instance + its live context state.
# --------------------------------------------------------------------------- #
@dataclass
class ArmedStrategy:
    """A pre-armed strategy instance the fast loop will drive on each bar.

    ``strategy`` is any object implementing the engine ``Strategy`` interface
    (``on_session_start``, ``on_bar``). ``levels`` carries the session levels
    (pdh/pdl/atr14/...) the strategy reads via ``ctx.levels``. ``ri`` optionally
    overrides the loop default risk index for conviction tiering.
    """

    name: str
    strategy: Any
    symbol: str
    asset_class: str = "equity"
    tick: float = 0.01
    levels: dict = field(default_factory=dict)
    ri: int | None = None
    armed: bool = True
    # conviction tier (-> RI in the gateway risk cap) + venue route.
    grade: str = "A"
    route: str = "paper"
    # vol-target / lifecycle config
    allow_fractional: bool = True
    min_qty: float = 0.0
    tp1_fraction: float = 0.5
    trail_mode: str = "prior_bar"
    time_stop_bars: int | None = None

    # runtime
    ctx: LiveContext | None = None
    bar_index: int = -1
    session_started: bool = False
    session_bars: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Latency tracking
# --------------------------------------------------------------------------- #
@dataclass
class LatencyStats:
    """Rolling hot-path latency stats (milliseconds)."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0
    budget_ms: float = DEFAULT_LATENCY_BUDGET_MS
    over_budget: int = 0

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms
        if ms > self.budget_ms:
            self.over_budget += 1

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def summary(self) -> dict:
        return {
            "count": self.count,
            "avg_ms": self.avg_ms,
            "max_ms": self.max_ms,
            "last_ms": self.last_ms,
            "budget_ms": self.budget_ms,
            "over_budget": self.over_budget,
        }


# --------------------------------------------------------------------------- #
# The fast loop
# --------------------------------------------------------------------------- #
class FastLoop:
    """Deterministic real-time entry+lifecycle engine (no LLM/MCP/network).

    See module docstring for the public API. Inject the bus so it can be unit-
    tested with a fake in-memory bus.
    """

    def __init__(
        self,
        *,
        bus,
        armed: list[ArmedStrategy],
        equity_source: Callable[[], float],
        limits: Limits,
        ri: int = 5,
        clock: Callable[[], datetime] | None = None,
        latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
        source: str = "fast_loop",
        event_factory: Callable[..., Any] | None = None,
    ):
        self.bus = bus
        self.armed = list(armed)
        self.equity_source = equity_source
        self.limits = limits
        self.ri = ri
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.source = source
        self._event_factory = event_factory

        # Stand-down flags (F3 interrupts preempt all entries).
        self.halted: bool = False          # CIRCUIT_BREAKER_TRIPPED
        self.no_trade_window: bool = False  # NO_TRADE_WINDOW active

        # One managed position at a time in the fast loop (single-symbol slice).
        self._managed: ManagedPosition | None = None
        self._lifecycle: PositionLifecycle | None = None
        self._pending_entry: dict | None = None  # last entry intent awaiting fill

        self._seq: int = 0
        self.latency = LatencyStats(budget_ms=latency_budget_ms)

        # Build each armed strategy's live context.
        for a in self.armed:
            a.ctx = LiveContext(self, a)

    # ------------------------------------------------------------ wiring
    def start(self) -> None:
        """Subscribe to the bus. Idempotent enough for a single process boot."""
        self.bus.subscribe([EVT_BAR, EVT_PRICE_CROSS_LEVEL], self._on_bar_event)
        self.bus.subscribe([EVT_ORDER_FILLED], self._on_order_filled)
        self.bus.subscribe([EVT_ORDER_PARTIAL], self._on_order_partial)
        self.bus.subscribe([EVT_ORDER_VETOED, EVT_ORDER_REJECTED], self._on_entry_rejected)
        self.bus.subscribe([EVT_POSITION_CLOSED], self._on_position_closed)
        self.bus.subscribe([EVT_CIRCUIT_BREAKER_TRIPPED], self._on_circuit_breaker)
        self.bus.subscribe([EVT_NO_TRADE_WINDOW], self._on_no_trade_window)

    # ------------------------------------------------------------ stand-down
    def _on_circuit_breaker(self, event) -> None:
        self.halted = True

    def _on_no_trade_window(self, event) -> None:
        data = _event_data(event)
        # data.active (default True) lets a window open AND close.
        self.no_trade_window = bool(data.get("active", True))

    @property
    def standing_down(self) -> bool:
        """True when no NEW entries may fire (halt or active no-trade window)."""
        return self.halted or self.no_trade_window

    # ------------------------------------------------------------ bar handler
    def _on_bar_event(self, event) -> None:
        """The hot path: time it, manage the position, then evaluate entries."""
        t0 = time.perf_counter()
        try:
            data = _event_data(event)
            symbol = data.get("symbol")
            bar = LiveBar.from_data(data)

            # 1) LIFECYCLE FIRST (F2): manage an open position against this bar
            #    before considering any new entry (an exit may free the slot).
            self._manage_open_position(bar, symbol)

            # 2) ENTRIES (F1): drive armed strategies through their live context.
            self._drive_strategies(bar, symbol)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.latency.record(dt_ms)

    def _manage_open_position(self, bar: LiveBar, symbol) -> None:
        if self._lifecycle is None or self._managed is None:
            return
        if symbol is not None and self._managed.symbol != symbol:
            return
        if not self._managed.filled:
            return  # entry not yet filled; nothing to manage
        # Capture the managed position + lifecycle locally. on_bar() emits intents
        # SYNCHRONOUSLY through the real bus, and a close intent round-trips to a
        # POSITION_CLOSED that resets ``self._managed`` to None mid-call. We read
        # the marker fields off the captured object so reentrancy is safe.
        managed = self._managed
        actions = self._lifecycle.on_bar(bar)
        # Surface lifecycle markers (deterministic record of what we did).
        for act in actions:
            if act.reason == "tp1_partial_exit":
                self._publish(EVT_TP1_HIT, {
                    "symbol": managed.symbol,
                    "strategy": managed.strategy,
                    "price": managed.tp1_price,
                })
            elif act.reason in ("stop", "trail_stop"):
                self._publish(EVT_STOP_HIT, {
                    "symbol": managed.symbol,
                    "strategy": managed.strategy,
                    "price": managed.stop_price,
                    "kind": act.reason,
                })

    def _drive_strategies(self, bar: LiveBar, symbol) -> None:
        for a in self.armed:
            if not a.armed:
                continue
            if symbol is not None and a.symbol != symbol:
                continue
            ctx = a.ctx
            # New session detection: first bar, or an is_eod boundary reset.
            if not a.session_started:
                ctx.levels = dict(a.levels or {})
                a.strategy.on_session_start(ctx)
                a.session_started = True
                a.bar_index = -1
                a.session_bars = []

            a.bar_index += 1
            ctx.bar = bar
            ctx.bar_index = a.bar_index
            ctx.prev_bars = list(a.session_bars)
            ctx.is_eod = bar.is_eod
            ctx.in_session = True

            # Strategy reads ctx and may queue an entry via ctx.enter_*; we do
            # NOT let it open a NEW position on the EOD bar (can't be managed).
            if not (bar.is_eod and self._managed is None):
                a.strategy.on_bar(ctx)

            a.session_bars.append(bar)

            # On EOD, reset session state for the next session.
            if bar.is_eod:
                a.session_started = False

    # ------------------------------------------------------------ entry intent
    def _request_entry(self, armed: ArmedStrategy, side, stop, target, partial) -> None:
        """Called by LiveContext.enter_*; size + publish an ORDER_INTENT.

        Honors the stand-down (no new entries on halt / no-trade window) and the
        one-position-at-a-time invariant. Sizing is vol-target (§1.B).
        """
        if self.standing_down:
            return  # F3 stand-down: suppress all new entries
        if self._managed is not None:
            return  # already in / entering a position (single slot)

        ctx = armed.ctx
        signal_close = ctx.bar.close
        equity = float(self.equity_source())
        ri = armed.ri if armed.ri is not None else self.ri

        sizing = vol_target_qty(
            equity=equity,
            ri=ri,
            limits=self.limits,
            entry_price=signal_close,
            stop_price=stop,
            allow_fractional=armed.allow_fractional,
            min_qty=armed.min_qty,
        )
        if sizing.skipped or sizing.qty <= 0:
            return  # degenerate sizing -> do not enter

        client_id = self._client_id(armed, side)
        order_side = "buy" if side == "long" else "sell"

        # Determine TP1 (partial) vs fixed target.
        tp1_price = None
        tp1_fraction = armed.tp1_fraction
        target_price = target
        bracket = {"stop_price": stop}
        if partial is not None:
            tp1_price = getattr(partial, "tp1", None)
            tp1_fraction = getattr(partial, "fraction", armed.tp1_fraction)
            trail_mode = getattr(partial, "trail_mode", armed.trail_mode)
            bracket["target_price"] = tp1_price  # broker bracket TP at the partial level
        else:
            trail_mode = armed.trail_mode
            if target is not None:
                bracket["target_price"] = target

        # Per-share risk -> stop_distance_pct (for the gateway per-trade cap).
        stop_distance_pct = (
            (sizing.stop_distance / signal_close) if signal_close else None
        )

        intent = {
            # --- identity / correlation ---
            "client_id": client_id,
            # --- core order (gateway-required) ---
            "symbol": armed.symbol,
            "side": order_side,
            "qty": sizing.qty,
            "order_type": "market",
            # --- pricing: the signal close is BOTH the slippage base
            #     (intended_price) and the paper-fill reference (ref_price). ---
            "intended_price": signal_close,
            "ref_price": signal_close,
            # --- risk-cap inputs (gateway): equity from equity_source, the
            #     vol-target stop distance as a fraction, conviction grade -> RI.
            "equity": equity,
            "stop_distance_pct": stop_distance_pct,
            "portfolio_heat_pct": 0.0,  # single-slot loop; no concurrent heat
            "grade": armed.grade,
            "route": armed.route,
            # --- OCO bracket spec (stop + optional target) ---
            "bracket": bracket,
            # --- provenance ---
            "strategy": armed.name,
            "reason": f"entry_{side}",
        }
        self._pending_entry = intent

        # Stand up the deterministic managed-position state machine.
        self._managed = ManagedPosition(
            client_id=client_id,
            symbol=armed.symbol,
            side=side,
            strategy=armed.name,
            entry_price=signal_close,
            stop_price=stop,
            initial_stop=stop,
            qty=sizing.qty,
            initial_qty=sizing.qty,
            tp1_price=tp1_price,
            tp1_fraction=tp1_fraction,
            trail_mode=trail_mode,
            time_stop_bars=armed.time_stop_bars,
            target_price=target_price if partial is None else None,
            state=LifecycleState.OPEN_PENDING,
        )
        self._lifecycle = PositionLifecycle(self._managed, emit=self._emit_lifecycle_intent)

        self._publish(EVT_ORDER_INTENT, intent)

    def _request_strategy_close(self, armed: ArmedStrategy) -> None:
        """A strategy explicitly requested a close (ctx.close())."""
        if self._lifecycle is None or self._managed is None:
            return
        if self._managed.strategy != armed.name:
            return
        # _close publishes the close intent via the lifecycle's emit callback.
        self._lifecycle._close(reason="strategy_close")

    # ------------------------------------------------------------ fill handlers
    def _on_order_filled(self, event) -> None:
        t0 = time.perf_counter()
        try:
            data = _event_data(event)
            cid = data.get("client_id")
            if self._lifecycle is None or self._managed is None:
                return
            # The entry fill advances OPEN_PENDING -> OPEN.
            if cid == self._managed.client_id or self._managed.state is LifecycleState.OPEN_PENDING:
                self._lifecycle.on_entry_filled()
            # A close fill (market exit) is reported via POSITION_CLOSED normally,
            # but if the gateway reports the close as an ORDER_FILLED on the
            # close client_id, treat it as terminal too.
            elif cid and cid.endswith(":close"):
                self._lifecycle.on_position_closed()
                self._reset_position()
        finally:
            self.latency.record((time.perf_counter() - t0) * 1000.0)

    def _on_order_partial(self, event) -> None:
        t0 = time.perf_counter()
        try:
            if self._lifecycle is None or self._managed is None:
                return
            actions = self._lifecycle.on_partial_filled()
            # stop->BE move is already emitted by the lifecycle.
        finally:
            self.latency.record((time.perf_counter() - t0) * 1000.0)

    def _on_position_closed(self, event) -> None:
        t0 = time.perf_counter()
        try:
            if self._lifecycle is None or self._managed is None:
                return
            self._lifecycle.on_position_closed()
            self._reset_position()
        finally:
            self.latency.record((time.perf_counter() - t0) * 1000.0)

    def _on_entry_rejected(self, event) -> None:
        """Free the single slot when our PENDING entry is vetoed/rejected.

        ``_request_entry`` stands up ``_managed`` (OPEN_PENDING) BEFORE publishing
        the entry intent. If the gateway then VETOES (risk/halt) or the venue
        REJECTS that intent, no ORDER_FILLED ever arrives — so without this the
        slot would stay OPEN_PENDING forever and block every future entry. We
        reset only when the event matches our pending (unfilled) entry's
        client_id, so an already-filled position is never clobbered.
        """
        t0 = time.perf_counter()
        try:
            if self._managed is None or self._managed.filled:
                return
            data = _event_data(event)
            cid = data.get("client_id")
            if cid is None or cid == self._managed.client_id:
                self._reset_position()
        finally:
            self.latency.record((time.perf_counter() - t0) * 1000.0)

    def _reset_position(self) -> None:
        self._managed = None
        self._lifecycle = None
        self._pending_entry = None

    # ------------------------------------------------------------ emit helpers
    def _emit_lifecycle_intent(self, intent: dict, reason: str) -> None:
        """Lifecycle callback: publish an ORDER_INTENT for a management action.

        Lifecycle intents (tp1 partial-exit, stop-move, market close) are built
        by the pure :mod:`lifecycle` state machine with only the order fields it
        knows (side/qty/order_type/price). The fast loop ENRICHES them here with
        the same gateway-facing schema the entry intent uses (``ref_price`` so the
        paper venue can fill, the venue ``route``, and the conviction ``grade``)
        — the management legs are exits, so they skip the per-trade risk cap.
        """
        enriched = dict(intent)
        armed = self._armed_for_managed()
        enriched.setdefault("route", armed.route if armed else "paper")
        enriched.setdefault("grade", armed.grade if armed else "A")
        # Provide a fill reference for the venue. A management exit fills against
        # its own resting price (limit/stop); a market close fills at the next
        # mark we know (entry as a deterministic fallback when no mark is fed).
        ref = (
            intent.get("limit_price")
            or intent.get("stop_price")
            or (self._managed.entry_price if self._managed else None)
        )
        enriched.setdefault("ref_price", ref)
        enriched.setdefault("intended_price", ref)
        self._publish(EVT_ORDER_INTENT, enriched)

    def _armed_for_managed(self) -> "ArmedStrategy | None":
        if self._managed is None:
            return None
        for a in self.armed:
            if a.name == self._managed.strategy:
                return a
        return None

    def _client_id(self, armed: ArmedStrategy, side: str) -> str:
        return f"{armed.name}:{armed.symbol}:{side}:{self._seq}"

    def _publish(self, evt_type: str, data: dict) -> None:
        """Publish an event onto the bus, building an Event record if possible."""
        self._seq += 1  # internal counter: client_id uniqueness only (NOT bus seq)
        event = self._build_event(evt_type, data)
        self.bus.publish(event)

    def _build_event(self, evt_type: str, data: dict):
        """Build the bus event payload.

        Prefers an injected ``event_factory`` (the sibling agent's typed
        ``Event``); else falls back to a lightweight namespace the fake test bus
        understands. We never hard-import the sibling's class so the loop stays
        decoupled and unit-testable.

        ``seq`` is left None: the EventBus is the SINGLE ordering authority and
        assigns the monotonic, replayable seq on publish (see bus.py). The loop's
        own ``_seq`` is only used to disambiguate ``client_id``s.
        """
        ts = self.clock()
        if self._event_factory is not None:
            return self._event_factory(
                type=evt_type, data=data, ts_utc=ts, seq=None, source=self.source
            )
        return _SimpleEvent(type=evt_type, data=data, ts_utc=ts, seq=None, source=self.source)

    # ------------------------------------------------------------ introspection
    def latency_summary(self) -> dict:
        """Return the rolling hot-path latency summary (ms)."""
        return self.latency.summary()

    def assert_within_budget(self) -> None:
        """Raise AssertionError if any hot-path event exceeded the budget."""
        assert self.latency.over_budget == 0, (
            f"fast loop exceeded latency budget {self.latency.budget_ms}ms "
            f"on {self.latency.over_budget}/{self.latency.count} events "
            f"(max {self.latency.max_ms:.3f}ms)"
        )


# --------------------------------------------------------------------------- #
# Event helpers — duck-typed access to whatever the bus delivers.
# --------------------------------------------------------------------------- #
@dataclass
class _SimpleEvent:
    """Fallback Event record matching the contract Event(type,data,ts_utc,seq,source)."""

    type: Any
    data: dict
    ts_utc: datetime
    seq: int
    source: str


def _event_data(event) -> dict:
    """Extract the ``data`` dict from a bus event (duck-typed)."""
    if event is None:
        return {}
    data = getattr(event, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(event, dict):
        return event.get("data", event)
    return {}
