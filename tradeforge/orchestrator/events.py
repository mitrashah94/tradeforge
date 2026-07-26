"""Typed event catalog + Event record for the event-driven core (MASTER_PLAN.md §4).

This module is the canonical taxonomy of bus events AND the serializable record
that flows through :mod:`orchestrator.bus`. It deliberately depends on nothing
heavy (stdlib only) so any module — fast loop, breakers, gateway, tests — can
import it without pulling in DuckDB, pydantic, or pandas.

Determinism contract: an :class:`Event` is a plain, JSON-serializable record.
``seq`` (assigned by the bus on publish) is the ordering authority for replay.
``data`` MUST be JSON-serializable (it is persisted as a JSON text column).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class EventType(str, Enum):
    """Canonical event names published on the bus. Value == member name (str).

    The 26-member catalog from MASTER_PLAN.md §4 is preserved verbatim; the
    order-lifecycle additions (SUBMITTED / WORKING / CANCELLED / EXPIRED) plus
    the resilience events (RECONCILED / HEARTBEAT) are appended so the state
    machine, gateway, and reconciler can express the full FSM on the bus.
    """

    # --- Market data / signal / setup (fast loop upstream) ---
    # BAR is the deterministic OHLCV input event the fast loop ingests on the hot
    # path (P3 integration: the FastLoop subscribes to BAR or PRICE_CROSS_LEVEL).
    BAR = "BAR"
    PRICE_CROSS_LEVEL = "PRICE_CROSS_LEVEL"
    LEVEL_BREAK_CONFIRMED = "LEVEL_BREAK_CONFIRMED"
    SETUP_FORMING = "SETUP_FORMING"
    SETUP_CONFIRMED = "SETUP_CONFIRMED"
    SETUP_INVALIDATED = "SETUP_INVALIDATED"

    # --- Order lifecycle ---
    ORDER_INTENT = "ORDER_INTENT"
    ORDER_APPROVED = "ORDER_APPROVED"
    ORDER_VETOED = "ORDER_VETOED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_WORKING = "ORDER_WORKING"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIAL = "ORDER_PARTIAL"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_EXPIRED = "ORDER_EXPIRED"

    # --- Position management ---
    TP1_HIT = "TP1_HIT"
    STOP_HIT = "STOP_HIT"
    POSITION_CLOSED = "POSITION_CLOSED"

    # --- Risk / breakers / halts (consumed by the gateway as halt gates) ---
    CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"
    COOLDOWN_STARTED = "COOLDOWN_STARTED"
    NO_TRADE_WINDOW = "NO_TRADE_WINDOW"
    VOL_SPIKE = "VOL_SPIKE"
    DATA_ANOMALY = "DATA_ANOMALY"

    # --- Slow-loop policy (LLM-overseen, deterministic compute) ---
    # Published once per session by the regime-reader (orchestrator/agents/
    # regime_reader.py): the daily regime/vol read + armed strategies + the
    # exposure scalar the fast-loop sizing consumes. See MASTER_PLAN.md §1
    # (regime-scaled exposure) and §4 (slow-loop agents).
    REGIME_TAGGED = "REGIME_TAGGED"

    # --- Strategy / growth ---
    STRATEGY_DEMOTED = "STRATEGY_DEMOTED"
    MILESTONE_REACHED = "MILESTONE_REACHED"
    RATCHET_SWEEP = "RATCHET_SWEEP"
    DEPOSIT_LOGGED = "DEPOSIT_LOGGED"
    # Mark-to-market equity refresh consumed by the breaker service (drawdown
    # tracking). Part of the §4 growth/risk plumbing; named here so it can flow
    # on the typed bus (the breaker service subscribes to it by name).
    EQUITY_UPDATE = "EQUITY_UPDATE"
    WATCHLIST_UPDATED = "WATCHLIST_UPDATED"
    SYMBOL_PROMOTED = "SYMBOL_PROMOTED"
    SYMBOL_DEMOTED = "SYMBOL_DEMOTED"

    # --- Resilience ---
    RECONCILED = "RECONCILED"
    HEARTBEAT = "HEARTBEAT"
    # Watchdog enters SAFE_MODE when the engine heartbeat goes stale (hung
    # engine). Published alongside a CIRCUIT_BREAKER_TRIPPED so the gateway
    # refuses NEW orders while the engine is suspect (MASTER_PLAN.md §7).
    SAFE_MODE = "SAFE_MODE"
    # The dead-man's switch fired: broker/data lost beyond the timeout WITH an
    # open position whose LOCAL bracket can no longer be managed. Carries the
    # action taken ("flatten" or "alert_and_halt"). MANDATORY because brackets
    # are local (MASTER_PLAN.md §3, §7).
    DEAD_MANS_SWITCH_TRIPPED = "DEAD_MANS_SWITCH_TRIPPED"


# Backwards-compatibility alias. The original stub exported ``Event`` as the
# enum; the contract renames the enum to ``EventType`` and reuses ``Event`` for
# the record dataclass (below). ``EventType`` is the name new code should import.


def _utcnow() -> datetime:
    """Return a tz-naive UTC timestamp (matches the DB's TIMESTAMP convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class Event:
    """One event flowing through the bus and persisted in the events log.

    Attributes:
        type:   the :class:`EventType` of this event.
        data:   JSON-serializable payload dict.
        ts_utc: tz-naive UTC timestamp; filled by the bus on publish if None.
        seq:    monotonic sequence number; the ORDERING AUTHORITY. Assigned by
                the bus on publish (None before that).
        source: free-form producer tag (e.g. "order_gateway", "fast_loop").
    """

    type: EventType
    data: dict = field(default_factory=dict)
    ts_utc: datetime | None = None
    seq: int | None = None
    source: str = ""

    def __post_init__(self) -> None:
        # Accept a bare string for ``type`` and coerce to the enum so callers
        # constructing events from rows or loose strings stay ergonomic.
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)

    # ------------------------------------------------------------------ #
    # DuckDB persistence helpers                                         #
    # ------------------------------------------------------------------ #
    def to_row(self) -> tuple:
        """Serialize to a tuple matching the events table column order.

        Columns: (seq, ts_utc, type, source, data) where ``data`` is JSON text.
        """
        return (
            self.seq,
            self.ts_utc,
            self.type.value,
            self.source,
            json.dumps(self.data, default=str, sort_keys=True),
        )

    @classmethod
    def from_row(cls, row) -> "Event":
        """Rebuild an Event from a persisted row (seq, ts_utc, type, source, data).

        ``data`` is parsed back from its JSON text column. Accepts either a
        sequence (tuple/list from a DuckDB fetch) in the canonical column order.
        """
        seq, ts_utc, type_, source, data = row
        parsed = json.loads(data) if isinstance(data, str) else (data or {})
        return cls(
            type=EventType(type_),
            data=parsed,
            ts_utc=ts_utc,
            seq=seq,
            source=source or "",
        )
