"""EventBus — synchronous, deterministic pub/sub over a replayable DuckDB log.

The event-driven core (MASTER_PLAN.md §4, §7). Design contract the sibling
fast-loop and breaker agents build against:

- ``seq`` is the single ordering authority. ``publish`` assigns a strictly
  monotonic ``seq``, persists the row, THEN dispatches synchronously and in
  subscription order. Dispatch is ordered and replayable; asyncio may wrap the
  OUTER run loop / IO, but event *dispatch order* is deterministic.
- Handlers are deterministic sync callables — ``handler(event)``. **No LLM and
  no MCP calls in a handler**: this is the hot path that touches money.
- The bus is duck-typed / injectable: any object exposing ``publish`` and
  ``subscribe`` can stand in for tests (see the fake bus used by the gateway
  tests). The concrete class here adds durable persistence + replay.

DuckDB is imported lazily inside :meth:`__init__` so merely importing this
module never hard-requires the dependency (matches the hooks/data convention).
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

from orchestrator.events import Event, EventType

DEFAULT_DB_PATH = "orderbook/../events.duckdb"  # overridden below to repo root
# Resolve to a stable absolute-ish default alongside the other artifact DBs.
DEFAULT_DB_PATH = "events.duckdb"

Handler = Callable[[Event], None]


class EventBus:
    """Synchronous event bus with a durable, replayable DuckDB event log.

    Public API (stable for sibling agents):
      - ``EventBus(db_path=...)``
      - ``subscribe(types, handler)`` — ``types`` is an EventType, a list of
        them, or the string ``"*"`` for all events. Dispatch is in subscription
        registration order.
      - ``publish(event) -> Event`` — assigns ``seq``, persists, dispatches.
      - ``replay(handler=None, from_seq=0)`` — re-emit persisted events in seq
        order to rebuild state. If ``handler`` is None, replays to all currently
        registered subscribers (so a fresh bus can rebuild downstream state).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        import duckdb  # lazy

        self.db_path = db_path
        if db_path != ":memory:":
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._con = duckdb.connect(db_path)
        self._init_schema()

        # Subscriptions preserve registration order for deterministic dispatch.
        # Each entry: (frozenset_of_types_or_None_for_wildcard, handler).
        self._subs: list[tuple[frozenset | None, Handler]] = []

        # Monotonic seq counter, seeded from the persisted max so restarts
        # continue the sequence rather than colliding on the PRIMARY KEY.
        row = self._con.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()
        self._next_seq = int(row[0]) + 1

    # ------------------------------------------------------------------ #
    # schema                                                             #
    # ------------------------------------------------------------------ #
    def _init_schema(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                seq    BIGINT PRIMARY KEY,
                ts_utc TIMESTAMP,
                type   VARCHAR,
                source VARCHAR,
                data   VARCHAR
            )
            """
        )

    # ------------------------------------------------------------------ #
    # subscribe                                                          #
    # ------------------------------------------------------------------ #
    def subscribe(self, types, handler: Handler) -> None:
        """Register ``handler`` for ``types``.

        ``types`` may be a single :class:`EventType`, an iterable of them, or
        the string ``"*"`` to receive every event. Handlers fire in the order
        they were registered (deterministic dispatch).
        """
        if types == "*":
            key: frozenset | None = None
        elif isinstance(types, EventType):
            key = frozenset({types})
        elif isinstance(types, (list, tuple, set, frozenset)):
            key = frozenset(EventType(t) for t in types)
        else:
            # Allow a bare string event name for convenience.
            key = frozenset({EventType(types)})
        self._subs.append((key, handler))

    # ------------------------------------------------------------------ #
    # publish                                                            #
    # ------------------------------------------------------------------ #
    def publish(self, event: Event) -> Event:
        """Assign a monotonic ``seq``, persist the row, then dispatch in order.

        Returns the same event with ``seq`` / ``ts_utc`` populated. Persistence
        happens BEFORE dispatch so the log is the durable source of truth even
        if a handler raises.
        """
        if event.seq is None:
            event.seq = self._next_seq
            self._next_seq += 1
        else:
            # Honor an externally-assigned seq but keep the counter ahead of it.
            self._next_seq = max(self._next_seq, event.seq + 1)

        if event.ts_utc is None:
            from orchestrator.events import _utcnow

            event.ts_utc = _utcnow()

        self._persist(event)
        self._dispatch(event)
        return event

    def _persist(self, event: Event) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO events (seq, ts_utc, type, source, data) "
            "VALUES (?, ?, ?, ?, ?)",
            list(event.to_row()),
        )

    def _dispatch(self, event: Event) -> None:
        """Synchronous, in-registration-order dispatch to matching handlers."""
        for key, handler in self._subs:
            if key is None or event.type in key:
                handler(event)

    # ------------------------------------------------------------------ #
    # replay                                                             #
    # ------------------------------------------------------------------ #
    def replay(self, handler: Handler | None = None, from_seq: int = 0) -> list[Event]:
        """Re-emit persisted events in ``seq`` order (>= ``from_seq``).

        If ``handler`` is provided, every replayed event is passed only to it
        (state-rebuild use case where one consumer wants the full stream). If
        ``handler`` is None, events are dispatched to all currently registered
        subscribers — rebuilding their state from the durable log.

        Returns the list of replayed events (in seq order) for inspection/tests.
        Replay does NOT re-persist (the rows already exist) and does NOT mutate
        the live seq counter.
        """
        rows = self._con.execute(
            "SELECT seq, ts_utc, type, source, data FROM events "
            "WHERE seq >= ? ORDER BY seq ASC",
            [from_seq],
        ).fetchall()

        events = [Event.from_row(r) for r in rows]
        for ev in events:
            if handler is not None:
                handler(ev)
            else:
                self._dispatch(ev)
        return events

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #
    def events(self, from_seq: int = 0) -> list[Event]:
        """Return persisted events (>= ``from_seq``) in seq order, no dispatch."""
        rows = self._con.execute(
            "SELECT seq, ts_utc, type, source, data FROM events "
            "WHERE seq >= ? ORDER BY seq ASC",
            [from_seq],
        ).fetchall()
        return [Event.from_row(r) for r in rows]

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        try:
            self._con.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass
