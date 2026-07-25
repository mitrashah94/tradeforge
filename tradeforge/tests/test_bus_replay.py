"""tests/test_bus_replay.py — event bus persistence + deterministic replay.

Deterministic, offline. Uses a temp-file DuckDB so a SECOND bus can reopen the
same log and prove the persisted stream replays in the same seq order (the
ordering authority for replayable, deterministic dispatch).
"""

from __future__ import annotations

import pytest

from orchestrator.bus import EventBus
from orchestrator.events import Event, EventType


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "events.duckdb")


def test_publish_assigns_monotonic_seq(db_path):
    bus = EventBus(db_path=db_path)
    e1 = bus.publish(Event(EventType.ORDER_INTENT, {"symbol": "SPY"}))
    e2 = bus.publish(Event(EventType.ORDER_APPROVED, {"symbol": "SPY"}))
    e3 = bus.publish(Event(EventType.ORDER_FILLED, {"symbol": "SPY"}))
    assert [e1.seq, e2.seq, e3.seq] == [1, 2, 3]
    assert e1.ts_utc is not None
    bus.close()


def test_subscribe_dispatch_in_registration_order(db_path):
    bus = EventBus(db_path=db_path)
    calls = []
    bus.subscribe(EventType.ORDER_INTENT, lambda e: calls.append("a"))
    bus.subscribe(EventType.ORDER_INTENT, lambda e: calls.append("b"))
    bus.subscribe("*", lambda e: calls.append("wild"))
    bus.publish(Event(EventType.ORDER_INTENT, {}))
    assert calls == ["a", "b", "wild"]
    bus.close()


def test_subscribe_list_and_wildcard_filtering(db_path):
    bus = EventBus(db_path=db_path)
    got = []
    bus.subscribe([EventType.ORDER_FILLED, EventType.ORDER_REJECTED],
                  lambda e: got.append(("multi", e.type)))
    bus.subscribe(EventType.ORDER_INTENT, lambda e: got.append(("single", e.type)))
    bus.publish(Event(EventType.ORDER_INTENT, {}))
    bus.publish(Event(EventType.ORDER_FILLED, {}))
    bus.publish(Event(EventType.VOL_SPIKE, {}))  # no subscriber
    assert ("single", EventType.ORDER_INTENT) in got
    assert ("multi", EventType.ORDER_FILLED) in got
    assert all(t != EventType.VOL_SPIKE for _, t in got)
    bus.close()


def test_replay_reconstructs_ordered_seq_stream(db_path):
    bus = EventBus(db_path=db_path)
    published = [
        bus.publish(Event(EventType.PRICE_CROSS_LEVEL, {"i": 0})),
        bus.publish(Event(EventType.ORDER_INTENT, {"i": 1})),
        bus.publish(Event(EventType.ORDER_APPROVED, {"i": 2})),
        bus.publish(Event(EventType.ORDER_FILLED, {"i": 3})),
    ]
    bus.close()

    # Fresh bus reopens the SAME log and replays.
    fresh = EventBus(db_path=db_path)
    replayed = []
    fresh.replay(handler=lambda e: replayed.append(e))

    assert [e.seq for e in replayed] == [1, 2, 3, 4]
    assert [e.type for e in replayed] == [e.type for e in published]
    # data round-trips through the JSON column.
    assert [e.data["i"] for e in replayed] == [0, 1, 2, 3]
    fresh.close()


def test_replay_to_subscribers_rebuilds_state(db_path):
    bus = EventBus(db_path=db_path)
    bus.publish(Event(EventType.DEPOSIT_LOGGED, {"amount": 50}))
    bus.publish(Event(EventType.DEPOSIT_LOGGED, {"amount": 50}))
    bus.close()

    fresh = EventBus(db_path=db_path)
    total = {"sum": 0}
    fresh.subscribe(EventType.DEPOSIT_LOGGED,
                    lambda e: total.__setitem__("sum", total["sum"] + e.data["amount"]))
    # No handler arg -> replay dispatches to registered subscribers.
    fresh.replay()
    assert total["sum"] == 100
    fresh.close()


def test_replay_from_seq(db_path):
    bus = EventBus(db_path=db_path)
    for i in range(5):
        bus.publish(Event(EventType.HEARTBEAT, {"i": i}))
    got = bus.replay(handler=lambda e: None, from_seq=3)
    assert [e.seq for e in got] == [3, 4, 5]
    bus.close()


def test_seq_continues_after_reopen(db_path):
    bus = EventBus(db_path=db_path)
    bus.publish(Event(EventType.HEARTBEAT, {}))
    bus.publish(Event(EventType.HEARTBEAT, {}))
    bus.close()
    # Reopen: next seq must continue from the persisted max, not collide.
    bus2 = EventBus(db_path=db_path)
    e = bus2.publish(Event(EventType.HEARTBEAT, {}))
    assert e.seq == 3
    bus2.close()


def test_event_to_row_from_row_roundtrip():
    ev = Event(EventType.ORDER_FILLED, {"symbol": "SPY", "qty": 10}, seq=7,
               source="gw")
    row = ev.to_row()
    back = Event.from_row(row)
    assert back.type == EventType.ORDER_FILLED
    assert back.data == {"symbol": "SPY", "qty": 10}
    assert back.seq == 7
    assert back.source == "gw"
