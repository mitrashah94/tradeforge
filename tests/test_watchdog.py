"""tests/test_watchdog.py — the INDEPENDENT watchdog (MASTER_PLAN §7).

The watchdog monitors an engine HEARTBEAT and broker/data CONNECTIVITY via
INJECTED callables (so it is an independent process in prod, a pure unit here):

  - a STALE heartbeat (now - last > heartbeat_timeout) -> ``check()`` enters
    SAFE-MODE: publishes SAFE_MODE + CIRCUIT_BREAKER_TRIPPED (gateway refuses new
    orders) and sends an alert (to the in-test sink).
  - a FRESH heartbeat -> no trip.
  - broker/data loss + an open position past dms_timeout -> routes into the
    dead-man's switch (which flattens / alert-and-halts).

Deterministic, offline. Clock injected; bus + notify + sources are fakes. No real
iMessage, no real DB, no engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from orchestrator.dead_mans_switch import ACTION_ALERT_AND_HALT, ACTION_FLATTEN
from orchestrator.events import Event, EventType
from orchestrator.watchdog import Watchdog


# --------------------------------------------------------------------------- #
# Fakes (shared shape with the DMS tests)                                      #
# --------------------------------------------------------------------------- #
@dataclass
class FakePosition:
    position_id: str
    symbol: str
    side: str
    qty: float


@dataclass
class FakeOrder:
    order_id: str
    symbol: str
    side: str


class FakeBus:
    def __init__(self):
        self.events: list[Event] = []

    def publish(self, event: Event) -> Event:
        self.events.append(event)
        return event

    def of_type(self, etype: EventType) -> list[Event]:
        return [e for e in self.events if e.type == etype]


class FakeClock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class NotifySink:
    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, message, *, title=None, channel=None, **kw) -> str:
        text = f"[{title}] {message}" if title else message
        self.messages.append(text)
        return text


class Flattener:
    def __init__(self):
        self.flattened: list[str] = []

    def __call__(self, position) -> dict:
        self.flattened.append(position.symbol)
        return {"status": "filled", "symbol": position.symbol}


class Canceller:
    def __init__(self):
        self.cancelled: list[str] = []

    def __call__(self, order) -> dict:
        self.cancelled.append(order.order_id)
        return {"status": "cancelled"}


def _make_watchdog(*, clock, bus, notify, heartbeat_source, connectivity,
                   positions, working_orders=None, flatten=None, cancel=None,
                   heartbeat_timeout=15.0, dms_timeout=30.0):
    return Watchdog(
        bus=bus,
        heartbeat_source=heartbeat_source,
        connectivity=connectivity,
        positions_source=lambda: positions,
        working_orders_source=lambda: (working_orders or []),
        flatten_fn=flatten or Flattener(),
        cancel_fn=cancel or Canceller(),
        ri=5,
        heartbeat_timeout=heartbeat_timeout,
        dms_timeout=dms_timeout,
        clock=clock,
        notify_fn=notify,
    )


# --------------------------------------------------------------------------- #
# 1) Stale heartbeat -> SAFE MODE                                             #
# --------------------------------------------------------------------------- #
def test_stale_heartbeat_enters_safe_mode():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()

    # Last heartbeat is 20s old; timeout is 15s -> STALE.
    last_hb = start - timedelta(seconds=20)
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: last_hb,
        connectivity=lambda: {"broker": True, "data": True},
        positions=[],
        heartbeat_timeout=15.0,
    )

    state = wd.check()

    assert state.safe_mode is True
    # SAFE_MODE + CIRCUIT_BREAKER_TRIPPED published (gateway refuses new orders).
    assert len(bus.of_type(EventType.SAFE_MODE)) == 1
    cb = bus.of_type(EventType.CIRCUIT_BREAKER_TRIPPED)
    assert len(cb) == 1
    assert cb[0].data["kind"] == "watchdog_safe_mode"
    # An alert went to the (in-test) file sink.
    assert any("SAFE-MODE" in m for m in notify.messages)


def test_safe_mode_alert_is_idempotent_on_latch():
    """Re-checking while still stale does not re-spam SAFE_MODE / alerts."""
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    last_hb = start - timedelta(seconds=20)
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: last_hb,
        connectivity=lambda: {"broker": True, "data": True},
        positions=[], heartbeat_timeout=15.0,
    )

    wd.check()
    clock.advance(1)
    wd.check()  # still stale, but already latched

    assert len(bus.of_type(EventType.SAFE_MODE)) == 1  # only the EDGE published
    assert len(notify.messages) == 1


def test_missing_heartbeat_is_treated_as_stale():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: None,  # engine never beat
        connectivity=lambda: {"broker": True, "data": True},
        positions=[], heartbeat_timeout=15.0,
    )
    state = wd.check()
    assert state.safe_mode is True
    assert bus.of_type(EventType.SAFE_MODE)


# --------------------------------------------------------------------------- #
# 2) Fresh heartbeat -> no trip                                              #
# --------------------------------------------------------------------------- #
def test_fresh_heartbeat_does_not_trip():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()

    last_hb = start - timedelta(seconds=3)  # well within the 15s timeout
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: last_hb,
        connectivity=lambda: {"broker": True, "data": True},
        positions=[], heartbeat_timeout=15.0,
    )

    state = wd.check()

    assert state.safe_mode is False
    assert bus.events == []          # nothing published
    assert notify.messages == []     # no alert


# --------------------------------------------------------------------------- #
# 3) Broker/data loss + open position -> routes into the DMS                   #
# --------------------------------------------------------------------------- #
def test_connectivity_loss_with_position_routes_into_dms_flatten():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()

    pos = FakePosition(position_id="pos1", symbol="QQQ", side="buy", qty=5)
    orders = [FakeOrder(order_id="stop1", symbol="QQQ", side="sell")]

    # Data link down, broker REACHABLE. Heartbeat is fresh (engine is alive; only
    # the broker/data feed dropped).
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: clock(),  # always fresh
        connectivity=lambda: {"broker": True, "data": False},
        positions=[pos], working_orders=orders,
        flatten=flatten, cancel=cancel,
        heartbeat_timeout=15.0, dms_timeout=30.0,
    )

    # First tick records lost_since; DMS not yet armed (0s elapsed).
    wd.check()
    assert wd.state.lost_since == start
    assert wd.state.dms_fired is False
    assert flatten.flattened == []

    # Advance PAST dms_timeout and tick again -> DMS fires and FLATTENS.
    clock.advance(31)
    state = wd.check()

    assert state.dms_fired is True
    assert state.last_action.action == ACTION_FLATTEN
    assert flatten.flattened == ["QQQ"]
    assert cancel.cancelled == ["stop1"]
    dms = bus.of_type(EventType.DEAD_MANS_SWITCH_TRIPPED)
    assert dms and dms[0].data["action"] == ACTION_FLATTEN
    assert state.safe_mode is False  # engine heartbeat was fresh


def test_connectivity_loss_broker_unreachable_routes_into_dms_halt():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()
    pos = FakePosition(position_id="pos1", symbol="QQQ", side="buy", qty=5)

    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: clock(),
        connectivity=lambda: {"broker": False, "data": False},  # fully dark
        positions=[pos], flatten=flatten, cancel=cancel,
        heartbeat_timeout=15.0, dms_timeout=30.0,
    )

    wd.check()              # mark loss
    clock.advance(31)
    state = wd.check()      # fire

    assert state.dms_fired is True
    assert state.last_action.action == ACTION_ALERT_AND_HALT
    assert flatten.flattened == []  # no phantom flatten
    assert bus.of_type(EventType.CIRCUIT_BREAKER_TRIPPED)
    dms = bus.of_type(EventType.DEAD_MANS_SWITCH_TRIPPED)
    assert dms and dms[0].data["action"] == ACTION_ALERT_AND_HALT


def test_recovered_connectivity_resets_loss_window():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    pos = FakePosition(position_id="pos1", symbol="QQQ", side="buy", qty=5)

    conn = {"broker": True, "data": False}
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: clock(),
        connectivity=lambda: conn,
        positions=[pos], dms_timeout=30.0,
    )

    wd.check()
    assert wd.state.lost_since == start

    # Link recovers BEFORE the timeout -> the loss window resets, no DMS fire.
    clock.advance(10)
    conn["data"] = True
    state = wd.check()
    assert state.lost_since is None
    assert state.dms_fired is False
    assert bus.of_type(EventType.DEAD_MANS_SWITCH_TRIPPED) == []


# --------------------------------------------------------------------------- #
# 4) arm() satisfies the state machine's DMS requirement                      #
# --------------------------------------------------------------------------- #
def test_arm_satisfies_state_machine_requirement():
    """An armed watchdog makes OrderBook.assert_dead_mans_switch_armed pass."""
    from orderbook.state_machine import OrderBook

    ob = OrderBook(db_path=":memory:")
    # Plant a live bracket so the orderbook REQUIRES the switch.
    e = ob.create_order("SPY", "buy", 10, order_type="market")
    s = ob.create_order("SPY", "sell", 10, order_type="stop", stop_price=99.0)
    t = ob.create_order("SPY", "sell", 10, order_type="limit", limit_price=110.0)
    ob.create_bracket("SPY", e, s, t)
    assert ob.requires_dead_mans_switch is True

    clock = FakeClock(datetime(2026, 6, 13, 14, 30, 0))
    wd = _make_watchdog(
        clock=clock, bus=FakeBus(), notify=NotifySink(),
        heartbeat_source=lambda: clock(),
        connectivity=lambda: {"broker": True, "data": True},
        positions=[],
    )

    # Before arming: the requirement is NOT satisfied.
    try:
        ob.assert_dead_mans_switch_armed(wd.armed)
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True

    # After arming: it passes.
    wd.arm()
    assert wd.armed is True
    wd.assert_armed(ob)  # does not raise


# --------------------------------------------------------------------------- #
# 5) run() polls check() for a bounded number of ticks (process loop)         #
# --------------------------------------------------------------------------- #
def test_run_loop_polls_check_without_real_sleep():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    wd = _make_watchdog(
        clock=clock, bus=bus, notify=notify,
        heartbeat_source=lambda: clock(),  # fresh -> no trips
        connectivity=lambda: {"broker": True, "data": True},
        positions=[],
    )

    sleeps: list[float] = []
    state = wd.run(max_ticks=3, sleep=lambda s: sleeps.append(s))

    assert state.ticks == 3
    # Slept BETWEEN ticks only (n-1 sleeps), never after the last.
    assert len(sleeps) == 2
    assert all(s == wd.poll_interval for s in sleeps)
