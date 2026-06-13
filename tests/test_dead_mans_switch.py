"""tests/test_dead_mans_switch.py — the MANDATORY dead-man's switch (MASTER_PLAN §3, §7).

Brackets are LOCAL, so a lost broker/data connection with an OPEN position means
the local stop/target can no longer be managed. The dead-man's switch must act
WITHIN ``dms_timeout``:

  - broker REACHABLE  -> cancel working orders then FLATTEN (market close) every
    open position; emit DEAD_MANS_SWITCH_TRIPPED(action=flatten); positions flat.
  - broker UNREACHABLE -> ALERT-AND-HALT: publish a HARD CIRCUIT_BREAKER_TRIPPED +
    DEAD_MANS_SWITCH_TRIPPED(action=alert_and_halt); NO phantom flatten claimed.

And it must NOT fire when connectivity is healthy or no position is open.

Deterministic, offline. Clock is injected; bus + notify are fakes (no real
iMessage, no real DB). No engine, no MCP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from orchestrator.dead_mans_switch import (
    ACTION_ALERT_AND_HALT,
    ACTION_FLATTEN,
    ACTION_NOOP,
    dead_mans_switch,
)
from orchestrator.events import Event, EventType


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
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
    """Captures published events (duck-typed: only needs ``publish``)."""

    def __init__(self):
        self.events: list[Event] = []

    def publish(self, event: Event) -> Event:
        self.events.append(event)
        return event

    def of_type(self, etype: EventType) -> list[Event]:
        return [e for e in self.events if e.type == etype]


class FakeClock:
    """Monotonic injected clock; ``advance(seconds)`` moves it forward."""

    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


class Flattener:
    """Records flattens; can be told to FAIL (broker can't confirm a close)."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.flattened: list[str] = []

    def __call__(self, position) -> dict:
        if self.fail:
            raise RuntimeError("broker refused / could not confirm close")
        self.flattened.append(position.symbol)
        return {"status": "filled", "symbol": position.symbol}


class Canceller:
    def __init__(self):
        self.cancelled: list[str] = []

    def __call__(self, order) -> dict:
        self.cancelled.append(order.order_id)
        return {"status": "cancelled", "order_id": order.order_id}


class NotifySink:
    """In-test notify fake — NEVER touches iMessage or the filesystem."""

    def __init__(self):
        self.messages: list[str] = []

    def __call__(self, message, *, title=None, channel=None, **kw) -> str:
        text = f"[{title}] {message}" if title else message
        self.messages.append(text)
        return text


# A minimal stand-in for risk.config.Limits (the switch only threads it through).
class FakeLimits:
    default_ri = 5


def _open_position():
    return FakePosition(position_id="pos1", symbol="SPY", side="buy", qty=10)


def _working_orders():
    return [
        FakeOrder(order_id="stop1", symbol="SPY", side="sell"),
        FakeOrder(order_id="tgt1", symbol="SPY", side="sell"),
    ]


# --------------------------------------------------------------------------- #
# 1) Broker REACHABLE -> cancel + flatten within the timeout                   #
# --------------------------------------------------------------------------- #
def test_dms_flattens_when_broker_reachable():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()

    # Connection lost (data link dark) at t0; broker still reachable.
    lost_since = start
    conn = {"broker": True, "data": False}

    # Advance the clock PAST dms_timeout so the switch is armed.
    clock.advance(31)

    action = dead_mans_switch(
        open_positions=[_open_position()],
        working_orders=_working_orders(),
        connectivity=conn,
        ri=5,
        limits=FakeLimits(),
        flatten_fn=flatten,
        cancel_fn=cancel,
        notify_fn=notify,
        clock=clock,
        lost_since=lost_since,
        dms_timeout=30.0,
        bus=bus,
    )

    # Fired, flattened, cancelled both protective legs.
    assert action.fired is True
    assert action.action == ACTION_FLATTEN
    assert flatten.flattened == ["SPY"]
    assert action.flattened == ["SPY"]
    assert set(cancel.cancelled) == {"stop1", "tgt1"}
    assert set(action.cancelled_orders) == {"stop1", "tgt1"}
    assert action.halted is False

    # Decision+action completed WITHIN the timeout (instantaneous on the fake clock).
    assert action.within_timeout is True
    assert action.elapsed_s <= 30.0

    # DEAD_MANS_SWITCH_TRIPPED(action=flatten) emitted; no hard halt published.
    dms_evts = bus.of_type(EventType.DEAD_MANS_SWITCH_TRIPPED)
    assert len(dms_evts) == 1
    assert dms_evts[0].data["action"] == ACTION_FLATTEN
    assert dms_evts[0].data["flattened"] == ["SPY"]
    assert bus.of_type(EventType.CIRCUIT_BREAKER_TRIPPED) == []  # flatten, not halt
    assert notify.messages  # a human alert was sent (to the in-test sink)


# --------------------------------------------------------------------------- #
# 2) Broker UNREACHABLE -> alert-and-halt, no phantom flatten                  #
# --------------------------------------------------------------------------- #
def test_dms_alert_and_halt_when_broker_unreachable():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()

    # Broker AND data both down — we cannot confirm a flatten.
    conn = {"broker": False, "data": False}
    clock.advance(45)  # well past dms_timeout

    action = dead_mans_switch(
        open_positions=[_open_position()],
        working_orders=_working_orders(),
        connectivity=conn,
        ri=5,
        limits=FakeLimits(),
        flatten_fn=flatten,
        cancel_fn=cancel,
        notify_fn=notify,
        clock=clock,
        lost_since=start,
        dms_timeout=30.0,
        bus=bus,
    )

    assert action.fired is True
    assert action.action == ACTION_ALERT_AND_HALT
    assert action.halted is True
    # NO phantom flatten: nothing was actually closed and none is CLAIMED.
    assert flatten.flattened == []
    assert action.flattened == []
    assert action.within_timeout is True

    # A HARD halt is published (gateway latches on CIRCUIT_BREAKER_TRIPPED) ...
    cb = bus.of_type(EventType.CIRCUIT_BREAKER_TRIPPED)
    assert len(cb) == 1
    assert cb[0].data["kind"] == "dead_mans_switch_halt"
    assert "reason" in cb[0].data
    # ... and DEAD_MANS_SWITCH_TRIPPED records the alert_and_halt action.
    dms_evts = bus.of_type(EventType.DEAD_MANS_SWITCH_TRIPPED)
    assert len(dms_evts) == 1
    assert dms_evts[0].data["action"] == ACTION_ALERT_AND_HALT
    assert dms_evts[0].data["open_symbols"] == ["SPY"]
    assert notify.messages


# --------------------------------------------------------------------------- #
# 3) Flatten that FAILS mid-way degrades to alert-and-halt (no false claim)    #
# --------------------------------------------------------------------------- #
def test_dms_failed_flatten_degrades_to_halt():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    # Broker says reachable, but the flatten call raises (could not confirm).
    flatten, cancel = Flattener(fail=True), Canceller()
    clock.advance(31)

    action = dead_mans_switch(
        open_positions=[_open_position()],
        working_orders=_working_orders(),
        connectivity={"broker": True, "data": False},
        ri=5,
        limits=FakeLimits(),
        flatten_fn=flatten,
        cancel_fn=cancel,
        notify_fn=notify,
        clock=clock,
        lost_since=start,
        dms_timeout=30.0,
        bus=bus,
    )

    assert action.fired is True
    assert action.action == ACTION_ALERT_AND_HALT
    assert action.halted is True
    assert action.flattened == []  # nothing confirmed -> nothing claimed
    assert bus.of_type(EventType.CIRCUIT_BREAKER_TRIPPED)  # hard halt published


# --------------------------------------------------------------------------- #
# 4) Does NOT fire when healthy / no position / loss not yet past timeout      #
# --------------------------------------------------------------------------- #
def test_dms_noop_when_connectivity_healthy():
    clock = FakeClock(datetime(2026, 6, 13, 14, 30, 0))
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()

    action = dead_mans_switch(
        open_positions=[_open_position()],
        working_orders=_working_orders(),
        connectivity={"broker": True, "data": True},  # healthy
        ri=5, limits=FakeLimits(),
        flatten_fn=flatten, cancel_fn=cancel, notify_fn=notify, clock=clock,
        lost_since=None, dms_timeout=30.0, bus=bus,
    )
    assert action.fired is False
    assert action.action == ACTION_NOOP
    assert flatten.flattened == []
    assert bus.events == []


def test_dms_noop_when_no_open_position():
    clock = FakeClock(datetime(2026, 6, 13, 14, 30, 0))
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()

    action = dead_mans_switch(
        open_positions=[],  # nothing to protect
        working_orders=[],
        connectivity={"broker": False, "data": False},  # lost, but no position
        ri=5, limits=FakeLimits(),
        flatten_fn=flatten, cancel_fn=cancel, notify_fn=notify, clock=clock,
        lost_since=datetime(2026, 6, 13, 14, 0, 0), dms_timeout=30.0, bus=bus,
    )
    assert action.fired is False
    assert action.action == ACTION_NOOP
    assert bus.events == []


def test_dms_noop_before_timeout_elapses():
    start = datetime(2026, 6, 13, 14, 30, 0)
    clock = FakeClock(start)
    bus, notify = FakeBus(), NotifySink()
    flatten, cancel = Flattener(), Canceller()

    clock.advance(10)  # only 10s of loss; dms_timeout is 30s -> not yet armed

    action = dead_mans_switch(
        open_positions=[_open_position()],
        working_orders=_working_orders(),
        connectivity={"broker": True, "data": False},
        ri=5, limits=FakeLimits(),
        flatten_fn=flatten, cancel_fn=cancel, notify_fn=notify, clock=clock,
        lost_since=start, dms_timeout=30.0, bus=bus,
    )
    assert action.fired is False
    assert action.action == ACTION_NOOP
    assert flatten.flattened == []
    assert bus.events == []
