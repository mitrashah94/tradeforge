"""Tests for the deterministic, bus-driven BreakerService (risk/breaker_service.py).

Everything here is offline and deterministic: a FAKE in-memory bus is injected,
and every period boundary is driven by the event's ``ts_utc`` (never a wall
clock), so the same event sequence always trips the same breakers in the same
order.

These tests also assert the EXISTING pure functions in risk/breakers.py are
unchanged and still pass (see the bottom section).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from risk import breakers as breakers_mod
from risk.breaker_service import (
    CIRCUIT_BREAKER_TRIPPED,
    COOLDOWN_STARTED,
    DEPOSIT_LOGGED,
    EQUITY_UPDATE,
    ORDER_FILLED,
    POSITION_CLOSED,
    SCOPE_DAILY,
    SCOPE_MONTHLY,
    SCOPE_PROGRAM,
    SCOPE_WEEKLY,
    BreakerService,
)
from risk.breakers import (
    daily_halt_breached,
    program_abort_state,
    weekly_halt_breached,
)
from risk.config import load_limits


# --- Fake bus + event ---------------------------------------------------------
@dataclass
class FakeEvent:
    """Mirror of the duck-typed bus event contract."""

    type: str
    data: dict
    ts_utc: datetime
    seq: int | None = None
    source: str = "test"


class FakeBus:
    """In-memory pub/sub bus for deterministic, offline tests."""

    def __init__(self) -> None:
        self._subs: list[tuple[set, object]] = []
        self.published: list[FakeEvent] = []

    def subscribe(self, types, handler) -> None:
        self._subs.append(({getattr(t, "value", t) for t in types}, handler))

    def publish(self, event) -> None:
        self.published.append(event)
        etype = getattr(event.type, "value", event.type)
        for types, handler in self._subs:
            if etype in types:
                handler(event)

    # convenience for assertions
    def of_type(self, etype: str) -> list[FakeEvent]:
        return [e for e in self.published if getattr(e.type, "value", e.type) == etype]

    def trips(self, scope: str | None = None) -> list[FakeEvent]:
        evs = self.of_type(CIRCUIT_BREAKER_TRIPPED)
        if scope is not None:
            evs = [e for e in evs if e.data.get("scope") == scope]
        return evs


# --- helpers ------------------------------------------------------------------
RI = 5  # default floor; daily_halt 2%, weekly_halt 5%, per the limits table

BASE_TS = datetime(2026, 6, 8, 14, 30)  # a Monday, RTH


def make_service(bus: FakeBus, *, ri: int = RI, cooldown_losses: int = 3,
                 start_equity: float = 10_000.0) -> BreakerService:
    svc = BreakerService(
        bus,
        limits=load_limits(),
        ri=ri,
        cooldown_losses=cooldown_losses,
        event_factory=FakeEvent,
    )
    # seed equity / peak via an EQUITY_UPDATE so drawdown math has a base
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": start_equity}, BASE_TS))
    return svc


def fill(bus: FakeBus, pnl: float, ts: datetime, etype: str = ORDER_FILLED) -> None:
    bus.publish(FakeEvent(etype, {"realized_pnl": pnl}, ts))


# ============================================================================
# Daily halt
# ============================================================================
def test_daily_halt_trips_once_and_sets_halted():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)

    # RI5 daily_halt = 2% of 10k = $200. Feed losers crossing it.
    fill(bus, -80, BASE_TS)
    assert not svc.is_halted()
    fill(bus, -80, BASE_TS)
    assert not svc.is_halted()  # -160 still under -200
    fill(bus, -80, BASE_TS)  # -240 -> breach
    assert svc.is_halted()

    daily_trips = bus.trips(SCOPE_DAILY)
    assert len(daily_trips) == 1
    assert daily_trips[0].data["level"] == "scoped"
    assert "daily" in daily_trips[0].data["reason"]

    # No double-trip: more losses on the same day don't re-publish.
    fill(bus, -500, BASE_TS)
    assert len(bus.trips(SCOPE_DAILY)) == 1
    assert svc.halt_reason() is not None


def test_daily_halt_not_tripped_by_gains():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)
    for _ in range(10):
        fill(bus, +500, BASE_TS)
    assert not svc.is_halted()
    assert bus.trips(SCOPE_DAILY) == []


def test_daily_halt_resets_next_day():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)

    fill(bus, -250, BASE_TS)  # breach day 1
    assert svc.is_halted()

    next_day = BASE_TS + timedelta(days=1)
    # A fresh event on the next calendar day rolls the period and clears the
    # scoped daily halt (the gateway sees is_halted go False).
    fill(bus, -10, next_day)
    assert not svc.state.daily_halted
    # weekly may still accumulate, but it hasn't breached weekly (5%).
    assert not svc.is_halted()


# ============================================================================
# Weekly halt
# ============================================================================
def test_weekly_halt_trips_across_days_within_week():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)

    # RI5 weekly_halt = 5% of 10k = $500. Spread losses across days so the daily
    # halt doesn't fire first on a single day (each day -$150 < $200 daily cap).
    fill(bus, -150, BASE_TS)                         # Mon
    fill(bus, -150, BASE_TS + timedelta(days=1))     # Tue
    fill(bus, -150, BASE_TS + timedelta(days=2))     # Wed
    assert not svc.state.weekly_halted               # -450 < 500
    fill(bus, -150, BASE_TS + timedelta(days=3))     # Thu -> -600 breach
    assert svc.state.weekly_halted
    assert svc.is_halted()

    weekly_trips = bus.trips(SCOPE_WEEKLY)
    assert len(weekly_trips) == 1
    assert weekly_trips[0].data["scope"] == "weekly"
    # No daily trip should have fired (each day stayed under the daily cap).
    assert bus.trips(SCOPE_DAILY) == []


def test_weekly_halt_resets_next_week():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)
    # breach weekly across the first week
    for i in range(4):
        fill(bus, -150, BASE_TS + timedelta(days=i))
    assert svc.state.weekly_halted

    next_week = BASE_TS + timedelta(days=7)  # following Monday
    fill(bus, -10, next_week)
    assert not svc.state.weekly_halted


# ============================================================================
# Monthly -20% review (soft flag, not a hard stop)
# ============================================================================
def test_monthly_review_trips_at_20pct_drawdown():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)

    # Drive a -20% calendar-month drawdown via a mark-to-market equity update
    # (so daily/weekly realized accumulators don't dominate the test).
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 8_000.0}, BASE_TS))  # -20% from 10k

    review_trips = bus.trips(SCOPE_MONTHLY)
    assert len(review_trips) == 1
    assert review_trips[0].data["level"] == "review"
    assert svc.needs_review() is True
    # A -20% month review is NOT a hard halt by itself.
    assert svc.is_program_halted() is False


def test_monthly_review_resets_next_month():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 8_000.0}, BASE_TS))
    assert svc.needs_review()

    # An event in the next calendar month re-anchors month-start to current
    # equity and clears the review flag.
    next_month = datetime(2026, 7, 1, 14, 30)
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 8_000.0}, next_month))
    assert svc.needs_review() is False


# ============================================================================
# Program hard halt (-35% from all-time peak), never auto-clears
# ============================================================================
def test_program_hard_halt_at_35pct_from_peak_does_not_auto_clear():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)

    # Push the peak up first, then crash -35% from that peak.
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 12_000.0}, BASE_TS))  # peak = 12k
    assert svc.state.equity_peak == 12_000.0

    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 7_800.0}, BASE_TS))  # -35% from 12k

    program_trips = bus.trips(SCOPE_PROGRAM)
    assert len(program_trips) == 1
    assert program_trips[0].data["level"] == "hard"
    assert svc.is_program_halted() is True
    assert svc.is_halted() is True

    # Recovery in equity must NOT auto-clear the hard halt.
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 12_500.0}, BASE_TS + timedelta(days=1)))
    assert svc.is_program_halted() is True
    assert svc.is_halted() is True
    # A new month also must NOT clear it.
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 13_000.0}, datetime(2026, 7, 1, 14, 30)))
    assert svc.is_program_halted() is True

    # Only an explicit operator restart clears it.
    svc.manual_restart()
    assert svc.is_program_halted() is False


def test_program_halt_only_trips_once():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 6_000.0}, BASE_TS))  # -40% from peak 10k
    bus.publish(FakeEvent(EQUITY_UPDATE, {"equity": 5_000.0}, BASE_TS))  # deeper
    assert len(bus.trips(SCOPE_PROGRAM)) == 1


# ============================================================================
# Cooldown: distinct from a halt, fires after N consecutive losers
# ============================================================================
def test_cooldown_fires_after_n_consecutive_losers_and_is_not_a_halt():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0, cooldown_losses=3)

    # Small losers that never approach the daily/weekly halt thresholds.
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    assert not svc.in_cooldown()
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)  # third consecutive loser

    assert svc.in_cooldown() is True
    cooldowns = bus.of_type(COOLDOWN_STARTED)
    assert len(cooldowns) == 1
    assert cooldowns[0].data["consecutive_losses"] == 3
    # Cooldown is softer than a halt — is_halted stays False.
    assert svc.is_halted() is False


def test_winner_resets_consecutive_loss_streak():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0, cooldown_losses=3)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    fill(bus, +20, BASE_TS, etype=POSITION_CLOSED)  # winner resets the streak
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    assert not svc.in_cooldown()  # only 2 in a row since the reset
    assert bus.of_type(COOLDOWN_STARTED) == []


def test_cooldown_only_fires_once_until_cleared():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0, cooldown_losses=2)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    fill(bus, -5, BASE_TS, etype=POSITION_CLOSED)
    assert len(bus.of_type(COOLDOWN_STARTED)) == 1
    svc.clear_cooldown()
    assert not svc.in_cooldown()


# ============================================================================
# Deposits are not PnL
# ============================================================================
def test_deposit_does_not_count_as_loss_or_streak():
    bus = FakeBus()
    svc = make_service(bus, start_equity=10_000.0)
    bus.publish(FakeEvent(DEPOSIT_LOGGED, {"amount": 50.0}, BASE_TS))
    assert svc.state.equity == 10_050.0
    assert svc.state.consecutive_losses == 0
    assert not svc.is_halted()
    # A deposit also lifts the peak; no spurious drawdown trip.
    assert bus.trips() == []


# ============================================================================
# Gateway integration: trip events reach a subscriber too (poll OR subscribe)
# ============================================================================
def test_gateway_can_subscribe_to_trip_events():
    bus = FakeBus()
    received: list[str] = []
    bus.subscribe([CIRCUIT_BREAKER_TRIPPED], lambda e: received.append(e.data["scope"]))
    svc = make_service(bus, start_equity=10_000.0)

    fill(bus, -250, BASE_TS)  # breach daily
    assert received == [SCOPE_DAILY]
    assert svc.is_halted()  # ...and the poll query agrees


# ============================================================================
# The existing pure functions are UNCHANGED and still behave as specified.
# ============================================================================
def test_pure_functions_unchanged_behaviour():
    limits = load_limits()
    # daily: RI5 threshold 2.0%
    assert daily_halt_breached(-2.0, 5, limits) is True
    assert daily_halt_breached(-1.99, 5, limits) is False
    assert daily_halt_breached(+5.0, 5, limits) is False  # a gain never halts
    # weekly: RI5 threshold 5.0%
    assert weekly_halt_breached(-5.0, 5, limits) is True
    assert weekly_halt_breached(-4.99, 5, limits) is False
    # program-abort ladder: peak halt takes precedence over monthly review
    assert program_abort_state(10.0, 35.0, limits) == "halt"
    assert program_abort_state(20.0, 10.0, limits) == "review"
    assert program_abort_state(5.0, 5.0, limits) == "ok"
    assert program_abort_state(40.0, 40.0, limits) == "halt"


def test_breakerservice_reexported_from_breakers_module():
    # ``from risk.breakers import BreakerService`` must work (same class).
    assert breakers_mod.BreakerService is BreakerService
