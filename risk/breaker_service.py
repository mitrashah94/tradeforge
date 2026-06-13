"""risk/breaker_service.py — deterministic, bus-driven circuit-breaker service.

This is the P3 wrapper around the pure-function halt checks in
``risk/breakers.py``. The pure functions decide *whether a rule is breached*
given explicit numbers; this service keeps the running state (daily / weekly /
monthly realized PnL, drawdown-from-peak), reacts to bus events, and publishes
the trip / cooldown events the rest of the system listens for.

Design rules (MASTER_PLAN.md §2, §4 / CLAUDE.md):
  * Deterministic, pure-Python, NO LLM, NO network.
  * No ``datetime.now`` in the deterministic path — every period boundary is
    derived from the *event's* ``ts_utc``. Replaying the same event log always
    produces the same trips, in the same order.
  * The breaker RULES live unchanged in ``risk/breakers.py``; this module only
    feeds them state and wires them to the bus.
  * Daily / weekly halts are scoped to the risk dial and reset next period.
  * Program-abort (-20% calendar month -> review, -35% from all-time peak ->
    hard halt) is NOT on the dial and the hard halt never auto-clears.

Bus contract (a sibling agent owns ``bus.py`` / ``events.py``; we depend only on
this interface and the bus is INJECTED so tests can use a fake in-memory bus):
  * ``bus.subscribe(types, handler)`` — ``types`` is an iterable of event-type
    values; ``handler(event)`` is called for each matching published event.
  * ``bus.publish(event)`` — publishes an event object.
  * ``event`` is duck-typed: ``event.type`` (an event-type value), ``event.data``
    (a dict), ``event.ts_utc`` (a ``datetime``). ``seq`` / ``source`` are
    optional and ignored by the breaker logic.

Consumes : ORDER_FILLED, POSITION_CLOSED (realized PnL),
           EQUITY_UPDATE, DEPOSIT_LOGGED (equity / peak / drawdown).
Emits    : CIRCUIT_BREAKER_TRIPPED, COOLDOWN_STARTED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

from risk.breakers import (
    daily_halt_breached,
    program_abort_state,
    weekly_halt_breached,
)
from risk.config import Limits, load_limits

# --- Event-type name constants -------------------------------------------------
# These match the catalog member names in orchestrator/events.py (whose Enum
# values equal their names). Using bare strings keeps the service decoupled from
# the sibling's concrete Enum class: a string, an ``Enum`` member whose value is
# the string, all compare/normalise the same way (see ``_type_value``).
ORDER_FILLED = "ORDER_FILLED"
POSITION_CLOSED = "POSITION_CLOSED"
EQUITY_UPDATE = "EQUITY_UPDATE"
DEPOSIT_LOGGED = "DEPOSIT_LOGGED"

CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"
COOLDOWN_STARTED = "COOLDOWN_STARTED"

# Trip scopes published in CIRCUIT_BREAKER_TRIPPED.data["scope"].
SCOPE_DAILY = "daily"
SCOPE_WEEKLY = "weekly"
SCOPE_MONTHLY = "monthly"
SCOPE_PROGRAM = "program"

# Trip "level" — how hard the stop is.
LEVEL_SCOPED = "scoped"  # daily/weekly: clears next period
LEVEL_REVIEW = "review"  # monthly -20%: mandatory review, soft flag
LEVEL_HARD = "hard"  # program -35% from peak: manual restart only


def _type_value(t: Any) -> str:
    """Normalise an event-type (str or Enum member) to its string value."""
    return getattr(t, "value", t)


def _as_date(ts: datetime) -> date:
    """The calendar date of an event timestamp."""
    return ts.date()


def _iso_week(d: date) -> tuple[int, int]:
    """ISO (year, week) key — the calendar week a date falls in (Mon-anchored)."""
    iso = d.isocalendar()
    return (iso[0], iso[1])


def _month_key(d: date) -> tuple[int, int]:
    """Calendar (year, month) key."""
    return (d.year, d.month)


@dataclass
class _SimpleEvent:
    """Minimal event object used to PUBLISH when no factory is injected.

    Mirrors the duck-typed bus contract: ``type``, ``data``, ``ts_utc``,
    ``seq``, ``source``. The sibling's real ``Event`` works too — inject an
    ``event_factory`` to use it.
    """

    type: str
    data: dict
    ts_utc: datetime
    seq: int | None = None
    source: str = "breaker_service"


@dataclass
class BreakerState:
    """All running state the breaker service derives from the event stream.

    Everything here is a pure function of the events seen so far (no clocks),
    so replaying an event log reconstructs identical state.
    """

    equity: float | None = None
    equity_peak: float | None = None  # all-time high water mark
    month_start_equity: float | None = None  # equity at the first event of the month

    day_key: date | None = None
    week_key: tuple[int, int] | None = None
    month_key: tuple[int, int] | None = None

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

    consecutive_losses: int = 0

    # Latched flags. Scoped halts clear on the next period boundary; the hard
    # program halt never auto-clears.
    daily_halted: bool = False
    weekly_halted: bool = False
    monthly_review: bool = False
    program_halted: bool = False  # HARD: manual restart only

    cooldown_active: bool = False

    halt_reasons: dict[str, str] = field(default_factory=dict)


class BreakerService:
    """Stateful, bus-driven circuit-breaker service.

    Subscribe it to the bus (it does so itself in ``__init__`` when ``subscribe``
    is True), then feed it events. On a breach it publishes the matching event
    and latches an internal flag the order gateway can query. The gateway may
    *either* poll the query methods (:meth:`is_halted`, :meth:`halt_reason`,
    :meth:`in_cooldown`) *or* subscribe to the trip / cooldown events — both are
    supported and consistent.

    Parameters
    ----------
    bus:
        Injected event bus with ``subscribe(types, handler)`` / ``publish(event)``.
    limits:
        Validated :class:`~risk.config.Limits`. Defaults to ``load_limits()``.
    ri:
        Current risk index (the dial). Daily/weekly thresholds read from this row.
    cooldown_losses:
        Number of *consecutive* losing trades that triggers a COOLDOWN_STARTED
        stand-down (default 3). This is softer than a halt.
    event_factory:
        Optional callable ``(type, data, ts_utc) -> event`` used to build events
        to publish. Defaults to an internal :class:`_SimpleEvent`. Inject the
        sibling's real ``Event`` constructor in production.
    subscribe:
        If True (default), the service subscribes itself to the consumed event
        types on construction.
    """

    def __init__(
        self,
        bus: Any,
        *,
        limits: Limits | None = None,
        ri: int | None = None,
        cooldown_losses: int = 3,
        event_factory: Callable[[Any, dict, datetime], Any] | None = None,
        subscribe: bool = True,
    ) -> None:
        self.bus = bus
        self.limits = limits if limits is not None else load_limits()
        self.ri = ri if ri is not None else self.limits.default_ri
        self.cooldown_losses = cooldown_losses
        self._event_factory = event_factory or _SimpleEvent
        self.state = BreakerState()

        self._consumes = (ORDER_FILLED, POSITION_CLOSED, EQUITY_UPDATE, DEPOSIT_LOGGED)
        if subscribe:
            self.bus.subscribe(list(self._consumes), self.handle)

    # --- bus entry point ------------------------------------------------------
    def handle(self, event: Any) -> None:
        """Single handler for every consumed event. Order: roll periods (from the
        event ts), fold in the event, then evaluate breakers."""
        etype = _type_value(event.type)
        ts: datetime = event.ts_utc
        data: dict = getattr(event, "data", {}) or {}

        self._roll_periods(ts)

        if etype == DEPOSIT_LOGGED:
            self._apply_deposit(data)
        elif etype in (ORDER_FILLED, POSITION_CLOSED):
            self._apply_realized(data)
        elif etype == EQUITY_UPDATE:
            self._apply_equity_update(data)

        self._evaluate(ts)

    # --- period bookkeeping ---------------------------------------------------
    def _roll_periods(self, ts: datetime) -> None:
        """Reset day/week/month accumulators (and clear scoped halts) when the
        event's timestamp crosses the corresponding calendar boundary.

        All boundaries come from the event timestamp — never a wall clock — so
        the deterministic path stays replayable.
        """
        d = _as_date(ts)
        wk = _iso_week(d)
        mo = _month_key(d)

        if self.state.day_key != d:
            self.state.day_key = d
            self.state.daily_pnl = 0.0
            self.state.daily_halted = False  # scoped: a new day clears the daily halt

        if self.state.week_key != wk:
            self.state.week_key = wk
            self.state.weekly_pnl = 0.0
            self.state.weekly_halted = False  # scoped: a new week clears it

        if self.state.month_key != mo:
            self.state.month_key = mo
            self.state.monthly_pnl = 0.0
            self.state.monthly_review = False  # the -20% review flag is monthly-scoped
            # Anchor the month's starting equity to current equity (if known) so
            # the -20%-in-a-calendar-month drawdown is measured from the month's
            # open, independent of realized-PnL accounting.
            self.state.month_start_equity = self.state.equity

    # --- folding events into state -------------------------------------------
    def _apply_realized(self, data: dict) -> None:
        """Fold a realized-PnL event (ORDER_FILLED / POSITION_CLOSED) into state.

        Reads ``data["realized_pnl"]`` (alias ``pnl``). A fill with no realized
        component (an opening fill) contributes 0 and does not affect the
        consecutive-loss counter.
        """
        pnl = data.get("realized_pnl", data.get("pnl"))
        if pnl is None:
            return
        pnl = float(pnl)

        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.monthly_pnl += pnl

        if self.state.equity is not None:
            self.state.equity += pnl
            self._touch_peak()

        # Consecutive-loss counter for the cooldown.
        if pnl < 0:
            self.state.consecutive_losses += 1
        elif pnl > 0:
            self.state.consecutive_losses = 0
        # pnl == 0 (scratch) leaves the streak unchanged.

    def _apply_deposit(self, data: dict) -> None:
        """Fold a DEPOSIT_LOGGED event: deposits add to equity but are NOT PnL
        (they must not move the loss accumulators or the streak)."""
        amount = float(data.get("amount", data.get("equity_delta", 0.0)))
        if self.state.equity is None:
            self.state.equity = amount
        else:
            self.state.equity += amount
        # A deposit lifts the month-start baseline too, so injected cash is not
        # mistaken for a recovery / drawdown against the month.
        if self.state.month_start_equity is not None:
            self.state.month_start_equity += amount
        self._touch_peak()

    def _apply_equity_update(self, data: dict) -> None:
        """Fold a mark-to-market EQUITY_UPDATE: set absolute equity and refresh
        the peak (so drawdown-from-peak tracks unrealized swings)."""
        if "equity" in data:
            self.state.equity = float(data["equity"])
        elif "equity_delta" in data:
            base = self.state.equity or 0.0
            self.state.equity = base + float(data["equity_delta"])
        else:
            return
        if self.state.month_start_equity is None:
            self.state.month_start_equity = self.state.equity
        self._touch_peak()

    def _touch_peak(self) -> None:
        """Maintain the all-time equity high-water mark."""
        if self.state.equity is None:
            return
        if self.state.equity_peak is None or self.state.equity > self.state.equity_peak:
            self.state.equity_peak = self.state.equity

    # --- breaker evaluation ---------------------------------------------------
    def _pct(self, pnl: float) -> float:
        """Signed PnL as a % of the reference equity (the equity *before* the
        period's losses, approximated by current equity less the period PnL).

        Pure-function breakers expect a signed pnl-% of equity. We use the
        period-start equity (current equity minus this period's PnL) as the base
        so the threshold is measured against capital at risk, not post-loss
        capital. Falls back gracefully when equity is unknown.
        """
        if self.state.equity is None:
            return 0.0
        base = self.state.equity - pnl
        if base <= 0:
            base = self.state.equity if self.state.equity > 0 else 1.0
        return 100.0 * pnl / base

    def _evaluate(self, ts: datetime) -> None:
        """Run the pure-function breakers against current state and trip on any
        newly-breached, not-yet-latched rule. Program-abort is checked first
        (it is the hardest stop and independent of the dial)."""
        # --- Program-abort (always on, not on the dial) -----------------------
        month_dd = self._month_drawdown_pct()
        peak_dd = self._peak_drawdown_pct()
        pa = program_abort_state(month_dd, peak_dd, self.limits)

        if pa == "halt" and not self.state.program_halted:
            self.state.program_halted = True
            self.state.halt_reasons[SCOPE_PROGRAM] = (
                f"peak drawdown {peak_dd:.2f}% >= "
                f"{self.limits.program_abort.peak_halt_drawdown_pct}% "
                "— HARD halt, manual restart required"
            )
            self._trip(SCOPE_PROGRAM, LEVEL_HARD, self.state.halt_reasons[SCOPE_PROGRAM], ts)
        if pa == "review" and not self.state.monthly_review:
            self.state.monthly_review = True
            reason = (
                f"calendar-month drawdown {month_dd:.2f}% >= "
                f"{self.limits.program_abort.monthly_review_drawdown_pct}% "
                "— mandatory review"
            )
            self.state.halt_reasons[SCOPE_MONTHLY] = reason
            self._trip(SCOPE_MONTHLY, LEVEL_REVIEW, reason, ts)

        # --- Daily / weekly scoped halts (on the dial) ------------------------
        if not self.state.daily_halted and daily_halt_breached(
            self._pct(self.state.daily_pnl), self.ri, self.limits
        ):
            self.state.daily_halted = True
            reason = (
                f"daily loss {-self._pct(self.state.daily_pnl):.2f}% >= "
                f"RI{self.ri} daily_halt {self.limits.level(self.ri).daily_halt_pct}%"
            )
            self.state.halt_reasons[SCOPE_DAILY] = reason
            self._trip(SCOPE_DAILY, LEVEL_SCOPED, reason, ts)

        if not self.state.weekly_halted and weekly_halt_breached(
            self._pct(self.state.weekly_pnl), self.ri, self.limits
        ):
            self.state.weekly_halted = True
            reason = (
                f"weekly loss {-self._pct(self.state.weekly_pnl):.2f}% >= "
                f"RI{self.ri} weekly_halt {self.limits.level(self.ri).weekly_halt_pct}%"
            )
            self.state.halt_reasons[SCOPE_WEEKLY] = reason
            self._trip(SCOPE_WEEKLY, LEVEL_SCOPED, reason, ts)

        # --- Cooldown (softer than a halt) ------------------------------------
        if (
            not self.state.cooldown_active
            and self.state.consecutive_losses >= self.cooldown_losses
        ):
            self.state.cooldown_active = True
            self._cooldown(self.state.consecutive_losses, ts)

    def _month_drawdown_pct(self) -> float:
        """Positive % drawdown within the current calendar month, measured from
        the month-start equity. 0 if unknown or in profit for the month."""
        base = self.state.month_start_equity
        if base is None or base <= 0 or self.state.equity is None:
            return 0.0
        dd = (base - self.state.equity) / base * 100.0
        return dd if dd > 0 else 0.0

    def _peak_drawdown_pct(self) -> float:
        """Positive % drawdown from the all-time equity peak. 0 if at/above peak
        or unknown."""
        peak = self.state.equity_peak
        if peak is None or peak <= 0 or self.state.equity is None:
            return 0.0
        dd = (peak - self.state.equity) / peak * 100.0
        return dd if dd > 0 else 0.0

    # --- publishing -----------------------------------------------------------
    def _trip(self, scope: str, level: str, reason: str, ts: datetime) -> None:
        ev = self._event_factory(
            CIRCUIT_BREAKER_TRIPPED,
            {"scope": scope, "reason": reason, "level": level},
            ts,
        )
        self.bus.publish(ev)

    def _cooldown(self, losses: int, ts: datetime) -> None:
        ev = self._event_factory(
            COOLDOWN_STARTED,
            {
                "reason": f"{losses} consecutive losing trades",
                "consecutive_losses": losses,
                "trigger": self.cooldown_losses,
            },
            ts,
        )
        self.bus.publish(ev)

    # --- queryable state (for the order gateway) ------------------------------
    def is_halted(self) -> bool:
        """True if ANY halt is active: a scoped daily/weekly halt or the hard
        program halt. The order gateway must refuse approvals while True.

        Note: the monthly -20% review flag is a soft flag, not a hard stop, so it
        is intentionally NOT counted here (use :meth:`needs_review`)."""
        return (
            self.state.daily_halted
            or self.state.weekly_halted
            or self.state.program_halted
        )

    def halt_reason(self) -> str | None:
        """Human-readable reason for the current halt, hardest scope first
        (program > weekly > daily), or None if not halted."""
        if self.state.program_halted:
            return self.state.halt_reasons.get(SCOPE_PROGRAM)
        if self.state.weekly_halted:
            return self.state.halt_reasons.get(SCOPE_WEEKLY)
        if self.state.daily_halted:
            return self.state.halt_reasons.get(SCOPE_DAILY)
        return None

    def in_cooldown(self) -> bool:
        """True while a consecutive-loss cooldown stand-down is active."""
        return self.state.cooldown_active

    def needs_review(self) -> bool:
        """True if the -20%-calendar-month review flag is raised (soft; not a
        hard stop)."""
        return self.state.monthly_review

    def is_program_halted(self) -> bool:
        """True if the HARD program-abort halt is latched (never auto-clears)."""
        return self.state.program_halted

    # --- manual lifecycle -----------------------------------------------------
    def clear_cooldown(self) -> None:
        """Clear an active cooldown (e.g. after the timed stand-down elapses).
        Also resets the consecutive-loss streak so the next loss starts fresh."""
        self.state.cooldown_active = False
        self.state.consecutive_losses = 0

    def manual_restart(self) -> None:
        """Operator-only: clear the HARD program halt. The deterministic path
        never calls this — a -35%-from-peak halt requires a human restart per
        MASTER_PLAN.md §2."""
        self.state.program_halted = False
        self.state.halt_reasons.pop(SCOPE_PROGRAM, None)
