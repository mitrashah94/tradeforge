"""risk/breakers.py — deterministic, pure-function halt checks.

These are the protective circuit-breakers of the risk core. They are pure
functions of explicit inputs: no I/O, no event bus, no live-state reads yet.

In P3 this module will subscribe to the event bus and read live state
(realized/unrealized P&L, open heat, equity peak) to evaluate these breakers in
the hot path. For now the breaker logic itself lives here as deterministic,
testable predicates so the wiring can be added later without changing the rules.

Daily/weekly/heat breakers are tied to the risk dial via the RI table.
Program-abort is NOT on the risk dial — it is always-on insurance against bugs,
disconnects, and death-spirals, and is never adjusted by the dial.
"""

from __future__ import annotations

from risk.config import Limits


def daily_halt_breached(daily_pnl_pct: float, ri: int, limits: Limits) -> bool:
    """True if the day's loss has reached the RI's daily-halt threshold.

    `daily_pnl_pct` is signed (negative for a loss). The comparison is on the
    magnitude of the loss, so a gain never breaches.
    """
    loss = -daily_pnl_pct
    return loss >= limits.level(ri).daily_halt_pct


def weekly_halt_breached(weekly_pnl_pct: float, ri: int, limits: Limits) -> bool:
    """True if the week's loss has reached the RI's weekly-halt threshold.

    `weekly_pnl_pct` is signed (negative for a loss); compared on loss magnitude.
    """
    loss = -weekly_pnl_pct
    return loss >= limits.level(ri).weekly_halt_pct


def heat_breached(open_heat_pct: float, ri: int, limits: Limits) -> bool:
    """True if open portfolio heat exceeds the RI's portfolio-heat cap."""
    return open_heat_pct > limits.level(ri).portfolio_heat_pct


def program_abort_state(
    month_drawdown_pct: float, peak_drawdown_pct: float, limits: Limits
) -> str:
    """Evaluate the always-on program-abort ladder.

    NOTE: program-abort is never adjusted by the risk dial — these thresholds
    are fixed insurance, independent of the current RI.

    Inputs are positive magnitudes of loss (e.g. 36.0 means down 36%). Returns:
      - "halt"   if peak drawdown has reached the peak-halt threshold,
      - "review" if month drawdown has reached the monthly-review threshold,
      - "ok"     otherwise.
    The peak-halt check takes precedence over the monthly-review check.
    """
    pa = limits.program_abort
    if peak_drawdown_pct >= pa.peak_halt_drawdown_pct:
        return "halt"
    if month_drawdown_pct >= pa.monthly_review_drawdown_pct:
        return "review"
    return "ok"
