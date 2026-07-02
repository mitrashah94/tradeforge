"""forecast/kronos/leakage.py — the HARD look-ahead guard for Kronos.

Kronos was pretrained on market history through an undisclosed cutoff (~2025). A
"forecast" whose TARGET dates fall inside that training history is contaminated:
the model may have memorized the very moves it is "predicting", so any backtest
using it there silently leaks the future and corrupts the OOS discipline the
platform rests on.

:func:`assert_post_cutoff` is the non-negotiable guard every Kronos path calls
before it forecasts. It RAISES :class:`LeakageError` unless:
  * the decision date ``asof`` (and hence every forecast-target day) is on/after
    :data:`KRONOS_TRAINING_CUTOFF` — the TARGET must postdate training; and
  * NO input bar is dated after ``asof`` — the classic structural look-ahead
    check (the model must never see a bar from its own future).

CONTEXT bars from BEFORE the cutoff are allowed by default: the model having
trained on its own historical context is exactly the DEPLOYMENT condition (any
live use hands it history it has seen) and leaks nothing about the post-cutoff
target. ``require_context_post_cutoff=True`` restores the stricter mode (every
input bar post-cutoff) for maximal paranoia at the cost of most of the usable
window. Pure / deterministic; no torch, no I/O.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

# Conservative ASSUMED training cutoff. Kronos discloses no exact cutoff, so we
# pick a date safely at/after any plausible one and only trust forecasts whose
# every input bar is on/after it. Hand-set; never tuned from results.
KRONOS_TRAINING_CUTOFF = date(2025, 8, 1)


class LeakageError(Exception):
    """Raised when a Kronos forecast would use look-ahead-contaminated bars."""


def _as_date(d) -> Optional[date]:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        try:
            return d.date()
        except Exception:  # noqa: BLE001
            return None
    return None


def _bar_dates(bars) -> list:
    """Extract bar dates from a DataFrame (``ts_utc`` col or index), Series, or iterable."""
    # pandas DataFrame with a ts_utc column
    cols = getattr(bars, "columns", None)
    if cols is not None and "ts_utc" in cols:
        return [_as_date(x) for x in bars["ts_utc"].tolist()]
    # pandas object with a DatetimeIndex / date index
    idx = getattr(bars, "index", None)
    if idx is not None and cols is not None:
        return [_as_date(x) for x in list(idx)]
    # a Series indexed by date
    if idx is not None:
        return [_as_date(x) for x in list(idx)]
    # a plain iterable of dates / timestamps
    if isinstance(bars, Iterable):
        return [_as_date(x) for x in bars]
    return []


def assert_post_cutoff(
    asof, bars, cutoff: date = KRONOS_TRAINING_CUTOFF,
    *, require_context_post_cutoff: bool = False,
) -> None:
    """Raise :class:`LeakageError` unless the forecast is honestly out-of-sample.

    Enforced (see the module docstring for the reasoning):
      * ``asof`` >= ``cutoff`` — every forecast-TARGET day postdates the model's
        training history (the condition that makes the forecast OOS);
      * no input bar is dated AFTER ``asof`` — structural no-look-ahead;
      * (``require_context_post_cutoff``) every input bar >= ``cutoff`` — the
        stricter optional mode that also refuses pre-cutoff context.

    Parameters
    ----------
    asof
        The decision date the forecast is made for.
    bars
        The exact input bars to be fed to the model — a DataFrame (``ts_utc``
        column or a date index), a date-indexed Series, or an iterable of
        dates/timestamps.
    cutoff
        The assumed training cutoff (defaults to :data:`KRONOS_TRAINING_CUTOFF`).
    """
    a = _as_date(asof)
    if a is None:
        raise LeakageError("assert_post_cutoff: asof date is unparseable")
    if a < cutoff:
        raise LeakageError(
            f"Kronos look-ahead: asof {a} predates the training cutoff {cutoff} "
            "(the forecast target would be inside Kronos's training history)"
        )
    dates = [d for d in _bar_dates(bars) if d is not None]
    if not dates:
        raise LeakageError("assert_post_cutoff: no parseable bar dates to check")
    latest = max(dates)
    if latest > a:
        raise LeakageError(
            f"Kronos look-ahead: input bar {latest} is AFTER asof {a} — the model "
            "would be reading its own future"
        )
    if require_context_post_cutoff:
        earliest = min(dates)
        if earliest < cutoff:
            raise LeakageError(
                f"Kronos strict mode: input bar {earliest} predates the training "
                f"cutoff {cutoff}. Restrict the context window to the post-cutoff "
                "slice (or drop require_context_post_cutoff)."
            )


def is_post_cutoff(
    asof, bars, cutoff: date = KRONOS_TRAINING_CUTOFF,
    *, require_context_post_cutoff: bool = False,
) -> bool:
    """Non-raising form of :func:`assert_post_cutoff` (True if safe to forecast)."""
    try:
        assert_post_cutoff(
            asof, bars, cutoff,
            require_context_post_cutoff=require_context_post_cutoff,
        )
        return True
    except LeakageError:
        return False
