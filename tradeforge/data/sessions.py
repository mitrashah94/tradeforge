"""Timezone and session helpers (stdlib only).

Bars in TradeForge are stored tz-naive in UTC. These helpers accept either a
tz-naive datetime (interpreted as UTC) or a tz-aware datetime, normalize to
UTC internally, and convert to the relevant session calendar.

Equity sessions are evaluated in America/New_York and are DST-correct via
``zoneinfo`` (Python stdlib). Windows (ET, weekdays only):

    premarket = [04:00, 09:30)
    RTH       = [09:30, 16:00)

Crypto uses a synthetic daily session on the UTC calendar with the boundary
at 00:00 UTC; "weekend" means Saturday/Sunday in UTC.

Pure functions, no I/O — safe to import anywhere in the hot path.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

PREMARKET_START = time(4, 0)
RTH_START = time(9, 30)
RTH_END = time(16, 0)


def _as_utc(ts: datetime) -> datetime:
    """Return a tz-aware UTC datetime.

    A tz-naive input is treated as UTC (TradeForge's storage convention); a
    tz-aware input is converted to UTC.
    """
    if not isinstance(ts, datetime):
        raise TypeError(f"expected datetime, got {type(ts)!r}")
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def to_et(ts: datetime) -> datetime:
    """Convert ``ts`` to a tz-aware America/New_York datetime (DST-correct)."""
    return _as_utc(ts).astimezone(ET)


def is_weekday_et(ts: datetime) -> bool:
    """True if the bar falls on a weekday (Mon-Fri) in ET."""
    return to_et(ts).weekday() < 5


def is_premarket(ts: datetime) -> bool:
    """True if ``ts`` is in the equity premarket window: weekday, [04:00, 09:30) ET."""
    et = to_et(ts)
    if et.weekday() >= 5:
        return False
    return PREMARKET_START <= et.time() < RTH_START


def is_rth(ts: datetime) -> bool:
    """True if ``ts`` is in regular trading hours: weekday, [09:30, 16:00) ET."""
    et = to_et(ts)
    if et.weekday() >= 5:
        return False
    return RTH_START <= et.time() < RTH_END


def classify_session(ts: datetime) -> str:
    """Classify an equity bar: 'premarket' | 'rth' | 'afterhours' | 'closed'.

    'closed' covers weekends and overnight (before 04:00 ET). 'afterhours'
    covers [16:00, 24:00) ET on a weekday.
    """
    et = to_et(ts)
    if et.weekday() >= 5:
        return "closed"
    t = et.time()
    if PREMARKET_START <= t < RTH_START:
        return "premarket"
    if RTH_START <= t < RTH_END:
        return "rth"
    if RTH_END <= t:
        return "afterhours"
    return "closed"


def et_session_date(ts: datetime) -> date:
    """The ET calendar date of the bar (the session it belongs to)."""
    return to_et(ts).date()


def crypto_session_date(ts: datetime) -> date:
    """The UTC calendar date of the bar (synthetic daily crypto session)."""
    return _as_utc(ts).date()


def is_crypto_weekend(ts: datetime) -> bool:
    """True if the bar falls on a Saturday or Sunday in UTC."""
    return _as_utc(ts).weekday() in (5, 6)
