"""backtest/stats/oos.py — the LOCKED out-of-sample vault.

MASTER_PLAN.md §5: *"Locked OOS vault — a history slice never touched in
development and never reused."* The most-recent slice of history is reserved as
out-of-sample, written to ``oos_vault.yaml`` with a ``locked: true`` flag and a
checksum so reuse is detectable, and a documented guard
(:func:`assert_not_tuned_on_oos`) that callers in the research path invoke to
keep development off the locked range.

Discipline (why this is structural, not advisory)
-------------------------------------------------
* ``split_is_oos`` reserves the *most-recent* ``oos_fraction`` of the date range
  as OOS — recent data is the hardest test (no lookahead, closest to live).
* The locked range is persisted ONCE to ``oos_vault.yaml`` with a checksum over
  the range + lock date. Re-locking a *different* range without explicitly
  unlocking is refused — that is the "never reused" rule made enforceable.
* ``assert_not_tuned_on_oos(start, end)`` raises if a development/tuning run's
  date window overlaps the locked OOS range. Research code calls this before any
  in-sample fit so the vault cannot be touched by accident.

This module reads/writes YAML and dates only; it does not run backtests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

DEFAULT_VAULT_PATH = Path(__file__).resolve().parent / "oos_vault.yaml"


# --------------------------------------------------------------------------- #
# Date coercion
# --------------------------------------------------------------------------- #
def _as_date(d) -> date:
    """Coerce a date / datetime / 'YYYY-MM-DD' string into a ``date``."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):  # pandas Timestamp
        return d.date()
    raise TypeError(f"cannot coerce {d!r} ({type(d)}) to a date")


def _iso(d) -> str:
    return _as_date(d).isoformat()


# --------------------------------------------------------------------------- #
# IS / OOS split
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DateRange:
    """A simple inclusive [start, end] date range."""

    start: date
    end: date

    def as_tuple(self) -> tuple[str, str]:
        return (self.start.isoformat(), self.end.isoformat())

    def overlaps(self, other: "DateRange") -> bool:
        return self.start <= other.end and other.start <= self.end


def split_is_oos(
    start, end, oos_fraction: float = 0.2
) -> tuple[tuple[str, str], tuple[str, str]]:
    """Split ``[start, end]`` into (in-sample, out-of-sample) date ranges.

    The most-recent ``oos_fraction`` of the *calendar span* is reserved as OOS.
    Returns ``((is_start, is_end), (oos_start, oos_end))`` as ISO date-string
    pairs. The IS and OOS ranges are contiguous and non-overlapping: IS ends the
    day before OOS begins.

    Example: 1000 days with oos_fraction=0.2 -> last 200 days are OOS.
    """
    if not (0.0 < oos_fraction < 1.0):
        raise ValueError(f"oos_fraction must be in (0,1), got {oos_fraction!r}")
    s = _as_date(start)
    e = _as_date(end)
    if e < s:
        raise ValueError(f"end {e} is before start {s}")

    total_days = (e - s).days
    oos_days = int(round(total_days * oos_fraction))
    oos_days = max(1, min(oos_days, total_days - 1)) if total_days >= 2 else 1

    from datetime import timedelta

    oos_start = e - timedelta(days=oos_days - 1)
    is_end = oos_start - timedelta(days=1)
    if is_end < s:
        is_end = s  # degenerate tiny ranges: keep IS non-empty

    return (
        (s.isoformat(), is_end.isoformat()),
        (oos_start.isoformat(), e.isoformat()),
    )


# --------------------------------------------------------------------------- #
# The vault file
# --------------------------------------------------------------------------- #
def _checksum(oos_start: str, oos_end: str, locked_on: str) -> str:
    """Stable marker over the locked range + lock date (reuse detection)."""
    payload = f"{oos_start}|{oos_end}|{locked_on}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def lock_oos(
    oos_start,
    oos_end,
    locked_on,
    path: str | Path = DEFAULT_VAULT_PATH,
    symbol: str | None = None,
    timeframe: str | None = None,
    note: str | None = None,
    force: bool = False,
) -> dict:
    """Write/maintain the locked OOS range in ``oos_vault.yaml``.

    Idempotent: re-locking the SAME range is a no-op (returns the existing
    record). Attempting to lock a DIFFERENT range when one is already locked
    raises ``RuntimeError`` unless ``force=True`` — that refusal is the "never
    reused / never re-tuned" guarantee.

    ``locked_on`` is a caller-provided ISO date (determinism; no wall-clock).
    """
    path = Path(path)
    oos_start_s = _iso(oos_start)
    oos_end_s = _iso(oos_end)
    locked_on_s = _iso(locked_on)

    existing = read_vault(path)
    if existing and existing.get("locked"):
        same = (
            existing.get("oos_start") == oos_start_s
            and existing.get("oos_end") == oos_end_s
        )
        if same:
            return existing
        if not force:
            raise RuntimeError(
                "OOS vault is already locked to "
                f"[{existing.get('oos_start')} .. {existing.get('oos_end')}]; "
                f"refusing to re-lock to [{oos_start_s} .. {oos_end_s}]. "
                "The locked OOS range must never be reused or moved during "
                "development. Pass force=True only for a deliberate re-baseline."
            )

    record = {
        "locked": True,
        "oos_start": oos_start_s,
        "oos_end": oos_end_s,
        "locked_on": locked_on_s,
        "symbol": symbol,
        "timeframe": timeframe,
        "checksum": _checksum(oos_start_s, oos_end_s, locked_on_s),
        "note": note
        or (
            f"DO NOT TUNE ON THIS RANGE - locked {locked_on_s}. "
            "This out-of-sample slice is reserved for a single final validation "
            "and must never be touched during development or reused."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            record, fh, sort_keys=True, default_flow_style=False, allow_unicode=True
        )
    return record


def read_vault(path: str | Path = DEFAULT_VAULT_PATH) -> dict | None:
    """Read the locked OOS record from ``oos_vault.yaml`` (None if absent)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or None


def vault_is_locked(path: str | Path = DEFAULT_VAULT_PATH) -> bool:
    """True if a locked OOS range exists and is marked ``locked: true``."""
    rec = read_vault(path)
    return bool(rec and rec.get("locked"))


def verify_checksum(path: str | Path = DEFAULT_VAULT_PATH) -> bool:
    """Recompute and verify the vault checksum (tamper / reuse detection).

    Returns True if the stored checksum matches the recomputed one over the
    locked range + lock date. A mismatch means the file was edited by hand —
    treat that as a discipline breach.
    """
    rec = read_vault(path)
    if not rec or not rec.get("locked"):
        return False
    expect = _checksum(rec["oos_start"], rec["oos_end"], rec["locked_on"])
    return expect == rec.get("checksum")


def oos_range(path: str | Path = DEFAULT_VAULT_PATH) -> DateRange | None:
    """The locked OOS :class:`DateRange`, or None if nothing is locked."""
    rec = read_vault(path)
    if not rec or not rec.get("locked"):
        return None
    return DateRange(_as_date(rec["oos_start"]), _as_date(rec["oos_end"]))


# --------------------------------------------------------------------------- #
# The guard
# --------------------------------------------------------------------------- #
def assert_not_tuned_on_oos(
    start, end, path: str | Path = DEFAULT_VAULT_PATH
) -> None:
    """Raise if a development/tuning window overlaps the locked OOS range.

    GUARD SEMANTICS
    ---------------
    Any research/tuning code that fits or selects on a date window MUST call this
    first with that window's ``[start, end]``. If no OOS range is locked yet, the
    call is a permissive no-op (nothing to protect). If a range IS locked and the
    requested window overlaps it, this raises ``AssertionError`` — development is
    structurally blocked from touching the vault. The single, final OOS
    evaluation is the ONLY code allowed to run on the locked range, and it does
    not call this guard.
    """
    rng = oos_range(path)
    if rng is None:
        return  # nothing locked -> nothing to protect
    req = DateRange(_as_date(start), _as_date(end))
    if req.overlaps(rng):
        raise AssertionError(
            "Refusing to run: requested window "
            f"[{req.start} .. {req.end}] overlaps the LOCKED OOS vault "
            f"[{rng.start} .. {rng.end}]. The OOS slice must never be tuned on "
            "or reused during development (MASTER_PLAN §5). Use the in-sample "
            "range, or run the single final OOS validation explicitly."
        )
