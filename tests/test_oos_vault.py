"""tests/test_oos_vault.py — locked OOS vault + IS/OOS split.

Verifies (MASTER_PLAN §5):
  - ``split_is_oos`` reserves the most-recent slice as OOS, contiguous and
    non-overlapping with IS;
  - locking writes a ``locked: true`` record with a verifiable checksum and the
    'DO NOT TUNE' note;
  - re-locking a DIFFERENT range is refused (the never-reused guarantee);
  - ``assert_not_tuned_on_oos`` blocks windows that overlap the locked range and
    permits in-sample windows.
"""

from __future__ import annotations

from datetime import date

import pytest

from backtest.stats.oos import (
    assert_not_tuned_on_oos,
    lock_oos,
    oos_range,
    read_vault,
    split_is_oos,
    vault_is_locked,
    verify_checksum,
)


def test_split_reserves_recent_slice_as_oos():
    (is_s, is_e), (oos_s, oos_e) = split_is_oos("2024-01-01", "2025-12-31", 0.2)
    # OOS ends at the overall end; IS starts at the overall start.
    assert oos_e == "2025-12-31"
    assert is_s == "2024-01-01"
    # Contiguous + non-overlapping: IS ends the day before OOS starts.
    assert date.fromisoformat(is_e) < date.fromisoformat(oos_s)
    assert (date.fromisoformat(oos_s) - date.fromisoformat(is_e)).days == 1


def test_split_oos_fraction_sizing():
    (_is, (oos_s, oos_e)) = split_is_oos("2020-01-01", "2020-12-31", 0.25)
    span = (date.fromisoformat("2020-12-31") - date.fromisoformat("2020-01-01")).days
    oos_span = (date.fromisoformat(oos_e) - date.fromisoformat(oos_s)).days
    # Roughly a quarter of the span (allow rounding).
    assert abs(oos_span - 0.25 * span) <= 2


def test_split_invalid_fraction_raises():
    with pytest.raises(ValueError):
        split_is_oos("2024-01-01", "2025-01-01", 0.0)
    with pytest.raises(ValueError):
        split_is_oos("2024-01-01", "2025-01-01", 1.0)


def test_lock_writes_locked_record_with_checksum(tmp_path):
    vp = tmp_path / "oos_vault.yaml"
    rec = lock_oos("2025-06-01", "2025-12-31", "2026-06-13", path=vp, symbol="QQQ")
    assert rec["locked"] is True
    assert rec["oos_start"] == "2025-06-01"
    assert "DO NOT TUNE" in rec["note"]
    assert vault_is_locked(vp)
    assert verify_checksum(vp)


def test_lock_is_idempotent_for_same_range(tmp_path):
    vp = tmp_path / "oos_vault.yaml"
    a = lock_oos("2025-06-01", "2025-12-31", "2026-06-13", path=vp)
    b = lock_oos("2025-06-01", "2025-12-31", "2026-06-13", path=vp)
    assert a["checksum"] == b["checksum"]


def test_relock_different_range_refused(tmp_path):
    vp = tmp_path / "oos_vault.yaml"
    lock_oos("2025-06-01", "2025-12-31", "2026-06-13", path=vp)
    with pytest.raises(RuntimeError):
        lock_oos("2025-01-01", "2025-03-31", "2026-06-13", path=vp)
    # force=True allows a deliberate re-baseline.
    rec = lock_oos("2025-01-01", "2025-03-31", "2026-06-13", path=vp, force=True)
    assert rec["oos_start"] == "2025-01-01"


def test_guard_blocks_overlap_and_allows_is(tmp_path):
    vp = tmp_path / "oos_vault.yaml"
    lock_oos("2025-06-01", "2025-12-31", "2026-06-13", path=vp)
    # An in-sample window before the OOS range is allowed.
    assert_not_tuned_on_oos("2024-01-01", "2025-05-31", path=vp)
    # Any overlap with the locked range raises.
    with pytest.raises(AssertionError):
        assert_not_tuned_on_oos("2025-05-01", "2025-07-01", path=vp)
    with pytest.raises(AssertionError):
        assert_not_tuned_on_oos("2025-12-31", "2026-01-15", path=vp)


def test_guard_noop_when_nothing_locked(tmp_path):
    vp = tmp_path / "absent.yaml"
    # No vault file -> nothing to protect -> permissive no-op.
    assert_not_tuned_on_oos("2025-01-01", "2025-12-31", path=vp)
    assert oos_range(vp) is None
    assert read_vault(vp) is None
