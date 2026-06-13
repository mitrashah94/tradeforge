"""Tests for the equity premarket / RTH session windows (DST-correct)."""

from datetime import datetime
from zoneinfo import ZoneInfo

from data.sessions import is_premarket, is_rth

UTC = ZoneInfo("UTC")


def _utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


# --- EST (winter); 2024-01-16 is a Tuesday, ET = UTC-5 ---


def test_est_premarket_open_edge():
    # 09:00 UTC = 04:00 ET -> premarket starts.
    assert is_premarket(_utc(2024, 1, 16, 9, 0)) is True
    assert is_rth(_utc(2024, 1, 16, 9, 0)) is False


def test_est_premarket_last_minute():
    # 14:29 UTC = 09:29 ET -> still premarket.
    assert is_premarket(_utc(2024, 1, 16, 14, 29)) is True


def test_est_rth_open():
    # 14:30 UTC = 09:30 ET -> RTH, no longer premarket.
    ts = _utc(2024, 1, 16, 14, 30)
    assert is_rth(ts) is True
    assert is_premarket(ts) is False


def test_est_before_premarket():
    # 08:59 UTC = 03:59 ET -> before premarket.
    assert is_premarket(_utc(2024, 1, 16, 8, 59)) is False


# --- EDT (summer); 2024-07-16 is a Tuesday, ET = UTC-4 ---


def test_edt_premarket_open_edge():
    # 08:00 UTC = 04:00 EDT -> premarket.
    assert is_premarket(_utc(2024, 7, 16, 8, 0)) is True


def test_edt_premarket_last_minute():
    # 13:29 UTC = 09:29 EDT -> premarket.
    assert is_premarket(_utc(2024, 7, 16, 13, 29)) is True


def test_edt_rth_open():
    # 13:30 UTC = 09:30 EDT -> RTH.
    assert is_rth(_utc(2024, 7, 16, 13, 30)) is True


# --- Weekend ---


def test_weekend_not_premarket():
    # 2024-01-13 is a Saturday; 09:00 UTC would be premarket on a weekday.
    assert is_premarket(_utc(2024, 1, 13, 9, 0)) is False


# --- Naive-UTC inputs are treated as UTC ---


def test_naive_utc_treated_as_utc():
    assert is_premarket(datetime(2024, 1, 16, 9, 0)) is True
    assert is_rth(datetime(2024, 1, 16, 14, 30)) is True
