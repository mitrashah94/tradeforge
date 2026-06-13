"""Tests for crypto session date + weekend classification (UTC calendar)."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from data.sessions import crypto_session_date, is_crypto_weekend

UTC = ZoneInfo("UTC")


def _utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=UTC)


def test_crypto_session_date_before_boundary():
    # 2024-03-09 23:59 UTC -> still the 9th.
    assert crypto_session_date(_utc(2024, 3, 9, 23, 59)) == date(2024, 3, 9)


def test_crypto_session_date_after_boundary():
    # 2024-03-10 00:01 UTC -> the 10th (boundary is 00:00 UTC).
    assert crypto_session_date(_utc(2024, 3, 10, 0, 1)) == date(2024, 3, 10)


def test_crypto_weekend_saturday():
    # 2024-03-09 is a Saturday.
    assert is_crypto_weekend(_utc(2024, 3, 9, 12, 0)) is True


def test_crypto_weekend_sunday():
    # 2024-03-10 is a Sunday.
    assert is_crypto_weekend(_utc(2024, 3, 10, 12, 0)) is True


def test_crypto_weekend_monday_false():
    # 2024-03-11 is a Monday.
    assert is_crypto_weekend(_utc(2024, 3, 11, 12, 0)) is False


def test_crypto_session_naive_utc():
    # Naive datetimes are treated as UTC.
    assert crypto_session_date(datetime(2024, 3, 9, 23, 59)) == date(2024, 3, 9)
