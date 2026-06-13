"""tests/test_screeners.py — watchlist screeners (thresholds + unusual volume).

Deterministic, offline: synthetic OHLCV frames with known liquidity so we can
assert the dollar-volume / price thresholds filter correctly (a low-dollar-volume
name is excluded), and the relative-volume spike detector flags the right names.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from watchlist.screeners import (
    screen_crypto,
    screen_stocks,
    summarize_symbol,
    unusual_volume_flags,
)
from watchlist.screeners.common import daily_dollar_volume

BASE = datetime(2025, 1, 6, 14, 30)  # a Monday, ~09:30 ET


def _session(day_offset: int, price: float, volume: float, n: int = 5) -> list[dict]:
    """One RTH session of ``n`` flat-ish bars at ``price`` with per-bar ``volume``."""
    base = BASE + timedelta(days=day_offset)
    rows = []
    for i in range(n):
        ts = base + timedelta(minutes=5 * i)
        rows.append(
            {
                "ts_utc": ts,
                "open": price,
                "high": price + 0.1,
                "low": price - 0.1,
                "close": price,
                "volume": volume,
            }
        )
    return rows


def _frame(price: float, per_bar_volume: float, n_days: int = 5) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        rows.extend(_session(d, price, per_bar_volume))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Liquidity primitives
# --------------------------------------------------------------------------- #
def test_daily_dollar_volume_sums_per_session():
    # 5 bars/session, price ~100, volume 1000 -> ~ 5 * 100 * 1000 = 500_000/day.
    bars = _frame(price=100.0, per_bar_volume=1000.0, n_days=3)
    dv = daily_dollar_volume(bars, "equity")
    assert len(dv) == 3
    for v in dv:
        assert abs(v - 500_000.0) < 1e-6


def test_summarize_symbol_reports_average_dollar_volume():
    bars = _frame(price=50.0, per_bar_volume=2000.0, n_days=4)
    stats = summarize_symbol("FOO", bars, "equity")
    assert stats is not None
    assert stats.n_sessions == 4
    assert stats.last_price == 50.0
    # 5 bars * 50 * 2000 = 500_000 per session.
    assert abs(stats.avg_dollar_volume - 500_000.0) < 1e-6


def test_summarize_empty_returns_none():
    assert summarize_symbol("EMPTY", pd.DataFrame(), "equity") is None


# --------------------------------------------------------------------------- #
# The core requirement: thresholds filter correctly
# --------------------------------------------------------------------------- #
def test_screen_stocks_excludes_low_dollar_volume_name():
    bars_by_symbol = {
        # ~100M/session: 5 bars * 200 * 100_000 = 100_000_000.
        "BIG": _frame(price=200.0, per_bar_volume=100_000.0),
        # ~2.5M/session: 5 bars * 50 * 1000 = 250_000 (LOW).
        "SMALL": _frame(price=50.0, per_bar_volume=1000.0),
    }
    survivors = screen_stocks(bars_by_symbol, min_dollar_volume=50_000_000, min_price=5)
    syms = [s.symbol for s in survivors]
    assert "BIG" in syms
    assert "SMALL" not in syms  # excluded on dollar-volume


def test_screen_stocks_excludes_low_price_penny_name():
    bars_by_symbol = {
        # High dollar-volume but a $2 price (below a $5 CORE min).
        "PENNY": _frame(price=2.0, per_bar_volume=50_000_000.0),
        "OK": _frame(price=100.0, per_bar_volume=200_000.0),
    }
    survivors = screen_stocks(bars_by_symbol, min_dollar_volume=50_000_000, min_price=5)
    syms = [s.symbol for s in survivors]
    assert "PENNY" not in syms  # excluded on price
    assert "OK" in syms


def test_screen_stocks_sorted_by_liquidity_desc():
    bars_by_symbol = {
        "MID": _frame(price=100.0, per_bar_volume=200_000.0),
        "TOP": _frame(price=100.0, per_bar_volume=2_000_000.0),
    }
    survivors = screen_stocks(bars_by_symbol, min_dollar_volume=1_000_000, min_price=5)
    assert [s.symbol for s in survivors] == ["TOP", "MID"]


def test_screen_crypto_uses_utc_calendar_and_filters():
    bars_by_symbol = {
        "BTC/USD": _frame(price=60000.0, per_bar_volume=10.0),  # ~3M/session
        "TINY/USD": _frame(price=0.5, per_bar_volume=1.0),       # tiny + sub-$1
    }
    survivors = screen_crypto(bars_by_symbol, min_dollar_volume=1_000_000, min_price=1)
    syms = [s.symbol for s in survivors]
    assert "BTC/USD" in syms
    assert "TINY/USD" not in syms


# --------------------------------------------------------------------------- #
# Unusual volume
# --------------------------------------------------------------------------- #
def test_unusual_volume_flags_a_spike():
    # 20 baseline sessions at volume 1000, then a final session at 5000 (5x RVOL).
    rows = []
    for d in range(20):
        rows.extend(_session(d, 100.0, 1000.0))
    rows.extend(_session(20, 100.0, 5000.0))  # the spike day
    spike = pd.DataFrame(rows)

    # A calm name: flat volume throughout.
    calm = _frame(price=100.0, per_bar_volume=1000.0, n_days=21)

    flags = unusual_volume_flags(
        {"SPIKE": spike, "CALM": calm}, "equity", lookback=20, threshold=2.0
    )
    fmap = {f.symbol: f for f in flags}
    assert fmap["SPIKE"].unusual is True
    assert fmap["SPIKE"].rvol > 2.0
    assert fmap["CALM"].unusual is False
    # Flags sorted with the unusual / highest RVOL first.
    assert flags[0].symbol == "SPIKE"


def test_relative_volume_nan_without_baseline():
    flags = unusual_volume_flags(
        {"ONEDAY": _frame(100.0, 1000.0, n_days=1)}, "equity"
    )
    assert flags[0].unusual is False  # no baseline -> never flagged
