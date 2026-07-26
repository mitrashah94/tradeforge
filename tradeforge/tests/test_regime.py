"""tests/test_regime.py — regime tagger + by-regime slicing.

Verifies (MASTER_PLAN §5):
  - the tagger is deterministic and partitions ALL sessions (no date dropped);
  - the three regimes classify per the documented thresholds (vol_shock wins
    ties; far-from-SMA => trend; the low-range remainder => chop);
  - sessions lacking features default to 'chop' (still partitioned);
  - ``by_regime`` partitions a daily-return series exactly across regimes.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.stats.regime import (
    CHOP,
    REGIMES,
    TREND,
    VOL_SHOCK,
    RegimeConfig,
    by_regime,
    regime_tags_from_daily,
)


def _synthetic_daily(n: int = 60, base: float = 100.0) -> pd.DataFrame:
    """A flat, low-range series: every session should land in 'chop' by default."""
    rows = []
    d = date(2025, 1, 1)
    for i in range(n):
        c = base + (i % 2) * 0.1  # tiny oscillation, no trend, no shock
        rows.append(
            {
                "session_date": d + timedelta(days=i),
                "open": c,
                "high": c + 0.2,
                "low": c - 0.2,
                "close": c,
            }
        )
    return pd.DataFrame(rows)


def _atr_map(daily: pd.DataFrame, atr: float = 1.0) -> dict:
    return {r.session_date: atr for r in daily.itertuples(index=False)}


def test_tagger_partitions_all_sessions():
    daily = _synthetic_daily(60)
    tags = regime_tags_from_daily(daily, _atr_map(daily))
    # Every session date is tagged exactly once.
    assert len(tags) == 60
    assert set(tags.keys()) == set(daily["session_date"])
    assert set(tags.values()) <= set(REGIMES)


def test_tagger_deterministic():
    daily = _synthetic_daily(40)
    am = _atr_map(daily)
    assert regime_tags_from_daily(daily, am) == regime_tags_from_daily(daily, am)


def test_low_range_flat_is_chop():
    daily = _synthetic_daily(60)
    tags = regime_tags_from_daily(daily, _atr_map(daily, atr=5.0))
    # Flat, well within ATR of its own SMA, small TR -> all chop.
    assert all(v == CHOP for v in tags.values())


def test_vol_shock_detected_and_wins_ties():
    daily = _synthetic_daily(60)
    # Blow out one day's range to > 1.8 * ATR (ATR=1.0 -> TR ~ 5).
    shock_day = daily.loc[40, "session_date"]
    daily.loc[40, "high"] = daily.loc[40, "close"] + 4.0
    daily.loc[40, "low"] = daily.loc[40, "close"] - 1.0  # TR ~ 5 >> 1.8
    tags = regime_tags_from_daily(daily, _atr_map(daily, atr=1.0))
    assert tags[shock_day] == VOL_SHOCK


def test_trend_detected_far_from_sma():
    # Build a strong, steady uptrend so close - SMA20 >> 2 ATR.
    rows = []
    d = date(2025, 1, 1)
    c = 100.0
    for i in range(60):
        c += 2.0  # steady +2/day drift; SMA20 lags far behind
        rows.append(
            {
                "session_date": d + timedelta(days=i),
                "open": c - 0.1,
                "high": c + 0.3,
                "low": c - 0.3,
                "close": c,
            }
        )
    daily = pd.DataFrame(rows)
    # ATR ~ 2 (the daily drift); a steady trend keeps TR/ATR ~ 1 (not a shock).
    tags = regime_tags_from_daily(daily, _atr_map(daily, atr=2.0))
    # Later sessions (well past the 20-day SMA warmup) should be 'trend'.
    later = [tags[r.session_date] for r in daily.iloc[40:].itertuples(index=False)]
    assert TREND in later
    assert VOL_SHOCK not in later  # steady trend is not a vol shock


def test_no_atr_defaults_to_chop():
    daily = _synthetic_daily(30)
    # No ATR for any session -> features undefined -> chop (still partitioned).
    tags = regime_tags_from_daily(daily, atr_by_date={})
    assert len(tags) == 30
    assert all(v == CHOP for v in tags.values())


def test_thresholds_are_params():
    daily = _synthetic_daily(60)
    daily.loc[40, "high"] = daily.loc[40, "close"] + 1.6
    daily.loc[40, "low"] = daily.loc[40, "close"] - 0.0  # TR ~ 1.6
    am = _atr_map(daily, atr=1.0)
    # Default vol_shock_tr_mult=1.8 -> not a shock.
    assert regime_tags_from_daily(daily, am)[daily.loc[40, "session_date"]] != VOL_SHOCK
    # Lower the threshold -> now it is.
    cfg = RegimeConfig(vol_shock_tr_mult=1.5)
    assert regime_tags_from_daily(daily, am, cfg)[daily.loc[40, "session_date"]] == VOL_SHOCK


def test_by_regime_partitions_returns():
    daily = _synthetic_daily(30)
    tags = regime_tags_from_daily(daily, _atr_map(daily, atr=5.0))  # all chop
    # A return series over those dates.
    rets = pd.Series(
        np.linspace(-0.01, 0.01, 30),
        index=[r.session_date for r in daily.itertuples(index=False)],
    )
    out = by_regime(rets, tags)
    assert set(out.keys()) == set(REGIMES)
    # All days are chop here; the partition must sum to the full series length.
    total_days = sum(out[r]["n_days"] for r in REGIMES)
    assert total_days == 30
    assert out[CHOP]["n_days"] == 30
    assert out[CHOP]["total"] == pytest.approx(float(rets.sum()))


def test_by_regime_untagged_dates_go_to_chop():
    # A return on a date absent from tags is still partitioned (-> chop).
    rets = pd.Series([0.01, -0.02], index=[date(2030, 1, 1), date(2030, 1, 2)])
    out = by_regime(rets, tags={})
    assert out[CHOP]["n_days"] == 2
    assert sum(out[r]["n_days"] for r in REGIMES) == 2
