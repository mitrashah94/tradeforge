"""tests/test_regime_reader.py — the daily regime-reader policy module.

Offline + deterministic: every assessment is built from a synthetic daily OHLC
frame + ATR map (no DB), and publishing goes to a fake in-memory bus. Verifies
(MASTER_PLAN.md §1 regime-scaled exposure, §4 slow-loop agents):

  * assess() returns a regime in {trend, chop, vol_shock};
  * the exposure scalar is HIGHER in trend than in chop / vol_shock (lean in vs
    cut), and vol_shock stands the book down (scalar 0, nothing armed);
  * the armed list matches the regime;
  * IV is None (never fabricated — no options feed this phase);
  * publishing emits exactly one REGIME_TAGGED event carrying the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from backtest.stats.regime import CHOP, REGIMES, TREND, VOL_SHOCK
from orchestrator.agents.regime_reader import (
    ARMED_BY_REGIME,
    RegimeAssessment,
    assess_from_daily,
    exposure_scalar_for,
    publish,
)
from orchestrator.events import EventType


# --- fake bus -----------------------------------------------------------------
@dataclass
class FakeEvent:
    type: object
    data: dict
    ts_utc: datetime | None = None
    seq: int | None = None
    source: str = "test"


class FakeBus:
    """In-memory bus that records published events (duck-typed publish)."""

    def __init__(self) -> None:
        self.published: list = []

    def publish(self, event):
        self.published.append(event)
        return event


# --- synthetic daily builders -------------------------------------------------
def _flat_daily(n: int = 60, base: float = 100.0) -> pd.DataFrame:
    """Flat, low-range series -> every session is 'chop'."""
    rows = []
    d = date(2025, 1, 1)
    for i in range(n):
        c = base + (i % 2) * 0.1
        rows.append(
            {"session_date": d + timedelta(days=i), "open": c,
             "high": c + 0.2, "low": c - 0.2, "close": c}
        )
    return pd.DataFrame(rows)


def _trend_daily(n: int = 60, base: float = 100.0) -> pd.DataFrame:
    """Strong steady uptrend -> far from SMA -> 'trend'."""
    rows = []
    d = date(2025, 1, 1)
    c = base
    for i in range(n):
        c += 2.0
        rows.append(
            {"session_date": d + timedelta(days=i), "open": c - 0.1,
             "high": c + 0.3, "low": c - 0.3, "close": c}
        )
    return pd.DataFrame(rows)


def _atr_map(daily: pd.DataFrame, atr: float) -> dict:
    return {r.session_date: atr for r in daily.itertuples(index=False)}


# --- regime + armed -----------------------------------------------------------
def test_assess_chop():
    daily = _flat_daily(60)
    target = daily["session_date"].iloc[-1]
    a = assess_from_daily(daily, _atr_map(daily, atr=5.0), target)
    assert a.regime == CHOP
    assert a.regime in REGIMES
    assert tuple(a.armed) == ARMED_BY_REGIME[CHOP] == ("level_meanrev",)
    assert a.iv is None  # never fabricated


def test_assess_trend_arms_continuation():
    daily = _trend_daily(60)
    target = daily["session_date"].iloc[-1]
    a = assess_from_daily(daily, _atr_map(daily, atr=2.0), target)
    assert a.regime == TREND
    assert set(a.armed) == {"breakout_retest", "momentum_thrust"}
    assert a.iv is None


def test_assess_vol_shock_stands_down():
    daily = _flat_daily(60)
    # Blow out the target day's range to >> 1.8 * ATR.
    idx = 59
    daily.loc[idx, "high"] = daily.loc[idx, "close"] + 4.0
    daily.loc[idx, "low"] = daily.loc[idx, "close"] - 1.0
    target = daily.loc[idx, "session_date"]
    a = assess_from_daily(daily, _atr_map(daily, atr=1.0), target)
    assert a.regime == VOL_SHOCK
    assert tuple(a.armed) == ()  # stand down
    assert a.exposure_scalar == 0.0
    assert a.iv is None


# --- exposure scalar ordering -------------------------------------------------
def test_exposure_scalar_higher_in_trend_than_chop_and_shock():
    trend_daily = _trend_daily(60)
    chop_daily = _flat_daily(60)
    t = assess_from_daily(trend_daily, _atr_map(trend_daily, 2.0),
                          trend_daily["session_date"].iloc[-1])
    c = assess_from_daily(chop_daily, _atr_map(chop_daily, 5.0),
                          chop_daily["session_date"].iloc[-1])
    assert t.exposure_scalar > c.exposure_scalar
    assert c.exposure_scalar > 0.0  # chop cuts but still trades

    # vol_shock is the lowest (stand down).
    shock = _flat_daily(60)
    shock.loc[59, "high"] = shock.loc[59, "close"] + 4.0
    shock.loc[59, "low"] = shock.loc[59, "close"] - 1.0
    s = assess_from_daily(shock, _atr_map(shock, 1.0), shock.loc[59, "session_date"])
    assert t.exposure_scalar > s.exposure_scalar
    assert s.exposure_scalar == 0.0


def test_exposure_scalar_for_helper_ordering_and_clamp():
    assert exposure_scalar_for(TREND) > exposure_scalar_for(CHOP)
    assert exposure_scalar_for(CHOP) > exposure_scalar_for(VOL_SHOCK)
    assert exposure_scalar_for(VOL_SHOCK) == 0.0
    # Clamp: scalar never exceeds the configured ceiling (default 1.25).
    assert exposure_scalar_for(TREND) <= 1.25
    assert exposure_scalar_for(TREND) >= 0.0


# --- realized vol read --------------------------------------------------------
def test_realized_vol_is_tr_over_atr():
    daily = _flat_daily(60)
    a = assess_from_daily(daily, _atr_map(daily, atr=5.0),
                          daily["session_date"].iloc[-1])
    # Flat day TR ~ 0.4 (high-low), ATR 5 -> realized_vol ~ 0.08, a real number.
    assert a.realized_vol is not None
    assert a.realized_vol == pytest.approx(0.4 / 5.0, abs=1e-6)


# --- publish ------------------------------------------------------------------
def test_publish_emits_one_regime_tagged_event():
    daily = _trend_daily(60)
    a = assess_from_daily(daily, _atr_map(daily, 2.0),
                          daily["session_date"].iloc[-1])
    bus = FakeBus()
    ev = publish(bus, a)

    assert len(bus.published) == 1
    assert bus.published[0] is ev
    assert ev.type == EventType.REGIME_TAGGED

    d = ev.data
    # The full consumer schema is present.
    assert set(d) == {
        "date", "symbol", "regime", "realized_vol",
        "iv", "exposure_scalar", "armed", "rationale",
    }
    assert d["iv"] is None  # explicit null, not omitted
    assert d["regime"] == TREND
    assert d["armed"] == ["breakout_retest", "momentum_thrust"]
    assert d["exposure_scalar"] > 1.0  # trend leans in
    assert d["date"] == a.date.isoformat()


def test_event_payload_is_json_serializable():
    import json

    daily = _flat_daily(40)
    a = assess_from_daily(daily, _atr_map(daily, 5.0),
                          daily["session_date"].iloc[-1])
    # to_event_data() must round-trip through JSON (it is persisted as JSON text).
    payload = a.to_event_data()
    assert json.loads(json.dumps(payload)) == payload
