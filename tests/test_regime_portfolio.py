"""tests/test_regime_portfolio.py — the daily multi-sleeve portfolio extension
of the regime-reader (arm/disarm + vol-target exposure scalar).

Offline + deterministic: every assessment is built from a synthetic wide
ADJUSTED-close panel (no DB), and publishing goes to a fake in-memory bus.
This file is ADDITIVE — it does not import or touch the intraday-read tests in
``test_regime_reader.py`` (those still pass unchanged).

What is verified (the task's acceptance criteria):

  * the 200d-trend ARM flag FLIPS exactly at the SMA boundary (armed just above,
    disarmed just below) for the "market"/"or" rules;
  * own absolute-momentum gates a sleeve under the "own" rule;
  * the vol-target EXPOSURE SCALAR rises as realized vol FALLS and falls as it
    rises (and is clamped to [scalar_min, scalar_max]);
  * a disarmed sleeve gets the risk-off weights (the rotate-to-safety fallback);
  * the assessment is point-in-time (no lookahead) and JSON-serializable;
  * publishing emits exactly one REGIME_TAGGED event carrying scope='portfolio'.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from orchestrator.agents.regime_reader import (
    DEFAULT_RISK_OFF_SYMBOL,
    PortfolioRegimeAssessment,
    SleeveArmConfig,
    arm_signals,
    assess_portfolio_from_panel,
    publish_portfolio,
    realized_vol_annualized,
    vol_target_scalar,
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


# --- synthetic panel builders -------------------------------------------------
def _bdays(n: int, start=date(2023, 1, 2)) -> list:
    """N consecutive Mon-Fri business dates from ``start``."""
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _panel(series: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(series.values())))
    return pd.DataFrame(series, index=_bdays(n))


def _const_then_step(n: int, base: float, last: float) -> list[float]:
    """``n-1`` flat closes at ``base``, then a single final close at ``last``.

    The 200d SMA of a flat-then-step series is ~``base`` (one stepped sample
    barely moves it), so setting the FINAL close just above/below ``base`` puts
    the price cleanly on the chosen side of its own SMA — a precise boundary lever.
    """
    return [base] * (n - 1) + [last]


# =========================================================================== #
# 200d trend ARM flips at the SMA boundary
# =========================================================================== #
def test_arm_flips_at_200d_boundary_market_rule():
    cfg = SleeveArmConfig(arm_rule="market", trend_sma_window=200)
    n = 260
    # SMA over the trailing 200 closes is essentially 100.0 (199 flat + 1 step).
    base = 100.0

    above = _panel({"SPY": _const_then_step(n, base, base + 5.0)})
    below = _panel({"SPY": _const_then_step(n, base, base - 5.0)})
    asof_a = above.index[-1]
    asof_b = below.index[-1]

    # market rule: armed iff SPY >= its 200d SMA.
    armed_above = arm_signals(above, asof_a, ["momentum_rotation"], cfg=cfg)
    armed_below = arm_signals(below, asof_b, ["momentum_rotation"], cfg=cfg)
    assert armed_above["momentum_rotation"] is True
    assert armed_below["momentum_rotation"] is False


def test_arm_boundary_is_tight():
    """A tiny move across the SMA flips the flag — the boundary is exactly the SMA."""
    cfg = SleeveArmConfig(arm_rule="market", trend_sma_window=200)
    n = 260
    base = 100.0
    sma = base  # ~the trailing-200 mean of 199 flats + 1 step is ~base

    just_above = _panel({"SPY": _const_then_step(n, base, sma + 0.01)})
    just_below = _panel({"SPY": _const_then_step(n, base, sma - 0.01)})
    a = arm_signals(just_above, just_above.index[-1], ["s"], cfg=cfg)
    b = arm_signals(just_below, just_below.index[-1], ["s"], cfg=cfg)
    assert a["s"] is True
    assert b["s"] is False


def test_assessment_market_above_flag_and_risk_off_on_disarm():
    cfg = SleeveArmConfig(arm_rule="market", trend_sma_window=200)
    n = 260
    base = 100.0
    below = _panel(
        {"SPY": _const_then_step(n, base, base - 5.0),
         DEFAULT_RISK_OFF_SYMBOL: [100.0] * n}
    )
    pa = assess_portfolio_from_panel(below, below.index[-1], ["momentum_rotation"], cfg=cfg)
    assert pa.market_above_200d is False
    assert pa.armed_sleeves["momentum_rotation"] is False
    # A disarmed sleeve rotates to the risk-off destination (ready-to-return dict).
    assert pa.risk_off_weights == {DEFAULT_RISK_OFF_SYMBOL: 1.0}


# =========================================================================== #
# Own absolute-momentum gates the sleeve
# =========================================================================== #
def test_own_momentum_rule_arms_on_positive_disarms_on_negative():
    cfg = SleeveArmConfig(arm_rule="own", abs_mom_lookback=252, abs_mom_min=0.0)
    n = 300
    # Steady uptrend over the 252d lookback -> positive abs-momentum -> armed.
    up = [100.0 * (1.0 + 0.0008 * i) for i in range(n)]
    # Steady downtrend -> negative abs-momentum -> disarmed.
    down = [200.0 * (1.0 - 0.0008 * i) for i in range(n)]

    up_panel = _panel({"SPY": up})
    down_panel = _panel({"SPY": down})
    armed_up = arm_signals(up_panel, up_panel.index[-1], {"sec": "SPY"}, cfg=cfg)
    armed_down = arm_signals(down_panel, down_panel.index[-1], {"sec": "SPY"}, cfg=cfg)
    assert armed_up["sec"] is True
    assert armed_down["sec"] is False


def test_or_rule_arms_if_either_filter_true():
    """'or' rule: market trend DOWN but own momentum UP still arms."""
    cfg = SleeveArmConfig(arm_rule="or", trend_sma_window=200, abs_mom_lookback=252)
    n = 300
    # Price recovers at the very end (above 200d SMA) but is still net-down over
    # 252d (negative abs-mom). Construct the opposite per filter:
    # market DOWN (last < 200d SMA) yet own-mom UP (last > price 252d ago).
    base = 100.0
    closes = [base] * (n - 1) + [base - 3.0]  # last below the ~flat SMA -> market DOWN
    # but make 252d-ago price LOWER so abs-mom is positive:
    closes[n - 1 - 252] = base - 10.0
    panel = _panel({"SPY": closes})
    armed = arm_signals(panel, panel.index[-1], ["s"], cfg=cfg)
    # market filter False, own filter True -> 'or' arms.
    assert armed["s"] is True
    # but 'and' would NOT arm (market half is False).
    armed_and = arm_signals(panel, panel.index[-1], ["s"],
                            cfg=SleeveArmConfig(arm_rule="and"))
    assert armed_and["s"] is False


def test_insufficient_history_disarms_conservatively():
    cfg = SleeveArmConfig(arm_rule="or", trend_sma_window=200, abs_mom_lookback=252)
    # Only 50 sessions -> neither the 200d SMA nor the 252d momentum is defined.
    panel = _panel({"SPY": [100.0 + i for i in range(50)]})
    pa = assess_portfolio_from_panel(panel, panel.index[-1], ["s"], cfg=cfg)
    assert pa.market_above_200d is None
    assert pa.sleeve_momentum["s"] is None
    assert pa.armed_sleeves["s"] is False  # undecidable -> risk-off


# =========================================================================== #
# Vol-target exposure scalar rises/falls with vol
# =========================================================================== #
def test_vol_target_scalar_monotonic_in_vol():
    cfg = SleeveArmConfig(target_vol=0.16, scalar_max=1.25, scalar_min=0.0)
    low = vol_target_scalar(0.08, cfg)    # calm -> scale up
    mid = vol_target_scalar(0.16, cfg)    # at target -> ~1.0
    high = vol_target_scalar(0.32, cfg)   # stormy -> cut
    assert low > mid > high
    assert mid == pytest.approx(1.0, abs=1e-9)
    assert high == pytest.approx(0.5, abs=1e-9)
    # Clamp: very calm tape pins to the ceiling, not above it.
    assert vol_target_scalar(0.01, cfg) == pytest.approx(1.25, abs=1e-9)
    # None / degenerate handling.
    assert vol_target_scalar(None, cfg) == 1.0      # neutral, never fabricated
    assert vol_target_scalar(0.0, cfg) == 1.25      # flat tape -> ceiling


def test_assessment_scalar_rises_as_realized_vol_falls():
    """End-to-end: a calmer proxy yields a HIGHER exposure scalar than a stormy one."""
    cfg = SleeveArmConfig(vol_lookback=20, target_vol=0.16, trend_sma_window=200)
    n = 260
    rng = np.random.default_rng(0)

    def _series(daily_sigma: float) -> list[float]:
        # Gentle upward drift + controlled daily noise -> a known realized vol.
        rets = rng.normal(0.0003, daily_sigma, n)
        px = 100.0 * np.cumprod(1.0 + rets)
        return list(px)

    calm = _panel({"SPY": _series(0.004)})    # ~6%/yr realized
    storm = _panel({"SPY": _series(0.025)})   # ~40%/yr realized
    pa_calm = assess_portfolio_from_panel(calm, calm.index[-1], ["s"], cfg=cfg)
    pa_storm = assess_portfolio_from_panel(storm, storm.index[-1], ["s"], cfg=cfg)

    assert pa_calm.realized_vol is not None and pa_storm.realized_vol is not None
    assert pa_calm.realized_vol < pa_storm.realized_vol
    assert pa_calm.exposure_scalar > pa_storm.exposure_scalar
    # Both inside the band.
    for pa in (pa_calm, pa_storm):
        assert 0.0 <= pa.exposure_scalar <= cfg.scalar_max


def test_realized_vol_annualized_matches_hand_computation():
    # 21 closes -> 20 returns; a flat-ish series has a known small vol.
    closes = pd.Series([100.0 * (1.01 if i % 2 else 0.99) ** 1 for i in range(21)])
    rv = realized_vol_annualized(closes, lookback=20, trading_days=252)
    rets = closes.pct_change().dropna()
    expected = float(rets.std(ddof=1)) * math.sqrt(252)
    assert rv == pytest.approx(expected, rel=1e-9)
    # Too little history -> None (never fabricated).
    assert realized_vol_annualized(pd.Series([100.0]), 20) is None


# =========================================================================== #
# Point-in-time (no lookahead) + serialization + publish
# =========================================================================== #
def test_assessment_is_point_in_time():
    """A later crash is invisible to an earlier asof_date (no lookahead)."""
    cfg = SleeveArmConfig(arm_rule="market", trend_sma_window=200)
    n = 260
    base = 100.0
    closes = [base] * n
    # Crash AFTER the decision date should not change the decision date's read.
    decision_i = 230
    closes_with_future_crash = list(closes)
    for j in range(decision_i + 1, n):
        closes_with_future_crash[j] = base * 0.5
    panel = _panel({"SPY": closes_with_future_crash})
    asof = panel.index[decision_i]

    pa = assess_portfolio_from_panel(panel, asof, ["s"], cfg=cfg)
    # As of decision_i the price is still flat at base, on its SMA -> armed.
    assert pa.date == asof
    assert pa.market_above_200d is True
    assert pa.armed_sleeves["s"] is True


def test_assessment_resolves_non_session_asof():
    cfg = SleeveArmConfig(arm_rule="market")
    n = 260
    panel = _panel({"SPY": [100.0] * n})
    last = panel.index[-1]
    weekend = last + timedelta(days=1)  # a non-session date after the last bar
    pa = assess_portfolio_from_panel(panel, weekend, ["s"], cfg=cfg)
    assert pa.date == last  # resolves to the most recent session <= asof


def test_event_payload_is_json_serializable_and_scoped():
    cfg = SleeveArmConfig(arm_rule="or")
    n = 300
    panel = _panel({"SPY": [100.0 * (1.0 + 0.0008 * i) for i in range(n)],
                    DEFAULT_RISK_OFF_SYMBOL: [100.0] * n})
    pa = assess_portfolio_from_panel(panel, panel.index[-1], ["momentum_rotation"], cfg=cfg)
    payload = pa.to_event_data()
    assert payload["scope"] == "portfolio"  # distinguishes from the intraday read
    assert set(payload) == {
        "date", "scope", "market_proxy", "market_above_200d", "realized_vol",
        "exposure_scalar", "armed_sleeves", "sleeve_momentum",
        "risk_off_weights", "rationale",
    }
    assert json.loads(json.dumps(payload)) == payload


def test_publish_emits_one_portfolio_event():
    cfg = SleeveArmConfig(arm_rule="or")
    n = 300
    panel = _panel({"SPY": [100.0 * (1.0 + 0.0008 * i) for i in range(n)]})
    pa = assess_portfolio_from_panel(panel, panel.index[-1], ["momentum_rotation"], cfg=cfg)
    bus = FakeBus()
    ev = publish_portfolio(bus, pa)
    assert len(bus.published) == 1
    assert bus.published[0] is ev
    assert ev.type == EventType.REGIME_TAGGED
    assert ev.data["scope"] == "portfolio"
    assert ev.data["armed_sleeves"]["momentum_rotation"] is True
