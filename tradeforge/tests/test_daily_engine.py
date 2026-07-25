"""tests/test_daily_engine.py — daily-NAV backtester unit tests.

Deterministic, OFFLINE (no DB): every test builds a tiny synthetic ADJUSTED
daily-close panel with KNOWN behavior, runs ``run_daily``, and asserts the exact
NAV math, rebalance turnover, transaction cost, short-term tax accrual, and the
no-lookahead guarantee of the point-in-time history helper.

The math is hand-computed in each test so a regression in the engine's
accounting (cost timing, basis tracking, weight application) fails loudly.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.daily.engine import (
    DailyHistory,
    run_daily,
    run_param_grid,
    _rebalance_flags,
    _validate_weights,
)


# --------------------------------------------------------------------------- #
# Synthetic panel helpers
# --------------------------------------------------------------------------- #
def _bdays(n: int, start=date(2024, 1, 1)) -> list:
    """N consecutive Mon-Fri business dates starting on/after ``start``."""
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _panel(prices: dict[str, list[float]], dates=None) -> pd.DataFrame:
    """Build a wide ADJUSTED-close panel from {symbol: [closes]}."""
    n = len(next(iter(prices.values())))
    idx = dates if dates is not None else _bdays(n)
    return pd.DataFrame(prices, index=idx)


def _weekly_dates(n: int, start=date(2024, 1, 1)) -> list:
    """N dates each in a DISTINCT ISO week (every date is a weekly rebalance day).

    Using one trading date per week makes every scheduled weight change land on a
    'W' rebalance boundary, so a multi-step accounting test (add to a position,
    sell to realize a gain) executes each step deterministically.
    """
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:  # a weekday
            out.append(d)
            d += timedelta(days=7)  # jump a full week
        else:
            d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# Fixed-weight strategies (deterministic, no history dependence)
# --------------------------------------------------------------------------- #
class FixedWeights:
    """Always return the SAME target weights (a constant allocation)."""

    def __init__(self, weights: dict[str, float]):
        self._w = dict(weights)

    def target_weights(self, asof_date, history):
        return dict(self._w)


class WeightSchedule:
    """Return per-date weights from a {date: weights} schedule (else hold)."""

    def __init__(self, schedule: dict, default: dict | None = None):
        self._sched = schedule
        self._default = default or {}

    def target_weights(self, asof_date, history):
        return dict(self._sched.get(asof_date, self._default))


# --------------------------------------------------------------------------- #
# 1. Buy-and-hold NAV math (single symbol, zero cost)
# --------------------------------------------------------------------------- #
def test_buy_and_hold_nav_tracks_price():
    # One symbol, fully invested, price 100 -> 110 over the window.
    panel = _panel({"AAA": [100.0, 105.0, 110.0]})
    res = run_daily(
        FixedWeights({"AAA": 1.0}), ["AAA"], panel=panel, rebalance="W",
        cost_bps=0.0, initial_equity=100_000.0, short_term_tax_rate=0.0,
    )
    nav = res.nav
    # Day 0: rebalance into 1000 shares @100 -> NAV 100k.
    assert nav.iloc[0] == pytest.approx(100_000.0)
    # Day 1: 1000 * 105 = 105k. Day 2: 1000 * 110 = 110k.
    assert nav.iloc[1] == pytest.approx(105_000.0)
    assert nav.iloc[2] == pytest.approx(110_000.0)
    # No sells -> no realized gain, no tax, no cost.
    assert res.total_costs == pytest.approx(0.0)
    assert res.tax_reserve == pytest.approx(0.0)
    s = res.summary()
    assert s["total_return"] == pytest.approx(0.10)
    assert s["max_drawdown"] == pytest.approx(0.0)


def test_cash_remainder_is_uninvested():
    # 60% in AAA, 40% cash. Price doubles -> NAV = 0.6*200% + 0.4*flat = 1.6x.
    panel = _panel({"AAA": [100.0, 200.0]})
    res = run_daily(
        FixedWeights({"AAA": 0.6}), ["AAA"], panel=panel, rebalance="W",
        cost_bps=0.0, initial_equity=100_000.0, short_term_tax_rate=0.0,
    )
    # Day 0 NAV unchanged at 100k (60k in shares = 600 sh, 40k cash).
    assert res.nav.iloc[0] == pytest.approx(100_000.0)
    # Day 1: 600 sh * 200 = 120k + 40k cash = 160k.
    assert res.nav.iloc[1] == pytest.approx(160_000.0)


# --------------------------------------------------------------------------- #
# 2. Transaction cost on turnover
# --------------------------------------------------------------------------- #
def test_transaction_cost_charged_on_turnover():
    # Buy 100k of AAA on the single trading day. cost_bps=10 ->
    # cost = 100k * 10e-4 = 100. One date -> exactly one (initial) rebalance.
    panel = _panel({"AAA": [100.0]})
    res = run_daily(
        FixedWeights({"AAA": 1.0}), ["AAA"], panel=panel, rebalance="M",
        cost_bps=10.0, initial_equity=100_000.0, short_term_tax_rate=0.0,
    )
    # NAV day 0 = 100k - 100 cost = 99,900.
    assert res.total_costs == pytest.approx(100.0)
    assert res.nav.iloc[0] == pytest.approx(99_900.0)
    # Turnover on the initial allocation = full notional / nav = 1.0.
    assert res.rebalance_turnover.iloc[0] == pytest.approx(1.0)


def test_turnover_per_year_scales_with_cadence():
    # Two symbols, weekly flip A<->B every rebalance: turnover should accumulate.
    n = 25
    dates = _bdays(n)
    panel = _panel({"AAA": [100.0] * n, "BBB": [100.0] * n}, dates=dates)
    # Alternate the full allocation between A and B each calendar week.
    sched = {}
    flags = _rebalance_flags(dates, "W")
    flip = True
    for d, f in zip(dates, flags):
        if f:
            sched[d] = {"AAA": 1.0} if flip else {"BBB": 1.0}
            flip = not flip
    res = run_daily(
        WeightSchedule(sched), ["AAA", "BBB"], panel=panel, rebalance="W",
        cost_bps=0.0, initial_equity=100_000.0, short_term_tax_rate=0.0,
    )
    # The first week buys A (turnover 1.0); each later flip sells A buys B
    # (turnover 2.0). turnover_per_year must be > 0 and finite.
    s = res.summary()
    assert s["turnover_per_year"] > 0
    assert np.isfinite(s["turnover_per_year"])
    # Sum of one-way turnover fractions: 1.0 (first) + 2.0 * (n_flips - 1).
    assert res.rebalance_turnover.iloc[0] == pytest.approx(1.0)
    assert res.rebalance_turnover.iloc[1] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 3. Short-term tax accrual on realized gains
# --------------------------------------------------------------------------- #
def test_tax_accrues_on_realized_gain_only():
    # Buy 1000 sh AAA @100 (day0), price -> 120, SELL ALL on the last day.
    # Realized gain = (120-100)*1000 = 20,000. Tax reserve = 20k * 0.30 = 6,000.
    dates = _weekly_dates(3)
    panel = _panel({"AAA": [100.0, 110.0, 120.0]}, dates=dates)
    # Hold AAA fully through day0 and day1; go flat on day2 (every weekly date is
    # a rebalance boundary, so a non-listed day would otherwise re-target).
    sched = {dates[0]: {"AAA": 1.0}, dates[1]: {"AAA": 1.0}, dates[2]: {}}
    res = run_daily(
        WeightSchedule(sched), ["AAA"], panel=panel, rebalance="W",
        cost_bps=0.0, initial_equity=100_000.0, short_term_tax_rate=0.30,
    )
    assert res.realized_gains == pytest.approx(20_000.0)
    assert res.tax_reserve == pytest.approx(6_000.0)
    # Gross NAV day2 = all in cash after the sell = 120,000.
    assert res.nav.iloc[-1] == pytest.approx(120_000.0)
    # After-tax NAV = 120,000 - 6,000 reserve = 114,000.
    assert res.after_tax_nav.iloc[-1] == pytest.approx(114_000.0)
    s = res.summary()
    assert s["after_tax_total_return"] == pytest.approx(0.14)
    assert s["total_return"] == pytest.approx(0.20)


def test_realized_loss_does_not_create_negative_tax():
    # Buy @100, price -> 80, sell all. Realized loss -20k; reserve must NOT go
    # below 0 (a loss banks an offset against realized_gains, never a refund).
    dates = _bdays(3)
    panel = _panel({"AAA": [100.0, 90.0, 80.0]}, dates=dates)
    sched = {dates[0]: {"AAA": 1.0}, dates[2]: {}}
    res = run_daily(
        WeightSchedule(sched), ["AAA"], panel=panel, rebalance="W",
        cost_bps=0.0, initial_equity=100_000.0, short_term_tax_rate=0.30,
    )
    assert res.realized_gains == pytest.approx(-20_000.0)
    assert res.tax_reserve == pytest.approx(0.0)
    # After-tax == gross when no positive gain was taxed.
    assert res.after_tax_nav.iloc[-1] == pytest.approx(res.nav.iloc[-1])
    assert res.nav.iloc[-1] == pytest.approx(80_000.0)


def test_avg_cost_basis_merges_on_add():
    # Day0 buy 50% AAA @100 (=500 sh basis 100). Day1 price 100, rebalance to
    # 100% AAA -> buy 500 more @100. Basis stays 100. Day2 price 150, sell all:
    # gain = (150-100)*1000 = 50,000.
    dates = _weekly_dates(3)  # each date is its own weekly rebalance
    panel = _panel({"AAA": [100.0, 100.0, 150.0]}, dates=dates)
    sched = {dates[0]: {"AAA": 0.5}, dates[1]: {"AAA": 1.0}, dates[2]: {}}
    res = run_daily(
        WeightSchedule(sched), ["AAA"], panel=panel, rebalance="W",
        cost_bps=0.0, initial_equity=100_000.0, short_term_tax_rate=0.30,
    )
    assert res.realized_gains == pytest.approx(50_000.0)
    assert res.tax_reserve == pytest.approx(15_000.0)


# --------------------------------------------------------------------------- #
# 4. No-lookahead guarantee
# --------------------------------------------------------------------------- #
def test_history_has_no_lookahead():
    dates = _bdays(5)
    panel = _panel({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, dates=dates)
    seen = {}

    class Recorder:
        def target_weights(self, asof_date, history):
            # The visible window must never contain a date AFTER asof_date.
            px = history.prices()
            assert all(d <= asof_date for d in px.index), (
                f"lookahead: saw a date > asof {asof_date} in {list(px.index)}"
            )
            # asof price equals the panel's value AT asof_date (not a future one).
            seen[asof_date] = float(history.asof_prices()["AAA"])
            return {"AAA": 1.0}

    run_daily(
        Recorder(), ["AAA"], panel=panel, rebalance="W",
        cost_bps=0.0, short_term_tax_rate=0.0,
    )
    # Each decision day's asof price matches that day's bar exactly.
    for d, p in zip(dates, panel["AAA"]):
        if d in seen:
            assert seen[d] == pytest.approx(p)


def test_history_window_and_returns():
    dates = _bdays(4)
    panel = _panel({"AAA": [100.0, 110.0, 121.0, 133.1]}, dates=dates)
    h = DailyHistory(panel, dates[2])  # asof = 3rd date
    px = h.prices()
    assert list(px.index) == dates[:3]            # only <= asof
    assert h.asof_prices()["AAA"] == pytest.approx(121.0)
    rets = h.returns()
    # 110/100-1 = .10 ; 121/110-1 = .10
    assert rets["AAA"].tolist() == pytest.approx([0.10, 0.10])
    # lookback trims to the last row.
    assert list(h.prices(lookback=1).index) == [dates[2]]


# --------------------------------------------------------------------------- #
# 5. Weight-contract validation
# --------------------------------------------------------------------------- #
def test_negative_weight_rejected():
    panel = _panel({"AAA": [100.0, 100.0]})
    with pytest.raises(ValueError, match="LONG-ONLY"):
        run_daily(
            FixedWeights({"AAA": -0.5}), ["AAA"], panel=panel, rebalance="W",
            cost_bps=0.0, short_term_tax_rate=0.0,
        )


def test_weights_over_one_rejected():
    panel = _panel({"AAA": [100.0, 100.0], "BBB": [100.0, 100.0]})
    with pytest.raises(ValueError, match="sum"):
        run_daily(
            FixedWeights({"AAA": 0.7, "BBB": 0.7}), ["AAA", "BBB"],
            panel=panel, rebalance="W", cost_bps=0.0, short_term_tax_rate=0.0,
        )


def test_validate_weights_drops_zeros_and_nan():
    out = _validate_weights({"A": 0.5, "B": 0.0, "C": float("nan"), "D": None}, date(2024, 1, 1))
    assert out == {"A": 0.5}


# --------------------------------------------------------------------------- #
# 6. Rebalance calendar
# --------------------------------------------------------------------------- #
def test_rebalance_flags_month_end():
    # Span two months; month-end flags fire on the LAST trading day of each month
    # plus the first day (initial allocation).
    dates = (
        [date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31)]
        + [date(2024, 2, 1), date(2024, 2, 28), date(2024, 2, 29)]
    )
    flags = _rebalance_flags(dates, "M")
    assert flags[0] is True                  # initial
    assert flags[2] is True                  # Jan 31 (month end)
    assert flags[-1] is True                 # Feb 29 (last in sample)
    assert flags[3] is False                 # Feb 1 (mid-period)


def test_rebalance_flags_weekly():
    # Mon-Fri then next Mon-Tue: Friday is the week-end rebalance.
    dates = _bdays(7, start=date(2024, 1, 1))  # Mon 1/1 .. Tue 1/9
    flags = _rebalance_flags(dates, "W")
    assert flags[0] is True                  # initial (Mon)
    # Index 4 = Fri 1/5 -> last of ISO week 1 -> rebalance.
    assert flags[4] is True
    assert flags[5] is False                 # Mon 1/8 mid-week


# --------------------------------------------------------------------------- #
# 7. Robustness grid helper
# --------------------------------------------------------------------------- #
def test_run_param_grid_dispersion():
    # Strategy weight = the swept value; sweeping 0.2..1.0 over a rising market.
    n = 12
    dates = _bdays(n)
    prices = [100.0 * (1.01 ** i) for i in range(n)]
    panel = _panel({"AAA": prices}, dates=dates)

    def factory(w):
        return FixedWeights({"AAA": w})

    out = run_param_grid(
        factory, "weight", [0.25, 0.5, 0.75, 1.0], ["AAA"],
        rebalance="W", cost_bps=0.0, short_term_tax_rate=0.0, panel=panel,
    )
    assert out["param"] == "weight"
    assert len(out["grid"]) == 4
    # CAGR must rise monotonically with the equity weight in a rising market.
    cagrs = [g["CAGR"] for g in out["grid"]]
    assert cagrs == sorted(cagrs)
    disp = out["dispersion"]["CAGR"]
    assert disp["spread"] > 0
    assert np.isfinite(disp["cv"])


# --------------------------------------------------------------------------- #
# 8. Empty / degenerate inputs
# --------------------------------------------------------------------------- #
def test_empty_panel_returns_empty_result():
    res = run_daily(
        FixedWeights({"AAA": 1.0}), ["AAA"], panel=pd.DataFrame(),
        rebalance="W", cost_bps=2.0,
    )
    assert len(res.nav) == 0
    assert res.summary()["n_days"] == 0
