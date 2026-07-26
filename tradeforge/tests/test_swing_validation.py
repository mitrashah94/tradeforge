"""tests/test_swing_validation.py — unit tests for the daily bracket-validation
harness (``backtest/daily/validation.py``).

Deterministic, offline, no DB: hand-built :class:`BracketTrade` ledgers and a
synthetic continuous-run stand-in (a ``trades`` list + a dated ``daily_returns``
Series) drive every assertion, so the gate math is pinned independently of any
backtest.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.daily.result import BracketTrade
from backtest.daily.validation import (
    haircut_verdict,
    pool_windows,
    pooled_trade_metrics,
    returns_metrics,
    slice_result,
    yearly_breakdown,
)
from backtest.stats.multiple_testing import min_pf_threshold


def _trade(entry: date, pnl: float, r: float) -> BracketTrade:
    """A minimal closed BracketTrade carrying just the fields the gate reads."""
    return BracketTrade(
        symbol="T", entry_date=entry, exit_date=entry, entry_price=100.0,
        avg_exit_price=100.0 + pnl, shares=1.0, initial_stop=99.0, pnl=pnl,
        gross_pnl=pnl, costs=0.0, r_multiple=r, bars_held=1, exit_reason="x",
    )


class _Run:
    """Stand-in for a full-span BracketResult: a ledger + a dated return series."""

    def __init__(self, trades, returns):
        self.trades = trades
        self._returns = returns

    @property
    def daily_returns(self) -> pd.Series:
        return self._returns


# --------------------------------------------------------------------------- #
# pooled_trade_metrics — PF / win-rate / expectancy
# --------------------------------------------------------------------------- #
def test_pooled_trade_metrics_pf_and_winrate():
    # 3 winners (+2,+1,+3 = +6) and 2 losers (-1,-3 = -4) -> PF = 6/4 = 1.5.
    trades = [
        _trade(date(2020, 1, 1), 2.0, 1.0),
        _trade(date(2020, 1, 2), 1.0, 0.5),
        _trade(date(2020, 1, 3), 3.0, 1.5),
        _trade(date(2020, 1, 4), -1.0, -1.0),
        _trade(date(2020, 1, 5), -3.0, -1.0),
    ]
    m = pooled_trade_metrics(trades)
    assert m["n_trades"] == 5
    assert m["profit_factor"] == pytest.approx(1.5)
    assert m["win_rate"] == pytest.approx(3 / 5)
    assert m["gross_profit"] == pytest.approx(6.0)
    assert m["gross_loss"] == pytest.approx(4.0)


def test_pooled_trade_metrics_empty():
    m = pooled_trade_metrics([])
    assert m["n_trades"] == 0
    assert np.isnan(m["profit_factor"])


# --------------------------------------------------------------------------- #
# returns_metrics — annualized return, drawdown, higher moments
# --------------------------------------------------------------------------- #
def test_returns_metrics_annualizes_and_drawdown():
    # A flat +0.1%/day stream for exactly one trading year compounds to ~+28.6%;
    # since it never falls, max drawdown is 0.
    r = pd.Series([0.001] * 252)
    m = returns_metrics(r)
    assert m["n_obs"] == 252
    assert m["ann_return"] == pytest.approx((1.001 ** 252) - 1.0, rel=1e-6)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-9)
    assert m["ann_vol"] == pytest.approx(0.0, abs=1e-9)


def test_returns_metrics_drawdown_on_a_dip():
    # up 10%, down 20%, up 5% -> trough after the -20% is the max drawdown.
    r = pd.Series([0.10, -0.20, 0.05])
    m = returns_metrics(r)
    # curve: 1.1, 0.88, 0.924 ; peak 1.1 -> trough 0.88 -> dd = 1 - 0.88/1.1 = 0.20
    assert m["max_drawdown"] == pytest.approx(0.20, rel=1e-6)


# --------------------------------------------------------------------------- #
# slice_result — trades by ENTRY date, returns by date
# --------------------------------------------------------------------------- #
def test_slice_result_restricts_trades_and_returns():
    trades = [
        _trade(date(2018, 6, 1), 1.0, 1.0),   # in window
        _trade(date(2019, 6, 1), 2.0, 1.0),   # out (after)
        _trade(date(2017, 6, 1), 3.0, 1.0),   # out (before)
    ]
    idx = [date(2017, 12, 31), date(2018, 6, 1), date(2018, 12, 31), date(2019, 6, 1)]
    rets = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    run = _Run(trades, rets)
    sl = slice_result(run, "2018-01-01", "2018-12-31")
    assert [t.entry_date for t in sl["trades"]] == [date(2018, 6, 1)]
    assert list(sl["returns"].values) == pytest.approx([0.02, 0.03])


# --------------------------------------------------------------------------- #
# yearly_breakdown + pool_windows
# --------------------------------------------------------------------------- #
def test_yearly_breakdown_buckets_by_year():
    trades = [
        _trade(date(2018, 3, 1), 2.0, 1.0),
        _trade(date(2018, 9, 1), -1.0, -1.0),
        _trade(date(2019, 3, 1), 5.0, 2.0),
    ]
    idx = [date(2018, 6, 1), date(2019, 6, 1)]
    rets = pd.Series([0.01, 0.02], index=idx)
    run = _Run(trades, rets)
    rows = yearly_breakdown(run, "2018-01-01", "2019-12-31")
    by_year = {r["year"]: r for r in rows}
    assert by_year[2018]["n_trades"] == 2
    assert by_year[2018]["profit_factor"] == pytest.approx(2.0)  # +2 / 1
    assert by_year[2019]["n_trades"] == 1


def test_pool_windows_unions_trades():
    trades = [
        _trade(date(2018, 3, 1), 2.0, 1.0),
        _trade(date(2020, 3, 1), 4.0, 1.0),
        _trade(date(2019, 3, 1), -1.0, -1.0),   # excluded by the windows below
    ]
    rets = pd.Series([0.01], index=[date(2018, 6, 1)])
    run = _Run(trades, rets)
    pooled = pool_windows(run, [("2018-01-01", "2018-12-31"), ("2020-01-01", "2020-12-31")])
    assert pooled["n_pooled_trades"] == 2
    assert pooled["trade_metrics"]["profit_factor"] == float("inf")  # 2 winners, no losers


# --------------------------------------------------------------------------- #
# haircut_verdict — rising PF bar + deflated Sharpe gate
# --------------------------------------------------------------------------- #
def test_haircut_bar_rises_with_trials():
    # The PF bar is strictly increasing in n_trials (multiple-testing penalty).
    assert min_pf_threshold(40) > min_pf_threshold(5) > min_pf_threshold(1)
    v_few = haircut_verdict({"profit_factor": 1.6}, {"sharpe_per_period": 0.1, "n_obs": 500}, n_trials=5)
    v_many = haircut_verdict({"profit_factor": 1.6}, {"sharpe_per_period": 0.1, "n_obs": 500}, n_trials=40)
    assert v_many["pf_bar"] > v_few["pf_bar"]
    # PF 1.6 clears a 5-trial bar but not necessarily a 40-trial bar.
    assert v_few["clears_pf"] is True
    assert v_many["clears_pf"] == bool(1.6 >= v_many["pf_bar"])


def test_haircut_pf_is_the_gate_dsr_is_advisory():
    # PF is the BINARY gate (matching the intraday convention); DSR is advisory
    # context, not a veto. A strong PF with a weak DSR still CLEARS overall.
    v = haircut_verdict(
        {"profit_factor": 5.0},
        {"sharpe_per_period": 0.001, "n_obs": 30, "skew": 0.0, "kurt": 3.0},
        n_trials=10,
    )
    assert v["clears_pf"] is True
    assert v["clears_dsr"] is False     # advisory, does not veto
    assert v["clears"] is True          # the gate is the PF haircut


def test_haircut_dsr_meaningful_only_with_empirical_variance():
    # The deflated Sharpe is only meaningful for a daily strategy when the trial
    # Sharpe variance is supplied IN PER-PERIOD UNITS (~1e-4). A solid daily
    # Sharpe then reads as a real discovery (~>0.9); the textbook var=1.0 default
    # (an annualized-units assumption) instead crushes the same number to ~0.
    rm = {"sharpe_per_period": 0.07, "n_obs": 2000, "skew": 0.0, "kurt": 3.0}
    v_emp = haircut_verdict({"profit_factor": 2.0}, rm, n_trials=40, var_trials_sharpe=1e-4)
    v_def = haircut_verdict({"profit_factor": 2.0}, rm, n_trials=40, var_trials_sharpe=1.0)
    assert v_emp["deflated_sharpe"] > 0.9
    assert v_def["deflated_sharpe"] < 0.1
