"""tests/test_stats_confidence.py — bootstrap CIs + metrics.

Verifies (MASTER_PLAN §5):
  - the CI brackets the point estimate;
  - the CI WIDENS as the trade count shrinks (small samples are less certain);
  - every CI returns ``n`` alongside the interval;
  - ``is_underpowered`` flags samples under 100 trades;
  - the headline metrics agree with hand-computed values.
"""

from __future__ import annotations

import numpy as np
import pytest

from backtest.stats.confidence import (
    ci_width,
    expectancy_ci,
    is_underpowered,
    pf_ci,
)
from backtest.stats.metrics import (
    expectancy_dollar,
    expectancy_r,
    max_drawdown,
    profit_factor,
    sharpe,
    win_rate,
)


# ---- a reproducible trade-pnl sample with a known positive edge ----
def _make_pnls(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 45% winners of +2R, 55% losers of -1R (R = $100): a PF > 1 edge.
    wins = rng.random(n) < 0.45
    return np.where(wins, 200.0, -100.0)


# --------------------------------------------------------------------- metrics
def test_profit_factor_basic():
    pnls = [100.0, 50.0, -75.0, -25.0]
    # gross profit 150, gross loss 100 -> PF 1.5
    assert profit_factor(pnls) == pytest.approx(1.5)


def test_profit_factor_no_losers_is_inf():
    assert profit_factor([10.0, 20.0]) == float("inf")


def test_profit_factor_empty_is_nan():
    assert np.isnan(profit_factor([]))


def test_expectancy_and_win_rate():
    pnls = [100.0, -50.0, 100.0, -50.0]
    assert expectancy_dollar(pnls) == pytest.approx(25.0)
    assert win_rate(pnls) == pytest.approx(0.5)
    assert expectancy_r([2.0, -1.0, 2.0, -1.0]) == pytest.approx(0.5)


def test_sharpe_known():
    # Constant-mean series with known mean/std.
    rets = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)
    assert sharpe(rets) == pytest.approx(expected)


def test_sharpe_too_few_obs_is_nan():
    assert np.isnan(sharpe([0.01]))


def test_max_drawdown_fraction():
    eq = [100.0, 120.0, 90.0, 110.0]  # peak 120 -> trough 90 = 25% DD
    assert max_drawdown(eq) == pytest.approx(0.25)


def test_max_drawdown_monotone_up_is_zero():
    assert max_drawdown([100, 101, 102, 103]) == pytest.approx(0.0)


# ------------------------------------------------------------------ confidence
def test_pf_ci_brackets_point_estimate():
    pnls = _make_pnls(300)
    pf, lo, hi, n = pf_ci(pnls, n_boot=1000, seed=1)
    assert n == 300
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= pf <= hi
    assert pf == pytest.approx(profit_factor(pnls))


def test_expectancy_ci_brackets_point_estimate():
    pnls = _make_pnls(300)
    mean, lo, hi, n = expectancy_ci(pnls, kind="dollar", n_boot=1000, seed=1)
    assert n == 300
    assert lo <= mean <= hi
    assert mean == pytest.approx(expectancy_dollar(pnls))


def test_ci_widens_with_fewer_trades():
    # The same edge measured on fewer trades must give a WIDER expectancy CI.
    big = _make_pnls(400, seed=3)
    small = _make_pnls(40, seed=3)
    w_big = ci_width(expectancy_ci(big, n_boot=1500, seed=5))
    w_small = ci_width(expectancy_ci(small, n_boot=1500, seed=5))
    assert w_small > w_big


def test_pf_ci_widens_with_fewer_trades():
    big = _make_pnls(400, seed=9)
    small = _make_pnls(50, seed=9)
    w_big = ci_width(pf_ci(big, n_boot=1500, seed=2))
    w_small = ci_width(pf_ci(small, n_boot=1500, seed=2))
    assert w_small > w_big


def test_ci_returns_n_always():
    # Even an empty sample returns the count (0) — never just an interval.
    res = pf_ci([], n_boot=100)
    assert res[3] == 0
    res2 = expectancy_ci([], n_boot=100)
    assert res2[3] == 0


def test_ci_is_deterministic_with_seed():
    pnls = _make_pnls(200)
    a = pf_ci(pnls, n_boot=500, seed=42)
    b = pf_ci(pnls, n_boot=500, seed=42)
    assert a == b


def test_is_underpowered_threshold():
    assert is_underpowered(99) is True
    assert is_underpowered(100) is False
    assert is_underpowered(50, threshold=150) is True
