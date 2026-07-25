"""tests/test_prop_simulate.py — the prop simulator: P(pass), payout, EV, leverage.

Deterministic, offline. Uses CONSTANT return streams (so the block bootstrap is
degenerate and every path is identical -> exact, hand-computable P(pass) and
payout) plus a mixed stream for a between-0-and-1 sanity. Asserts the returns->P&L
scaling, the bootstrap shape/determinism, that a positive-drift edge passes and a
negative one busts, the funded-year payout arithmetic, and the end-to-end EV +
$100k scaling answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from prop.rules import load_firm
from prop.simulate import (
    block_bootstrap,
    expected_value,
    funded_year_payout,
    historical_eval,
    monte_carlo_eval,
    returns_to_pnl,
    sweep_leverage,
)


def test_returns_to_pnl_scales_by_account_and_leverage():
    pnl = returns_to_pnl([0.01, -0.02], account_size=50_000, leverage=2.0)
    assert list(pnl) == pytest.approx([1_000.0, -2_000.0])


def test_block_bootstrap_shape_and_determinism():
    r = [0.01, -0.01, 0.02, -0.02, 0.0]
    a = block_bootstrap(r, horizon=10, n_paths=7, block=3, seed=42)
    b = block_bootstrap(r, horizon=10, n_paths=7, block=3, seed=42)
    assert a.shape == (7, 10)
    assert np.array_equal(a, b)                       # same seed -> identical
    assert set(np.unique(a)).issubset(set(r))         # values come from the input


def test_positive_edge_passes_negative_edge_busts():
    r = load_firm("futures_50k")   # target +3000, trailing DD 2500, min 7 days
    # +0.004/day at 3x -> +600/day, monotone up -> passes at day 7, never breaches DD.
    up = monte_carlo_eval([0.004] * 40, r, leverage=3.0, horizon=30, n_paths=50, seed=1)
    assert up["p_pass"] == pytest.approx(1.0)
    # -0.004/day at 3x -> -600/day -> breaches the trailing floor -> all bust.
    down = monte_carlo_eval([-0.004] * 40, r, leverage=3.0, horizon=30, n_paths=50, seed=1)
    assert down["p_pass"] == pytest.approx(0.0)
    assert down["p_fail"] == pytest.approx(1.0)


def test_historical_eval_rolls_real_windows():
    r = load_firm("futures_50k")
    res = historical_eval([0.004] * 40, r, leverage=3.0, horizon=30)
    assert res["n_windows"] >= 1
    assert res["p_pass"] == pytest.approx(1.0)         # constant up stream always passes


def test_funded_year_payout_arithmetic():
    r = load_firm("futures_50k")   # split 0.90, payout_min 2% ($1000)
    # +0.008/day at 1x -> +400/day; 21-day cycle profit 8400 -> keep 0.90*8400=7560;
    # 252/21 = 12 cycles -> 90720; monotone up -> survives.
    fund = funded_year_payout([0.008] * 5, r, leverage=1.0, days=252,
                              n_paths=20, block=5, seed=3)
    assert fund["mean_payout"] == pytest.approx(90_720.0, rel=1e-6)
    assert fund["survival_rate"] == pytest.approx(1.0)


def test_expected_value_and_target_scaling():
    r = load_firm("futures_50k")
    ev = expected_value([0.006] * 5, r, leverage=2.0, horizon=30,
                        funded_days=252, n_paths=30, seed=0, target_payout=100_000.0)
    assert set(ev) >= {"p_pass", "mean_annual_payout_per_account", "ev_one_attempt",
                       "accounts_for_target_payout", "expected_fee_to_pass"}
    assert ev["p_pass"] == pytest.approx(1.0)          # constant up edge
    assert ev["mean_annual_payout_per_account"] > 0
    assert ev["ev_one_attempt"] > 0                    # EV net of the $150 fee is positive
    assert ev["accounts_for_target_payout"] >= 1


def test_sweep_leverage_returns_row_per_leverage():
    r = load_firm("futures_50k")
    rows = sweep_leverage([0.002, -0.001, 0.003, -0.002, 0.004] * 4, r,
                          leverages=(1.0, 2.0, 4.0), horizon=40, n_paths=40, seed=0)
    assert [row["leverage"] for row in rows] == [1.0, 2.0, 4.0]
    for row in rows:
        assert 0.0 <= row["p_pass"] <= 1.0
