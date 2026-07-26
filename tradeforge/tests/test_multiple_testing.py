"""tests/test_multiple_testing.py — data-mining guards.

Verifies (MASTER_PLAN §5/§6):
  - ``min_pf_threshold`` is monotonically increasing in n_trials;
  - ``deflated_sharpe`` always returns a probability in [0, 1] and behaves
    correctly (decreasing in n_trials, increasing in the observed Sharpe);
  - the standard-normal CDF/PPF helpers are accurate (no scipy);
  - ``HypothesisLog`` appends every hypothesis and round-trips them back,
    including non-finite (inf/nan) profit factors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtest.stats.multiple_testing import (
    HypothesisLog,
    deflated_sharpe,
    expected_max_sharpe,
    min_pf_threshold,
    norm_cdf,
    norm_ppf,
)


# ------------------------------------------------------------ normal helpers
def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-6)


# ----------------------------------------------------------- min_pf_threshold
def test_min_pf_threshold_monotonic_increasing():
    ns = [1, 2, 3, 5, 10, 25, 50, 100, 500]
    vals = [min_pf_threshold(n) for n in ns]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_min_pf_threshold_base_at_one_trial():
    assert min_pf_threshold(1) == pytest.approx(1.3)


def test_min_pf_threshold_clamps_below_one():
    # n<1 is clamped to 1, never below base.
    assert min_pf_threshold(0) == pytest.approx(1.3)


# ------------------------------------------------------------ deflated_sharpe
def test_deflated_sharpe_in_unit_interval():
    for sr in (-2.0, -0.1, 0.0, 0.1, 0.5, 2.0):
        for nt in (1, 5, 50, 500):
            d = deflated_sharpe(sr, n_trials=nt, n_obs=200)
            assert 0.0 <= d <= 1.0


def test_deflated_sharpe_decreasing_in_n_trials():
    # More trials -> the false-discovery benchmark rises -> DSR falls.
    vals = [
        deflated_sharpe(0.3, n_trials=nt, n_obs=250, var_trials_sharpe=0.01)
        for nt in (1, 2, 5, 10, 50, 100)
    ]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[0] > vals[-1]  # strictly lower with many trials


def test_deflated_sharpe_increasing_in_sharpe():
    lo = deflated_sharpe(0.1, n_trials=10, n_obs=250, var_trials_sharpe=0.01)
    hi = deflated_sharpe(0.4, n_trials=10, n_obs=250, var_trials_sharpe=0.01)
    assert hi > lo


def test_deflated_sharpe_fat_tails_lower_it():
    normal = deflated_sharpe(1.0, 5, 60, skew=0.0, kurt=3.0, var_trials_sharpe=0.05)
    fat = deflated_sharpe(1.0, 5, 60, skew=-1.0, kurt=8.0, var_trials_sharpe=0.05)
    assert fat < normal


def test_deflated_sharpe_too_few_obs_is_zero():
    assert deflated_sharpe(1.0, 1, 1) == 0.0


def test_expected_max_sharpe_rises_with_trials():
    vals = [expected_max_sharpe(n) for n in (1, 2, 10, 100)]
    assert vals[0] == 0.0  # one trial: no selection bias
    assert all(b > a for a, b in zip(vals, vals[1:]))


# -------------------------------------------------------------- HypothesisLog
def test_hypothesis_log_appends_and_round_trips(tmp_path):
    log = HypothesisLog(path=tmp_path / "hyp.jsonl")
    assert log.count() == 0

    log.append(
        name="V0",
        params={"r_multiple": 2.0, "entry": "retest"},
        n_trades=188,
        pf=1.41,
        expectancy_r=0.13,
        sharpe=1.75,
        timestamp="2026-06-13T00:00:00",
        passed=True,
    )
    log.append(
        name="V1_ntz",
        params={"ntz_filter": True},
        n_trades=5,
        pf=float("inf"),  # degenerate: no losers -> inf PF must survive a round trip
        expectancy_r=float("nan"),
        sharpe=0.2,
        timestamp="2026-06-13T00:05:00",
        passed=False,
        note="too few trades",
    )

    assert log.count() == 2
    recs = log.read_all()
    assert [r.name for r in recs] == ["V0", "V1_ntz"]
    assert recs[0].params == {"r_multiple": 2.0, "entry": "retest"}
    assert recs[0].passed is True
    assert math.isinf(recs[1].pf)
    assert math.isnan(recs[1].expectancy_r)
    assert recs[1].note == "too few trades"


def test_hypothesis_log_is_append_only(tmp_path):
    p = tmp_path / "hyp.jsonl"
    log = HypothesisLog(path=p)
    for i in range(3):
        log.append(f"h{i}", {}, 10, 1.2, 0.05, 0.3, f"2026-06-13T00:0{i}:00", True)
    # A second handle to the same file sees all prior appends.
    log2 = HypothesisLog(path=p)
    assert log2.count() == 3
    assert [r.name for r in log2.read_all()] == ["h0", "h1", "h2"]


def test_hypothesis_log_logs_failures_too(tmp_path):
    # The discipline: log EVERY hypothesis, survivors and failures alike.
    log = HypothesisLog(path=tmp_path / "hyp.jsonl")
    log.append("winner", {}, 200, 1.6, 0.2, 1.0, "2026-06-13T00:00:00", True)
    log.append("loser", {}, 200, 0.8, -0.1, -0.5, "2026-06-13T00:01:00", False)
    passed = [r.passed for r in log.read_all()]
    assert passed == [True, False]
    assert log.count() == 2
