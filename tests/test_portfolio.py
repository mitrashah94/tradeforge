"""tests/test_portfolio.py — the edge-portfolio / maximization analysis.

Unit tests for the math (min-variance solver, simplex projection, blend stats,
the g proxy) that need no DB, plus an integration test against the real
market.duckdb that asserts the analysis runs end-to-end and the verdict is
internally consistent.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from data.schema import DEFAULT_DB_PATH, connect

from backtest.portfolio import (  # noqa: E402
    _g,
    _project_simplex,
    _series_stats,
    analyze,
    blend_returns,
    correlation_matrix,
    min_variance_weights,
)


# --------------------------------------------------------------------------- #
# Pure math (no DB)
# --------------------------------------------------------------------------- #
def test_project_simplex_sums_to_one_and_nonneg():
    v = np.array([0.5, -2.0, 3.0, 0.1])
    w = _project_simplex(v)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all()


def test_min_variance_prefers_the_lower_variance_asset():
    # Two uncorrelated assets, one far more volatile -> minvar tilts to the calm one.
    cov = np.array([[1e-4, 0.0], [0.0, 4e-4]])
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()
    assert w[0] > w[1]  # more weight on the lower-variance asset
    # For two uncorrelated assets minvar weight ~ inverse-variance.
    assert w[0] == pytest.approx(4.0 / 5.0, abs=0.02)


def test_min_variance_negative_correlation_lowers_blend_variance():
    # Perfectly anti-correlated equal-variance assets -> minvar variance ~ 0.
    s = 1e-2
    cov = np.array([[s * s, -s * s], [-s * s, s * s]])
    w = min_variance_weights(cov)
    blend_var = float(w @ cov @ w)
    assert blend_var < min(cov[0, 0], cov[1, 1])
    assert w[0] == pytest.approx(0.5, abs=0.05)


def test_g_proxy_matches_formula():
    assert _g(0.01, 0.0004) == pytest.approx(0.01 - 0.5 * 0.0004)


def test_blend_returns_is_weighted_sum():
    mat = pd.DataFrame(
        {"a": [0.1, -0.2, 0.3], "b": [0.0, 0.4, -0.1]},
        index=pd.Index([1, 2, 3], name="session_date"),
    )
    w = np.array([0.5, 0.5])
    b = blend_returns(mat, w)
    assert list(b.values) == pytest.approx([0.05, 0.1, 0.1])


def test_correlation_matrix_is_square_and_diag_one():
    mat = pd.DataFrame({"a": [0.1, -0.2, 0.3, 0.0], "b": [-0.1, 0.2, -0.3, 0.05]})
    c = correlation_matrix(mat)
    assert c.shape == (2, 2)
    assert c.loc["a", "a"] == pytest.approx(1.0)
    assert c.loc["a", "b"] == pytest.approx(c.loc["b", "a"])


def test_series_stats_g_consistency():
    s = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
    st = _series_stats(s)
    assert st["g"] == pytest.approx(st["mean_daily"] - 0.5 * st["var_daily"])
    assert st["n_days"] == 5


# --------------------------------------------------------------------------- #
# Integration (real DB)
# --------------------------------------------------------------------------- #
pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_DB_PATH), reason="market.duckdb not present"
)


@pytest.fixture(scope="module")
def con():
    c = connect(DEFAULT_DB_PATH)
    yield c
    c.close()


def test_analyze_runs_end_to_end(con):
    # Use the in-sample window to stay off the locked OOS vault.
    an = analyze(con=con, start="2024-06-13", end="2026-01-17")
    # Three sleeves, aligned calendar, square correlation matrix.
    assert len(an.names) == 3
    assert an.corr.shape == (3, 3)
    assert an.returns.shape[1] == 3
    # Min-variance weights are a valid long-only allocation.
    assert an.minvar_weights.sum() == pytest.approx(1.0)
    assert (an.minvar_weights >= -1e-9).all()
    # g-table has every single + the two blends.
    assert "BLEND_equal" in an.g_table.index
    assert "BLEND_minvar" in an.g_table.index


def test_blend_variance_below_best_single(con):
    # The structural maximization claim: the min-variance blend's variance is
    # below the best single's (decorrelation cuts variance). This is the honest
    # win even when the blend loses on raw g.
    an = analyze(con=con, start="2024-06-13", end="2026-01-17")
    v = an.verdict
    assert v["blend_lowers_variance_vs_best_single"] is True
    # And the verdict's g uplift is internally consistent with the g-table.
    blends = an.g_table[an.g_table["is_blend"]]
    assert v["best_blend_g"] == pytest.approx(float(blends["g"].max()))


def test_minvar_blend_has_highest_sharpe(con):
    # The scale-invariant maximization signal: the decorrelated min-variance
    # blend should be the highest-Sharpe sleeve.
    an = analyze(con=con, start="2024-06-13", end="2026-01-17")
    assert an.verdict["blend_has_best_sharpe"] is True
