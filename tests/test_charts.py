"""tests/test_charts.py — headless chart rendering smoke tests.

Asserts the Agg-backed renderers in ``reporting.charts`` actually write a
non-empty PNG for each entry point, from both a bare return Series (a blend) and
a duck-typed BacktestResult-like object. No display, no DB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reporting.charts import (
    blend_vs_singles,
    correlation_heatmap,
    tear_sheet,
)


def _png_nonempty(path) -> bool:
    import os

    return os.path.exists(path) and os.path.getsize(path) > 1000


def test_tear_sheet_from_returns_series(tmp_path):
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0002, 0.005, size=120))
    out = tmp_path / "ts_returns.png"
    p = tear_sheet(rets, "test blend", out)
    assert _png_nonempty(p)


def test_tear_sheet_from_result_like(tmp_path):
    # A minimal duck-typed BacktestResult: trades df + equity_curve + summary().
    class FakeResult:
        def __init__(self):
            self.initial_equity = 100_000.0
            self.trades = pd.DataFrame(
                {
                    "pnl": [100.0, -50.0, 200.0, -30.0],
                    "r_multiple": [2.0, -1.0, 2.0, -1.0],
                    "exit_ts": pd.to_datetime(
                        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
                    ),
                }
            )
            self.equity_curve = pd.Series(
                [100_000, 100_100, 100_050, 100_250, 100_220],
                index=pd.RangeIndex(5),
            )

        def summary(self):
            return {
                "n_trades": 4,
                "profit_factor": 3.75,
                "expectancy_dollar": 55.0,
                "expectancy_R": 0.5,
                "win_rate": 0.5,
                "net_profit": 220.0,
                "max_drawdown_pct": 0.05,
            }

    out = tmp_path / "ts_result.png"
    p = tear_sheet(FakeResult(), "fake result", out)
    assert _png_nonempty(p)


def test_correlation_heatmap(tmp_path):
    names = ["a", "b", "c"]
    m = np.array([[1.0, -0.2, 0.05], [-0.2, 1.0, -0.1], [0.05, -0.1, 1.0]])
    corr = pd.DataFrame(m, index=names, columns=names)
    out = tmp_path / "corr.png"
    p = correlation_heatmap(corr, out)
    assert _png_nonempty(p)


def test_blend_vs_singles(tmp_path):
    df = pd.DataFrame(
        {
            "mean_daily": [0.0002, 0.0001, 0.00015, 0.00012],
            "var_daily": [9e-6, 2e-5, 7e-5, 5e-6],
            "sharpe": [0.9, 0.3, 0.5, 1.1],
            "g": [0.00019, 0.00009, 0.0001, 0.000118],
            "is_blend": [False, False, False, True],
            "n_days": [300, 300, 300, 300],
        },
        index=["s1", "s2", "s3", "BLEND_minvar"],
    )
    out = tmp_path / "g.png"
    p = blend_vs_singles(df, out)
    assert _png_nonempty(p)
