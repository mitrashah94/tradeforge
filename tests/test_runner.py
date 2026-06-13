"""tests/test_runner.py — the canonical run harness + walk-forward.

Integration-flavored tests against the real market.duckdb (QQQ 5m), mirroring how
the gate loads data. They assert the runner's public contract that downstream
stages depend on:
  - ``load_bars_levels`` returns RTH bars + levels, clippable by date;
  - ``run_strategy`` accepts an instance OR a (factory, params) tuple and matches
    the gate's V0 result;
  - ``daily_returns`` is indexed by exit session_date and sums to net profit;
  - ``walk_forward`` produces folds whose OOS windows tile the span.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from data.schema import DEFAULT_DB_PATH, connect

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_DB_PATH),
    reason="market.duckdb not present",
)

from backtest.runner import (  # noqa: E402
    daily_returns,
    daily_returns_pct,
    load_bars_levels,
    levels_map,
    run_strategy,
)
from backtest.stats.walk_forward import fold_windows, walk_forward  # noqa: E402
from strategies.breakout_retest.strategy import (  # noqa: E402
    BreakoutRetestStrategy,
    load_params,
)


@pytest.fixture(scope="module")
def con():
    c = connect(DEFAULT_DB_PATH)
    yield c
    c.close()


def test_load_bars_levels_rth_and_columns(con):
    bars, levels = load_bars_levels("QQQ", "5m", con=con)
    assert len(bars) > 0
    assert list(bars.columns) == ["ts_utc", "open", "high", "low", "close", "volume"]
    assert {"session_date", "pdh", "pdl", "atr14"} <= set(levels.columns)
    # RTH filter: every bar is within regular trading hours.
    from data.sessions import is_rth

    assert bool(bars["ts_utc"].apply(is_rth).all())


def test_load_bars_levels_date_clip(con):
    full, _ = load_bars_levels("QQQ", "5m", con=con)
    clipped, _ = load_bars_levels(
        "QQQ", "5m", start="2025-01-01", end="2025-03-31", con=con
    )
    assert 0 < len(clipped) < len(full)
    from data.sessions import et_session_date

    sess = clipped["ts_utc"].apply(et_session_date)
    assert sess.min() >= pd.Timestamp("2025-01-01").date()
    assert sess.max() <= pd.Timestamp("2025-03-31").date()


def test_levels_map_keys_are_dates(con):
    _bars, levels = load_bars_levels("QQQ", "5m", con=con)
    m = levels_map(levels)
    assert len(m) > 0
    k = next(iter(m))
    from datetime import date

    assert isinstance(k, date)
    assert {"pdh", "pdl", "atr14", "ntz_valid"} <= set(m[k].keys())


def test_run_strategy_instance_matches_gate(con):
    strat = BreakoutRetestStrategy(params=load_params("V0"))
    res = run_strategy(strat, "QQQ", "5m", cost_profile="tv_style", con=con)
    s = res.summary()
    # The V0 reproduction: a few hundred trades, a real (finite, >1) PF.
    assert s["n_trades"] > 100
    assert s["profit_factor"] > 1.0
    assert s["profit_factor"] < 5.0


def test_run_strategy_accepts_factory_params_tuple(con):
    inst = run_strategy(
        BreakoutRetestStrategy(params=load_params("V0")),
        "QQQ", "5m", cost_profile="tv_style", con=con,
    )
    fac = run_strategy(
        (lambda p: BreakoutRetestStrategy(params=p), load_params("V0")),
        "QQQ", "5m", cost_profile="tv_style", con=con,
    )
    # Same config via either path -> identical headline result.
    assert inst.summary()["n_trades"] == fac.summary()["n_trades"]
    assert inst.summary()["profit_factor"] == pytest.approx(
        fac.summary()["profit_factor"]
    )


def test_run_strategy_bad_arg_raises(con):
    with pytest.raises(TypeError):
        run_strategy("not a strategy", "QQQ", "5m", con=con)


def test_daily_returns_indexed_by_exit_date_and_sums_to_net(con):
    strat = BreakoutRetestStrategy(params=load_params("V0"))
    res = run_strategy(strat, "QQQ", "5m", cost_profile="tv_style", con=con)
    dr = daily_returns(res)
    from datetime import date

    assert len(dr) > 0
    assert isinstance(dr.index[0], date)
    assert dr.index.is_monotonic_increasing
    # Bucketed pnl sums to the ledger net profit.
    assert dr.sum() == pytest.approx(res.summary()["net_profit"])


def test_daily_returns_pct_scales_by_equity(con):
    strat = BreakoutRetestStrategy(params=load_params("V0"))
    res = run_strategy(strat, "QQQ", "5m", cost_profile="tv_style", con=con)
    pct = daily_returns_pct(res)
    dollars = daily_returns(res)
    assert pct.sum() == pytest.approx(dollars.sum() / res.initial_equity)


def test_daily_returns_empty_result():
    from backtest.engine.result import BacktestResult

    empty = BacktestResult(
        trades=pd.DataFrame(),
        equity_curve=pd.Series(dtype="float64"),
        initial_equity=100_000.0,
    )
    assert len(daily_returns(empty)) == 0


def test_fold_windows_tile_and_are_ordered():
    w = fold_windows("2024-06-13", "2026-06-12", is_months=6, oos_months=1)
    assert len(w) > 0
    # OOS windows are contiguous, ordered, and inside the span.
    from datetime import date

    assert all(oe <= date(2026, 6, 12) for (_s, _e, _os, oe) in w)
    for (a, b) in zip(w, w[1:]):
        assert b[2] > a[2]  # next OOS start after prior OOS start


def test_walk_forward_runs_folds(con):
    def run_fn(s, e):
        strat = BreakoutRetestStrategy(params=load_params("V0"))
        return run_strategy(
            strat, "QQQ", "5m", start=s, end=e, cost_profile="tv_style", con=con
        )

    folds = walk_forward(
        run_fn, "2024-06-13", "2026-06-12", is_months=6, oos_months=1
    )
    assert len(folds) > 1
    assert all(f.oos_start < f.oos_end for f in folds)
    assert all(f.is_end < f.oos_start for f in folds)
