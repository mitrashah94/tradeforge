"""tests/test_portfolio_backtester.py — run_portfolio offline checks + the REAL-DB
blended-book acceptance (the maximization verdict).

Two layers:
  * offline (no DB): the backtester's plumbing — empty-universe guard, deposit /
    TWR / MWR wiring on a flat synthetic panel.
  * real-DB (skipped if ``market.duckdb`` is absent): run the three daily sleeves
    (rotation + swing_meanrev + swing_breakout) as standalone books AND as one
    blended book, then put the result through the SAME maximization machinery the
    registry verdict uses (``backtest.portfolio`` correlation / min-variance / g)
    and the daily promotion gate (``validation.haircut_verdict``). The durable
    claim asserted is the one the thesis actually rests on: the long-only
    MIN-VARIANCE blend's variance is BELOW every single sleeve's (decorrelation
    cuts variance -> safer size -> faster compounding), and the blended book's TWR
    curve is gate-checkable. Whether the blend also wins vol-targeted g vs the
    single best sleeve is DATA-dependent (here the breakout single edges it), so —
    exactly as ``tests/test_portfolio.py`` does for the intraday sleeves — that is
    reported, not hard-asserted.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.daily.bracket_engine import BracketConfig
from backtest.daily.portfolio_backtester import run_portfolio
from backtest.daily.validation import (
    haircut_verdict,
    pooled_trade_metrics,
    returns_metrics,
)
from backtest.portfolio import (
    blend_returns,
    correlation_matrix,
    min_variance_weights,
)
from data.schema import DEFAULT_DB_PATH
from portfolio.model import SleeveSpec


# --------------------------------------------------------------------------- #
# offline plumbing
# --------------------------------------------------------------------------- #
def _bdays(n, start=date(2024, 1, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_empty_universe_returns_empty_result():
    res = run_portfolio([], ["NOPE"], panel={
        "open": pd.DataFrame(), "high": pd.DataFrame(),
        "low": pd.DataFrame(), "close": pd.DataFrame(),
    })
    assert len(res.nav) == 0
    assert res.total_deposited == 0.0


def test_deposit_grows_nav_but_not_twr():
    idx = _bdays(4)
    flat = [(100, 100, 100, 100)] * 4
    panel = {
        f: pd.DataFrame({"AAA": [r[i] for r in flat]}, index=idx)
        for i, f in enumerate(("open", "high", "low", "close"))
    }

    class Idle:
        def target_weights(self, asof, history):
            return {}

    sleeves = [SleeveSpec(name="idle", strategy=Idle(), kind="weight")]
    res = run_portfolio(sleeves, ["AAA"], initial_equity=1000.0,
                        contributions={idx[1]: 500.0}, panel=panel)
    assert res.nav.iloc[-1] == pytest.approx(1500.0)        # deposit grew NAV
    assert res.total_deposited == pytest.approx(500.0)
    assert res.twr_returns.loc[idx[1]] == pytest.approx(0.0)  # but not the edge


# --------------------------------------------------------------------------- #
# real-DB blended book + maximization verdict
# --------------------------------------------------------------------------- #
dbmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_DB_PATH), reason="market.duckdb not present"
)

_START, _END = "2018-01-01", "2025-12-31"


def _build_sleeves():
    from strategies.momentum_rotation.strategy import (
        MomentumRotationStrategy, load_params as rl,
    )
    from strategies.swing_meanrev.strategy import (
        SwingMeanRevStrategy, load_params as ml,
    )
    from strategies.swing_breakout.strategy import (
        SwingBreakoutStrategy, load_params as bl,
    )
    rot = MomentumRotationStrategy(rl("DEFAULT"))
    mr = SwingMeanRevStrategy(ml("DEFAULT"))
    br = SwingBreakoutStrategy(bl("DEFAULT"))
    br_universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "JPM", "XLK", "QQQ", "SPY"]
    universe = sorted(set(rot.extra_symbols()) | set(mr.universe) | set(br_universe))
    sleeves = [
        SleeveSpec(name="rotation", strategy=rot, kind="weight", grade="B",
                   family="equity", benchmark="SPY", allocation=0.4),
        SleeveSpec(name="swing_meanrev", strategy=mr, kind="weight", grade="B",
                   family="meanrev", benchmark="SPY", allocation=0.3),
        SleeveSpec(name="swing_breakout", strategy=br, kind="score", grade="A",
                   family="trend", benchmark="SPY", allocation=0.3,
                   bracket=BracketConfig(atr_window=14)),
    ]
    return sleeves, universe


@pytest.fixture(scope="module")
def book_runs():
    from data.schema import connect
    con = connect(DEFAULT_DB_PATH)
    try:
        sleeves, universe = _build_sleeves()
        singles = {}
        for sp in sleeves:
            singles[sp.name] = run_portfolio(
                [sp], universe, start=_START, end=_END, initial_equity=1000.0, con=con
            )
        blend = run_portfolio(
            sleeves, universe, start=_START, end=_END, initial_equity=1000.0, con=con
        )
    finally:
        con.close()
    mat = pd.DataFrame({n: r.twr_returns for n, r in singles.items()}).fillna(0.0).sort_index()
    return {"singles": singles, "blend": blend, "mat": mat}


@dbmark
def test_blended_book_runs_end_to_end(book_runs):
    res = book_runs["blend"]
    s = res.summary()
    assert s["n_days"] > 1000
    assert s["n_trades"] > 0
    assert np.isfinite(s["twr_cagr"])
    assert np.isfinite(s["mwr_irr"])
    # attribution names every sleeve that traded.
    assert set(res.sleeve_attribution) <= {"rotation", "swing_meanrev", "swing_breakout"}


@dbmark
def test_minvariance_blend_cuts_variance_below_every_single(book_runs):
    # THE maximization thesis (durable form): the long-only min-variance blend's
    # variance is strictly below the lowest single sleeve's. True by construction
    # for a decorrelated set (the blend can always fall back to the lowest-var
    # single, and low/negative cross-correlation pushes it strictly lower).
    mat = book_runs["mat"]
    assert mat.shape[1] == 3
    cov = np.cov(mat.to_numpy().T, ddof=1)
    w = min_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()
    blend = blend_returns(mat, w)
    single_vars = {c: float(mat[c].var(ddof=1)) for c in mat.columns}
    blend_var = float(blend.var(ddof=1))
    assert blend_var < min(single_vars.values())   # strictly below the best single


@dbmark
def test_cross_sleeve_correlations_are_low(book_runs):
    # The decorrelation the variance cut depends on: every off-diagonal well below 1.
    corr = correlation_matrix(book_runs["mat"])
    off = corr.to_numpy()[~np.eye(3, dtype=bool)]
    assert (np.abs(off) < 0.8).all()


@dbmark
def test_blended_book_twr_is_gate_checkable(book_runs):
    # Run the blended TWR curve + pooled trade ledger through the daily promotion
    # haircut. The book carries a positive net edge (PF > 1) and — at the honest
    # small trial count for a 3-sleeve blend — clears the rising PF bar.
    res = book_runs["blend"]
    rm = returns_metrics(res.twr_returns)
    tm = pooled_trade_metrics(res.trades)
    hv = haircut_verdict(tm, rm, n_trials=3, var_trials_sharpe=1e-4)
    assert tm["n_trades"] > 0
    assert tm["profit_factor"] > 1.0
    assert set(hv) >= {"pf_bar", "profit_factor", "clears_pf", "clears"}
    assert hv["clears_pf"] is True
