"""tests/test_kronos_forecast.py — the Kronos overlay (leakage guard, stats, store,
engine consumption). All offline; the model is MOCKED so no torch / weights.

Asserts the honesty constraint provably blocks pre-cutoff use; the distribution
reduction is deterministic; the store round-trips; and a SEEDED forecast table
deterministically changes the portfolio engine's ranking/veto (with the overlay
off by default so the engine runs identically without torch).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from forecast.kronos.leakage import (
    KRONOS_TRAINING_CUTOFF,
    LeakageError,
    assert_post_cutoff,
    is_post_cutoff,
)
from forecast.kronos.predictor import KronosForecaster, forecast_stats
from forecast.kronos.store import (
    make_forecast_provider,
    read_forecast,
    read_forecasts_asof,
    write_forecast,
    write_forecasts,
)


# --------------------------------------------------------------------------- #
# leakage guard — the honesty constraint
# --------------------------------------------------------------------------- #
def _bars_df(dates):
    return pd.DataFrame({"ts_utc": [datetime(d.year, d.month, d.day) for d in dates],
                         "close": [100.0] * len(dates)})


def test_leakage_raises_on_pre_cutoff_asof():
    # asof before the cutoff -> the forecast TARGET is inside training history.
    bars = _bars_df([date(2024, 12, 1), date(2024, 12, 31)])
    with pytest.raises(LeakageError):
        assert_post_cutoff(date(2025, 1, 1), bars)


def test_leakage_allows_pre_cutoff_context_by_default():
    # Pre-cutoff CONTEXT is the deployment condition (the model trained on its own
    # history) and leaks nothing about a post-cutoff TARGET -> allowed by default.
    asof = date(2026, 1, 5)
    bars = _bars_df([date(2025, 1, 1), asof])
    assert_post_cutoff(asof, bars)            # no raise
    # ... but the STRICT opt-in still refuses it.
    with pytest.raises(LeakageError):
        assert_post_cutoff(asof, bars, require_context_post_cutoff=True)


def test_leakage_raises_on_future_input_bar():
    # A bar dated AFTER asof is structural look-ahead -> always refused.
    asof = date(2026, 1, 5)
    bars = _bars_df([date(2025, 12, 1), date(2026, 1, 8)])
    with pytest.raises(LeakageError):
        assert_post_cutoff(asof, bars)


def test_leakage_passes_when_all_post_cutoff():
    asof = date(2026, 1, 5)
    bars = _bars_df([date(2025, 9, 1), date(2025, 12, 1), asof])
    assert_post_cutoff(asof, bars)            # no raise
    assert is_post_cutoff(asof, bars) is True
    assert is_post_cutoff(date(2024, 1, 1), bars) is False


def test_cutoff_is_conservative_2025():
    assert KRONOS_TRAINING_CUTOFF >= date(2025, 1, 1)


# --------------------------------------------------------------------------- #
# distribution reduction (pure, deterministic)
# --------------------------------------------------------------------------- #
def test_forecast_stats_shape_and_values():
    rets = [0.10, 0.05, -0.02, -0.08, 0.03, 0.01]
    s = forecast_stats(rets, alpha=0.34)   # ceil(0.34*6)=3 -> worst 3 of 6
    assert set(s) == {"exp_return", "vol", "downside_cvar", "prob_up"}
    assert s["exp_return"] == pytest.approx(np.mean(rets))
    assert s["prob_up"] == pytest.approx(4 / 6)
    # worst 3 returns are -0.08, -0.02, 0.01 -> CVaR = -mean = 0.03.
    assert s["downside_cvar"] == pytest.approx(0.03)


def test_forecast_stats_no_downside_is_zero_cvar():
    s = forecast_stats([0.01, 0.02, 0.03])
    assert s["downside_cvar"] == 0.0
    assert s["prob_up"] == 1.0


# --------------------------------------------------------------------------- #
# predictor with an INJECTED sampler (no torch)
# --------------------------------------------------------------------------- #
def test_forecast_distribution_is_deterministic_and_guarded():
    asof = date(2026, 1, 5)
    bars = _bars_df([date(2025, 9, 1), date(2025, 12, 1), asof])

    def sampler(hist, horizon, n_paths, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(0.01, 0.02, n_paths)

    fc = KronosForecaster(sampler=sampler)
    a = fc.forecast_distribution("NVDA", asof, bars, horizon=5, n_paths=64, seed=7)
    b = fc.forecast_distribution("NVDA", asof, bars, horizon=5, n_paths=64, seed=7)
    assert a["exp_return"] == b["exp_return"]   # fixed seed -> identical
    assert a["symbol"] == "NVDA" and a["n_paths"] == 64
    # the leakage guard fires even with a sampler:
    with pytest.raises(LeakageError):
        fc.forecast_distribution("NVDA", date(2024, 1, 1), _bars_df([date(2024, 1, 1)]))


# --------------------------------------------------------------------------- #
# store round-trip
# --------------------------------------------------------------------------- #
def test_store_write_read_round_trip():
    con = duckdb.connect(":memory:")
    sd = date(2026, 1, 5)
    write_forecast(con, "NVDA", sd, {"exp_return": 0.03, "vol": 0.02,
                                     "downside_cvar": 0.01, "prob_up": 0.6})
    got = read_forecast(con, "NVDA", sd)
    assert got["exp_return"] == pytest.approx(0.03)
    assert read_forecast(con, "AAPL", sd) is None
    # asof lookup + provider closure.
    write_forecasts(con, [{"symbol": "AAPL", "session_date": sd, "exp_return": -0.01,
                           "vol": 0.03, "downside_cvar": 0.05, "prob_up": 0.4}])
    asof_map = read_forecasts_asof(con, sd)
    assert set(asof_map) == {"NVDA", "AAPL"}
    provider = make_forecast_provider(con)
    assert provider(sd, "NVDA")["exp_return"] == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# engine consumption: a seeded table deterministically vetoes / re-ranks
# --------------------------------------------------------------------------- #
def _engine_with_kronos(provider, *, veto=False, blend=0.0):
    from backtest.daily.bracket_engine import BracketConfig
    from portfolio.config import KronosOverlay, PortfolioConfig, SyntheticStop
    from portfolio.engine import PortfolioEngine
    from portfolio.model import SleeveSpec

    pcfg = PortfolioConfig(
        synthetic_stop=SyntheticStop(window=2),
        kronos=KronosOverlay(use_kronos=True, rank_blend=blend, veto_negative_return=veto),
    )

    frm = date(2024, 6, 5)  # any date; arm defaults on (no SPY proxy)

    class TwoNames:
        def entry_score(self, symbol, asof, history):
            return {"AAA": 1.0, "BBB": 1.0}.get(symbol)

    sleeves = [SleeveSpec(name="brk", strategy=TwoNames(), kind="score", grade="A+",
                          family="trend", bracket=BracketConfig(atr_window=2))]
    return PortfolioEngine(sleeves, pcfg=pcfg, forecast_provider=provider), frm


def _bdays(n, start=date(2024, 6, 3)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _step_engine(eng, syms, days):
    from backtest.daily.bracket_engine import _atr_series
    from backtest.daily.engine import DailyHistory
    from portfolio.model import BookState, DayBars

    idx = _bdays(days)
    rows = {s: [(100, 101, 99, 100)] * days for s in syms}
    close = pd.DataFrame({s: [r[3] for r in rows[s]] for s in syms}, index=idx)
    high = pd.DataFrame({s: [r[1] for r in rows[s]] for s in syms}, index=idx)
    low = pd.DataFrame({s: [r[2] for r in rows[s]] for s in syms}, index=idx)
    opn = pd.DataFrame({s: [r[0] for r in rows[s]] for s in syms}, index=idx)
    atr2 = {s: _atr_series(high[s], low[s], close[s], 2) for s in syms}
    book = BookState(cash=100_000.0, peak_equity=100_000.0, month_start_equity=100_000.0,
                     week_start_equity=100_000.0, prev_nav=100_000.0)
    last = None
    for d in idx:
        bars = DayBars(asof=d, open=opn.loc[d], high=high.loc[d], low=low.loc[d],
                       close=close.loc[d], atr={2: pd.Series({s: atr2[s].loc[d] for s in syms})})
        last = eng.step(d, DailyHistory(close, d), bars, book)
    return book, last


def test_engine_kronos_veto_drops_negative_forecast():
    # BBB has a negative Kronos exp_return -> vetoed; AAA survives.
    forecasts = {"AAA": {"exp_return": 0.05, "downside_cvar": 0.01},
                 "BBB": {"exp_return": -0.05, "downside_cvar": 0.01}}
    eng, _ = _engine_with_kronos(lambda asof, sym: forecasts.get(sym), veto=True)
    book, alloc = _step_engine(eng, ["AAA", "BBB"], 4)
    assert "AAA" in book.lots
    assert "BBB" not in book.lots
    assert any(c.symbol == "BBB" and "kronos veto" in why for c, why in alloc.rejected)


def test_premarket_kronos_batch_guards_by_asof():
    # An in-memory bars table with two symbols. Under the amended guard,
    # pre-cutoff CONTEXT is allowed (deployment condition), so a post-cutoff asof
    # forecasts BOTH; a PRE-cutoff asof (target inside training history) is
    # leakage-refused for everything -> nothing written.
    from orchestrator.workflows.premarket import run_kronos_batch

    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE bars (symbol VARCHAR, timeframe VARCHAR, ts_utc TIMESTAMP, "
        "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)"
    )
    asof = date(2026, 1, 20)
    dates_a = [date(2025, 9, 1), date(2025, 12, 1), asof]
    dates_b = [date(2024, 1, 1), asof]     # pre-cutoff CONTEXT (allowed)
    for sym, dts in [("AAA", dates_a), ("BBB", dates_b)]:
        for d in dts:
            con.execute("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)",
                        [sym, "1d", datetime(d.year, d.month, d.day),
                         100.0, 101.0, 99.0, 100.0, 1000.0])

    def sampler(hist, horizon, n_paths, seed):
        return np.full(n_paths, 0.01)

    fc = KronosForecaster(sampler=sampler)
    n = run_kronos_batch(con, ["AAA", "BBB"], asof, fc, horizon=3, n_paths=8, lookback=10)
    assert n == 2                                   # both written (context OK)
    assert read_forecast(con, "AAA", asof, horizon=3) is not None
    assert read_forecast(con, "BBB", asof, horizon=3) is not None
    # a pre-cutoff ASOF is refused wholesale — the target would be in training.
    n_pre = run_kronos_batch(con, ["AAA", "BBB"], date(2025, 1, 10), fc,
                             horizon=3, n_paths=8, lookback=10)
    assert n_pre == 0


def test_engine_without_kronos_keeps_both():
    # Same setup but the overlay is OFF (no provider) -> BBB is NOT vetoed.
    #
    # Grade is "B" (not "A+") deliberately: this test only cares that nothing
    # gets vetoed by Kronos, not about the book's heat/concurrency admission
    # walk (that has its own dedicated tests). "B" keeps each candidate's
    # resolved-RI dollar risk small enough to fit two names under the book's
    # current heat cap for any operator-set risk_index.default within the
    # normal [5, 8] band (an A+ candidate's fixed 2%-of-equity risk does not).
    from backtest.daily.bracket_engine import BracketConfig
    from portfolio.config import load_portfolio_config
    from portfolio.engine import PortfolioEngine
    from portfolio.model import SleeveSpec
    from risk.config import load_limits
    from risk.sizing import per_trade_dollar_risk, resolve_ri

    pcfg = load_portfolio_config()
    import dataclasses
    pcfg = dataclasses.replace(pcfg, synthetic_stop=dataclasses.replace(pcfg.synthetic_stop, window=2))

    limits = load_limits()
    row = limits.level(limits.default_ri)
    cand_risk = per_trade_dollar_risk(100_000.0, resolve_ri("B", limits), limits)
    assert 2 * cand_risk <= row.portfolio_heat_pct / 100.0 * 100_000.0 + 1e-9, (
        "test setup: two grade-B candidates no longer fit under this config's "
        "book heat cap -- the fixture needs a smaller grade/equity combination"
    )
    assert row.max_concurrent >= 2, "test setup: this config's max_concurrent < 2"

    class TwoNames:
        def entry_score(self, symbol, asof, history):
            return {"AAA": 1.0, "BBB": 1.0}.get(symbol)
    sleeves = [SleeveSpec(name="brk", strategy=TwoNames(), kind="score", grade="B",
                          family="trend", bracket=BracketConfig(atr_window=2))]
    eng = PortfolioEngine(sleeves, pcfg=pcfg)   # no forecast_provider
    book, _ = _step_engine(eng, ["AAA", "BBB"], 4)
    assert "AAA" in book.lots and "BBB" in book.lots
