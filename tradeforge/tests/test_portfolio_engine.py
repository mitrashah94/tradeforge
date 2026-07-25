"""tests/test_portfolio_engine.py — the cross-strategy book engine, end-to-end.

Deterministic, OFFLINE (no DB): every test hand-builds a tiny synthetic ADJUSTED
daily OHLC panel and either drives ``run_portfolio`` or steps the
:class:`PortfolioEngine` directly, asserting the exact multi-sleeve mechanics the
plan specifies:

  * multi-sleeve: breakout fires on A, meanrev wants B, rotation wants C ->
    exactly those three open, with hand-computed fractional shares;
  * same-symbol conflict: two sleeves want A -> the higher conviction grade holds;
  * cluster dedup: two ~1.0-correlated names -> only the top-ranked opens;
  * heat: admission stops exactly at portfolio_heat_pct;
  * drawdown: NAV -36% from peak -> opens halted, closes (risk-reducers) execute;
  * the inherited bracket mechanics (TP1 -> breakeven, gap-down fill) on a score lot;
  * deposits: TWR excludes the flow, MWR includes it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.daily.bracket_engine import BracketConfig, _atr_series
from backtest.daily.engine import DailyHistory
from backtest.daily.portfolio_backtester import run_portfolio
from portfolio.config import load_portfolio_config
from portfolio.engine import PortfolioEngine
from portfolio.model import BookState, DayBars, OpenLot, SleeveSpec
from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk, resolve_ri


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bdays(n, start=date(2024, 1, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _panel(bars: dict, dates=None):
    """{symbol: [(o,h,l,c), ...]} -> {'open'/'high'/'low'/'close': wide df}."""
    n = len(next(iter(bars.values())))
    idx = dates if dates is not None else _bdays(n)
    fields = {"open": {}, "high": {}, "low": {}, "close": {}}
    for sym, rows in bars.items():
        o, h, l, c = zip(*rows)
        fields["open"][sym] = list(o)
        fields["high"][sym] = list(h)
        fields["low"][sym] = list(l)
        fields["close"][sym] = list(c)
    return {f: pd.DataFrame(fields[f], index=idx) for f in fields}


def _flat(price, rng=0.0):
    return (price, price + rng, price - rng, price)


def _small_synth_pcfg():
    """A pcfg with a 2-day synthetic-stop window so ATR warms in tiny fixtures."""
    pcfg = load_portfolio_config()
    object.__setattr__(pcfg.synthetic_stop, "window", 2)
    return pcfg


def _limits_with_min_concurrent(n):
    """A test-local copy of the real risk/limits.yaml with the operator's current
    floor row's max_concurrent raised to at least ``n``.

    Used only to isolate multi-sleeve OPEN mechanics (exact share counts,
    exposure) from the book's concurrency cap, which has its own dedicated
    tests (test_heat_cap_limits_concurrent_opens /
    test_portfolio_budget::test_concurrency_cap_stops_walk). Every other
    config value -- per_trade_pct, heat, halts -- stays exactly what the
    operator set in risk/limits.yaml; this never touches the file itself.
    """
    limits = load_limits()
    row = limits.table[limits.default_ri]
    if row.max_concurrent >= n:
        return limits
    new_row = row.model_copy(update={"max_concurrent": n})
    table = dict(limits.table)
    table[limits.default_ri] = new_row
    return limits.model_copy(update={"table": table})


def _corr(mapping):
    syms = list(mapping.keys())
    df = pd.DataFrame(index=syms, columns=syms, dtype="float64")
    for a in syms:
        for b in syms:
            df.at[a, b] = mapping[a].get(b, 0.0)
        df.at[a, a] = 1.0
    return df


# --- test strategies ---
class ScoreFrom:
    """entry_score 1.0 for ``sym`` on/after ``frm`` (a one-name breakout)."""
    def __init__(self, sym, frm, score=1.0):
        self.sym, self.frm, self._s = sym, frm, score
    def entry_score(self, symbol, asof, history):
        return self._s if (symbol == self.sym and asof >= self.frm) else None


class WantFrom:
    """target_weights = ``targets`` on/after ``frm`` (a standing allocation)."""
    def __init__(self, targets, frm):
        self.targets, self.frm = dict(targets), frm
    def target_weights(self, asof, history):
        return dict(self.targets) if asof >= self.frm else {}


# --------------------------------------------------------------------------- #
# (1) multi-sleeve: breakout A + meanrev B + rotation C -> exact opens
# --------------------------------------------------------------------------- #
def test_multi_sleeve_opens_exact_shares():
    idx = _bdays(6)
    frm = idx[3]
    panel = _panel({
        "AAA": [_flat(100.0, 1.0)] * 6,   # score name (range 1 -> ATR 2 via h/l? see below)
        "BBB": [_flat(500.0, 0.5)] * 6,   # meanrev weight name
        "CCC": [_flat(500.0, 0.5)] * 6,   # rotation weight name
    })
    # AAA bars are (100,101,99,100): TR=2 -> ATR(2)=2 -> stop 100-2.5*2=95, rps 5.
    sleeves = [
        SleeveSpec(name="brk", strategy=ScoreFrom("AAA", frm), kind="score", grade="B",
                   family="trend", bracket=BracketConfig(atr_window=2)),
        SleeveSpec(name="mr", strategy=WantFrom({"BBB": 0.2}, frm), kind="weight",
                   grade="B", family="meanrev"),
        SleeveSpec(name="rot", strategy=WantFrom({"CCC": 0.2}, frm), kind="weight",
                   grade="B", family="equity"),
    ]
    # 3 concurrent opens need max_concurrent >= 3 at the book's current floor row
    # (see _limits_with_min_concurrent); every dollar amount below is still
    # read straight from that config, not hardcoded.
    limits = _limits_with_min_concurrent(3)
    res = run_portfolio(sleeves, ["AAA", "BBB", "CCC"], initial_equity=100_000.0,
                        pcfg=_small_synth_pcfg(), panel=panel, limits=limits)
    # Reconstruct holdings from the engine via a fresh run capturing the book.
    # Simplest: assert exposure + that all three names traded (open or held).
    # AAA risk = per_trade_dollar_risk(equity, resolve_ri("B", limits), limits) / rps(5);
    # BBB/CCC 0.2*100k/500 = 40 sh each.
    # Mark-to-close on the final bar with flat prices -> NAV ~ 100k (minus costs).
    assert res.nav.iloc[-1] == pytest.approx(100_000.0, rel=2e-3)
    aaa_dollar_risk = per_trade_dollar_risk(100_000.0, resolve_ri("B", limits), limits)
    aaa_shares = aaa_dollar_risk / 5.0
    expected_exposure = (aaa_shares * 100.0 + 40.0 * 500.0 + 40.0 * 500.0) / 100_000.0
    assert res.exposure.iloc[-1] == pytest.approx(expected_exposure, rel=2e-2)


def test_multi_sleeve_book_state_exact():
    # Same setup, but step the engine directly so we can read the lots.
    idx = _bdays(6)
    frm = idx[3]
    panel = _panel({
        "AAA": [(100, 101, 99, 100)] * 6,
        "BBB": [(500, 500.5, 499.5, 500)] * 6,
        "CCC": [(500, 500.5, 499.5, 500)] * 6,
    })
    close = panel["close"]
    pcfg = _small_synth_pcfg()
    atr2 = {s: _atr_series(panel["high"][s], panel["low"][s], close[s], 2) for s in close.columns}
    sleeves = [
        SleeveSpec(name="brk", strategy=ScoreFrom("AAA", frm), kind="score", grade="B",
                   family="trend", bracket=BracketConfig(atr_window=2)),
        SleeveSpec(name="mr", strategy=WantFrom({"BBB": 0.2}, frm), kind="weight",
                   grade="B", family="meanrev"),
        SleeveSpec(name="rot", strategy=WantFrom({"CCC": 0.2}, frm), kind="weight",
                   grade="B", family="equity"),
    ]
    # 3 concurrent opens need max_concurrent >= 3 at the book's current floor row.
    limits = _limits_with_min_concurrent(3)
    eng = PortfolioEngine(sleeves, pcfg=pcfg, limits=limits)
    book = BookState(cash=100_000.0, peak_equity=100_000.0, month_start_equity=100_000.0,
                     week_start_equity=100_000.0, prev_nav=100_000.0)
    entry_alloc = None
    for d in idx:
        bars = DayBars(asof=d, open=panel["open"].loc[d], high=panel["high"].loc[d],
                       low=panel["low"].loc[d], close=close.loc[d],
                       atr={2: pd.Series({s: atr2[s].loc[d] for s in close.columns})})
        alloc = eng.step(d, DailyHistory(close, d), bars, book)
        if d == frm:
            entry_alloc = alloc
    # Assert the EXACT hand-computed opens on the entry day (before any reconcile
    # drift as cash/equity shifts): AAA risk = per_trade_dollar_risk(equity,
    # resolve_ri("B", limits), limits) / rps(5); BBB/CCC 0.2*100k/500 -> 40 sh each.
    aaa_dollar_risk = per_trade_dollar_risk(100_000.0, resolve_ri("B", limits), limits)
    aaa_shares = aaa_dollar_risk / 5.0
    opens = {o.symbol: o for o in entry_alloc.opens}
    assert opens["AAA"].shares == pytest.approx(aaa_shares)
    assert opens["AAA"].stop == pytest.approx(95.0)
    assert opens["BBB"].shares == pytest.approx(40.0)
    assert opens["CCC"].shares == pytest.approx(40.0)
    assert {s: l.sleeve for s, l in book.lots.items()} == {"AAA": "brk", "BBB": "mr", "CCC": "rot"}
    assert book.lots["AAA"].shares == pytest.approx(aaa_shares)  # score lot: no reconcile drift


# --------------------------------------------------------------------------- #
# (2) same-symbol conflict -> higher grade holds
# --------------------------------------------------------------------------- #
def test_same_symbol_conflict_higher_grade_holds():
    idx = _bdays(5)
    frm = idx[2]
    panel = _panel({"AAA": [(100, 101, 99, 100)] * 5})
    close = panel["close"]
    atr2 = _atr_series(panel["high"]["AAA"], panel["low"]["AAA"], close["AAA"], 2)
    sleeves = [
        # weight sleeve (grade B) wants AAA; score sleeve (grade A+) also wants AAA.
        SleeveSpec(name="rot", strategy=WantFrom({"AAA": 0.3}, frm), kind="weight", grade="B",
                   family="equity"),
        SleeveSpec(name="brk", strategy=ScoreFrom("AAA", frm), kind="score", grade="A+",
                   family="trend", bracket=BracketConfig(atr_window=2)),
    ]
    eng = PortfolioEngine(sleeves, pcfg=_small_synth_pcfg())
    book = BookState(cash=100_000.0, peak_equity=100_000.0, month_start_equity=100_000.0,
                     week_start_equity=100_000.0, prev_nav=100_000.0)
    alloc = None
    for d in idx:
        bars = DayBars(asof=d, open=panel["open"].loc[d], high=panel["high"].loc[d],
                       low=panel["low"].loc[d], close=close.loc[d],
                       atr={2: pd.Series({"AAA": atr2.loc[d]})})
        alloc = eng.step(d, DailyHistory(close, d), bars, book)
    # AAA held by the A+ score sleeve (a long bracket lot), not the weight sleeve.
    assert "AAA" in book.lots
    assert book.lots["AAA"].sleeve == "brk" and book.lots["AAA"].kind == "score"


# --------------------------------------------------------------------------- #
# (3) cluster dedup -> only the top-ranked of a correlated pair opens
# --------------------------------------------------------------------------- #
def test_cluster_dedup_only_top_opens():
    idx = _bdays(5)
    frm = idx[2]
    panel = _panel({
        "AAA": [(100, 101, 99, 100)] * 5,
        "ZZZ": [(100, 101, 99, 100)] * 5,
    })
    close = panel["close"]
    atr2 = {s: _atr_series(panel["high"][s], panel["low"][s], close[s], 2) for s in close.columns}
    # one score sleeve scores AAA higher than ZZZ; they are ~1.0 correlated.
    class TwoNames:
        def entry_score(self, symbol, asof, history):
            if asof < frm:
                return None
            return {"AAA": 2.0, "ZZZ": 1.0}.get(symbol)
    sleeves = [SleeveSpec(name="brk", strategy=TwoNames(), kind="score", grade="B",
                          family="trend", bracket=BracketConfig(atr_window=2))]
    corr = _corr({"AAA": {"ZZZ": 0.99}, "ZZZ": {"AAA": 0.99}})
    eng = PortfolioEngine(sleeves, pcfg=_small_synth_pcfg(), correlation_matrix=corr)
    book = BookState(cash=100_000.0, peak_equity=100_000.0, month_start_equity=100_000.0,
                     week_start_equity=100_000.0, prev_nav=100_000.0)
    for d in idx:
        bars = DayBars(asof=d, open=panel["open"].loc[d], high=panel["high"].loc[d],
                       low=panel["low"].loc[d], close=close.loc[d],
                       atr={2: pd.Series({s: atr2[s].loc[d] for s in close.columns})})
        eng.step(d, DailyHistory(close, d), bars, book)
    assert "AAA" in book.lots and "ZZZ" not in book.lots


# --------------------------------------------------------------------------- #
# (4) heat -> admission stops at portfolio_heat_pct
# --------------------------------------------------------------------------- #
def test_heat_cap_limits_concurrent_opens():
    idx = _bdays(5)
    frm = idx[2]
    # three A+ score names, each fixed at RI8's per_trade_pct of equity (A+'s
    # base tier IS band_high, so it resolves to RI8 regardless of the
    # operator's floor). How many fit under the CURRENT book heat cap (also
    # clamped by max_concurrent) is derived from config, not hardcoded.
    syms = ["AAA", "BBB", "CCC"]
    panel = _panel({s: [(100, 101, 99, 100)] * 5 for s in syms})
    close = panel["close"]
    atr2 = {s: _atr_series(panel["high"][s], panel["low"][s], close[s], 2) for s in syms}

    class AllThree:
        def entry_score(self, symbol, asof, history):
            if asof < frm:
                return None
            return {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}.get(symbol)
    # distinct families so the per-family cap never binds; book heat is the gate.
    sleeves = [SleeveSpec(name="brk", strategy=AllThree(), kind="score", grade="A+",
                          family="trend", bracket=BracketConfig(atr_window=2))]
    limits = load_limits()
    row = limits.level(limits.default_ri)
    cand_risk = per_trade_dollar_risk(100_000.0, resolve_ri("A+", limits), limits)
    heat_cap = row.portfolio_heat_pct / 100.0 * 100_000.0
    n_admit = min(int(heat_cap // cand_risk), row.max_concurrent, len(syms))
    assert n_admit >= 1, "test setup: no A+ candidate fits under this config's heat cap"

    eng = PortfolioEngine(sleeves, pcfg=_small_synth_pcfg())
    book = BookState(cash=100_000.0, peak_equity=100_000.0, month_start_equity=100_000.0,
                     week_start_equity=100_000.0, prev_nav=100_000.0)
    alloc = None
    for d in idx:
        bars = DayBars(asof=d, open=panel["open"].loc[d], high=panel["high"].loc[d],
                       low=panel["low"].loc[d], close=close.loc[d],
                       atr={2: pd.Series({s: atr2[s].loc[d] for s in syms})})
        alloc = eng.step(d, DailyHistory(close, d), bars, book)
        if d == frm:
            break
    # the top-n_admit-ranked names are admitted (AAA=3.0 > BBB=2.0 > CCC=1.0).
    expected = set(syms[:n_admit])
    assert len(book.lots) == n_admit
    assert expected == set(book.lots)
    if n_admit < len(syms):
        assert any("heat cap" in why or "concurrency" in why for _c, why in alloc.rejected)


# --------------------------------------------------------------------------- #
# (5) drawdown -> opens halted, closes (risk-reducers) execute
# --------------------------------------------------------------------------- #
def test_drawdown_halts_opens_but_closes_run():
    d = date(2024, 6, 3)
    pcfg = _small_synth_pcfg()
    # Book: a weight lot (BBB, no stop) marked down hard + a score lot (AAA) whose
    # stop is gapped through today. Peak is high -> today's equity is -36% from peak.
    book = BookState(
        cash=10_000.0,
        peak_equity=100_000.0, month_start_equity=100_000.0,
        week_start_equity=100_000.0, prev_nav=100_000.0,
    )
    book.lots["BBB"] = OpenLot(sleeve="rot", symbol="BBB", kind="weight", entry_date=date(2024, 5, 1),
                               entry_price=500.0, shares=120.0, grade="B", family="equity",
                               initial_stop=425.0, initial_shares=120.0, atr_entry=1.0,
                               stop=425.0, target_weight=0.6)
    book.lots["AAA"] = OpenLot(sleeve="brk", symbol="AAA", kind="score", entry_date=date(2024, 5, 20),
                               entry_price=100.0, shares=200.0, grade="B", family="trend",
                               initial_stop=95.0, initial_shares=200.0, atr_entry=2.0, stop=95.0)
    # Today: BBB craters to 300 (120 sh -> $36k, was $60k) and AAA gaps to 90 (< stop 95).
    # Equity after AAA stops out: cash 10k + AAA proceeds (200*90=18k) + BBB 120*300=36k = 64k
    # = -36% from the 100k peak -> program-abort halt. A fresh CCC breakout is suppressed.
    close = pd.DataFrame({"AAA": [90.0], "BBB": [300.0], "CCC": [100.0]}, index=[d])
    opn = pd.DataFrame({"AAA": [90.0], "BBB": [300.0], "CCC": [100.0]}, index=[d])
    high = pd.DataFrame({"AAA": [90.0], "BBB": [300.0], "CCC": [101.0]}, index=[d])
    low = pd.DataFrame({"AAA": [89.0], "BBB": [300.0], "CCC": [99.0]}, index=[d])
    atr2 = pd.Series({"AAA": 2.0, "BBB": 1.0, "CCC": 2.0})

    class CCCBreakout:
        def entry_score(self, symbol, asof, history):
            return 1.0 if symbol == "CCC" else None
    # rot wants to keep BBB at its weight (so it is NOT closed by reconcile).
    sleeves = [
        SleeveSpec(name="rot", strategy=WantFrom({"BBB": 0.6}, date(2024, 1, 1)), kind="weight",
                   grade="B", family="equity"),
        SleeveSpec(name="brk", strategy=CCCBreakout(), kind="score", grade="B",
                   family="trend", bracket=BracketConfig(atr_window=2)),
    ]
    eng = PortfolioEngine(sleeves, pcfg=pcfg)
    bars = DayBars(asof=d, open=opn.loc[d], high=high.loc[d], low=low.loc[d], close=close.loc[d],
                   atr={2: atr2})
    hist = DailyHistory(close, d)
    alloc = eng.step(d, hist, bars, book)
    assert alloc.halted is True
    # the AAA stop (a risk-reducer) executed despite the halt:
    assert any(c.symbol == "AAA" for c in alloc.closes)
    assert "AAA" not in book.lots
    # the fresh CCC breakout was suppressed:
    assert "CCC" not in book.lots
    assert any(c.symbol == "CCC" and "halted" in why for c, why in alloc.rejected)


# --------------------------------------------------------------------------- #
# (6) inherited bracket mechanics on a score lot (TP1 -> breakeven, gap fill)
# --------------------------------------------------------------------------- #
def test_score_lot_tp1_moves_stop_to_breakeven():
    idx = _bdays(6)
    # warm ATR(2) flat at 100 (TR 2 -> ATR 2), enter day 2 (one-shot), push to TP1
    # on day 3, then stay WELL above the trailed stop (~102) so the runner survives.
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100),
            (100, 108, 100, 107.5), (108, 112, 107, 110), (110, 112, 108, 110)]
    panel = _panel({"AAA": rows})
    close = panel["close"]
    atr2 = _atr_series(panel["high"]["AAA"], panel["low"]["AAA"], close["AAA"], 2)

    class ScoreOn:
        def entry_score(self, symbol, asof, history):
            return 1.0 if (symbol == "AAA" and asof == idx[2]) else None
    sleeves = [SleeveSpec(name="brk", strategy=ScoreOn(), kind="score", grade="B",
                          family="trend",
                          bracket=BracketConfig(atr_window=2, tp1_R=1.5, tp1_fraction=0.5,
                                                use_trail=True, trail_atr_mult=3.0))]
    eng = PortfolioEngine(sleeves, pcfg=_small_synth_pcfg())
    book = BookState(cash=100_000.0, peak_equity=100_000.0, month_start_equity=100_000.0,
                     week_start_equity=100_000.0, prev_nav=100_000.0)
    seen_tp1 = False
    for d in idx:
        bars = DayBars(asof=d, open=panel["open"].loc[d], high=panel["high"].loc[d],
                       low=panel["low"].loc[d], close=close.loc[d],
                       atr={2: pd.Series({"AAA": atr2.loc[d]})})
        alloc = eng.step(d, DailyHistory(close, d), bars, book)
        if any(r.reason == "tp1_partial" for r in alloc.resizes):
            seen_tp1 = True
    # entry 100, stop 95, rps 5, TP1 = 100 + 1.5*5 = 107.5 (hit on day idx[3] high 108).
    # after TP1 the runner stop ratchets to breakeven (>= entry 100).
    assert seen_tp1
    assert "AAA" in book.lots
    assert book.lots["AAA"].tp1_done is True
    assert book.lots["AAA"].stop >= 100.0
    # entry size = per_trade_dollar_risk(equity, resolve_ri("B", limits), limits) / rps(5);
    # tp1_fraction=0.5 (set in the bracket config above) scales half out.
    limits = load_limits()
    entry_shares = per_trade_dollar_risk(100_000.0, resolve_ri("B", limits), limits) / 5.0
    assert book.lots["AAA"].shares == pytest.approx(entry_shares * 0.5)


# --------------------------------------------------------------------------- #
# (7) deposits: TWR excludes the flow, gross NAV includes it
# --------------------------------------------------------------------------- #
def test_deposit_separates_twr_from_nav():
    idx = _bdays(4)
    panel = _panel({"AAA": [(100, 100, 100, 100)] * 4})  # flat, nothing trades
    class Idle:
        def target_weights(self, asof, history):
            return {}
    sleeves = [SleeveSpec(name="idle", strategy=Idle(), kind="weight", grade="B")]
    contributions = {idx[2]: 50_000.0}
    res = run_portfolio(sleeves, ["AAA"], initial_equity=100_000.0,
                        contributions=contributions, panel=panel)
    # NAV jumps by the deposit on idx[2]; TWR that day is ~0 (flat book).
    assert res.nav.loc[idx[2]] == pytest.approx(150_000.0)
    assert res.twr_returns.loc[idx[2]] == pytest.approx(0.0, abs=1e-9)
    assert res.total_deposited == pytest.approx(50_000.0)
    # gross daily return on the deposit day is +50% — the trap TWR avoids.
    assert res.daily_returns.loc[idx[2]] == pytest.approx(0.5)
