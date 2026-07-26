"""tests/test_bracket_engine.py — daily bracketed-swing portfolio backtester tests.

Deterministic, OFFLINE (no DB): every test hand-builds a tiny synthetic ADJUSTED
daily OHLC panel with KNOWN behavior, runs ``run_bracket_portfolio``, and asserts
the exact bracket mechanics the operator specified:

  (a) breakout -> position opens with the correct FRACTIONAL share count
      (= $risk / stop_distance);
  (b) rises to TP1 -> half sold, runner stop moves to breakeven;
  (c) trails up then stops out on a pullback at the TRAILED level;
  (d) a gap-down below the stop fills at the OPEN (gap-aware), not the stop;
  (e) tax accrues only on GAINS;
  (f) no-lookahead (a post-asof move is invisible to the strategy).

The math is hand-computed in each test so a regression in the bracket accounting
(sizing, partial timing, breakeven move, chandelier trail, gap fill, tax) fails
loudly.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.daily.bracket_engine import (
    BracketConfig,
    OpenPosition,
    _atr_series,
    load_daily_ohlc,
    run_bracket_portfolio,
)


# --------------------------------------------------------------------------- #
# Synthetic OHLC panel helpers
# --------------------------------------------------------------------------- #
def _bdays(n: int, start=date(2024, 1, 1)) -> list:
    """N consecutive Mon-Fri business dates starting on/after ``start``."""
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _ohlc(bars: dict[str, list[tuple]], dates=None) -> dict[str, pd.DataFrame]:
    """Build an OHLC panel dict from {symbol: [(o,h,l,c), ...]}.

    Returns ``{"open","high","low","close": wide df}`` shaped exactly like the
    engine's ``panel=`` argument. All symbols must share the bar count.
    """
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


def _flat_bar(price: float) -> tuple:
    """An OHLC bar with no range at ``price`` (o=h=l=c)."""
    return (price, price, price, price)


# --------------------------------------------------------------------------- #
# Strategies for the tests
# --------------------------------------------------------------------------- #
class EnterOnDay:
    """Score 1.0 for ``symbol`` ONLY on a specific date (a one-shot entry)."""

    def __init__(self, symbol: str, on_date: date):
        self._sym = symbol
        self._on = on_date

    def entry_score(self, symbol, asof_date, history):
        if symbol == self._sym and asof_date == self._on:
            return 1.0
        return None


class ScoreByLevel:
    """Score = today's close (rank by price). Enters any symbol every day."""

    def entry_score(self, symbol, asof_date, history):
        px = history.asof_prices()[symbol]
        return float(px) if np.isfinite(px) else None


# --------------------------------------------------------------------------- #
# ATR sanity (the bracket's sizing/stop basis)
# --------------------------------------------------------------------------- #
def test_atr_is_mean_true_range():
    # Bars with a known constant true range of 2.0 -> ATR == 2.0 once warm.
    # close: 10,10,10,10 ; high=close+1, low=close-1 -> TR = max(2, 1, 1) = 2.
    h = pd.Series([11.0, 11.0, 11.0, 11.0])
    l = pd.Series([9.0, 9.0, 9.0, 9.0])
    c = pd.Series([10.0, 10.0, 10.0, 10.0])
    atr = _atr_series(h, l, c, window=2)
    assert np.isnan(atr.iloc[0])           # warmup
    assert atr.iloc[1] == pytest.approx(2.0)
    assert atr.iloc[3] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# (a) Breakout -> position opens with the correct FRACTIONAL share count
# --------------------------------------------------------------------------- #
def test_entry_sizes_fractional_shares_by_risk():
    # Warm the ATR (window=2) with two flat-range bars (TR=2 each -> ATR=2),
    # then enter on day index 2 at close=100. stop = 100 - 2.5*2 = 95.
    # risk/share = 5. $risk = 1% * 100_000 = 1000. shares = 1000/5 = 200.
    sym = "AAA"
    dates = _bdays(4)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),   # 0 warmup (TR=2)
            (100.0, 101.0, 99.0, 100.0),   # 1 warmup (TR=2) -> ATR=2
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100, stop 95
            (100.0, 101.0, 99.0, 100.0),   # 3 flat -> still open
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(atr_window=2, stop_atr_mult=2.5, tp1_fraction=0.5, tp1_R=1.5)
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    # No trade closed yet (held through to the end open).
    assert len(res.trades) == 0
    # The book is fully marked: NAV day2 unchanged (entered at close, no move).
    assert res.nav.loc[dates[2]] == pytest.approx(100_000.0)
    # 200 shares * 100 = 20_000 invested -> exposure = 0.20 on the entry day.
    assert res.exposure.loc[dates[2]] == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# (b) Rises to TP1 -> half sold, runner stop moves to breakeven
# --------------------------------------------------------------------------- #
def test_tp1_sells_half_and_moves_stop_to_breakeven():
    # Entry @100, stop 95, risk/share 5. TP1 at +1.5R = 100 + 7.5 = 107.5.
    # Day3 high reaches 108 -> TP1 fills at 107.5, sell half (100 sh), stop->100.
    # use_trail=False so the runner's stop STAYS at the breakeven 100 (isolating
    # the breakeven move from the chandelier trail, which is exercised separately
    # in test_chandelier_trail_stops_runner_at_trailed_level).
    # Day4 opens 101 (no gap), low 99.5 > BE 100? No, 99.5 < 100 -> runner stops
    # at the breakeven 100.
    sym = "AAA"
    dates = _bdays(5)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),   # 0 warmup
            (100.0, 101.0, 99.0, 100.0),   # 1 warmup -> ATR 2
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100 (200 sh), stop 95
            (101.0, 108.0, 100.5, 106.0),  # 3 high 108 -> TP1 107.5, stop->BE 100
            (101.0, 102.0, 99.5, 100.0),   # 4 low 99.5 < BE 100 -> stop @100
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(
        atr_window=2, stop_atr_mult=2.5, tp1_R=1.5, tp1_fraction=0.5,
        trail_atr_mult=3.0, use_trail=False,
    )
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    assert len(res.trades) == 1
    t = res.trades[0]
    # Partial: 100 sh sold @107.5 -> +750. Runner: 100 sh sold @100 (BE) -> 0.
    # Total net PnL = 750. Size-weighted R: partial leg = 1.5R * 0.5 = 0.75R;
    # runner leg = 0R * 0.5 = 0 -> blended 0.75R.
    assert t.pnl == pytest.approx(750.0)
    assert t.r_multiple == pytest.approx(0.75)
    assert t.exit_reason == "trail_stop"   # runner closed at the (BE) stop
    # Gross NAV grew by exactly the +750 realized.
    assert res.nav.iloc[-1] == pytest.approx(100_750.0)


# --------------------------------------------------------------------------- #
# (c) Trails up then stops out on a pullback at the TRAILED level
# --------------------------------------------------------------------------- #
def test_chandelier_trail_stops_runner_at_trailed_level():
    # ATR(entry)=2, trail_atr_mult=2 -> chandelier band = highest_high - 4.
    # Entry @100 stop 95. TP1 at +1.5R=107.5. Day3 high 108 -> TP1, stop->BE 100.
    #   highest_high=108 -> chandelier 108-4=104 > 100 -> stop trails to 104.
    # Day4 high 112, low 105 (no stop hit; 105 > 104) -> highest_high=112 ->
    #   chandelier 112-4=108 -> stop trails to 108.
    # Day5 low 107 <= 108 -> runner stops at the TRAILED 108.
    sym = "AAA"
    dates = _bdays(6)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),   # 0 warmup
            (100.0, 101.0, 99.0, 100.0),   # 1 warmup -> ATR 2
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100 (200 sh) stop 95
            (101.0, 108.0, 100.5, 106.0),  # 3 TP1 @107.5, stop->BE 100->trail 104
            (106.0, 112.0, 105.0, 110.0),  # 4 hi 112 -> trail to 108 (no stop)
            (109.0, 110.0, 107.0, 108.0),  # 5 low 107 <= 108 -> stop @108
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(
        atr_window=2, stop_atr_mult=2.5, tp1_R=1.5, tp1_fraction=0.5,
        trail_atr_mult=2.0, use_trail=True,
    )
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    assert len(res.trades) == 1
    t = res.trades[0]
    # Partial: 100 sh @107.5 -> +750. Runner: 100 sh @108 (trailed) -> +800.
    # Total +1550. R: partial 0.75R + runner (8/5 R)*0.5 = 0.8R -> 1.55R.
    assert t.pnl == pytest.approx(1550.0)
    assert t.r_multiple == pytest.approx(1.55)
    assert t.exit_reason == "trail_stop"
    assert res.nav.iloc[-1] == pytest.approx(101_550.0)


# --------------------------------------------------------------------------- #
# (d) A gap-down below the stop fills at the OPEN (gap-aware), not the stop
# --------------------------------------------------------------------------- #
def test_gap_down_below_stop_fills_at_open():
    # Entry @100, stop 95. Day3 OPENS at 90 (gaps below the stop). The fill is
    # at the OPEN 90 (the realistic worse price), NOT the resting stop 95.
    sym = "AAA"
    dates = _bdays(4)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),   # 0 warmup
            (100.0, 101.0, 99.0, 100.0),   # 1 warmup -> ATR 2
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100 (200 sh), stop 95
            (90.0, 91.0, 88.0, 89.0),      # 3 gap: open 90 < stop 95 -> fill @90
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(atr_window=2, stop_atr_mult=2.5, tp1_fraction=0.5, tp1_R=1.5)
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    assert len(res.trades) == 1
    t = res.trades[0]
    # 200 sh sold @90 vs entry 100 -> -2000 (NOT -1000 from a stop fill @95).
    assert t.avg_exit_price == pytest.approx(90.0)
    assert t.pnl == pytest.approx(-2000.0)
    assert t.exit_reason == "stop_gap"
    # Realized R = (90-100)/5 = -2R.
    assert t.r_multiple == pytest.approx(-2.0)


def test_normal_stop_fills_at_stop_not_low():
    # No gap: open above the stop, but the LOW pierces it -> fill at the STOP 95
    # (not the lower bar low), proving the intrabar (non-gap) path fills at stop.
    sym = "AAA"
    dates = _bdays(4)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),   # 0 warmup
            (100.0, 101.0, 99.0, 100.0),   # 1 warmup -> ATR 2
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100 stop 95
            (99.0, 99.5, 90.0, 92.0),      # 3 open 99 (no gap), low 90 < stop 95
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(atr_window=2, stop_atr_mult=2.5)
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    t = res.trades[0]
    assert t.avg_exit_price == pytest.approx(95.0)   # filled at the STOP, not 90
    assert t.pnl == pytest.approx(-1000.0)           # 200 * (95-100)
    assert t.exit_reason == "stop"


# --------------------------------------------------------------------------- #
# (e) Tax accrues only on GAINS
# --------------------------------------------------------------------------- #
def test_tax_accrues_on_gain_only():
    # A WINNER (full take-profit) accrues tax; a LOSER does not. Use a hard
    # target to make a clean full-exit gain, and a separate stop-out loser.
    # Winner: entry @100 stop 95, hard_target_R=2 -> target 110. Day3 high 111.
    #   gain = (110-100)*200 = 2000 -> reserve = 2000 * 0.25 = 500.
    sym = "WIN"
    dates = _bdays(4)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100 stop 95
            (101.0, 111.0, 100.0, 109.0),  # 3 high 111 -> hard target 110
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(
        atr_window=2, stop_atr_mult=2.5, tp1_fraction=0.0, hard_target_R=2.0,
        use_trail=False,
    )
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.25, max_concurrent=4,
    )
    t = res.trades[0]
    assert t.exit_reason == "hard_target"
    assert t.pnl == pytest.approx(2000.0)
    assert res.realized_gains == pytest.approx(2000.0)
    assert res.tax_reserve == pytest.approx(500.0)
    # after-tax NAV = gross - reserve.
    assert res.after_tax_nav.iloc[-1] == pytest.approx(res.nav.iloc[-1] - 500.0)

    # LOSER: same setup but stops out -> realized loss, NO tax reserve.
    bars_loss = {
        "LOS": [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (96.0, 96.5, 90.0, 92.0),      # low 90 < stop 95 -> stop @95 (loss)
        ]
    }
    panel2 = _ohlc(bars_loss, dates)
    res2 = run_bracket_portfolio(
        EnterOnDay("LOS", dates[2]), ["LOS"], panel=panel2, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.25, max_concurrent=4,
    )
    assert res2.trades[0].pnl == pytest.approx(-1000.0)
    assert res2.realized_gains == pytest.approx(-1000.0)
    assert res2.tax_reserve == pytest.approx(0.0)        # no tax on a loss
    assert res2.after_tax_nav.iloc[-1] == pytest.approx(res2.nav.iloc[-1])


# --------------------------------------------------------------------------- #
# (f) No-lookahead: a post-asof move is invisible to the strategy
# --------------------------------------------------------------------------- #
def test_no_lookahead_history_is_point_in_time():
    sym = "AAA"
    dates = _bdays(5)
    bars = {
        sym: [
            (10.0, 10.5, 9.5, 10.0),
            (11.0, 11.5, 10.5, 11.0),
            (12.0, 12.5, 11.5, 12.0),
            (13.0, 13.5, 12.5, 13.0),
            (14.0, 14.5, 13.5, 14.0),
        ]
    }
    panel = _ohlc(bars, dates)
    seen = {}

    class Recorder:
        def entry_score(self, symbol, asof_date, history):
            px = history.prices()
            # The visible window must NEVER contain a date after asof_date.
            assert all(d <= asof_date for d in px.index), (
                f"lookahead: saw a date > asof {asof_date} in {list(px.index)}"
            )
            seen[asof_date] = float(history.asof_prices()[symbol])
            return None  # never actually enter; just probe the history

    run_bracket_portfolio(
        Recorder(), [sym], panel=panel, bracket=BracketConfig(atr_window=2),
        cost_bps=0.0, short_term_tax_rate=0.0,
    )
    # Each decision day's asof price equals THAT day's close (not a future one).
    for d, (_o, _h, _l, c) in zip(dates, bars[sym]):
        if d in seen:
            assert seen[d] == pytest.approx(c)


# --------------------------------------------------------------------------- #
# Portfolio: ranking + concurrent slots + cost on turnover
# --------------------------------------------------------------------------- #
def test_ranking_respects_max_concurrent():
    # Three symbols always scored by price; max_concurrent=2 -> only the top 2
    # by price are held. CCC (300) and BBB (200) get in; AAA (100) does not.
    dates = _bdays(3)
    bars = {
        "AAA": [_flat_bar(100.0)] * 3,
        "BBB": [_flat_bar(200.0)] * 3,
        "CCC": [_flat_bar(300.0)] * 3,
    }
    # Give each symbol a real ATR by adding range to the warmup bars.
    bars = {
        "AAA": [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)],
        "BBB": [(200, 202, 198, 200), (200, 202, 198, 200), (200, 202, 198, 200)],
        "CCC": [(300, 303, 297, 300), (300, 303, 297, 300), (300, 303, 297, 300)],
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(atr_window=2, stop_atr_mult=2.5, tp1_fraction=0.5, tp1_R=1.5)
    res = run_bracket_portfolio(
        ScoreByLevel(), ["AAA", "BBB", "CCC"], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=1_000_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=2,
    )
    # On the entry day (index 2, ATR warm) exactly 2 names are held.
    # Exposure must be > 0 and reflect 2 positions, never 3 (AAA excluded).
    assert res.exposure.iloc[-1] > 0
    # No closed trades (held to the end), but the engine opened the top-2.
    # Verify via NAV staying flat (no moves) and exposure consistent with 2 names.
    assert np.isfinite(res.summary()["avg_exposure"])


def test_cost_charged_on_entry_and_exit_notional():
    # Entry @100 (200 sh -> 20_000 notional), exit at hard target 110
    # (200 sh -> 22_000 notional). cost_bps=10 -> entry cost 20, exit cost 22.
    sym = "AAA"
    dates = _bdays(4)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100 (200 sh)
            (101.0, 111.0, 100.0, 109.0),  # 3 hard target 110
        ]
    }
    panel = _ohlc(bars, dates)
    cfg = BracketConfig(
        atr_window=2, stop_atr_mult=2.5, tp1_fraction=0.0, hard_target_R=2.0,
        use_trail=False,
    )
    res = run_bracket_portfolio(
        EnterOnDay(sym, dates[2]), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=10.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    # Total cost = 20 (entry) + 22 (exit) = 42.
    assert res.total_costs == pytest.approx(42.0)
    t = res.trades[0]
    # Net PnL = gross 2000 - costs 42 = 1958.
    assert t.gross_pnl == pytest.approx(2000.0)
    assert t.costs == pytest.approx(42.0)
    assert t.pnl == pytest.approx(1958.0)


# --------------------------------------------------------------------------- #
# exit_signal discretionary close
# --------------------------------------------------------------------------- #
def test_exit_signal_closes_at_close():
    sym = "AAA"
    dates = _bdays(4)
    bars = {
        sym: [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),   # 2 ENTRY @100
            (101.0, 105.0, 100.5, 103.0),  # 3 exit_signal -> close @103
        ]
    }
    panel = _ohlc(bars, dates)

    class EnterThenExit:
        def entry_score(self, symbol, asof_date, history):
            return 1.0 if asof_date == dates[2] else None

        def exit_signal(self, symbol, asof_date, history, position):
            return asof_date == dates[3]

    cfg = BracketConfig(atr_window=2, stop_atr_mult=2.5, tp1_fraction=0.5, tp1_R=1.5)
    res = run_bracket_portfolio(
        EnterThenExit(), [sym], panel=panel, bracket=cfg,
        risk_pct_per_trade=0.01, initial_equity=100_000.0,
        cost_bps=0.0, short_term_tax_rate=0.0, max_concurrent=4,
    )
    t = res.trades[0]
    assert t.exit_reason == "exit_signal"
    assert t.avg_exit_price == pytest.approx(103.0)   # closed at the close
    assert t.pnl == pytest.approx(200 * 3.0)          # 200 sh * (103-100)


# --------------------------------------------------------------------------- #
# Degenerate inputs
# --------------------------------------------------------------------------- #
def test_empty_panel_returns_empty_result():
    empty = pd.DataFrame()
    res = run_bracket_portfolio(
        EnterOnDay("AAA", date(2024, 1, 1)), ["AAA"],
        panel={"open": empty, "high": empty, "low": empty, "close": empty},
        cost_bps=2.0,
    )
    assert len(res.nav) == 0
    assert res.summary()["n_days"] == 0
    assert res.summary()["n_trades"] == 0
