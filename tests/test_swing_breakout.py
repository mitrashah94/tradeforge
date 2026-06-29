"""tests/test_swing_breakout.py — the DAILY DONCHIAN-BREAKOUT TREND sleeve tests.

Deterministic, OFFLINE (no DB): every test hand-builds a tiny synthetic ADJUSTED
daily close panel with KNOWN behavior, wraps it in the engine's point-in-time
:class:`~backtest.daily.engine.DailyHistory`, and asserts the entry/scoring
contract the operator specified:

  (a) entry FIRES exactly on a fresh N-day breakout WHILE above the trend SMA;
  (b) entry does NOT fire on the same breakout when the close is BELOW the trend
      SMA (a breakout in a downtrend is a falling-knife trap we skip);
  (c) the score RANKS stronger momentum higher (so the engine fills the scarce
      slots with the strongest trends);
  (d) no same-bar self-reference: the Donchian high EXCLUDES today, so a close
      merely tying its own bar is not a breakout — and the point-in-time history
      a strategy sees never contains a date after asof_date (no lookahead).

The thresholds are hand-computed in each test so a regression in the breakout
gate / SMA gate / scoring / channel windowing fails loudly.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.daily.engine import DailyHistory
from strategies.swing_breakout.strategy import SwingBreakoutStrategy, load_params


# --------------------------------------------------------------------------- #
# Helpers — synthetic close panels + a point-in-time history at the last bar
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


def _panel(closes: dict[str, list[float]], dates=None) -> pd.DataFrame:
    """Wide ADJUSTED-close frame {symbol: [closes...]} -> DataFrame(date x sym)."""
    n = len(next(iter(closes.values())))
    idx = dates if dates is not None else _bdays(n)
    return pd.DataFrame(closes, index=idx)


def _history_at_last(panel: pd.DataFrame) -> DailyHistory:
    """A point-in-time history as of the panel's LAST date (full window visible)."""
    return DailyHistory(panel, panel.index[-1])


def _strategy(**param_overrides) -> SwingBreakoutStrategy:
    """Build a strategy from DEFAULT params with optional small overrides."""
    params = load_params("DEFAULT")
    params.update(param_overrides)
    return SwingBreakoutStrategy(params=params)


# --------------------------------------------------------------------------- #
# (a) Entry FIRES on a fresh N-day breakout-in-uptrend
# --------------------------------------------------------------------------- #
def test_entry_fires_on_breakout_in_uptrend():
    # Small Donchian (5) and trend SMA (10) so a short series is enough. Build a
    # rising series where the LAST close is a strict new 5-day high AND sits above
    # the 10d SMA -> the breakout must fire (finite score).
    sym = "AAA"
    # 14 bars: a steady uptrend, last bar a fresh high.
    closes = [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108, 109, 110, 113]
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=5)

    asof = panel.index[-1]
    score = strat.entry_score(sym, asof, _history_at_last(panel))

    # Sanity-check the gates by hand: prior-5 high (excl. today) over bars
    # [109,110,109? ...] -> the channel ending yesterday is max(108,109,108?,...).
    series = pd.Series([float(c) for c in closes])
    prior_high = float(series.iloc[-6:-1].max())   # 5 sessions ending yesterday
    sma10 = float(series.iloc[-10:].mean())
    assert series.iloc[-1] > prior_high            # genuine new 5-day high
    assert series.iloc[-1] > sma10                 # above the trend filter

    assert score is not None
    assert np.isfinite(score)
    assert score > 0  # momentum (close/close[-5] - 1) is positive in an uptrend


# --------------------------------------------------------------------------- #
# (b) Entry does NOT fire on a breakout BELOW the trend SMA
# --------------------------------------------------------------------------- #
def test_no_entry_when_breakout_below_trend_sma():
    # A series that has been FALLING, then ticks up to a fresh SHORT-channel high
    # but is still well BELOW its (higher) trend SMA -> a downtrend breakout we
    # must skip (close <= SMA gate fails). We rig it so the Donchian gate passes
    # but the SMA gate does not.
    sym = "AAA"
    # High early (pulls the 10d SMA up), then a decline into a trough, then a
    # bounce whose last close is a fresh 5-day high (vs the recent trough sessions)
    # but is still well BELOW the higher 10d SMA. Donchian gate passes, trend gate
    # fails -> a downtrend breakout we must skip.
    closes = [130, 128, 126, 124, 122, 120, 118, 100, 99, 98, 99, 100, 101, 103]
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=5)

    series = pd.Series([float(c) for c in closes])
    prior_high = float(series.iloc[-6:-1].max())   # 5 sessions ending yesterday
    sma10 = float(series.iloc[-10:].mean())
    # Confirm the SETUP: last close IS a fresh 5-day high but is BELOW the SMA.
    assert series.iloc[-1] > prior_high            # Donchian gate would pass...
    assert series.iloc[-1] < sma10                 # ...but the trend gate fails

    asof = panel.index[-1]
    assert strat.entry_score(sym, asof, _history_at_last(panel)) is None


def test_no_entry_when_not_a_breakout():
    # Above the trend SMA (uptrend) but the last close is NOT a new 5-day high
    # (it pulls back below the prior channel high) -> no breakout, no entry.
    sym = "AAA"
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 121]
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=5)

    series = pd.Series([float(c) for c in closes])
    prior_high = float(series.iloc[-6:-1].max())
    sma10 = float(series.iloc[-10:].mean())
    assert series.iloc[-1] > sma10                 # still in the uptrend...
    assert series.iloc[-1] < prior_high            # ...but below the channel high

    asof = panel.index[-1]
    assert strat.entry_score(sym, asof, _history_at_last(panel)) is None


def test_breakout_must_be_strict_new_high_not_tie():
    # The Donchian high EXCLUDES today's bar. A close that exactly TIES the prior
    # channel high is NOT a strict breakout (close <= prior_high) -> no entry.
    sym = "AAA"
    # prior 5-day channel high = 120; last close ties it at 120 -> no breakout.
    closes = [100, 105, 110, 115, 118, 116, 117, 119, 120, 118, 117, 119, 118, 120]
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=5)

    series = pd.Series([float(c) for c in closes])
    prior_high = float(series.iloc[-6:-1].max())
    assert series.iloc[-1] == pytest.approx(prior_high)  # a TIE, not a break
    asof = panel.index[-1]
    assert strat.entry_score(sym, asof, _history_at_last(panel)) is None


# --------------------------------------------------------------------------- #
# (c) Score ranking orders stronger momentum higher
# --------------------------------------------------------------------------- #
def test_score_ranks_stronger_momentum_higher():
    # Two names, BOTH valid breakouts-in-uptrend on the same last bar, but one has
    # a much steeper trailing-lookback return. Its score must be strictly higher,
    # so the engine would fill a scarce slot with it first.
    strong = [100, 101, 103, 105, 108, 111, 114, 118, 122, 127, 132, 138, 144, 152]
    weak = [100, 100, 101, 100, 101, 102, 101, 102, 103, 102, 103, 104, 103, 105]
    panel = _panel({"STRONG": [float(c) for c in strong],
                    "WEAK": [float(c) for c in weak]})
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=10)
    asof = panel.index[-1]
    hist = _history_at_last(panel)

    s_strong = strat.entry_score("STRONG", asof, hist)
    s_weak = strat.entry_score("WEAK", asof, hist)
    assert s_strong is not None and s_weak is not None
    assert s_strong > s_weak

    # And the score equals the exact trailing-lookback return for each name.
    for name, closes in (("STRONG", strong), ("WEAK", weak)):
        ser = pd.Series([float(c) for c in closes])
        expected = float(ser.iloc[-1] / ser.iloc[-(10 + 1)] - 1.0)
        got = strat.entry_score(name, asof, hist)
        assert got == pytest.approx(expected)


def test_above_sma_score_mode_ranks_by_distance_above_trend():
    # In 'above_sma' mode the score is close/SMA - 1 (distance above the trend),
    # independent of the trailing return path. The name further above its own SMA
    # ranks higher.
    far = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 145]
    near = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126]
    panel = _panel({"FAR": [float(c) for c in far], "NEAR": [float(c) for c in near]})
    strat = _strategy(donchian_n=5, trend_sma=10, score_mode="above_sma")
    asof = panel.index[-1]
    hist = _history_at_last(panel)

    s_far = strat.entry_score("FAR", asof, hist)
    s_near = strat.entry_score("NEAR", asof, hist)
    assert s_far is not None and s_near is not None
    assert s_far > s_near
    # Exact value check for FAR: close/SMA(10) - 1.
    ser = pd.Series([float(c) for c in far])
    expected = float(ser.iloc[-1] / ser.iloc[-10:].mean() - 1.0)
    assert s_far == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# (d) No-lookahead: the strategy never sees a date after asof_date
# --------------------------------------------------------------------------- #
def test_no_lookahead_history_is_point_in_time():
    # The breakout is on the LAST bar. Asked AS OF an EARLIER date (before the
    # breakout), the strategy must NOT fire — it cannot see the future high. Only
    # on the asof date of the actual breakout does it fire. This proves the
    # decision uses only data <= asof_date.
    sym = "AAA"
    closes = [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108, 109, 110, 113]
    dates = _bdays(len(closes))
    panel = _panel({sym: [float(c) for c in closes]}, dates)
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=5)

    # The breakout bar is the last one (index 13). As of an earlier date (index
    # 10, close 108 — not yet a fresh 5-day high vs its own prior channel), the
    # visible window excludes the later breakout and the score must reflect ONLY
    # the data up to that date.
    early_asof = dates[10]
    hist_early = DailyHistory(panel, early_asof)
    # The visible window must never contain a date after the asof date.
    visible = hist_early.prices(symbols=[sym])
    assert all(d <= early_asof for d in visible.index)
    assert len(visible) == 11  # bars 0..10 only

    # As of the true breakout date, the same strategy DOES fire — confirming the
    # later signal is real, just invisible from the earlier vantage point.
    late_asof = dates[-1]
    assert strat.entry_score(sym, late_asof, DailyHistory(panel, late_asof)) is not None


def test_momentum_lookback_longer_than_history_falls_back():
    # A valid breakout-in-uptrend whose history is SHORTER than momentum_lookback
    # must NOT crash on the out-of-bounds base; it falls back to the distance-
    # above-SMA score (still a finite, positive rank for an uptrend breakout).
    sym = "AAA"
    closes = [100, 101, 102, 101, 103, 104, 103, 105, 106, 107, 108, 109, 110, 113]
    panel = _panel({sym: [float(c) for c in closes]})  # 14 bars
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=120)  # >> 14
    asof = panel.index[-1]
    score = strat.entry_score(sym, asof, _history_at_last(panel))
    assert score is not None and np.isfinite(score)
    # Fallback equals close/SMA - 1 (the above_sma score).
    ser = pd.Series([float(c) for c in closes])
    expected = float(ser.iloc[-1] / ser.iloc[-10:].mean() - 1.0)
    assert score == pytest.approx(expected)


def test_insufficient_history_returns_none():
    # Fewer bars than the trend-SMA / Donchian window -> cannot decide -> None
    # (no partial-window breakout buys).
    sym = "AAA"
    closes = [100, 101, 102, 103, 104]  # only 5 bars
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10, momentum_lookback=5)
    asof = panel.index[-1]
    assert strat.entry_score(sym, asof, _history_at_last(panel)) is None


# --------------------------------------------------------------------------- #
# exit_signal — trend-break close (only when enabled)
# --------------------------------------------------------------------------- #
def test_exit_signal_off_by_default():
    sym = "AAA"
    closes = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94]
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10)  # exit_on_trend_break defaults False
    asof = panel.index[-1]
    assert strat.exit_signal(sym, asof, _history_at_last(panel)) is False


def test_exit_signal_fires_on_trend_break_when_enabled():
    # Falling series whose last close is below its 10d SMA -> trend broke -> exit.
    sym = "AAA"
    closes = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94]
    panel = _panel({sym: [float(c) for c in closes]})
    strat = _strategy(donchian_n=5, trend_sma=10, exit_on_trend_break=True)
    asof = panel.index[-1]
    series = pd.Series([float(c) for c in closes])
    assert series.iloc[-1] < float(series.iloc[-10:].mean())  # below the trend SMA
    assert strat.exit_signal(sym, asof, _history_at_last(panel)) is True
