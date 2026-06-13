"""tests/test_engine_fills.py — engine fill-semantics unit tests.

Deterministic, offline. Each test builds a tiny synthetic single-session
sequence of bars and a trivial strategy, then asserts a specific engine
behavior: next-bar-open entry, ~+2R target exit, ~-1R stop exit, EOD-flat, and
adverse cost application.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.engine.cost import AssetCosts, CostModel
from backtest.engine.engine import Bar, BacktestEngine, Strategy

BASE = datetime(2025, 1, 2, 14, 30)  # 09:30 ET


def _bars(rows):
    """rows: list of (o,h,l,c) -> list[Bar] at 5-minute spacing."""
    out = []
    for i, (o, h, l, c) in enumerate(rows):
        out.append(Bar(ts=BASE + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=1000))
    return out


def _session_of(bar):  # single session
    return "S"


def _zero_cost_model():
    ac = AssetCosts(
        commission_per_order=0.0,
        half_spread_price=0.0,
        half_spread_bps=None,
        slippage_price=0.0,
        slippage_bps=None,
    )
    return CostModel("zero", {"equity": ac})


class EnterOnceLong(Strategy):
    """Enter long on the first bar (index 0) with a given stop/target."""

    def __init__(self, stop, target):
        self.stop = stop
        self.target = target
        self.done = False

    def on_bar(self, ctx):
        if not self.done and ctx.bar_index == 0 and ctx.position is None:
            ctx.enter_long(stop=self.stop, target=self.target)
            self.done = True


class EnterOnceShort(Strategy):
    def __init__(self, stop, target):
        self.stop = stop
        self.target = target
        self.done = False

    def on_bar(self, ctx):
        if not self.done and ctx.bar_index == 0 and ctx.position is None:
            ctx.enter_short(stop=self.stop, target=self.target)
            self.done = True


def _run(strategy, bars, cost=None, **kw):
    engine = BacktestEngine(
        strategy,
        cost or _zero_cost_model(),
        symbol="TEST",
        asset_class="equity",
        tick=0.01,
        initial_equity=100_000.0,
        percent_of_equity=1.0,
        **kw,
    )
    return engine.run(bars, {}, _session_of)


# --------------------------------------------------------------------------- #
def test_entry_fills_at_next_bar_open():
    # Signal on bar0 (close 100). Bar1 opens at 100.5 -> entry must fill there.
    bars = _bars([
        (100, 100, 100, 100.0),   # bar0: signal
        (100.5, 101, 100.4, 100.8),  # bar1: entry fills at open 100.5
        (100.8, 110, 100.6, 109.0),  # bar2: target reached
    ])
    strat = EnterOnceLong(stop=99.0, target=102.0)  # risk vs signal close = 1.0
    res = _run(strat, bars)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["entry_price"] == pytest.approx(100.5)        # next bar open
    assert tr["ref_entry_price"] == pytest.approx(100.5)
    assert tr["entry_ts"] == bars[1].ts                      # filled on bar1


def test_clean_2R_target_exit():
    # signal close 100, stop 99 -> risk 1.0, target = 100 + 2*1 = 102 (set by strat).
    # Entry fills exactly at signal close (bar1 open == 100) so realized R == +2.
    bars = _bars([
        (100, 100, 100, 100.0),   # bar0 signal
        (100, 100.2, 99.9, 100.1),  # bar1 entry at open 100.0, no exit
        (100.1, 102.5, 100.0, 102.4),  # bar2: high 102.5 >= target 102 -> exit @102
    ])
    strat = EnterOnceLong(stop=99.0, target=102.0)
    res = _run(strat, bars)
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "target"
    assert tr["exit_price"] == pytest.approx(102.0)
    assert tr["r_multiple"] == pytest.approx(2.0)
    assert tr["pnl"] > 0


def test_stop_out_yields_minus_1R():
    # Entry at open 100 (== signal close), stop 99 (risk 1.0). Bar2 low hits 99.
    bars = _bars([
        (100, 100, 100, 100.0),
        (100, 100.1, 99.95, 100.0),   # bar1 entry @100, no exit
        (100.0, 100.0, 98.5, 98.8),   # bar2 low 98.5 <= stop 99 -> exit @99
    ])
    strat = EnterOnceLong(stop=99.0, target=102.0)
    res = _run(strat, bars)
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "stop"
    assert tr["exit_price"] == pytest.approx(99.0)
    assert tr["r_multiple"] == pytest.approx(-1.0)
    assert tr["pnl"] < 0


def test_eod_flat_closes_open_position():
    # Position never hits stop/target -> closes at the last bar's close.
    bars = _bars([
        (100, 100, 100, 100.0),
        (100, 100.3, 99.9, 100.2),   # bar1 entry @100
        (100.2, 100.5, 99.9, 100.4),  # bar2 (final) -> EOD-flat @ close 100.4
    ])
    strat = EnterOnceLong(stop=98.0, target=110.0)  # neither reached
    res = _run(strat, bars)
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] == "eod_flat"
    assert tr["exit_price"] == pytest.approx(100.4)
    assert tr["exit_ts"] == bars[-1].ts


def test_cost_model_applies_adversely():
    # tv_style-like: $1/order commission, 1-tick slippage, no half-spread.
    ac = AssetCosts(
        commission_per_order=1.0,
        half_spread_price=0.0,
        half_spread_bps=None,
        slippage_price=0.01,
        slippage_bps=None,
    )
    cost = CostModel("t", {"equity": ac})
    bars = _bars([
        (100, 100, 100, 100.0),
        (100, 100.2, 99.9, 100.1),   # bar1 entry: market long -> +slip => 100.01
        (100.1, 102.5, 100.0, 102.4),  # bar2 target @102 (limit, half_spread 0) => 102.0
    ])
    strat = EnterOnceLong(stop=99.0, target=102.0)
    res = _run(strat, bars, cost=cost)
    tr = res.trades.iloc[0]
    # Long entry slips UP by slippage; resting-limit target pays no slippage.
    assert tr["entry_price"] == pytest.approx(100.01)
    assert tr["exit_price"] == pytest.approx(102.0)
    assert tr["costs"] == pytest.approx(2.0)  # commission on entry + exit


def test_short_stop_and_cost_direction():
    # Short entry SELLS -> adverse fill is LOWER by slippage; stop is above.
    ac = AssetCosts(
        commission_per_order=0.0,
        half_spread_price=0.0,
        half_spread_bps=None,
        slippage_price=0.05,
        slippage_bps=None,
    )
    cost = CostModel("t", {"equity": ac})
    bars = _bars([
        (100, 100, 100, 100.0),       # bar0 signal close 100
        (100, 100.1, 99.8, 99.9),     # bar1 entry: short sells @ open 100 - 0.05 = 99.95
        (99.9, 101.5, 99.8, 101.2),   # bar2 high 101.5 >= stop 101 -> stop fill @101
    ])
    strat = EnterOnceShort(stop=101.0, target=96.0)  # risk vs signal close = 1.0
    res = _run(strat, bars, cost=cost)
    tr = res.trades.iloc[0]
    assert tr["side"] == "short"
    assert tr["entry_price"] == pytest.approx(99.95)   # sold lower (adverse)
    assert tr["exit_reason"] == "stop"
    # Buying back at the stop also slips adversely (higher) by 0.05.
    assert tr["exit_price"] == pytest.approx(101.05)
    assert tr["pnl"] < 0


def test_no_bracket_resolution_on_entry_bar():
    # The entry bar's own range must NOT trigger the stop/target (managed only
    # on subsequent bars). Bar1 spans both stop and target but should be ignored
    # for exit purposes; the exit happens on bar2.
    bars = _bars([
        (100, 100, 100, 100.0),
        (100, 103, 98, 100.0),    # bar1 entry @100; range spans stop 99 & target 102
        (100, 100.1, 98.9, 99.5),  # bar2 low 98.9 <= stop 99 -> stop here
    ])
    strat = EnterOnceLong(stop=99.0, target=102.0)
    res = _run(strat, bars)
    tr = res.trades.iloc[0]
    assert tr["entry_ts"] == bars[1].ts
    assert tr["exit_ts"] == bars[2].ts      # not the entry bar
    assert tr["exit_reason"] == "stop"


def test_stop_gap_modeling_toggle():
    # With model_stop_gaps=True, a long stop whose bar OPENS below it fills at
    # the open (worse than the stop). With it False (TV-parity) it fills at stop.
    bars = _bars([
        (100, 100, 100, 100.0),
        (100, 100.2, 99.9, 100.0),    # bar1 entry @100
        (98.0, 98.0, 97.5, 97.6),     # bar2 opens at 98 < stop 99 (adverse gap)
    ])
    strat = EnterOnceLong(stop=99.0, target=110.0)
    res_gap = _run(EnterOnceLong(99.0, 110.0), bars, model_stop_gaps=True)
    res_nogap = _run(EnterOnceLong(99.0, 110.0), bars, model_stop_gaps=False)
    assert res_gap.trades.iloc[0]["exit_reason"] == "stop_gap"
    assert res_gap.trades.iloc[0]["exit_price"] == pytest.approx(98.0)
    assert res_nogap.trades.iloc[0]["exit_reason"] == "stop"
    assert res_nogap.trades.iloc[0]["exit_price"] == pytest.approx(99.0)
    # gap fill is strictly worse for a long.
    assert res_gap.trades.iloc[0]["pnl"] < res_nogap.trades.iloc[0]["pnl"]


def test_summary_profit_factor():
    # Two trades: one +2R win, one -1R loss -> PF = 2.0 / 1.0 with $-sizing equal.
    # Build a sequence with two non-overlapping trades.
    class TwoTrades(Strategy):
        def __init__(self):
            self.n = 0

        def on_bar(self, ctx):
            if ctx.position is None and ctx.bar_index in (0, 4):
                ctx.enter_long(stop=99.0, target=102.0)

    bars = _bars([
        (100, 100, 100, 100.0),       # 0 signal
        (100, 100.1, 99.9, 100.0),    # 1 entry @100
        (100, 102.5, 100, 102.4),     # 2 target @102 (+2R)
        (102, 102, 101.9, 101.95),    # 3 idle
        (100, 100, 100, 100.0),       # 4 signal again
        (100, 100.1, 99.9, 100.0),    # 5 entry @100
        (100, 100.0, 98.5, 98.8),     # 6 stop @99 (-1R)
        (98.8, 98.8, 98.7, 98.75),    # 7 final (idle/EOD)
    ])
    res = _run(TwoTrades(), bars)
    s = res.summary()
    assert s["n_trades"] == 2
    assert s["profit_factor"] == pytest.approx(2.0, rel=0.05)
    assert s["win_rate"] == pytest.approx(0.5)
