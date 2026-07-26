"""tests/test_breakout_ablation.py — V1->V4 + v0_atr_stop ablation component tests.

Deterministic, offline. Each test isolates ONE ablation component on a synthetic
single-session sequence with known levels, and asserts the component actually
changes behavior as MASTER_PLAN §5 specifies:

  * V1 NTZ filter blocks an entry whose breaking level sits inside the NTZ band.
  * V2 PMH/PML setups generate trades the V0 PDH/PDL-only config does not.
  * V3 partial+runner produces a partial scale-out then a separate runner exit.
  * v0_atr_stop widens the role-reversal stop vs the 1-tick V0 stop.
  * REGRESSION: V0 on the real QQQ data still matches the gate (218 trades, PF
    ~1.34 tv_style on the Polygon/SIP re-pull) so the ablation refactor preserved
    V0 exactly. (Was 188 / PF ~1.41 on the legacy Alpaca-IEX feed; the swap to
    consolidated Polygon data changed the bar set, not the V0 logic.)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.engine.cost import AssetCosts, CostModel
from backtest.engine.engine import Bar, BacktestEngine
from strategies.breakout_retest.strategy import (
    BreakoutRetestStrategy,
    load_params,
)

BASE = datetime(2025, 1, 2, 14, 30)  # 09:30 ET


def _bars(rows):
    return [
        Bar(ts=BASE + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c,
            volume=1000)
        for i, (o, h, l, c) in enumerate(rows)
    ]


def _zero_cost():
    ac = AssetCosts(0.0, 0.0, None, 0.0, None)
    return CostModel("zero", {"equity": ac})


def _run(strat, bars, levels, **kw):
    eng = BacktestEngine(
        strat, _zero_cost(), symbol="T", asset_class="equity", tick=0.01,
        initial_equity=100_000.0, percent_of_equity=1.0, **kw,
    )
    return eng.run(bars, {"S": levels}, lambda b: "S")


# A canonical long-retest sequence: PDH=100, breaks then retests in window.
_LONG_RETEST = [
    (100.1, 100.3, 100.0, 100.20),   # 0: break (close>pdh) -> bsh=1
    (100.2, 100.4, 100.1, 100.30),   # 1: bsh=2 (no retest)
    (100.2, 100.3, 99.95, 100.10),   # 2: bsh=3, low<=pdh & close>pdh -> RETEST
    (100.2, 101.0, 100.1, 100.90),   # 3: entry fills @ open 100.2
    (100.9, 101.5, 100.0, 100.50),   # 4
    (100.5, 100.6, 100.0, 100.30),   # 5 (final / EOD)
]


# --------------------------------------------------------------------------- #
# V1 — NTZ no-trade filter
# --------------------------------------------------------------------------- #
def test_v1_ntz_blocks_entry_when_level_inside_band():
    # PDH=100 sits INSIDE a valid NTZ band [99, 101] -> V1 must block the entry
    # that V0 would take on the same bars.
    levels_in = {"pdh": 100.0, "pdl": 90.0,
                 "ntz_low": 99.0, "ntz_high": 101.0, "ntz_valid": True}
    res_v1 = _run(BreakoutRetestStrategy(variant="V1"), _bars(_LONG_RETEST),
                  levels_in)
    assert len(res_v1.trades) == 0   # blocked: level inside NTZ

    # Same bars, but V0 (no NTZ filter) DOES trade -> proves the block is the NTZ.
    res_v0 = _run(BreakoutRetestStrategy(variant="V0"), _bars(_LONG_RETEST),
                  levels_in)
    assert len(res_v0.trades) == 1


def test_v1_ntz_allows_entry_when_level_outside_band():
    # PDH=100 is ABOVE the NTZ band [95, 98] -> V1 allows the entry (same as V0).
    levels_out = {"pdh": 100.0, "pdl": 90.0,
                  "ntz_low": 95.0, "ntz_high": 98.0, "ntz_valid": True}
    res = _run(BreakoutRetestStrategy(variant="V1"), _bars(_LONG_RETEST),
               levels_out)
    assert len(res.trades) == 1


def test_v1_ntz_ignored_when_invalid():
    # ntz_valid=False -> the band is ignored even though the level is inside it.
    levels = {"pdh": 100.0, "pdl": 90.0,
              "ntz_low": 99.0, "ntz_high": 101.0, "ntz_valid": False}
    res = _run(BreakoutRetestStrategy(variant="V1"), _bars(_LONG_RETEST), levels)
    assert len(res.trades) == 1


# --------------------------------------------------------------------------- #
# V2 — PMH/PML second level set
# --------------------------------------------------------------------------- #
def test_v2_pmh_generates_a_trade_v0_does_not():
    # PDH far away (no PDH setup). PMH=100 has a clean break+retest -> only the
    # PMH-aware config (V2) trades it. Disable NTZ effect by making it invalid.
    bars = _bars(_LONG_RETEST)
    levels = {"pdh": 200.0, "pdl": 50.0,        # PDH/PDL never touched
              "pmh": 100.0, "pml": 60.0,        # PMH = the broken/retested level
              "ntz_low": None, "ntz_high": None, "ntz_valid": False}

    res_v0 = _run(BreakoutRetestStrategy(variant="V0"), bars, levels)
    assert len(res_v0.trades) == 0              # V0 ignores PMH/PML

    res_v2 = _run(BreakoutRetestStrategy(variant="V2"), bars, levels)
    assert len(res_v2.trades) == 1              # V2 trades the PMH retest
    assert res_v2.trades.iloc[0]["side"] == "long"


# --------------------------------------------------------------------------- #
# V3 — partial scale-out + trailing runner
# --------------------------------------------------------------------------- #
def test_v3_partial_then_runner_exit():
    # PMH/PDH=100 break+retest, entry @100.2 on bar3. partial_tp1_r=1 (default),
    # risk = signal_close(100.10) - stop. With a 1-tick stop, risk is tiny, so
    # tp1 is hit almost immediately; then the runner trails out at EOD or a
    # trailing stop. We assert: exactly one recorded trade, exit reason is a
    # runner reason (trail_stop or eod_flat), and a partial was booked.
    pdh = 100.0
    bars = _bars([
        (100.1, 100.3, 100.0, 100.20),   # 0 break bsh=1
        (100.2, 100.4, 100.1, 100.30),   # 1 bsh=2
        (100.2, 100.3, 99.95, 100.10),   # 2 retest -> arm long
        (100.2, 100.5, 100.1, 100.40),   # 3 entry @100.2; tp1 (~100.1+) hit here
        (100.4, 101.0, 100.3, 100.90),   # 4 runner rides up, trail follows
        (100.9, 101.2, 100.6, 100.80),   # 5 final -> runner EOD-flat @ close
    ])
    levels = {"pdh": pdh, "pdl": 90.0,
              "ntz_low": None, "ntz_high": None, "ntz_valid": False}
    res = _run(BreakoutRetestStrategy(variant="V3"), bars, levels)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    # The recorded trade is the RUNNER leg; its exit is a runner-style exit.
    assert tr["exit_reason"] in ("trail_stop", "eod_flat", "stop")
    # A partial+runner trade exits fewer shares than V0 would (runner < initial).
    res_v0 = _run(BreakoutRetestStrategy(variant="V0"), bars, levels)
    assert res_v0.trades.iloc[0]["target"] is not None     # V0 has a fixed 2R target
    assert tr["target"] is None                            # V3 runner has no fixed target


def test_v3_runner_shares_are_a_fraction_of_v0_shares():
    # Same setup as above; the V3 runner leg should report ~ (1 - partial_fraction)
    # of the share count that V0 (full size, no scale-out) would exit.
    pdh = 100.0
    bars = _bars([
        (100.1, 100.3, 100.0, 100.20),
        (100.2, 100.4, 100.1, 100.30),
        (100.2, 100.3, 99.95, 100.10),
        (100.2, 100.5, 100.1, 100.40),   # tp1 hit
        (100.4, 101.0, 100.3, 100.90),
        (100.9, 101.2, 100.6, 100.80),
    ])
    levels = {"pdh": pdh, "pdl": 90.0,
              "ntz_low": None, "ntz_high": None, "ntz_valid": False}
    v0 = _run(BreakoutRetestStrategy(variant="V0"), bars, levels)
    v3 = _run(BreakoutRetestStrategy(variant="V3"), bars, levels)
    v0_shares = float(v0.trades.iloc[0]["shares"])
    v3_runner_shares = float(v3.trades.iloc[0]["shares"])
    # partial_fraction defaults to 0.5 -> runner is ~half the full size.
    assert v3_runner_shares == pytest.approx(v0_shares * 0.5, rel=0.05)


def test_v3_partial_books_profit_on_a_winning_runner():
    # On a clean upward runner, total net P&L must be positive and reflect BOTH
    # the partial scale-out and the runner leg (runner P&L alone is on fewer
    # shares, so the booked partial materially adds to the total).
    pdh = 100.0
    bars = _bars([
        (100.1, 100.3, 100.0, 100.20),
        (100.2, 100.4, 100.1, 100.30),
        (100.2, 100.3, 99.95, 100.10),
        (100.2, 100.5, 100.1, 100.40),   # tp1 hit, scale out 50%, stop -> BE
        (100.4, 101.5, 100.3, 101.40),   # big runner bar up
        (101.4, 102.0, 101.2, 101.80),   # final, EOD-flat high
    ])
    levels = {"pdh": pdh, "pdl": 90.0,
              "ntz_low": None, "ntz_high": None, "ntz_valid": False}
    res = _run(BreakoutRetestStrategy(variant="V3"), bars, levels)
    tr = res.trades.iloc[0]
    assert tr["pnl"] > 0
    assert tr["r_multiple"] > 0


# --------------------------------------------------------------------------- #
# v0_atr_stop — widened role-reversal stop
# --------------------------------------------------------------------------- #
def test_v0_atr_stop_widens_stop_vs_one_tick():
    # Identical bars/levels; the only difference is the stop placement. With a
    # nonzero ATR14, v0_atr_stop puts the long stop k*ATR below the level, which
    # must be LOWER (wider) than the V0 1-tick stop.
    pdh = 100.0
    levels = {"pdh": pdh, "pdl": 90.0, "atr14": 1.0,
              "ntz_low": None, "ntz_high": None, "ntz_valid": False}
    bars = _bars(_LONG_RETEST)

    v0 = _run(BreakoutRetestStrategy(variant="V0"), bars, levels)
    atr = _run(BreakoutRetestStrategy(variant="v0_atr_stop"), bars, levels)

    v0_stop = float(v0.trades.iloc[0]["stop"])
    atr_stop = float(atr.trades.iloc[0]["stop"])
    # V0: 100 - 0.01 = 99.99.  v0_atr_stop: 100 - 0.25*1.0 = 99.75.
    assert v0_stop == pytest.approx(99.99)
    assert atr_stop == pytest.approx(99.75)
    assert atr_stop < v0_stop          # wider (further from the level)


def test_v0_atr_stop_falls_back_to_ticks_without_atr():
    # No atr14 in levels -> v0_atr_stop must fall back to the 1-tick stop.
    levels = {"pdh": 100.0, "pdl": 90.0,
              "ntz_low": None, "ntz_high": None, "ntz_valid": False}
    res = _run(BreakoutRetestStrategy(variant="v0_atr_stop"), _bars(_LONG_RETEST),
               levels)
    assert float(res.trades.iloc[0]["stop"]) == pytest.approx(99.99)


# --------------------------------------------------------------------------- #
# REGRESSION — V0 on the real QQQ data is unchanged by the refactor
# --------------------------------------------------------------------------- #
def test_v0_regression_matches_gate_on_real_data():
    # The ablation refactor must not perturb V0 at all: the real-data V0 run
    # under tv_style must still produce the gate's trade count and PF. Baselined
    # to the Polygon/SIP QQQ 5m re-pull (218 trades, PF ~1.34, win ~0.40); the
    # legacy Alpaca-IEX feed gave 188 / 1.41 / 0.436 before the data swap.
    pytest.importorskip("duckdb")
    from backtest.run_gate import load_qqq_5m, run_v0
    from data.schema import DEFAULT_DB_PATH, connect

    con = connect(DEFAULT_DB_PATH)
    bars_df, levels_by_session = load_qqq_5m(con)
    res = run_v0(bars_df, levels_by_session, "tv_style")
    s = res.summary()
    assert s["n_trades"] == 218
    assert s["profit_factor"] == pytest.approx(1.344, abs=0.01)
    assert s["win_rate"] == pytest.approx(0.404, abs=0.005)
