"""tests/test_breakout_retest.py — strategy signal logic (params + break/retest).

Deterministic, offline: synthetic single-session bars with a known PDH/PDL so we
can assert the break detection, the bars_since_break window, and the
one-attempt-per-side-per-day rule.
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

BASE = datetime(2025, 1, 2, 14, 30)


def _bars(rows):
    return [
        Bar(ts=BASE + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=1000)
        for i, (o, h, l, c) in enumerate(rows)
    ]


def _zero_cost():
    ac = AssetCosts(0.0, 0.0, None, 0.0, None)
    return CostModel("zero", {"equity": ac})


def _run(strat, bars, levels):
    eng = BacktestEngine(
        strat, _zero_cost(), symbol="T", asset_class="equity", tick=0.01,
        initial_equity=100_000.0, percent_of_equity=1.0,
    )
    return eng.run(bars, {"S": levels}, lambda b: "S")


def test_params_v0_defaults():
    p = load_params("V0")
    assert p["entry_type"] == "retest"
    assert p["target_mode"] == "fixed_2r"
    assert p["retest_window_min"] == 2
    assert p["retest_window_max"] == 7
    assert p["r_multiple"] == 2.0
    assert p["_variant"] == "V0"


def test_variant_deltas_merge():
    v1 = load_params("V1")
    assert v1["ntz_filter"] is True            # V1 delta
    assert v1["entry_type"] == "retest"        # inherited default
    v2 = load_params("V2")
    assert v2["use_pmh_pml"] is True
    assert "pmh" in v2["levels"]


def test_retest_long_fires_in_window_and_enters_next_open():
    # PDH=100. bar0 breaks (close 100.2). Then within bars_since in [2,7] a bar
    # dips to PDH (low<=100) and closes above -> retest long, fills next open.
    pdh = 100.0
    bars = _bars([
        (100.1, 100.3, 100.0, 100.20),  # 0: break (close>pdh) -> bsh=1
        (100.2, 100.4, 100.1, 100.30),  # 1: bsh=2 (low 100.1 > pdh, no retest)
        (100.2, 100.3, 99.95, 100.10),  # 2: bsh=3, low<=pdh & close>pdh -> RETEST
        (100.2, 101.0, 100.1, 100.90),  # 3: entry fills @ open 100.2
        (100.9, 101.5, 100.0, 100.50),  # 4
        (100.5, 100.6, 100.0, 100.30),  # 5 (final / EOD)
    ])
    levels = {"pdh": pdh, "pdl": 95.0}
    strat = BreakoutRetestStrategy(variant="V0")
    res = _run(strat, bars, levels)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "long"
    assert tr["entry_ts"] == bars[3].ts           # next bar open after retest bar
    assert tr["entry_price"] == pytest.approx(100.2)
    # stop = pdh - 1 tick.
    assert tr["stop"] == pytest.approx(99.99)


def test_no_retest_before_window_min():
    # A retest-shaped bar at bars_since==1 (the break bar) must NOT fire (min=2).
    # bar0 breaks AND its low touches PDH (retest shape) but bsh==1 -> blocked.
    # Subsequent bars stay strictly above PDH (no further retest), so 0 trades.
    pdh = 100.0
    bars = _bars([
        (99.9, 100.2, 99.8, 100.05),    # 0: break + retest shape, bsh=1 -> NO entry
        (100.1, 100.2, 100.06, 100.15),  # 1: bsh=2 but low 100.06 > pdh (no touch)
        (100.16, 100.2, 100.07, 100.12),  # 2: bsh=3, low 100.07 > pdh (no touch, final)
    ])
    levels = {"pdh": pdh, "pdl": 95.0}
    res = _run(BreakoutRetestStrategy(variant="V0"), bars, levels)
    assert len(res.trades) == 0


def test_one_attempt_per_side_after_stop():
    # After a long stop-out, a second qualifying retest the same day is blocked.
    pdh = 100.0
    bars = _bars([
        (100.1, 100.3, 100.0, 100.20),  # 0 break, bsh=1
        (100.2, 100.3, 99.95, 100.10),  # 1 bsh=2 retest -> arm long
        (100.1, 100.2, 99.0, 99.10),    # 2 entry @100.1; then later bars stop it
        (99.1, 99.2, 98.0, 98.10),      # 3 low<=stop(99.99) -> STOP OUT
        (99.0, 100.3, 98.9, 100.20),    # 4 another retest-shaped bar
        (100.2, 100.4, 99.9, 100.10),   # 5 would-be retest again
        (100.1, 100.2, 100.0, 100.05),  # 6 final
    ])
    levels = {"pdh": pdh, "pdl": 90.0}
    res = _run(BreakoutRetestStrategy(variant="V0"), bars, levels)
    # Exactly one long trade despite a second qualifying setup later.
    assert (res.trades["side"] == "long").sum() == 1
    assert res.trades.iloc[0]["exit_reason"] in ("stop", "stop_gap")
