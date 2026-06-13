"""tests/test_momentum_thrust.py — momentum-thrust signal + trailing-exit logic.

Deterministic, offline: synthetic single-session bars (20 calm bars to seed the
range/volume averages, then an expansion thrust bar) so we can assert the thrust
entry (no fixed target, protective stop pinned to the thrust extreme), the ATR
chandelier TRAILING exit, the hard-stop backstop, and the chop=no-trade case.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.engine.cost import AssetCosts, CostModel
from backtest.engine.engine import Bar, BacktestEngine
from strategies.momentum_thrust.strategy import (
    DEFAULT_VARIANT,
    VARIANTS,
    MomentumThrustStrategy,
    load_params,
)

BASE = datetime(2025, 1, 2, 14, 30)


def _bars(rows):
    return [
        Bar(ts=BASE + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, volume=v)
        for i, (o, h, l, c, v) in enumerate(rows)
    ]


def _zero_cost():
    ac = AssetCosts(0.0, 0.0, None, 0.0, None)
    return CostModel("zero", {"equity": ac})


def _run(strat, bars, levels=None):
    levels = levels or {"pdh": 200.0, "pdl": 1.0, "atr14": 1.0}
    eng = BacktestEngine(
        strat, _zero_cost(), symbol="T", asset_class="equity", tick=0.01,
        initial_equity=100_000.0, percent_of_equity=1.0,
    )
    return eng.run(bars, {"S": levels}, lambda b: "S")


def _calm(n=20):
    """n calm bars: range 0.20, flat close 100.0 — seeds the averages."""
    return [(100.0, 100.10, 99.90, 100.0, 1000) for _ in range(n)]


# --------------------------------------------------------------------------- #
# params
# --------------------------------------------------------------------------- #
def test_params_default_and_variants():
    p = load_params(DEFAULT_VARIANT)
    assert p["thrust_bars"] == 2
    assert p["use_volume"] is True
    assert p["trail_atr"] == 1.5
    assert p["_variant"] == DEFAULT_VARIANT
    for v in VARIANTS:
        assert load_params(v)["_variant"] == v
    assert load_params("V0")["use_volume"] is False     # range-only thrust


# --------------------------------------------------------------------------- #
# entry + trailing exit
# --------------------------------------------------------------------------- #
def test_long_thrust_enters_with_no_target_and_trails_out():
    # ATR=1.0. 20 calm bars (avg_range 0.20). bar20 small up bar (range 0.20,
    # NOT expansion). bar21 THRUST UP: range 1.10 >= 1.3*0.20, closes top, close
    # 101.0 > prev close 100.05 (2-bar up run). Enter LONG @ bar22 open 101.0,
    # NO target, stop = thrust low(100.00) - 0.5*ATR = 99.50.
    # Ride up to best high 102.40, then bar23 closes 100.70 <= trail
    # (102.40 - 1.5 = 100.90) -> strategy CLOSE at next open.
    rows = _calm(20) + [
        (100.0, 100.15, 99.95, 100.05, 1000),    # 20 prime (close>prev)
        (100.05, 101.10, 100.00, 101.00, 3000),  # 21 THRUST up
        (101.0, 102.30, 100.90, 102.20, 2000),   # 22 entry @101.0; best 102.30
        (102.2, 102.40, 100.60, 100.70, 2000),   # 23 close<=trail -> CLOSE
        (100.7, 100.80, 100.40, 100.60, 1000),   # 24 final (close fills @100.7)
    ]
    res = _run(MomentumThrustStrategy(variant="V0"), _bars(rows))
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "long"
    assert tr["entry_price"] == pytest.approx(101.0)         # next-open after thrust
    assert tr["target"] is None                              # NO fixed target — it rides
    assert tr["stop"] == pytest.approx(99.50)                # thrust low - 0.5*ATR
    assert tr["exit_reason"] == "strategy_close"             # trailed out, not hard stop
    assert tr["exit_price"] == pytest.approx(100.70)


def test_short_thrust_enters_and_trails_out():
    # Symmetric down thrust. bar21 THRUST DOWN: range 1.10, closes bottom, close
    # 99.0 < prev 99.95. Enter SHORT @ bar22 open 99.0; stop = thrust high(100.0)
    # + 0.5 = 100.50. Best low 97.60; bar23 close 99.30 >= trail(97.60+1.5=99.10)
    # -> strategy CLOSE.
    rows = _calm(20) + [
        (100.0, 100.05, 99.85, 99.95, 1000),     # 20 prime (close<prev)
        (99.95, 100.00, 98.90, 99.00, 3000),     # 21 THRUST down
        (99.0, 99.10, 97.70, 97.80, 2000),       # 22 entry @99.0; best low 97.70
        (97.8, 99.40, 97.60, 99.30, 2000),       # 23 close>=trail -> CLOSE
        (99.3, 99.40, 99.00, 99.10, 1000),       # 24 final
    ]
    res = _run(MomentumThrustStrategy(variant="V0"), _bars(rows))
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "short"
    assert tr["entry_price"] == pytest.approx(99.0)
    assert tr["target"] is None
    assert tr["stop"] == pytest.approx(100.50)               # thrust high + 0.5*ATR
    assert tr["exit_reason"] == "strategy_close"


def test_hard_protective_stop_backstops_a_failed_thrust():
    # Thrust, a mild bar (no trail/stop), then a reversal that pierces the
    # protective stop on a MANAGED bar (the engine manages the bracket only on
    # bars after the entry bar). stop = thrust low(100.00) - 0.5*ATR = 99.50.
    rows = _calm(20) + [
        (100.0, 100.15, 99.95, 100.05, 1000),    # 20 prime
        (100.05, 101.10, 100.00, 101.00, 3000),  # 21 THRUST up; stop=99.50
        (101.0, 101.20, 100.80, 101.00, 2000),   # 22 entry @101.0; mild (no trail/stop)
        (101.0, 101.10, 99.40, 100.90, 2000),    # 23 managed; low 99.40<=99.50 -> STOP
        #    (close 100.90 sits high in range -> not a down thrust, so no re-entry)
        (100.9, 101.00, 100.70, 100.80, 1000),   # 24 final
    ]
    res = _run(MomentumThrustStrategy(variant="V0"), _bars(rows))
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["exit_reason"] in ("stop", "stop_gap")
    assert tr["exit_price"] == pytest.approx(99.50)


def test_chop_produces_no_thrust():
    # No expansion bar ever prints (all ranges == avg) -> zero entries.
    rows = [
        (100.0, 100.10, 99.90, 100.0 + (0.01 if i % 2 else -0.01), 1000)
        for i in range(25)
    ]
    res = _run(MomentumThrustStrategy(variant="V0"), _bars(rows))
    assert len(res.trades) == 0


def test_volume_filter_blocks_low_volume_expansion():
    # DEFAULT requires volume >= vol_mult*avg_vol (1.2x). An expansion bar on
    # AVERAGE volume (1000, same as the calm baseline) must NOT fire under
    # DEFAULT but DOES fire under V0 (use_volume=false). Proves the param works.
    rows = _calm(20) + [
        (100.0, 100.15, 99.95, 100.05, 1000),    # 20 prime
        (100.05, 101.10, 100.00, 101.00, 1000),  # 21 expansion but LOW volume (==avg)
        (101.0, 101.20, 100.90, 101.10, 1000),   # 22
        (101.1, 101.20, 100.50, 100.60, 1000),   # 23
        (100.6, 100.70, 100.40, 100.50, 1000),   # 24 final
    ]
    res_default = _run(MomentumThrustStrategy(variant="DEFAULT"), _bars(rows))
    assert len(res_default.trades) == 0                      # volume filter blocks it
    res_v0 = _run(MomentumThrustStrategy(variant="V0"), _bars(rows))
    assert len(res_v0.trades) == 1                           # range-only lets it fire
