"""tests/test_level_meanrev.py — level-fade mean-reversion signal logic.

Deterministic, offline: synthetic single-session bars with a known PDH/PDL +
ATR so we can assert the tag-and-rejection fade, the clean-break DISARM, the
fixed-R and session-midpoint targets, and the one-fade-per-side rule.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from backtest.engine.cost import AssetCosts, CostModel
from backtest.engine.engine import Bar, BacktestEngine
from strategies.level_meanrev.strategy import (
    DEFAULT_VARIANT,
    VARIANTS,
    LevelMeanRevStrategy,
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


# --------------------------------------------------------------------------- #
# params
# --------------------------------------------------------------------------- #
def test_params_default_and_variants():
    p = load_params(DEFAULT_VARIANT)
    assert p["mean_mode"] == "fixed_r"          # robust fixed-R fade is the default
    assert "pdh" in p["upper_levels"]
    assert "pdl" in p["lower_levels"]
    assert p["_variant"] == DEFAULT_VARIANT
    # Every declared VARIANT loads.
    for v in VARIANTS:
        assert load_params(v)["_variant"] == v
    # V0 is the lean PDH/PDL-only variant.
    v0 = load_params("V0")
    assert v0["upper_levels"] == ["pdh"]
    assert v0["lower_levels"] == ["pdl"]


# --------------------------------------------------------------------------- #
# core fade logic
# --------------------------------------------------------------------------- #
def test_short_fade_of_pdh_tag_targets_below():
    # PDH=100, ATR=1.0 -> tol=0.10, brk=0.05, stop_buffer=0.10.
    # bar2 tags PDH (high 100.05 >= 99.90) and REJECTS (close 99.80 < 100, and
    # 99.80 < 100.05 so not a clean break) -> arm SHORT, fills next open.
    # V0 fixed_r target_r=1.2: stop=max(100.05,100)+0.10=100.15; risk=
    # |99.80-100.15|=0.35; target=99.80-1.2*0.35=99.38. A later bar dips to
    # <=99.38 -> target hit.
    pdh = 100.0
    bars = _bars([
        (99.50, 99.60, 99.40, 99.55),   # 0 warmup
        (99.55, 99.70, 99.50, 99.65),   # 1 warmup
        (99.70, 100.05, 99.60, 99.80),  # 2 tag+reject PDH -> arm SHORT
        (99.80, 99.85, 99.30, 99.40),   # 3 entry @ open 99.80; low 99.30<=99.38 -> TARGET
        (99.40, 99.50, 99.20, 99.30),   # 4 (final)
    ])
    levels = {"pdh": pdh, "pdl": 90.0, "pmh": 105.0, "pml": 88.0, "atr14": 1.0}
    res = _run(LevelMeanRevStrategy(variant="V0"), bars, levels)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "short"
    assert tr["entry_ts"] == bars[3].ts                    # next bar open after signal
    assert tr["entry_price"] == pytest.approx(99.80)
    assert tr["stop"] == pytest.approx(100.15)             # max(high,pdh)+0.10*ATR
    assert tr["target"] == pytest.approx(99.38)            # close - 1.2R
    assert tr["exit_reason"] == "target"
    assert tr["exit_price"] == pytest.approx(99.38)        # limit fills at target


def test_long_fade_of_pdl_tag():
    # PDL=100, ATR=1.0. bar2 tags PDL from above (low 99.95<=100.10) and closes
    # back ABOVE (100.20 > 100) -> arm LONG. stop=min(99.95,100)-0.10=99.85;
    # risk=|100.20-99.85|=0.35; V0 target_r=1.2 -> target=100.20+1.2*0.35=100.62.
    pdl = 100.0
    bars = _bars([
        (100.6, 100.7, 100.5, 100.6),   # 0 warmup
        (100.5, 100.6, 100.4, 100.5),   # 1 warmup
        (100.3, 100.4, 99.95, 100.20),  # 2 tag+reject PDL -> arm LONG
        (100.2, 100.7, 100.1, 100.60),  # 3 entry @100.2; high 100.7>=100.62 -> TARGET
        (100.6, 100.7, 100.5, 100.6),   # 4 final
    ])
    levels = {"pdh": 110.0, "pdl": pdl, "pmh": 112.0, "pml": 99.0, "atr14": 1.0}
    res = _run(LevelMeanRevStrategy(variant="V0"), bars, levels)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "long"
    assert tr["stop"] == pytest.approx(99.85)
    assert tr["target"] == pytest.approx(100.62)
    assert tr["exit_reason"] == "target"


def test_clean_break_disarms_the_fade():
    # A bar that CLOSES beyond PDH by >= break_buffer_atr*ATR (0.05) is a clean
    # breakout, NOT a fade -> no short. PDH=100, ATR=1.0: close 100.20 >= 100.05
    # disarms. No qualifying rejection later -> zero trades.
    pdh = 100.0
    bars = _bars([
        (99.50, 99.60, 99.40, 99.55),   # 0 warmup
        (99.60, 99.80, 99.55, 99.70),   # 1 warmup
        (99.80, 100.30, 99.75, 100.20),  # 2 CLEAN BREAK (close 100.20 >= 100.05) -> NO fade
        (100.2, 100.5, 100.1, 100.40),  # 3 stays above, still no rejection
        (100.4, 100.6, 100.3, 100.50),  # 4 final
    ])
    levels = {"pdh": pdh, "pdl": 90.0, "pmh": 105.0, "pml": 88.0, "atr14": 1.0}
    res = _run(LevelMeanRevStrategy(variant="V0"), bars, levels)
    assert len(res.trades) == 0


def test_one_fade_per_side_per_session():
    # Two qualifying short-fade setups; with one_per_side only the first arms.
    pdh = 100.0
    bars = _bars([
        (99.50, 99.60, 99.40, 99.55),   # 0 warmup
        (99.55, 99.70, 99.50, 99.65),   # 1 warmup
        (99.70, 100.05, 99.60, 99.80),  # 2 tag+reject -> arm SHORT #1
        (99.80, 99.90, 99.40, 99.50),   # 3 entry @99.80; later target/exit
        (99.50, 100.04, 99.45, 99.55),  # 4 ANOTHER tag+reject (blocked by one_per_side)
        (99.55, 99.70, 99.50, 99.60),   # 5 final
    ])
    levels = {"pdh": pdh, "pdl": 90.0, "pmh": 105.0, "pml": 88.0, "atr14": 1.0}
    res = _run(LevelMeanRevStrategy(variant="V0"), bars, levels)
    assert (res.trades["side"] == "short").sum() == 1


def test_session_mid_target_mode():
    # mean_mode=session_mid (V2): target is the running session midpoint.
    # PDH=100, ATR=1.0. Build a session whose running [low,high] gives a midpoint
    # comfortably below the short entry so the fade has room.
    pdh = 100.0
    bars = _bars([
        (98.00, 98.20, 97.80, 98.00),   # 0 warmup -> session low ~97.80
        (98.10, 98.40, 98.00, 98.30),   # 1 warmup
        (99.20, 100.05, 99.10, 99.80),  # 2 tag+reject PDH -> session hi=100.05, lo=97.80
        #    midpoint=(100.05+97.80)/2=98.925; entry close 99.80; reward=0.875
        (99.80, 99.90, 98.50, 98.60),   # 3 entry @99.80; low 98.50<=98.925 -> TARGET=mid
        (98.60, 98.80, 98.40, 98.50),   # 4 final
    ])
    levels = {"pdh": pdh, "pdl": 90.0, "pmh": 105.0, "pml": 88.0, "atr14": 1.0}
    p = load_params("V2")
    res = _run(LevelMeanRevStrategy(params=p), bars, levels)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["side"] == "short"
    assert tr["target"] == pytest.approx(98.925)          # running session midpoint
    assert tr["exit_reason"] == "target"
