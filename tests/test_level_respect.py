"""tests/test_level_respect.py — the per-strategy level-respect mini-backtest score.

Deterministic, offline. Two layers:
  1. Pure score math (``score_from_arrays``): bounded in [0, 1], monotone in
     expectancy, 0 with no trades.
  2. Engine replay (``score_synthetic_bars``): a clean trending / level-respecting
     series scores HIGHER than a whipsaw / noise series for the breakout_retest
     strategy — "screening IS a backtest" (MASTER_PLAN §4).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pandas as pd

from data.sessions import et_session_date
from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params
from watchlist.level_respect import score_from_arrays, score_synthetic_bars

# ~09:30 ET in winter (EST = UTC-5) so et_session_date lands on the calendar day.
BASE = datetime(2025, 1, 6, 14, 30)


# --------------------------------------------------------------------------- #
# Pure score math
# --------------------------------------------------------------------------- #
def test_score_bounded_and_no_trades_is_zero():
    score, diag = score_from_arrays([], [])
    assert score == 0.0
    assert diag["n_trades"] == 0


def test_score_in_unit_interval():
    # A mixed set of trades.
    pnls = [100.0, -50.0, 80.0, -40.0, 120.0]
    rmults = [2.0, -1.0, 1.6, -1.0, 2.4]
    score, _ = score_from_arrays(pnls, rmults)
    assert 0.0 <= score <= 1.0


def test_score_monotone_winner_beats_loser_set():
    winners_pnl = [100.0, 90.0, 110.0, 80.0]
    winners_r = [2.0, 1.8, 2.2, 1.6]
    losers_pnl = [-100.0, -90.0, -110.0, -80.0]
    losers_r = [-1.0, -1.0, -1.0, -1.0]
    s_win, _ = score_from_arrays(winners_pnl, winners_r)
    s_lose, _ = score_from_arrays(losers_pnl, losers_r)
    assert s_win > 0.5 > s_lose
    assert s_win > s_lose


# --------------------------------------------------------------------------- #
# Engine replay: trending respects levels more than noise
# --------------------------------------------------------------------------- #
def _build_sessions(kind: str, n_days: int = 12, bars_per: int = 40):
    """Build (bars_df, levels_by_session) for a 'trend' or 'noise' series.

    PDH/PDL are fixed per session. 'trend': price breaks PDH cleanly, retests it
    (touch from above, close back above), then follows through up — the textbook
    respected level. 'noise': a deterministic zig-zag that poke-and-reverses
    around the level (whipsaw), so retests stop out.
    """
    rows = []
    levels = {}
    pdh, pdl, atr = 100.0, 95.0, 2.0
    for d in range(n_days):
        base = BASE + timedelta(days=d)
        prices = []
        if kind == "trend":
            px = 98.0
            for i in range(bars_per):
                if i < 5:
                    px += 0.4
                elif i == 6:
                    px = 100.7           # CLOSE above PDH -> break
                elif i in (8, 9):
                    px = 100.05          # retest: dip to ~PDH, close just above
                else:
                    px += 0.55           # follow-through up
                o = px
                c = px + 0.05
                h = max(o, c) + 0.15
                l = min(o, c) - 0.15
                prices.append((o, h, l, c))
                px = c
        else:  # noise — deterministic whipsaw around the level
            px = 99.0
            for i in range(bars_per):
                # Saw-tooth that pokes above PDH then collapses back below.
                step = 0.9 if (i % 4 in (0, 1)) else -0.9
                px += step
                o = px
                c = px - 0.2 if step > 0 else px + 0.2  # close fades the poke
                h = max(o, c) + 0.5
                l = min(o, c) - 0.5
                prices.append((o, h, l, c))
                px = c
        for i, (o, h, l, c) in enumerate(prices):
            ts = base + timedelta(minutes=5 * i)
            rows.append(
                {"ts_utc": ts, "open": o, "high": h, "low": l, "close": c,
                 "volume": 1000.0}
            )
        sd = et_session_date(rows[-1]["ts_utc"])
        levels[sd] = {
            "pdh": pdh, "pdl": pdl, "pmh": None, "pml": None,
            "ntz_low": None, "ntz_high": None, "ntz_valid": False, "atr14": atr,
        }
    return pd.DataFrame(rows), levels


def test_trending_series_scores_higher_than_noise():
    params = load_params("V0")
    bars_t, lv_t = _build_sessions("trend")
    bars_n, lv_n = _build_sessions("noise")

    score_t = score_synthetic_bars(
        bars_t, lv_t, BreakoutRetestStrategy(params=params), strategy_name="br"
    )
    score_n = score_synthetic_bars(
        bars_n, lv_n, BreakoutRetestStrategy(params=params), strategy_name="br"
    )

    # Both bounded.
    assert 0.0 <= score_t.score <= 1.0
    assert 0.0 <= score_n.score <= 1.0
    # The respecting series must actually trade and score clearly higher.
    assert score_t.n_trades > 0
    assert score_t.score > score_n.score
    assert score_t.expectancy_r > 0.0
