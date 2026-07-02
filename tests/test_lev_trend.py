"""tests/test_lev_trend.py — the leveraged trend-gate sleeve (synthetic, offline)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backtest.daily.engine import DailyHistory
from strategies.lev_trend.strategy import LevTrendStrategy, load_params


def _panel(bench_closes, extra=None):
    idx = [date(2020, 1, 1) + timedelta(days=i) for i in range(len(bench_closes))]
    data = {"SPY": bench_closes}
    for sym in ("QLD", "TQQQ", "BIL", "TLT"):
        data[sym] = [100.0] * len(bench_closes)
    if extra:
        data.update(extra)
    return pd.DataFrame(data, index=idx)


def _hist(panel):
    return DailyHistory(panel, panel.index[-1])


def test_risk_on_above_sma_holds_lev():
    s = LevTrendStrategy(load_params("DEFAULT"))  # SPY 200d gate -> QLD / BIL
    closes = list(np.linspace(100, 150, 210))     # rising: close > SMA
    w = s.target_weights(None, _hist(_panel(closes)))
    assert w == {"QLD": 1.0}


def test_risk_off_below_sma_holds_safe():
    s = LevTrendStrategy(load_params("TQQQ_TLT"))
    closes = list(np.linspace(150, 100, 210))     # falling: close < SMA
    w = s.target_weights(None, _hist(_panel(closes)))
    assert w == {"TLT": 1.0}


def test_insufficient_history_is_risk_off():
    s = LevTrendStrategy(load_params("DEFAULT"))
    w = s.target_weights(None, _hist(_panel([100.0] * 50)))  # < 200 rows
    assert w == {"BIL": 1.0}


def test_dual_gate_requires_positive_momentum():
    s = LevTrendStrategy(load_params("DUAL_GATE"))  # SMA AND 12m mom > 0
    # V-shape: big early drop, late recovery — close is above the (depressed) SMA
    # but the 252-day total return is still negative -> dual gate stays risk-off.
    closes = list(np.linspace(200, 80, 150)) + list(np.linspace(80, 120, 150))
    w = s.target_weights(None, _hist(_panel(closes)))
    assert w == {"BIL": 1.0}


def test_extra_symbols_lists_all_roles():
    s = LevTrendStrategy(load_params("TQQQ_TLT"))
    assert set(s.extra_symbols()) == {"SPY", "TQQQ", "TLT"}
