"""tests/test_chart_for_trade.py — the per-trade chart PNG (headless Agg).

Light/optional smoke test: a synthetic trade produces a non-empty PNG with no
display (matplotlib Agg backend). Covers both the with-bars path and the
degraded levels-only path (no bars available).
"""

from __future__ import annotations

import os
from datetime import datetime

from orchestrator.agents.journalist import Journalist, Trade


def _png_nonempty(path) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1000


def _trade_with_bars() -> Trade:
    bars = [
        {"open": 99.8, "high": 100.2, "low": 99.6, "close": 100.0},
        {"open": 100.0, "high": 100.4, "low": 99.9, "close": 100.3},
        {"open": 100.3, "high": 101.0, "low": 100.2, "close": 100.9},
        {"open": 100.9, "high": 102.1, "low": 100.8, "close": 102.0},
    ]
    return Trade(
        symbol="QQQ",
        strategy="breakout_retest",
        side="long",
        planned_entry=100.0,
        planned_stop=99.0,
        planned_target=102.0,
        levels={"PDH": 101.2, "PDL": 99.3},
        entry_price=100.05,
        exit_price=102.0,
        entry_ts=datetime(2026, 6, 1, 14, 35),
        exit_ts=datetime(2026, 6, 1, 15, 50),
        exit_reason="target",
        bars=bars,
    )


def test_chart_for_trade_with_bars(tmp_path):
    j = Journalist(journal_dir=tmp_path / "journal",
                   market_db=tmp_path / "nope.duckdb",
                   notifier=lambda *a, **k: "")
    out = tmp_path / "trade.png"
    p = j.chart_for_trade(_trade_with_bars(), out)
    assert _png_nonempty(p)


def test_chart_for_trade_degrades_without_bars(tmp_path):
    """No bars + no market.duckdb still renders a valid PNG (levels + markers)."""
    j = Journalist(journal_dir=tmp_path / "journal",
                   market_db=tmp_path / "nope.duckdb",
                   notifier=lambda *a, **k: "")
    t = _trade_with_bars()
    t.bars = None  # force the degraded path
    out = tmp_path / "trade_nobars.png"
    p = j.chart_for_trade(t, out)
    assert _png_nonempty(p)
