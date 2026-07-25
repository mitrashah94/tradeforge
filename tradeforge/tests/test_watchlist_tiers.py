"""tests/test_watchlist_tiers.py — CORE/ACTIVE/SCOUT tiering + the correlation check.

Deterministic, offline. Drives :func:`watchlist.weekly.assign_tiers` (and the
daily-return correlation helper) directly with synthetic candidates and bars so
we can assert:
  * tier max-counts are respected (overflow cascades CORE -> ACTIVE -> SCOUT);
  * level-respect gates exclude low-fit names;
  * the CORRELATION CHECK rejects a highly-correlated duplicate from CORE — feed
    two near-identical daily-return series and only ONE makes CORE (don't fill
    CORE with three SPY proxies, MASTER_PLAN §4).

A small end-to-end smoke of :func:`run_weekly` over the real market.duckdb (if
present) confirms the wiring.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.schema import DEFAULT_DB_PATH, connect
from watchlist.weekly import (
    SymbolEntry,
    assign_tiers,
    daily_price_returns,
    returns_correlation,
    run_weekly,
)

BASE = datetime(2025, 1, 6, 14, 30)


# --------------------------------------------------------------------------- #
# Synthetic bar builders driving daily price returns
# --------------------------------------------------------------------------- #
def _bars_from_daily_closes(closes: list[float]) -> pd.DataFrame:
    """One bar per session at the given close (so daily returns == pct_change)."""
    rows = []
    for d, c in enumerate(closes):
        ts = BASE + timedelta(days=d)
        rows.append(
            {"ts_utc": ts, "open": c, "high": c + 0.1, "low": c - 0.1,
             "close": c, "volume": 1000.0}
        )
    return pd.DataFrame(rows)


def _entry(symbol, fit, dv=100_000_000.0, px=100.0) -> SymbolEntry:
    return SymbolEntry(
        symbol=symbol, asset_class="equity", tier=None, fit_score=fit,
        best_strategy="breakout_retest", avg_dollar_volume=dv, last_price=px,
        rvol=float("nan"),
    )


def _criteria(core_max_corr=0.85, core_gate=0.55, active_gate=0.45, scout_gate=0.0):
    return {
        "tiers": {
            "CORE": {"max_symbols": 2, "min_dollar_volume": 50_000_000,
                     "min_price": 5, "require_score": core_gate},
            "ACTIVE": {"max_symbols": 3, "min_dollar_volume": 20_000_000,
                       "min_price": 3, "require_score": active_gate},
            "SCOUT": {"max_symbols": 5, "min_dollar_volume": 5_000_000,
                      "min_price": 1, "require_score": scout_gate},
        },
        "require_correlation_check": True,
        "core_max_correlation": core_max_corr,
        "require_level_respect_score": 0.0,
    }


# --------------------------------------------------------------------------- #
# Correlation helper
# --------------------------------------------------------------------------- #
def test_daily_price_returns_and_correlation_of_identical_series():
    rng = np.random.default_rng(0)
    closes = list(100.0 + np.cumsum(rng.normal(0, 1, 60)))
    a = _bars_from_daily_closes(closes)
    b = _bars_from_daily_closes([c + 0.0001 for c in closes])  # ~identical
    corr = returns_correlation(["A", "B"], {"A": a, "B": b})
    assert corr.loc["A", "B"] > 0.99


# --------------------------------------------------------------------------- #
# The headline requirement: correlation check rejects a duplicate from CORE
# --------------------------------------------------------------------------- #
def test_correlation_check_rejects_duplicate_from_core():
    rng = np.random.default_rng(7)
    # Two near-identical return streams (SPY/QQQ proxies) ...
    base_closes = list(100.0 + np.cumsum(rng.normal(0, 1, 80)))
    spy = _bars_from_daily_closes(base_closes)
    qqq = _bars_from_daily_closes([c * 1.001 for c in base_closes])  # ~1.0 corr
    # ... and a genuinely independent third name.
    indep_closes = list(100.0 + np.cumsum(rng.normal(0, 1, 80)))
    indep = _bars_from_daily_closes(indep_closes)

    bars_by_symbol = {"SPY": spy, "QQQ": qqq, "INDEP": indep}
    # All three clear the CORE liquidity + fit gates; SPY has the best fit so it
    # is admitted to CORE first, QQQ should be rejected (too correlated), INDEP
    # admitted (decorrelated).
    candidates = [
        _entry("SPY", fit=0.90),
        _entry("QQQ", fit=0.85),
        _entry("INDEP", fit=0.80),
    ]
    entries, core_corr = assign_tiers(candidates, _criteria(core_max_corr=0.85),
                                      bars_by_symbol)
    tier = {e.symbol: e.tier for e in entries}

    assert tier["SPY"] == "CORE"          # best fit, admitted first
    assert tier["INDEP"] == "CORE"        # decorrelated -> admitted
    assert tier["QQQ"] != "CORE"          # near-duplicate of SPY -> rejected
    assert tier["QQQ"] == "ACTIVE"        # falls through to ACTIVE
    # Exactly one of the correlated pair is in CORE.
    assert ["SPY", "INDEP"] == [s for s in ("SPY", "INDEP") if tier[s] == "CORE"]
    assert "SPY" in core_corr.columns and "INDEP" in core_corr.columns
    assert "QQQ" not in core_corr.columns


def test_correlation_check_off_admits_both_correlated():
    rng = np.random.default_rng(3)
    base_closes = list(100.0 + np.cumsum(rng.normal(0, 1, 60)))
    spy = _bars_from_daily_closes(base_closes)
    qqq = _bars_from_daily_closes([c * 1.001 for c in base_closes])
    bars_by_symbol = {"SPY": spy, "QQQ": qqq}
    crit = _criteria()
    crit["require_correlation_check"] = False
    candidates = [_entry("SPY", 0.90), _entry("QQQ", 0.85)]
    entries, _ = assign_tiers(candidates, crit, bars_by_symbol)
    tier = {e.symbol: e.tier for e in entries}
    assert tier["SPY"] == "CORE" and tier["QQQ"] == "CORE"  # both, no check


# --------------------------------------------------------------------------- #
# Max-count caps + gates
# --------------------------------------------------------------------------- #
def test_tier_max_counts_respected_with_cascade():
    rng = np.random.default_rng(11)
    # 6 mutually-decorrelated names so the correlation check never rejects; the
    # CORE cap (2) and ACTIVE cap (3) must do the limiting, overflow -> SCOUT.
    bars_by_symbol = {}
    candidates = []
    for i in range(6):
        closes = list(100.0 + np.cumsum(rng.normal(0, 1, 60)))
        sym = f"S{i}"
        bars_by_symbol[sym] = _bars_from_daily_closes(closes)
        candidates.append(_entry(sym, fit=0.9 - i * 0.05))  # all clear all gates

    entries, _ = assign_tiers(candidates, _criteria(), bars_by_symbol)
    counts = {}
    for e in entries:
        counts[e.tier] = counts.get(e.tier, 0) + 1
    assert counts.get("CORE", 0) == 2     # capped
    assert counts.get("ACTIVE", 0) == 3   # capped
    assert counts.get("SCOUT", 0) == 1    # the 6th cascades to SCOUT


def test_fit_gate_excludes_low_score_from_core_and_active():
    rng = np.random.default_rng(5)
    bars_by_symbol = {}
    candidates = []
    # Two strong (fit 0.6) and one weak (fit 0.2, below ACTIVE's 0.45 gate).
    for sym, fit in [("A", 0.60), ("B", 0.58), ("WEAK", 0.20)]:
        closes = list(100.0 + np.cumsum(rng.normal(0, 1, 60)))
        bars_by_symbol[sym] = _bars_from_daily_closes(closes)
        candidates.append(_entry(sym, fit))
    entries, _ = assign_tiers(candidates, _criteria(), bars_by_symbol)
    tier = {e.symbol: e.tier for e in entries}
    assert tier["A"] == "CORE"
    assert tier["B"] == "CORE"
    # WEAK fails CORE (0.55) and ACTIVE (0.45) gates but clears SCOUT (0.0).
    assert tier["WEAK"] == "SCOUT"


# --------------------------------------------------------------------------- #
# End-to-end smoke against the real DB (if present)
# --------------------------------------------------------------------------- #
def test_run_weekly_smoke_real_db(tmp_path):
    if not os.path.exists(DEFAULT_DB_PATH):
        return  # offline-only environments skip the integration smoke
    con = connect(DEFAULT_DB_PATH)
    try:
        universe_db = str(tmp_path / "universe.duckdb")

        class _FakeBus:
            def __init__(self):
                self.events = []

            def publish(self, ev):
                self.events.append(ev)
                return ev

        bus = _FakeBus()
        result = run_weekly(
            con=con, bus=bus, universe_db_path=universe_db,
            lookback_sessions=60,
        )
        # CORE never exceeds its configured cap (read from criteria.yaml so this
        # stays correct if the cap is re-tuned — it is 4, not the old hard-coded 2).
        from watchlist.weekly import load_criteria
        core_cap = int(load_criteria()["tiers"]["CORE"]["max_symbols"])
        assert len(result.core()) <= core_cap
        # SPY + QQQ are ~0.95 correlated, so they cannot BOTH be CORE.
        assert not (set(result.core()) >= {"SPY", "QQQ"})
        # A WATCHLIST_UPDATED event was emitted.
        types = [e.type.value for e in bus.events]
        assert "WATCHLIST_UPDATED" in types
        # Persistence wrote rows.
        ucon = connect(universe_db)
        try:
            n = ucon.execute("SELECT COUNT(*) FROM tiers").fetchone()[0]
            assert n >= 1
        finally:
            ucon.close()
    finally:
        con.close()
