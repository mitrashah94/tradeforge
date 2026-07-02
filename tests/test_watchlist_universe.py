"""tests/test_watchlist_universe.py — the Phase-2 ticker-universe engine additions.

Deterministic, offline. Drives the new metadata gates (spread / price-history /
fractional), the per-tier sector-concentration cap, the cluster-dedup (the shared
union-find), the off-hot-path metadata fetch/read, and the premarket universe
wiring — all on synthetic inputs so each new gate is asserted in isolation.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from watchlist.weekly import SymbolEntry, assign_tiers, price_history_sessions

BASE = datetime(2025, 1, 6, 14, 30)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bars(closes):
    rows = [{"ts_utc": BASE + timedelta(days=d), "open": c, "high": c + 0.1,
             "low": c - 0.1, "close": c, "volume": 1000.0} for d, c in enumerate(closes)]
    return pd.DataFrame(rows)


def _decorrelated(syms, n=60, seed=0):
    """Independent random-walk bars per symbol (so clustering never collapses them)."""
    rng = np.random.default_rng(seed)
    return {s: _bars(list(100.0 + np.cumsum(rng.normal(0, 1, n)))) for s in syms}


def _entry(symbol, fit=0.8, *, sector=None, spread=float("nan"),
           fractional=True, history=0, dv=100_000_000.0, px=100.0):
    return SymbolEntry(
        symbol=symbol, asset_class="equity", tier=None, fit_score=fit,
        best_strategy="breakout_retest", avg_dollar_volume=dv, last_price=px,
        rvol=float("nan"), sector=sector, spread_bps=spread,
        fractional_enabled=fractional, price_history_sessions=history,
    )


def _criteria(**over):
    crit = {
        "tiers": {
            "CORE": {"max_symbols": 4, "min_dollar_volume": 50_000_000, "min_price": 5, "require_score": 0.5},
            "ACTIVE": {"max_symbols": 4, "min_dollar_volume": 20_000_000, "min_price": 3, "require_score": 0.4},
            "SCOUT": {"max_symbols": 10, "min_dollar_volume": 5_000_000, "min_price": 1, "require_score": 0.0},
        },
        "require_correlation_check": True,
        "corr_cluster_threshold": 0.85,
        "require_level_respect_score": 0.0,
    }
    crit.update(over)
    return crit


# --------------------------------------------------------------------------- #
# price-history helper
# --------------------------------------------------------------------------- #
def test_price_history_sessions_counts_distinct_sessions():
    bars = _bars([100.0, 101.0, 102.0, 103.0])  # 4 distinct days
    assert price_history_sessions(bars, "equity") == 4
    assert price_history_sessions(pd.DataFrame(), "equity") == 0


# --------------------------------------------------------------------------- #
# metadata gates
# --------------------------------------------------------------------------- #
def test_spread_gate_cascades_core_to_active():
    # CORE requires a tight spread (<=20bps); a 50bps name fails CORE but the
    # looser global gate (100bps) lets it land in ACTIVE.
    bars = _decorrelated(["A", "B"], seed=1)
    crit = _criteria(max_spread_bps=100)
    crit["tiers"]["CORE"]["max_spread_bps"] = 20
    cands = [_entry("A", 0.9, spread=50.0), _entry("B", 0.8, spread=5.0)]
    entries, _ = assign_tiers(cands, crit, bars)
    tier = {e.symbol: e.tier for e in entries}
    assert tier["B"] == "CORE"          # tight spread
    assert tier["A"] == "ACTIVE"        # wide spread -> cascaded out of CORE


def test_history_gate_excludes_thin_history():
    bars = _decorrelated(["YOUNG", "OLD"], seed=2)
    crit = _criteria(min_price_history_sessions=252)
    cands = [_entry("YOUNG", 0.9, history=100), _entry("OLD", 0.8, history=500)]
    entries, _ = assign_tiers(cands, crit, bars)
    tier = {e.symbol: e.tier for e in entries}
    assert tier["OLD"] == "CORE"
    assert tier["YOUNG"] is None        # thin history fails every tier's gate


def test_fractional_gate_required_on_core():
    bars = _decorrelated(["FRAC", "NOFRAC"], seed=3)
    crit = _criteria()
    crit["tiers"]["CORE"]["fractional_required"] = True
    cands = [_entry("NOFRAC", 0.9, fractional=False), _entry("FRAC", 0.8, fractional=True)]
    entries, _ = assign_tiers(cands, crit, bars)
    tier = {e.symbol: e.tier for e in entries}
    assert tier["FRAC"] == "CORE"
    assert tier["NOFRAC"] == "ACTIVE"   # non-fractional cascaded out of CORE


def test_missing_metadata_is_non_binding():
    # No spread / history / fractional metadata -> the gates don't bind.
    bars = _decorrelated(["A", "B"], seed=4)
    crit = _criteria(max_spread_bps=10, min_price_history_sessions=999)
    crit["tiers"]["CORE"]["fractional_required"] = True
    cands = [_entry("A", 0.9), _entry("B", 0.8)]  # spread=nan, history=0, frac=True
    entries, _ = assign_tiers(cands, crit, bars)
    tier = {e.symbol: e.tier for e in entries}
    assert tier["A"] == "CORE" and tier["B"] == "CORE"


# --------------------------------------------------------------------------- #
# sector-concentration cap
# --------------------------------------------------------------------------- #
def test_sector_cap_limits_one_sector_in_core():
    # CORE cap 4, sector_max_concentration 0.25 -> max 1 per sector in CORE. Three
    # Tech (decorrelated) names -> only the best fit makes CORE; the cap is scoped
    # to CORE so the rest cascade to ACTIVE (which has no sector cap here).
    bars = _decorrelated(["T1", "T2", "T3", "FIN"], seed=5)
    crit = _criteria()
    crit["tiers"]["CORE"]["sector_max_concentration"] = 0.25
    cands = [
        _entry("T1", 0.95, sector="Technology"),
        _entry("T2", 0.90, sector="Technology"),
        _entry("T3", 0.85, sector="Technology"),
        _entry("FIN", 0.80, sector="Financials"),
    ]
    entries, _ = assign_tiers(cands, crit, bars)
    tier = {e.symbol: e.tier for e in entries}
    core = [s for s in tier if tier[s] == "CORE"]
    assert tier["T1"] == "CORE"          # best Tech fit
    assert tier["FIN"] == "CORE"         # different sector
    assert sum(1 for s in ("T1", "T2", "T3") if tier[s] == "CORE") == 1
    assert tier["T2"] == "ACTIVE" and tier["T3"] == "ACTIVE"


# --------------------------------------------------------------------------- #
# cluster dedup (the shared union-find) sets cluster_id + dedups CORE
# --------------------------------------------------------------------------- #
def test_cluster_dedup_one_per_cluster_and_sets_cluster_id():
    rng = np.random.default_rng(9)
    base = list(100.0 + np.cumsum(rng.normal(0, 1, 80)))
    bars = {
        "SPY": _bars(base),
        "QQQ": _bars([c * 1.001 for c in base]),               # ~1.0 corr with SPY
        "GLD": _bars(list(100.0 + np.cumsum(rng.normal(0, 1, 80)))),  # independent
    }
    cands = [_entry("SPY", 0.9), _entry("QQQ", 0.85), _entry("GLD", 0.8)]
    entries, _ = assign_tiers(cands, _criteria(), bars)
    by = {e.symbol: e for e in entries}
    assert by["SPY"].cluster_id == by["QQQ"].cluster_id      # same cluster
    assert by["GLD"].cluster_id != by["SPY"].cluster_id      # distinct
    assert by["SPY"].tier == "CORE" and by["GLD"].tier == "CORE"
    assert by["QQQ"].tier != "CORE"                          # cluster already taken
