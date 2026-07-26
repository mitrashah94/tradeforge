"""tests/test_watchlist_metadata.py — off-hot-path metadata fetch/read + premarket wiring.

Deterministic, offline. The metadata refresh's MCP calls are MOCKED (injected
callables), so this asserts the fetch/JOIN contract without any network: ETFs take
their sector from the hardcoded map, single names from the (mock) fundamentals
call, spread from the (mock) quote, fractional from the (mock) tradability. Then
``read_metadata`` round-trips, and the premarket universe wiring unions the
watchlist selection with the sleeves' structural symbols.
"""

from __future__ import annotations

import duckdb
import pytest

from watchlist.fetch_metadata import (
    _spread_bps_from_quote,
    etf_sector,
    read_metadata,
    refresh_metadata,
)


# --------------------------------------------------------------------------- #
# ETF sector map + spread math
# --------------------------------------------------------------------------- #
def test_etf_sector_map():
    assert etf_sector("XLK") == "Technology"
    assert etf_sector("xlf") == "Financials"
    assert etf_sector("TLT") == "Fixed Income"
    assert etf_sector("NVDA") is None   # single name -> not in the ETF map


def test_spread_bps_from_quote():
    # bid 99.95 / ask 100.05 -> mid 100, spread 0.10 -> 10 bps.
    assert _spread_bps_from_quote({"bid_price": 99.95, "ask_price": 100.05}) == pytest.approx(10.0)
    assert _spread_bps_from_quote({}) is None
    assert _spread_bps_from_quote({"bid_price": 100, "ask_price": 99}) is None  # crossed


# --------------------------------------------------------------------------- #
# refresh + read round-trip (MCP mocked)
# --------------------------------------------------------------------------- #
def test_refresh_and_read_round_trip():
    con = duckdb.connect(":memory:")
    quotes = {
        "NVDA": {"bid_price": 99.0, "ask_price": 101.0},   # 200/100 mid -> ~200bps
        "XLK": {"bid_price": 199.95, "ask_price": 200.05},  # ~5 bps
    }
    fundamentals = {"NVDA": {"sector": "Technology"}}
    tradability = {"NVDA": {"fractional_tradable": True}, "XLK": {"fractional_tradable": False}}
    n = refresh_metadata(
        ["NVDA", "XLK"], con,
        quotes_fn=lambda s: quotes.get(s),
        fundamentals_fn=lambda s: fundamentals.get(s),
        tradability_fn=lambda s: tradability.get(s),
    )
    assert n == 2
    meta = read_metadata(con)
    # NVDA: sector from fundamentals (single name), spread from quote, fractional.
    assert meta["NVDA"]["sector"] == "Technology"
    assert meta["NVDA"]["spread_bps"] == pytest.approx(200.0, rel=1e-3)
    assert meta["NVDA"]["fractional_enabled"] is True
    # XLK: sector from the ETF map (fundamentals NOT consulted), not fractional.
    assert meta["XLK"]["sector"] == "Technology"
    assert meta["XLK"]["fractional_enabled"] is False


def test_read_metadata_absent_table_is_empty():
    con = duckdb.connect(":memory:")
    assert read_metadata(con) == {}


def test_refresh_survives_a_failing_call():
    con = duckdb.connect(":memory:")

    def boom(_s):
        raise RuntimeError("mcp down")

    # A failing fundamentals call must not crash the batch; the ETF still resolves
    # its sector from the map and the row is written.
    n = refresh_metadata(["XLF", "NVDA"], con, fundamentals_fn=boom)
    assert n == 2
    meta = read_metadata(con)
    assert meta["XLF"]["sector"] == "Financials"
    assert meta["NVDA"]["sector"] is None   # single name, fundamentals failed


# --------------------------------------------------------------------------- #
# premarket universe wiring (Phase 2 -> Phase 1)
# --------------------------------------------------------------------------- #
def test_premarket_universe_unions_selection_and_sleeve_symbols():
    from datetime import datetime

    from orchestrator.workflows.premarket import portfolio_universe, select_universe
    from portfolio.model import SleeveSpec
    from watchlist.weekly import SymbolEntry, WatchlistResult

    entries = [
        SymbolEntry("NVDA", "equity", "CORE", 0.9, "b", 1e8, 100, float("nan")),
        SymbolEntry("AAPL", "equity", "ACTIVE", 0.7, "b", 1e8, 100, float("nan")),
        SymbolEntry("PENNY", "equity", "SCOUT", 0.1, "b", 1e7, 2, float("nan")),
    ]
    result = WatchlistResult(run_ts=datetime(2026, 6, 29), entries=entries,
                             correlation=__import__("pandas").DataFrame())

    assert select_universe(result, include_active=False) == ["NVDA"]
    assert select_universe(result, include_active=True) == ["NVDA", "AAPL"]

    # A weight sleeve exposing a `.universe` contributes its structural tickers.
    class WeightStrat:
        universe = ["SPY", "QQQ", "XLK"]

        def target_weights(self, asof, history):
            return {}

    sleeve = SleeveSpec(name="mr", strategy=WeightStrat(), kind="weight")
    uni = portfolio_universe(result, [sleeve], include_active=True)
    # union of CORE+ACTIVE selection and the sleeve's symbols, sorted, deduped.
    assert set(uni) == {"NVDA", "AAPL", "SPY", "QQQ", "XLK"}
    assert uni == sorted(uni)
