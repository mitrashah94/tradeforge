"""watchlist/fetch_metadata.py — off-hot-path symbol-metadata refresh + reader.

The weekly tiering wants three pieces of per-symbol metadata it cannot get from
OHLCV bars alone: the **sector** (for the sector-concentration cap), the bid/ask
**spread** (a liquidity gate the bars' volume can't see), and **fractional
eligibility** (so a $1k book can actually take a slice of a $500 name). These come
from the Robinhood MCP (``get_equity_quotes`` / ``get_equity_fundamentals`` /
``get_equity_tradability``) — which is SLOW and non-deterministic, so it runs in
the premarket/slow loop, BATCHED, and writes a table the deterministic
``run_weekly`` simply JOINs.

Two halves, with a hard separation so the deterministic path never touches MCP:

  * :func:`refresh_metadata` — the SLOW-LOOP refresh. It takes INJECTED callables
    (``quotes_fn`` / ``fundamentals_fn`` / ``tradability_fn``) rather than
    importing MCP, so it is testable with mocks and the module has no hard MCP
    dependency. ETFs short-circuit to the hardcoded :data:`ETF_SECTORS` map
    (mirrors the universe grouping in ``data/pipelines/yahoo_ingest.py``); single
    names use the fundamentals call. Writes ``universe.duckdb.equity_metadata``.
  * :func:`read_metadata` — what ``run_weekly`` calls: a pure read of that table
    into ``{symbol: {sector, spread_bps, fractional_enabled}}``. No MCP, no network.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Mapping, Optional, Sequence

# ETF -> sector / category, derived from the DEFAULT_ETF_UNIVERSE grouping in
# data/pipelines/yahoo_ingest.py. SPDR Select-Sector funds map to their GICS
# sector; broad / intl / bond / factor / real-asset / inverse / leveraged / crypto
# ETFs map to a category label so the sector-concentration cap still bounds them.
ETF_SECTORS: dict[str, str] = {
    # broad market
    "VTI": "Broad Equity", "SPY": "Broad Equity", "QQQ": "Broad Equity",
    # SPDR Select Sector -> GICS sector
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
    "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate",
    "XLC": "Communication Services",
    # international
    "VXUS": "International", "EFA": "International", "EEM": "International",
    # bonds / duration / cash
    "BND": "Fixed Income", "AGG": "Fixed Income", "TLT": "Fixed Income",
    "IEF": "Fixed Income", "BIL": "Fixed Income", "SGOV": "Fixed Income",
    "SHY": "Fixed Income",
    # factors
    "MTUM": "Factor", "QUAL": "Factor", "USMV": "Factor",
    # real assets
    "GLD": "Real Assets", "DBC": "Real Assets",
    # inverse / leveraged
    "PSQ": "Inverse", "SH": "Inverse", "RWM": "Inverse",
    "QLD": "Leveraged", "TQQQ": "Leveraged", "QID": "Leveraged", "SQQQ": "Leveraged",
    # crypto-ETF
    "IBIT": "Crypto", "ETHA": "Crypto",
}


def etf_sector(symbol: str) -> Optional[str]:
    """The hardcoded sector/category for a known ETF, else ``None``."""
    return ETF_SECTORS.get(symbol.upper())


# --------------------------------------------------------------------------- #
# Schema + read (the deterministic JOIN side)
# --------------------------------------------------------------------------- #
def init_metadata_schema(con) -> None:
    """Create ``equity_metadata`` in universe.duckdb if absent."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS equity_metadata (
            symbol             VARCHAR PRIMARY KEY,
            run_ts             TIMESTAMP,
            sector             VARCHAR,
            spread_bps         DOUBLE,
            fractional_enabled BOOLEAN
        )
        """
    )


def read_metadata(con) -> dict:
    """Read ``equity_metadata`` -> ``{symbol: {sector, spread_bps, fractional_enabled}}``.

    Returns an empty dict if the table does not exist yet (so ``run_weekly`` runs
    fine before any refresh — the metadata gates then simply don't bind). Pure read.
    """
    try:
        rows = con.execute(
            "SELECT symbol, sector, spread_bps, fractional_enabled FROM equity_metadata"
        ).fetchall()
    except Exception:  # noqa: BLE001 — table absent before the first refresh
        return {}
    out: dict = {}
    for sym, sector, spread, frac in rows:
        out[sym] = {
            "sector": sector,
            "spread_bps": (None if spread is None else float(spread)),
            "fractional_enabled": (True if frac is None else bool(frac)),
        }
    return out


# --------------------------------------------------------------------------- #
# The slow-loop refresh (MCP injected, never imported)
# --------------------------------------------------------------------------- #
def _spread_bps_from_quote(quote: Mapping) -> Optional[float]:
    """Compute the bid/ask spread in bps from a quote dict (``None`` if unavailable)."""
    if not quote:
        return None
    try:
        bid = float(quote.get("bid_price") or quote.get("bid"))
        ask = float(quote.get("ask_price") or quote.get("ask"))
    except (TypeError, ValueError):
        return None
    mid = 0.5 * (bid + ask)
    if mid <= 0 or ask < bid:
        return None
    return (ask - bid) / mid * 1e4


def refresh_metadata(
    symbols: Sequence[str],
    con,
    *,
    quotes_fn: Optional[Callable] = None,
    fundamentals_fn: Optional[Callable] = None,
    tradability_fn: Optional[Callable] = None,
    run_ts: Optional[datetime] = None,
) -> int:
    """Refresh ``equity_metadata`` for ``symbols`` (the SLOW-LOOP MCP batch).

    ``quotes_fn(symbol) -> quote dict`` (for the spread), ``fundamentals_fn(symbol)
    -> {"sector": ...}`` (single-name sector), ``tradability_fn(symbol) ->
    {"fractional_tradable": bool}``. All three are INJECTED so the module never
    imports MCP and is fully unit-testable; a real operator script wires the
    Robinhood MCP tools. ETFs in :data:`ETF_SECTORS` take their sector from the map
    and skip the fundamentals call. Returns the number of rows written.

    A symbol whose calls fail is written with whatever was resolved (sector from
    the ETF map if applicable, else ``None``) — a missing field simply leaves that
    gate non-binding for the name, never crashes the batch.
    """
    init_metadata_schema(con)
    rt = run_ts or datetime.utcnow()
    n = 0
    for sym in symbols:
        sector = etf_sector(sym)
        if sector is None and fundamentals_fn is not None:
            try:
                f = fundamentals_fn(sym) or {}
                sector = f.get("sector") or f.get("sector_name")
            except Exception:  # noqa: BLE001 — one bad symbol must not kill the batch
                sector = None
        spread_bps = None
        if quotes_fn is not None:
            try:
                spread_bps = _spread_bps_from_quote(quotes_fn(sym) or {})
            except Exception:  # noqa: BLE001
                spread_bps = None
        fractional = True
        if tradability_fn is not None:
            try:
                t = tradability_fn(sym) or {}
                fractional = bool(
                    t.get("fractional_tradable", t.get("tradable", True))
                )
            except Exception:  # noqa: BLE001
                fractional = True
        con.execute(
            "INSERT OR REPLACE INTO equity_metadata VALUES (?, ?, ?, ?, ?)",
            [sym, rt, sector, spread_bps, fractional],
        )
        n += 1
    return n
