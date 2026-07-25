"""watchlist/screeners/stocks.py — the equity liquidity screener.

Filters a candidate equity universe down to the names that clear a tier's
liquidity/price thresholds from ``criteria.yaml`` (MASTER_PLAN.md §4). A pure
function over caller-supplied bars: it never touches the network, so the same
call is deterministic offline against whatever is in ``market.duckdb``.

Liquidity reality (§3/§4): on a small account, a name you cannot get in and out
of cheaply silently eats the trade's expectancy. The screen therefore gates on
*average per-session dollar-volume* (typical dollars traded per day) and a
minimum LAST price (penny names have wide spreads and are excluded).
"""

from __future__ import annotations

import pandas as pd

from watchlist.screeners.common import (
    SymbolStats,
    passes_thresholds,
    summarize_symbol,
)


def screen_stocks(
    bars_by_symbol: dict[str, pd.DataFrame],
    min_dollar_volume: float,
    min_price: float,
) -> list[SymbolStats]:
    """Return the equity symbols that pass the tier thresholds, richest first.

    Parameters
    ----------
    bars_by_symbol
        ``{symbol -> OHLCV DataFrame}`` (ts_utc, open, high, low, close,
        volume), already loaded from ``market.duckdb`` by the caller.
    min_dollar_volume, min_price
        The tier's gates (e.g. CORE: 50e6 / $5). A symbol clears if its average
        per-session dollar-volume ≥ ``min_dollar_volume`` AND its last price ≥
        ``min_price``.

    Returns the surviving :class:`SymbolStats`, sorted by descending average
    dollar-volume (most liquid first) so tiering can take the top-N directly.
    """
    survivors: list[SymbolStats] = []
    for symbol, bars in bars_by_symbol.items():
        stats = summarize_symbol(symbol, bars, asset_class="equity")
        if stats is None:
            continue
        if passes_thresholds(stats, min_dollar_volume, min_price):
            survivors.append(stats)
    survivors.sort(key=lambda s: s.avg_dollar_volume, reverse=True)
    return survivors
