"""watchlist/screeners/crypto.py — the 24/7 crypto liquidity screener.

The crypto analogue of :mod:`watchlist.screeners.stocks`, with two differences
that matter for an honest small-account picture (MASTER_PLAN.md §3/§4):

  1. Sessions are UTC calendar days (crypto has no RTH), so per-day dollar-volume
     is summed over the full 24h day — handled by passing ``asset_class="crypto"``
     to the shared liquidity primitives.
  2. Robinhood crypto spreads are wide (their revenue), so this screen applies an
     EXTRA spread-cost honesty gate on top of the tier thresholds: a candidate is
     only admissible if its average per-session dollar-volume comfortably clears
     the tier minimum (we keep the same threshold semantics as equities but flag
     that the real edge must clear a wider cost — the cost model in
     ``backtest/costs.yaml`` carries the 25 bps half-spread used downstream).

Pure function over caller-supplied bars; no network, deterministic offline.
"""

from __future__ import annotations

import pandas as pd

from watchlist.screeners.common import (
    SymbolStats,
    passes_thresholds,
    summarize_symbol,
)


def screen_crypto(
    bars_by_symbol: dict[str, pd.DataFrame],
    min_dollar_volume: float,
    min_price: float,
) -> list[SymbolStats]:
    """Return the crypto symbols that pass the tier thresholds, richest first.

    Same contract as :func:`watchlist.screeners.stocks.screen_stocks` but the
    liquidity stats are computed on the UTC-day calendar. Sorted by descending
    average dollar-volume.

    ``min_price`` is still applied (it screens out fractional-cent tokens), but
    note crypto prices span many orders of magnitude, so the binding gate is
    almost always the dollar-volume one.
    """
    survivors: list[SymbolStats] = []
    for symbol, bars in bars_by_symbol.items():
        stats = summarize_symbol(symbol, bars, asset_class="crypto")
        if stats is None:
            continue
        if passes_thresholds(stats, min_dollar_volume, min_price):
            survivors.append(stats)
    survivors.sort(key=lambda s: s.avg_dollar_volume, reverse=True)
    return survivors
