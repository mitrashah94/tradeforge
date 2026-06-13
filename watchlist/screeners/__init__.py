"""Screener package: stocks, crypto, and unusual-volume universe screeners (MASTER_PLAN.md §4).

All screeners are pure functions over caller-supplied OHLCV bars (read from
``market.duckdb``); none touch the network, so the package is deterministic and
offline-testable.
"""

from watchlist.screeners.common import (  # noqa: F401
    SymbolStats,
    daily_dollar_volume,
    daily_volume,
    passes_thresholds,
    summarize_symbol,
)
from watchlist.screeners.crypto import screen_crypto  # noqa: F401
from watchlist.screeners.stocks import screen_stocks  # noqa: F401
from watchlist.screeners.unusual_volume import (  # noqa: F401
    VolumeFlag,
    relative_volume,
    unusual_volume_flags,
)

__all__ = [
    "SymbolStats",
    "VolumeFlag",
    "daily_dollar_volume",
    "daily_volume",
    "passes_thresholds",
    "summarize_symbol",
    "screen_crypto",
    "screen_stocks",
    "relative_volume",
    "unusual_volume_flags",
]
