"""Watchlist package: universe selection, screeners, and per-strategy level-respect scoring (MASTER_PLAN.md §4).

The watchlist-curator agent is DEFERRED and starts as a weekly script
(:func:`watchlist.weekly.run_weekly`): it builds the candidate universe from the
screeners, scores each symbol's per-strategy level respect via a ~90-day replay
mini-backtest, assigns CORE/ACTIVE/SCOUT tiers gated by that score, runs a
correlation check so CORE stays diversified, persists to ``universe.duckdb``, and
emits WATCHLIST_UPDATED / SYMBOL_PROMOTED / SYMBOL_DEMOTED events.
"""

from watchlist.level_respect import (  # noqa: F401
    LevelRespectScore,
    level_respect_score,
    score_from_arrays,
)
from watchlist.weekly import (  # noqa: F401
    SymbolEntry,
    WatchlistResult,
    run_weekly,
)

__all__ = [
    "LevelRespectScore",
    "level_respect_score",
    "score_from_arrays",
    "SymbolEntry",
    "WatchlistResult",
    "run_weekly",
]
