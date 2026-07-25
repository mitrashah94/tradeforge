"""backtest/daily — the daily-NAV backtester for the rotation / mean-reversion
sleeves (MASTER_PLAN §5; the day-frequency complement to ``backtest/engine``).

Public surface (the Phase-2 contract):

  * :class:`~backtest.daily.engine.DailyStrategy` — the point-in-time, long-only
    ``target_weights(asof_date, history) -> {symbol: fraction}`` interface the
    rotation/mean-reversion strategies implement.
  * :class:`~backtest.daily.engine.DailyHistory` — the no-lookahead price/return
    access helper handed to the strategy each decision day.
  * :func:`~backtest.daily.engine.run_daily` — step the calendar, rebalance on a
    M/W schedule, charge costs + accrue the short-term tax reserve, return a
    :class:`~backtest.daily.result.DailyResult` (NAV curve + summary).
  * :func:`~backtest.daily.engine.run_param_grid` — robustness sweep (dispersion
    of CAGR/Sharpe/maxDD across a parameter grid — the GEM-fragility guard).
  * :func:`~backtest.daily.engine.load_daily_bars` — load the wide ADJUSTED
    ``timeframe='1d'`` price panel from the DuckDB ``bars`` table.
  * :class:`~backtest.daily.result.DailyResult` — NAV curve, after-tax NAV, daily
    fractional-returns Series, and ``summary()``.
"""

from backtest.daily.engine import (
    DEFAULT_SHORT_TERM_TAX_RATE,
    DailyHistory,
    DailyStrategy,
    load_daily_bars,
    run_daily,
    run_param_grid,
)
from backtest.daily.result import DailyResult

__all__ = [
    "DailyStrategy",
    "DailyHistory",
    "DailyResult",
    "run_daily",
    "run_param_grid",
    "load_daily_bars",
    "DEFAULT_SHORT_TERM_TAX_RATE",
]
