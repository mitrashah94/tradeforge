"""backtest/stats/metrics.py — headline performance metrics.

Pure, dependency-light functions (stdlib + numpy/pandas) that compute the
metrics MASTER_PLAN.md §5/§9 reports on: profit factor, expectancy (in R and in
dollars), win rate, Sharpe (from a daily-return series), and max drawdown.

Everything is computed from one of three inputs, and every public function
accepts whichever is natural:

  - a *list/array of per-trade pnls* (dollars) — the resampling unit for the
    bootstrap CIs in ``confidence.py``;
  - a *list/array of per-trade R-multiples* — for expectancy_r;
  - a :class:`~backtest.engine.result.BacktestResult` — convenience adapters
    that pull the trade ledger / equity curve out for you.

Conventions
-----------
* Profit factor = gross_profit / gross_loss (sum of winners / |sum of losers|).
  This is sizing-robust, matching ``BacktestResult.summary()``. With no losers
  PF is ``inf``; with no trades it is ``nan``.
* Sharpe is computed from a *daily* return series (see ``runner.daily_returns``)
  and ANNUALIZED with ``periods_per_year`` (252 trading days by default). It is
  a *non-excess* Sharpe (risk-free assumed 0) unless ``rf_per_period`` is given.
* Max drawdown is returned as a POSITIVE fraction (0.10 == a 10% drawdown).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Default annualization factor for a daily equity return series.
TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Input coercion helpers
# --------------------------------------------------------------------------- #
def _to_array(x) -> np.ndarray:
    """Coerce a list / Series / ndarray of numbers into a float ndarray."""
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype="float64")
    return arr


def trade_pnls(result) -> np.ndarray:
    """Extract the per-trade net pnl ($) array from a BacktestResult."""
    t = result.trades
    if t is None or len(t) == 0:
        return np.asarray([], dtype="float64")
    return t["pnl"].astype("float64").to_numpy()


def trade_r_multiples(result) -> np.ndarray:
    """Extract the per-trade R-multiple array from a BacktestResult."""
    t = result.trades
    if t is None or len(t) == 0:
        return np.asarray([], dtype="float64")
    return t["r_multiple"].astype("float64").to_numpy()


# --------------------------------------------------------------------------- #
# Profit factor
# --------------------------------------------------------------------------- #
def profit_factor(pnls) -> float:
    """Gross profit / gross loss from per-trade pnls.

    ``inf`` if there are winners but no losers; ``nan`` if there are no trades.
    """
    arr = _to_array(pnls)
    if arr.size == 0:
        return float("nan")
    gross_profit = float(arr[arr > 0].sum())
    gross_loss = float(-arr[arr < 0].sum())  # positive magnitude
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


# --------------------------------------------------------------------------- #
# Expectancy
# --------------------------------------------------------------------------- #
def expectancy_dollar(pnls) -> float:
    """Mean net pnl ($) per trade. ``nan`` with no trades."""
    arr = _to_array(pnls)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def expectancy_r(r_multiples) -> float:
    """Mean R-multiple per trade (expectancy in units of risk). ``nan`` empty."""
    arr = _to_array(r_multiples)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def win_rate(pnls) -> float:
    """Fraction of trades with pnl > 0. ``nan`` with no trades."""
    arr = _to_array(pnls)
    if arr.size == 0:
        return float("nan")
    return float((arr > 0).mean())


# --------------------------------------------------------------------------- #
# Sharpe (from a daily-return series)
# --------------------------------------------------------------------------- #
def sharpe(
    daily_returns,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf_per_period: float = 0.0,
    ddof: int = 1,
) -> float:
    """Annualized Sharpe ratio from a per-period (daily) return series.

    ``daily_returns`` are simple returns *per period* (e.g. daily). The Sharpe
    is ``mean(excess) / std(excess) * sqrt(periods_per_year)`` where
    ``excess = returns - rf_per_period``. Returns ``nan`` when there are fewer
    than 2 observations or zero dispersion.
    """
    arr = _to_array(daily_returns)
    if arr.size < 2:
        return float("nan")
    excess = arr - rf_per_period
    sd = float(np.std(excess, ddof=ddof))
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    mean = float(np.mean(excess))
    return (mean / sd) * np.sqrt(periods_per_year)


def sharpe_per_period(daily_returns, rf_per_period: float = 0.0, ddof: int = 1) -> float:
    """Un-annualized (per-period) Sharpe — the raw mean/std ratio.

    Useful for ``deflated_sharpe`` (Bailey & López de Prado), which works in the
    units of the underlying return observations, not annualized units.
    """
    arr = _to_array(daily_returns)
    if arr.size < 2:
        return float("nan")
    excess = arr - rf_per_period
    sd = float(np.std(excess, ddof=ddof))
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean(excess)) / sd


# --------------------------------------------------------------------------- #
# Max drawdown
# --------------------------------------------------------------------------- #
def max_drawdown(equity_curve) -> float:
    """Max peak-to-trough drawdown of an equity curve, as a POSITIVE fraction.

    ``equity_curve`` is a sequence/Series of equity *levels* (not returns).
    0.10 == a 10% drawdown. Returns 0.0 for an empty/flat curve.
    """
    eq = _to_array(equity_curve)
    if eq.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(eq)
    # Guard divide-by-zero on a zero/negative peak.
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_peak > 0, (eq - running_peak) / running_peak, 0.0)
    worst = float(dd.min()) if dd.size else 0.0
    return -worst if worst < 0 else 0.0


def max_drawdown_dollar(equity_curve) -> float:
    """Max peak-to-trough drawdown of an equity curve in dollars (positive)."""
    eq = _to_array(equity_curve)
    if eq.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(eq)
    dd = eq - running_peak
    worst = float(dd.min()) if dd.size else 0.0
    return -worst if worst < 0 else 0.0


# --------------------------------------------------------------------------- #
# One-shot bundle
# --------------------------------------------------------------------------- #
def compute_metrics(
    result=None,
    *,
    pnls: Sequence[float] | None = None,
    r_multiples: Sequence[float] | None = None,
    daily_rets=None,
    equity_curve=None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Compute the headline metric bundle from a BacktestResult or raw arrays.

    Pass either ``result`` (and we extract everything we can) and/or any of the
    explicit arrays, which override what is extracted from ``result``. Returns a
    plain dict with: n_trades, profit_factor, expectancy_dollar, expectancy_r,
    win_rate, sharpe, max_drawdown (fraction), net_profit.
    """
    if result is not None:
        if pnls is None:
            pnls = trade_pnls(result)
        if r_multiples is None:
            r_multiples = trade_r_multiples(result)
        if equity_curve is None:
            equity_curve = result.equity_curve

    pnls_arr = _to_array(pnls) if pnls is not None else np.asarray([], dtype="float64")
    r_arr = (
        _to_array(r_multiples)
        if r_multiples is not None
        else np.asarray([], dtype="float64")
    )

    out = {
        "n_trades": int(pnls_arr.size),
        "profit_factor": profit_factor(pnls_arr),
        "expectancy_dollar": expectancy_dollar(pnls_arr),
        "expectancy_r": expectancy_r(r_arr),
        "win_rate": win_rate(pnls_arr),
        "net_profit": float(pnls_arr.sum()) if pnls_arr.size else 0.0,
        "max_drawdown": (
            max_drawdown(equity_curve) if equity_curve is not None else 0.0
        ),
        "sharpe": (
            sharpe(daily_rets, periods_per_year=periods_per_year)
            if daily_rets is not None
            else float("nan")
        ),
    }
    return out
