"""backtest/daily/validation.py — gate-validation utilities for the DAILY
BRACKETED-SWING portfolio (``backtest/daily/bracket_engine``).

WHY A SEPARATE MODULE
---------------------
``backtest/stats/{oos,walk_forward,multiple_testing}`` were built for the
*intraday* single-symbol R-multiple engine: ``walk_forward`` consumes a
``BacktestResult`` and reads intraday summary keys (``expectancy_R``,
``net_profit``). The daily bracket engine produces a :class:`BracketResult`
(a NAV curve + a closed-trade ledger of :class:`BracketTrade`), so the
promotion gate for a *daily swing* sleeve needs a parallel, bracket-aware
harness. This module is that harness — pure, deterministic, offline.

It deliberately REUSES the discipline primitives that are already engine-
agnostic:
  * ``backtest.stats.oos``          — IS/OOS split + the locked vault guard.
  * ``backtest.stats.multiple_testing`` — the rising PF bar + deflated Sharpe +
                                          the append-only hypothesis log.
  * ``backtest.stats.metrics``      — profit_factor / expectancy_r / win_rate /
                                      sharpe / max_drawdown (operate on the
                                      bracket ledger + NAV the same way).

What it adds is the bracket-specific glue:
  * :func:`pooled_trade_metrics`  — PF / expectancy-R / win-rate over a list of
    :class:`BracketTrade` (the gate's per-trade view).
  * :func:`returns_metrics`       — annualized return / vol / Sharpe (annual and
    per-period) / max-drawdown / skew / kurtosis over a daily-return series
    (the gate's curve view, and the higher moments the deflated Sharpe needs).
  * :func:`slice_result`          — restrict a full-span :class:`BracketResult`
    to a calendar window (trades by ENTRY date, returns by date) — the building
    block for a cold-start-free walk-forward and per-year breakdowns.
  * :func:`yearly_breakdown` / :func:`pool_windows` — temporal-stability views.
  * :func:`haircut_verdict`       — apply ``min_pf_threshold`` + ``deflated_
    sharpe`` to a pooled OOS result and return a structured pass/fail.

PURE / DETERMINISTIC / OFFLINE: no LLM, no MCP, no network, no wall-clock.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from backtest.stats.metrics import (
    max_drawdown,
    profit_factor,
    sharpe as _sharpe,
    sharpe_per_period,
)
from backtest.stats.multiple_testing import deflated_sharpe, min_pf_threshold

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Date coercion (mirrors backtest.daily.engine._as_date / oos._as_date)
# --------------------------------------------------------------------------- #
def _as_date(d):
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    raise TypeError(f"cannot coerce {d!r} to a date")


# --------------------------------------------------------------------------- #
# Per-trade (ledger) view — the gate's PF / expectancy / win-rate
# --------------------------------------------------------------------------- #
def pooled_trade_metrics(trades: Iterable) -> dict:
    """PF / expectancy-R / win-rate / counts over a list of :class:`BracketTrade`.

    Operates on whatever closed trades are passed in — a single run's ledger, or
    a POOLED ledger concatenated across walk-forward OOS folds (the honest
    out-of-sample number for promotion). ``profit_factor`` is reused verbatim
    from ``backtest.stats.metrics`` so the daily and intraday gates agree on the
    definition. Empty -> all-``nan`` with ``n_trades=0``.
    """
    trades = list(trades)
    if not trades:
        return {
            "n_trades": 0, "profit_factor": float("nan"),
            "expectancy_r": float("nan"), "win_rate": float("nan"),
            "avg_R": float("nan"), "gross_profit": 0.0, "gross_loss": 0.0,
        }
    pnls = np.asarray([float(t.pnl) for t in trades], dtype="float64")
    rs = np.asarray([float(t.r_multiple) for t in trades], dtype="float64")
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(-pnls[pnls < 0].sum())
    return {
        "n_trades": int(pnls.size),
        "profit_factor": profit_factor(pnls),
        "expectancy_r": float(rs.mean()) if rs.size else float("nan"),
        "win_rate": float((pnls > 0).mean()),
        "avg_R": float(rs.mean()) if rs.size else float("nan"),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


# --------------------------------------------------------------------------- #
# Curve view — annualized return / vol / Sharpe / drawdown / higher moments
# --------------------------------------------------------------------------- #
def _skew_kurt(arr: np.ndarray) -> tuple[float, float]:
    """Sample skewness and NON-excess kurtosis (normal -> 0, 3). scipy-free.

    Fewer than 3 finite observations, or zero dispersion -> (0.0, 3.0), the
    normal defaults the deflated Sharpe assumes (so it neither helps nor hurts).
    """
    a = arr[np.isfinite(arr)]
    if a.size < 3:
        return 0.0, 3.0
    mu = float(a.mean())
    sd = float(a.std(ddof=0))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0, 3.0
    z = (a - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # non-excess (normal == 3.0)
    return skew, kurt


def returns_metrics(
    daily_returns, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> dict:
    """Annualized return / vol / Sharpe (+ per-period) / maxDD / skew / kurt.

    ``daily_returns`` is a per-day simple-return series (e.g.
    ``BracketResult.daily_returns``, or a pooled/stitched OOS return stream). The
    annualized return is the geometric compounding of the stream scaled to a
    252-day year (NAV-free, so it works on a stitched series); ``max_drawdown``
    is taken on the implied ``cumprod(1+r)`` equity curve. ``sharpe_per_period``
    + ``skew`` + ``kurt`` are exactly what :func:`deflated_sharpe` consumes.
    """
    arr = np.asarray(list(daily_returns), dtype="float64")
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return {
            "n_obs": n, "ann_return": float("nan"), "ann_vol": float("nan"),
            "sharpe": float("nan"), "sharpe_per_period": float("nan"),
            "max_drawdown": float("nan"), "skew": 0.0, "kurt": 3.0,
        }
    curve = np.cumprod(1.0 + arr)
    years = n / float(periods_per_year)
    total_growth = float(curve[-1])
    ann_return = (total_growth ** (1.0 / years) - 1.0) if (years > 0 and total_growth > 0) else float("nan")
    ann_vol = float(np.std(arr, ddof=1) * np.sqrt(periods_per_year))
    skew, kurt = _skew_kurt(arr)
    return {
        "n_obs": n,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": _sharpe(arr, periods_per_year=periods_per_year),
        "sharpe_per_period": sharpe_per_period(arr),
        "max_drawdown": max_drawdown(pd.Series(curve)),
        "skew": skew,
        "kurt": kurt,
    }


# --------------------------------------------------------------------------- #
# Slicing a full-span BracketResult to a calendar window (cold-start-free WF)
# --------------------------------------------------------------------------- #
def slice_result(result, win_start, win_end) -> dict:
    """Restrict a full-span :class:`BracketResult` to ``[win_start, win_end]``.

    Returns ``{"trades": [...], "returns": Series}`` where trades are those whose
    ENTRY date falls inside the window (standard attribution — a trade belongs to
    the period it was opened in) and returns are the ``daily_returns`` rows dated
    inside the window. Because the source result is ONE continuous run with full
    warmup, every window is evaluated with no signal cold-start — the honest way
    to partition a daily strategy into folds/years.
    """
    s, e = _as_date(win_start), _as_date(win_end)
    trades = [
        t for t in result.trades
        if (s is None or _as_date(t.entry_date) >= s)
        and (e is None or _as_date(t.entry_date) <= e)
    ]
    rets = result.daily_returns
    if len(rets) and (s is not None or e is not None):
        idx_dates = [_as_date(d) for d in rets.index]
        mask = [
            (s is None or d >= s) and (e is None or d <= e) for d in idx_dates
        ]
        rets = rets[pd.Series(mask, index=rets.index).values]
    return {"trades": trades, "returns": rets}


def yearly_breakdown(result, start, end) -> list[dict]:
    """Per-calendar-year PF / n_trades / annualized-return for a full-span run.

    A temporal-stability view: for each year in ``[start, end]`` it slices the
    continuous run to that year and reports the year's trade PF + curve return.
    The dispersion across years tells you whether the edge is broad-based or
    concentrated in one or two windows (a single-window edge is fragile).
    """
    s, e = _as_date(start), _as_date(end)
    out: list[dict] = []
    for yr in range(s.year, e.year + 1):
        y0 = date(yr, 1, 1)
        y1 = date(yr, 12, 31)
        if y0 < s:
            y0 = s
        if y1 > e:
            y1 = e
        sl = slice_result(result, y0, y1)
        tm = pooled_trade_metrics(sl["trades"])
        rm = returns_metrics(sl["returns"])
        out.append({
            "year": yr,
            "n_trades": tm["n_trades"],
            "profit_factor": tm["profit_factor"],
            "expectancy_r": tm["expectancy_r"],
            "ann_return": rm["ann_return"],
        })
    return out


def pool_windows(result, windows: Sequence[tuple]) -> dict:
    """Pool the trades + stitch the returns of a run over several windows.

    ``windows`` is a sequence of ``(start, end)`` pairs (e.g. the OOS-era years).
    Returns ``{"trade_metrics": {...}, "returns_metrics": {...}, "returns":
    Series}`` over the union — the pooled out-of-sample number that the haircut
    is applied to. Overlapping windows are the caller's responsibility (years do
    not overlap).
    """
    pooled_trades: list = []
    parts: list[pd.Series] = []
    for w0, w1 in windows:
        sl = slice_result(result, w0, w1)
        pooled_trades.extend(sl["trades"])
        if len(sl["returns"]):
            parts.append(sl["returns"])
    stitched = pd.concat(parts) if parts else pd.Series(dtype="float64")
    return {
        "trade_metrics": pooled_trade_metrics(pooled_trades),
        "returns_metrics": returns_metrics(stitched),
        "returns": stitched,
        "n_pooled_trades": len(pooled_trades),
    }


# --------------------------------------------------------------------------- #
# The multiple-testing haircut verdict (PF bar + deflated Sharpe)
# --------------------------------------------------------------------------- #
def haircut_verdict(
    trade_metrics: dict,
    returns_metrics_: dict,
    n_trials: int,
    var_trials_sharpe: float = 1.0,
) -> dict:
    """Apply the rising PF bar (+ a correctly-scaled deflated Sharpe) to a result.

    Parameters
    ----------
    trade_metrics
        Output of :func:`pooled_trade_metrics` (needs ``profit_factor``).
    returns_metrics_
        Output of :func:`returns_metrics` (needs ``sharpe_per_period``,
        ``n_obs``, ``skew``, ``kurt``).
    n_trials
        The honest multiple-testing count (how many configs were tried to find
        this edge). Feeds BOTH ``min_pf_threshold`` and ``deflated_sharpe``.
    var_trials_sharpe
        Variance of the trial Sharpe ratios IN THE SAME (per-period) UNITS as
        ``sharpe_per_period``. This MUST be supplied for a daily strategy: the
        Bailey-LdP false-discovery benchmark ``E[max Sharpe]`` is computed in the
        units of the trial Sharpes, and a per-PERIOD (daily) Sharpe has variance
        of order ``1e-4`` across trials, NOT the textbook default of ``1.0``
        (which is an annualized-units assumption). Leaving it at ``1.0`` against a
        per-period Sharpe makes the benchmark ~100x too large and the DSR collapse
        to 0 — the artifact the intraday gate tolerates because it gates on PF,
        not DSR. Pass the empirical per-period trial variance to get a meaningful
        probability.

    Returns
    -------
    A dict with the PF bar, the observed PF, ``clears_pf`` (THE gate, matching the
    intraday convention); the deflated-Sharpe probability + ``clears_dsr``
    (advisory, >= 0.95); and ``clears`` == ``clears_pf`` (the PF haircut is the
    binary gate; DSR is reported as corroborating context, not a hard veto).
    """
    pf = float(trade_metrics.get("profit_factor", float("nan")))
    bar = min_pf_threshold(n_trials)
    clears_pf = bool(np.isfinite(pf) and pf >= bar)

    spp = float(returns_metrics_.get("sharpe_per_period", float("nan")))
    n_obs = int(returns_metrics_.get("n_obs", 0))
    skew = float(returns_metrics_.get("skew", 0.0))
    kurt = float(returns_metrics_.get("kurt", 3.0))
    dsr = (
        deflated_sharpe(spp, n_trials=n_trials, n_obs=n_obs, skew=skew, kurt=kurt,
                        var_trials_sharpe=var_trials_sharpe)
        if (np.isfinite(spp) and n_obs >= 2) else 0.0
    )
    clears_dsr = bool(dsr >= 0.95)

    return {
        "n_trials": int(n_trials),
        "pf_bar": bar,
        "profit_factor": pf,
        "clears_pf": clears_pf,
        "deflated_sharpe": dsr,
        "clears_dsr": clears_dsr,
        "var_trials_sharpe": float(var_trials_sharpe),
        "clears": clears_pf,
    }
