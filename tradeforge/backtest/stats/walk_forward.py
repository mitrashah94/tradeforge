"""backtest/stats/walk_forward.py — rolling IS/OOS walk-forward folds.

MASTER_PLAN.md §5/§6: an edge must *"hold OOS"* and survive scheduled
walk-forward (refits within pre-registered sweep ranges, each OOS-checked). This
module provides the rolling-window machinery: it carves ``[start, end]`` into a
sequence of folds, each an in-sample window of ``is_months`` immediately
followed by an out-of-sample window of ``oos_months``, then advances by
``oos_months`` (a standard *rolling* / non-anchored walk-forward).

For each fold it calls a user-supplied ``run_fn(start, end) -> BacktestResult``
on the OOS window and records that window's metrics in a :class:`FoldResult`.
The caller owns what ``run_fn`` does (e.g. it may fit on the IS window first,
then evaluate OOS) — this module only manages the window calendar and the OOS
metric collection, so it stays independent of any particular strategy or fit
procedure.

Aggregation helpers (:func:`aggregate_folds`) pool the OOS trades across folds to
report a single honest out-of-sample profit factor / expectancy — the number
that matters for promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from backtest.engine.result import BacktestResult
from backtest.stats.metrics import (
    expectancy_dollar,
    expectancy_r,
    profit_factor,
    win_rate,
)


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    raise TypeError(f"cannot coerce {d!r} to a date")


def _add_months(d: date, months: int) -> date:
    """Add ``months`` calendar months to ``d`` (clamping day-of-month)."""
    m = d.month - 1 + months
    year = d.year + m // 12
    month = m % 12 + 1
    # Clamp the day to the last valid day of the target month.
    next_month_first = date(year + (month // 12), (month % 12) + 1, 1)
    last_day = (next_month_first - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day))


@dataclass
class FoldResult:
    """One walk-forward fold's IS window, OOS window, and OOS metrics."""

    fold: int
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    n_trades: int
    profit_factor: float
    expectancy_r: float
    expectancy_dollar: float
    win_rate: float
    net_profit: float
    result: BacktestResult | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "fold": self.fold,
            "is_start": self.is_start.isoformat(),
            "is_end": self.is_end.isoformat(),
            "oos_start": self.oos_start.isoformat(),
            "oos_end": self.oos_end.isoformat(),
            "n_trades": self.n_trades,
            "profit_factor": self.profit_factor,
            "expectancy_r": self.expectancy_r,
            "expectancy_dollar": self.expectancy_dollar,
            "win_rate": self.win_rate,
            "net_profit": self.net_profit,
        }


def fold_windows(
    start, end, is_months: int = 6, oos_months: int = 1
) -> list[tuple[date, date, date, date]]:
    """Enumerate ``(is_start, is_end, oos_start, oos_end)`` rolling fold windows.

    Each fold: an IS window of ``is_months``, then an OOS window of
    ``oos_months`` immediately after it. The next fold starts ``oos_months``
    later (rolling). The last fold whose OOS window would exceed ``end`` is
    dropped (every returned OOS window is fully inside ``[start, end]``).
    """
    s = _as_date(start)
    e = _as_date(end)
    windows: list[tuple[date, date, date, date]] = []
    is_start = s
    while True:
        oos_start = _add_months(is_start, is_months)
        is_end = oos_start - timedelta(days=1)
        oos_end = _add_months(oos_start, oos_months) - timedelta(days=1)
        if oos_end > e:
            break
        windows.append((is_start, is_end, oos_start, oos_end))
        is_start = _add_months(is_start, oos_months)
    return windows


def walk_forward(
    run_fn: Callable[[object, object], BacktestResult],
    start,
    end,
    is_months: int = 6,
    oos_months: int = 1,
    keep_results: bool = False,
) -> list[FoldResult]:
    """Run a rolling IS/OOS walk-forward and return per-fold OOS metrics.

    Parameters
    ----------
    run_fn
        ``run_fn(start, end) -> BacktestResult`` — called once per fold on the
        OOS window (``run_fn(oos_start, oos_end)``). If ``run_fn`` needs to fit
        on the IS window first, it should close over that logic; this driver only
        supplies windows and collects OOS metrics. Dates are passed as
        ``datetime.date``.
    start, end
        Overall span to walk across.
    is_months, oos_months
        In-sample and out-of-sample window lengths (calendar months).
    keep_results
        If True, retain each fold's full ``BacktestResult`` on the FoldResult
        (off by default to keep memory light when there are many folds).

    Returns
    -------
    ``list[FoldResult]`` — one per fold, in chronological order. Empty if the
    span is too short to fit even one full IS+OOS fold.
    """
    results: list[FoldResult] = []
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(
        fold_windows(start, end, is_months=is_months, oos_months=oos_months)
    ):
        res = run_fn(oos_s, oos_e)
        s = res.summary()
        results.append(
            FoldResult(
                fold=i,
                is_start=is_s,
                is_end=is_e,
                oos_start=oos_s,
                oos_end=oos_e,
                n_trades=int(s["n_trades"]),
                profit_factor=float(s["profit_factor"]),
                expectancy_r=float(s["expectancy_R"]),
                expectancy_dollar=float(s["expectancy_dollar"]),
                win_rate=float(s["win_rate"]),
                net_profit=float(s["net_profit"]),
                result=res if keep_results else None,
            )
        )
    return results


def aggregate_folds(folds: list[FoldResult]) -> dict:
    """Pool the OOS folds into one honest out-of-sample metric bundle.

    Concatenates every fold's OOS trade ledger (requires the folds to have been
    run with ``keep_results=True``) and recomputes PF / expectancy / win rate on
    the pooled trades — the single OOS number that matters for promotion. If the
    results were not kept, falls back to trade-count-weighted means of the
    per-fold metrics.
    """
    if not folds:
        return {
            "n_folds": 0,
            "n_trades": 0,
            "profit_factor": float("nan"),
            "expectancy_r": float("nan"),
            "expectancy_dollar": float("nan"),
            "win_rate": float("nan"),
            "net_profit": 0.0,
        }

    have_results = all(f.result is not None for f in folds)
    if have_results:
        frames = [f.result.trades for f in folds if len(f.result.trades) > 0]
        if frames:
            pooled = pd.concat(frames, ignore_index=True)
            pnl = pooled["pnl"].astype("float64")
            r = pooled["r_multiple"].astype("float64")
            return {
                "n_folds": len(folds),
                "n_trades": int(len(pooled)),
                "profit_factor": profit_factor(pnl),
                "expectancy_r": expectancy_r(r),
                "expectancy_dollar": expectancy_dollar(pnl),
                "win_rate": win_rate(pnl),
                "net_profit": float(pnl.sum()),
            }

    # Fallback: trade-weighted aggregate of per-fold summaries.
    total_trades = sum(f.n_trades for f in folds)
    net = sum(f.net_profit for f in folds)
    if total_trades == 0:
        return {
            "n_folds": len(folds),
            "n_trades": 0,
            "profit_factor": float("nan"),
            "expectancy_r": float("nan"),
            "expectancy_dollar": float("nan"),
            "win_rate": float("nan"),
            "net_profit": net,
        }

    def _wmean(attr):
        num = sum(getattr(f, attr) * f.n_trades for f in folds if f.n_trades)
        return num / total_trades

    return {
        "n_folds": len(folds),
        "n_trades": total_trades,
        "profit_factor": _wmean("profit_factor"),
        "expectancy_r": _wmean("expectancy_r"),
        "expectancy_dollar": net / total_trades,
        "win_rate": _wmean("win_rate"),
        "net_profit": net,
    }
