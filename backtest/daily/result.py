"""backtest/daily/result.py — the daily NAV curve + its summary metrics.

The sibling of ``backtest/engine/result.BacktestResult``, but for the
*daily-NAV* world (rotation / mean-reversion sleeves that rebalance to target
weights on a M/W schedule), not the intraday round-trip world. Where the engine
result is built from a per-trade ledger, this one is built from the **daily
equity (NAV) curve** the daily engine steps out, plus a few accrued totals
(transaction costs, the realized-gain tax reserve, turnover).

What it holds
-------------
``nav``            pandas Series of end-of-day equity (gross of the tax reserve),
                  indexed by ``datetime.date`` on the trading calendar.
``after_tax_nav`` the SAME curve minus the tax reserve AS IT ACCRUED (the
                  running reserve at each day's close, not the final total) — the
                  CLAUDE.md "after-tax equity is a first-class metric" line made
                  literal (compounding uses after-tax dollars). The curve steps
                  down only on a gain-realizing rebalance, never retroactively.
``daily_returns`` the per-day fractional NAV return (``nav.pct_change``), the
                  series ``backtest/portfolio.py`` consumes for the blend/Sharpe
                  math (reuse with ``portfolio.blend_returns``).

``summary()`` returns the headline daily-world metrics:
    CAGR, total_return, max_drawdown, ann_vol, sharpe, sortino,
    turnover_per_year, after_tax_CAGR, after_tax_total_return, n_days, years.

Metric conventions (shared with ``backtest/stats/metrics.py`` where they fit)
----------------------------------------------------------------------------
* max_drawdown is a POSITIVE fraction (0.10 == a 10% drawdown), computed by
  :func:`backtest.stats.metrics.max_drawdown` on the NAV levels.
* sharpe is annualized from the daily fractional-return series via
  :func:`backtest.stats.metrics.sharpe` (rf = 0, 252 periods/yr).
* sortino is the same shape but divides by the *downside* deviation (returns
  below 0 only); ``inf`` when there is no downside, ``nan`` with <2 obs.
* CAGR / vol annualize on a 252-trading-day year. CAGR uses the actual elapsed
  trading days (``n_days / 252`` years), so a half-year sample annualizes
  honestly rather than assuming a full year.
* turnover_per_year is the summed one-way rebalance turnover (fraction of equity
  traded) scaled to a per-year rate — the friction dial the cost model bills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.stats.metrics import (
    TRADING_DAYS_PER_YEAR,
    max_drawdown,
    sharpe as _sharpe,
)


def _sortino(daily_returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sortino: mean / downside-deviation * sqrt(periods_per_year).

    Downside deviation is the RMS of the negative returns only (returns >= 0 do
    not contribute to the denominator). ``inf`` when the mean is positive and
    there is no downside, ``nan`` with fewer than 2 observations or a zero/non-
    positive mean with no downside.
    """
    arr = np.asarray(list(daily_returns), dtype="float64")
    if arr.size < 2:
        return float("nan")
    mean = float(arr.mean())
    downside = arr[arr < 0.0]
    if downside.size == 0:
        # No losing day: undefined denominator. Report inf for a positive mean,
        # nan otherwise (flat/negative with no variance is not meaningful).
        return float("inf") if mean > 0 else float("nan")
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if dd == 0.0 or not np.isfinite(dd):
        return float("nan")
    return (mean / dd) * np.sqrt(periods_per_year)


@dataclass
class DailyResult:
    """Holds the daily NAV curve (+ accrued cost/tax/turnover) and its summary.

    Built by :func:`backtest.daily.engine.run_daily`. ``nav`` is the gross
    end-of-day equity curve; ``after_tax_nav`` subtracts the running short-term
    tax reserve. ``rebalance_turnover`` is the per-rebalance one-way turnover
    (fraction of equity traded) keyed by rebalance date — its sum drives the
    cost model and ``turnover_per_year``.
    """

    nav: pd.Series                       # gross end-of-day equity, indexed by date
    after_tax_nav: pd.Series             # nav minus running tax reserve
    initial_equity: float
    total_costs: float                   # summed transaction cost ($) charged
    tax_reserve: float                   # final accrued short-term tax reserve ($)
    realized_gains: float                # cumulative realized short-term gains ($)
    rebalance_turnover: pd.Series        # one-way turnover per rebalance date
    short_term_tax_rate: float
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    # ------------------------------------------------------------- derived series
    @property
    def daily_returns(self) -> pd.Series:
        """Per-day fractional NAV return (``nav.pct_change`` minus the NaN head).

        This is the series ``backtest/portfolio.py`` consumes (blend/Sharpe). The
        first day has no prior NAV so it is dropped; the index stays
        ``datetime.date``. Empty/one-point NAV -> empty float Series.
        """
        if self.nav is None or len(self.nav) < 2:
            return pd.Series(dtype="float64", name="daily_return")
        out = self.nav.astype("float64").pct_change().iloc[1:]
        out.name = "daily_return"
        out.index.name = "date"
        return out

    @property
    def after_tax_daily_returns(self) -> pd.Series:
        """Per-day fractional return of the AFTER-TAX NAV curve."""
        if self.after_tax_nav is None or len(self.after_tax_nav) < 2:
            return pd.Series(dtype="float64", name="after_tax_daily_return")
        out = self.after_tax_nav.astype("float64").pct_change().iloc[1:]
        out.name = "after_tax_daily_return"
        out.index.name = "date"
        return out

    # ------------------------------------------------------------- helpers
    def _years(self) -> float:
        """Elapsed time in years, measured in TRADING days (n_days / 252)."""
        n = int(len(self.nav)) if self.nav is not None else 0
        # n NAV points span (n-1) day-steps; annualize on the step count.
        steps = max(n - 1, 0)
        return steps / float(self.periods_per_year) if steps else 0.0

    @staticmethod
    def _cagr(curve: pd.Series, years: float) -> float:
        """Compound annual growth rate of an equity curve over ``years``.

        ``nan`` when there is no elapsed time or a non-positive start/end. For a
        sub-year sample this still annualizes honestly off the fractional year.
        """
        if curve is None or len(curve) < 2 or years <= 0:
            return float("nan")
        start = float(curve.iloc[0])
        end = float(curve.iloc[-1])
        if start <= 0 or end <= 0:
            return float("nan")
        return (end / start) ** (1.0 / years) - 1.0

    @staticmethod
    def _total_return(curve: pd.Series) -> float:
        """End/start - 1 of an equity curve. ``nan`` if degenerate."""
        if curve is None or len(curve) < 1:
            return float("nan")
        start = float(curve.iloc[0])
        if start <= 0:
            return float("nan")
        return float(curve.iloc[-1]) / start - 1.0

    def turnover_per_year(self) -> float:
        """Summed one-way rebalance turnover scaled to a per-year rate.

        ``rebalance_turnover`` is a fraction-of-equity-traded per rebalance; its
        sum over the sample, divided by the elapsed years, is the annual turnover
        the friction budget pays for. ``0.0`` when nothing traded; ``nan`` with
        no elapsed time.
        """
        if self.rebalance_turnover is None or len(self.rebalance_turnover) == 0:
            return 0.0
        years = self._years()
        if years <= 0:
            return float("nan")
        return float(self.rebalance_turnover.astype("float64").sum()) / years

    # ------------------------------------------------------------- summary
    def summary(self) -> dict:
        """Return the headline daily-world metrics as a plain dict."""
        years = self._years()
        rets = self.daily_returns
        at_rets = self.after_tax_daily_returns

        ann_vol = (
            float(np.std(rets.to_numpy(), ddof=1) * np.sqrt(self.periods_per_year))
            if len(rets) >= 2
            else float("nan")
        )

        return {
            "n_days": int(len(self.nav)) if self.nav is not None else 0,
            "years": years,
            "total_return": self._total_return(self.nav),
            "CAGR": self._cagr(self.nav, years),
            "max_drawdown": max_drawdown(self.nav) if self.nav is not None else 0.0,
            "ann_vol": ann_vol,
            "sharpe": _sharpe(rets, periods_per_year=self.periods_per_year),
            "sortino": _sortino(rets, periods_per_year=self.periods_per_year),
            "turnover_per_year": self.turnover_per_year(),
            "total_costs": float(self.total_costs),
            "realized_gains": float(self.realized_gains),
            "tax_reserve": float(self.tax_reserve),
            "short_term_tax_rate": float(self.short_term_tax_rate),
            "after_tax_total_return": self._total_return(self.after_tax_nav),
            "after_tax_CAGR": self._cagr(self.after_tax_nav, years),
        }


# --------------------------------------------------------------------------- #
# BracketResult — the ACTIVELY-MANAGED bracketed-swing portfolio result.
# --------------------------------------------------------------------------- #
@dataclass
class BracketTrade:
    """One CLOSED bracketed-swing round trip (entry -> partials/exits -> flat).

    A single logical position can scale out in pieces (a TP1 partial, then a
    trailed-runner exit), so ``pnl`` / ``r_multiple`` are aggregated over ALL
    fills of that position and ``shares`` is the qty at entry. ``r_multiple`` is
    the SIZE-WEIGHTED realized R against the position's INITIAL risk
    (``entry - initial_stop`` per share), so a half-scaled +1.5R and a runner
    stopped at breakeven net the right blended R. This mirrors the intraday
    engine's ``TradeRecord`` R bookkeeping for the partial/runner path.
    """

    symbol: str
    entry_date: object               # datetime.date the position opened
    exit_date: object                # datetime.date the LAST piece closed
    entry_price: float
    avg_exit_price: float            # size-weighted exit over all pieces
    shares: float                    # qty at entry (initial size)
    initial_stop: float
    pnl: float                       # net $ over all pieces (after costs)
    gross_pnl: float                 # gross $ over all pieces (before costs)
    costs: float                     # total transaction cost ($) on this trade
    r_multiple: float                # size-weighted realized R (initial risk)
    bars_held: int                   # trading days from entry to final exit
    exit_reason: str                 # reason of the FINAL exit piece


@dataclass
class BracketResult:
    """Daily NAV curve + the closed-trade ledger for the bracket portfolio.

    Built by :func:`backtest.daily.bracket_engine.run_bracket_portfolio`. The
    NAV / after-tax / cost / tax / turnover machinery is shared verbatim with
    :class:`DailyResult` (this engine marks the SAME way), but a bracketed-swing
    book is a *trade machine*, so this result also carries a closed-trade ledger
    and an ``exposure`` series (the fraction of equity invested each day) to
    report the trade-level metrics the operator asked for: win_rate, avg_R,
    n_trades, trades_per_year, avg_exposure.
    """

    nav: pd.Series                       # gross end-of-day equity, indexed by date
    after_tax_nav: pd.Series             # nav minus running tax reserve
    initial_equity: float
    total_costs: float                   # summed transaction cost ($) charged
    tax_reserve: float                   # final accrued short-term tax reserve ($)
    realized_gains: float                # cumulative realized short-term gains ($)
    rebalance_turnover: pd.Series        # one-way turnover per trading date
    short_term_tax_rate: float
    trades: list = field(default_factory=list)   # list[BracketTrade], closed
    exposure: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    periods_per_year: int = TRADING_DAYS_PER_YEAR

    # ---- NAV-curve metric machinery (identical convention to DailyResult) -----
    # This engine marks the NAV the same way DailyResult's does, so the derived
    # series and CAGR/turnover math are the same; they are re-expressed here as
    # thin forwards to the shared, already-tested DailyResult implementations so
    # there is ONE source of truth for the curve math.
    @property
    def daily_returns(self) -> pd.Series:
        """Per-day fractional NAV return (see :meth:`DailyResult.daily_returns`)."""
        return DailyResult.daily_returns.fget(self)

    @property
    def after_tax_daily_returns(self) -> pd.Series:
        """Per-day fractional AFTER-TAX NAV return."""
        return DailyResult.after_tax_daily_returns.fget(self)

    def _years(self) -> float:
        return DailyResult._years(self)

    @staticmethod
    def _cagr(curve: pd.Series, years: float) -> float:
        return DailyResult._cagr(curve, years)

    @staticmethod
    def _total_return(curve: pd.Series) -> float:
        return DailyResult._total_return(curve)

    def turnover_per_year(self) -> float:
        return DailyResult.turnover_per_year(self)

    # ----------------------------------------------------------- trade metrics
    def _trade_arrays(self):
        """Return (pnls, r_multiples) ndarrays over the closed-trade ledger."""
        if not self.trades:
            return (np.asarray([], dtype="float64"), np.asarray([], dtype="float64"))
        pnls = np.asarray([t.pnl for t in self.trades], dtype="float64")
        rs = np.asarray([t.r_multiple for t in self.trades], dtype="float64")
        return pnls, rs

    def avg_exposure(self) -> float:
        """Mean fraction of equity invested across the trading calendar.

        ``0.0`` with no exposure series; ignores non-finite marks.
        """
        if self.exposure is None or len(self.exposure) == 0:
            return 0.0
        arr = self.exposure.to_numpy(dtype="float64")
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else 0.0

    # ----------------------------------------------------------- summary
    def summary(self) -> dict:
        """Return the headline bracket-portfolio metrics as a plain dict.

        Combines the NAV-curve metrics (CAGR, after_tax_CAGR, max_drawdown,
        sharpe, ann_vol) with the closed-trade ledger metrics (win_rate, avg_R,
        n_trades, trades_per_year) and the average daily exposure / turnover.
        """
        years = self._years()
        rets = self.daily_returns

        ann_vol = (
            float(np.std(rets.to_numpy(), ddof=1) * np.sqrt(self.periods_per_year))
            if len(rets) >= 2
            else float("nan")
        )

        pnls, rs = self._trade_arrays()
        n_trades = int(pnls.size)
        win_rate = float((pnls > 0).mean()) if n_trades else float("nan")
        avg_R = float(rs.mean()) if rs.size else float("nan")
        trades_per_year = (n_trades / years) if years > 0 else float("nan")

        return {
            "n_days": int(len(self.nav)) if self.nav is not None else 0,
            "years": years,
            "total_return": self._total_return(self.nav),
            "CAGR": self._cagr(self.nav, years),
            "after_tax_CAGR": self._cagr(self.after_tax_nav, years),
            "after_tax_total_return": self._total_return(self.after_tax_nav),
            "max_drawdown": max_drawdown(self.nav) if self.nav is not None else 0.0,
            "ann_vol": ann_vol,
            "sharpe": _sharpe(rets, periods_per_year=self.periods_per_year),
            "win_rate": win_rate,
            "avg_R": avg_R,
            "n_trades": n_trades,
            "trades_per_year": trades_per_year,
            "avg_exposure": self.avg_exposure(),
            "turnover_per_year": self.turnover_per_year(),
            "total_costs": float(self.total_costs),
            "realized_gains": float(self.realized_gains),
            "tax_reserve": float(self.tax_reserve),
            "short_term_tax_rate": float(self.short_term_tax_rate),
        }
