"""backtest/daily/engine.py — the daily-NAV backtester + the DAILY STRATEGY
INTERFACE the rotation / mean-reversion sleeves plug into (Phase 2 builds here).

WHY A SEPARATE DAILY ENGINE (not the intraday ``backtest/engine``)
------------------------------------------------------------------
The intraday ``BacktestEngine`` is a path-dependent *single-symbol* round-trip
state machine (break -> retest -> OCO bracket -> EOD-flat). The rotation and
mean-reversion sleeves are a different animal: a **cross-sectional, long-only,
daily-rebalanced weight allocator** over the whole universe. Its natural output
is not a trade ledger but a **target-weight vector** ``{symbol: fraction}`` that
the engine marks-to-market every day and rebalances to on a M/W schedule. So
this engine steps the trading calendar day by day, holds positions through, and
the "strategy" only ever answers ONE question:

    given everything knowable AS OF today's close, what fraction of equity do I
    want in each symbol tomorrow?

THE DAILY STRATEGY INTERFACE  (the Phase-2 contract — implement against THIS)
============================================================================
A daily strategy is any object with a ``target_weights`` method::

    class DailyStrategy(Protocol):
        def target_weights(
            self, asof_date: datetime.date, history: "DailyHistory"
        ) -> dict[str, float]:
            ...

CONTRACT (every rule is enforced / relied on by the engine):

  * ``asof_date`` is the CURRENT trading date being decided. The returned
    weights are applied at ``asof_date``'s close (the rebalance fills at the
    close of the decision day — see "Fill timing" below).
  * ``history`` is a :class:`DailyHistory` exposing **point-in-time ADJUSTED**
    daily bars (``timeframe='1d'``) for the universe, sliced to dates
    ``<= asof_date``. There is **NO LOOKAHEAD**: the strategy can never see a
    bar dated after ``asof_date``. ``history.prices()`` /
    ``history.returns()`` return frames already clipped to ``<= asof_date``;
    asking for a future date is impossible by construction.
  * The returned dict maps ``symbol -> weight`` where each weight is a FRACTION
    of current equity. Weights are **LONG-ONLY** (each ``>= 0``) and must sum to
    ``<= 1`` (the remainder is held as CASH). A short / negative weight is a
    contract violation and is rejected by the engine. Inverse exposure is
    expressed as a POSITIVE weight on an inverse ETF ticker (no shorting) — the
    universe must include that ticker for the engine to price it.
  * Symbols omitted from the dict (or given weight 0) are flat. Symbols the
    engine has no price for ``asof_date`` are dropped with their weight forfeited
    to cash (a missing price cannot be traded).

HISTORY ACCESS HELPER (no-lookahead, point-in-time)
---------------------------------------------------
:class:`DailyHistory` wraps the full universe price panel ONCE and hands the
strategy cheap, already-sliced views:

    history.prices(symbols=None, lookback=None) -> pd.DataFrame
        Wide ADJUSTED-close frame, index = trading dates ``<= asof_date``,
        columns = symbols. ``lookback`` keeps only the last N rows.
    history.returns(symbols=None, lookback=None) -> pd.DataFrame
        Simple daily returns of that price frame (``pct_change`` dropna head).
    history.asof_prices(symbols=None) -> pd.Series
        The single row at ``asof_date`` (the marks the rebalance fills at).
    history.dates -> list[date]         (all dates <= asof_date, ascending)
    history.universe -> list[str]

Because the view is sliced to ``<= asof_date`` BEFORE the strategy sees it, a
no-lookahead bug is structurally impossible: the future rows are simply not
present in any frame the strategy can touch.

FILL TIMING (and why it is honest)
----------------------------------
On a rebalance date, the new target weights are computed from data up to AND
including ``asof_date`` close, then the portfolio is traded to those weights at
``asof_date``'s ADJUSTED close. Decision and fill use the same day's close, so
there is no peeking at the next day. Between rebalances the held share counts are
fixed and the NAV simply marks to each day's close (positions drift with price).
This is the standard close-to-close daily convention; it does not assume a
next-open fill it cannot model on daily bars, and it never uses a price the
strategy could not have seen.

COSTS & SHORT-TERM TAX (CLAUDE.md §10.1: after-tax equity is first-class)
-------------------------------------------------------------------------
* Transaction cost = ``cost_bps * 1e-4 * (notional traded)`` charged out of NAV
  on every rebalance, where notional traded is the sum of the absolute dollar
  change in each position (one-way turnover in dollars). 2 bps default.
* Short-term tax: the program runs in a TAXABLE wrapper, so realized gains are
  short-term/ordinary income. On every rebalance, for each position being
  reduced or closed we realize ``(sale_price - avg_cost) * shares_sold``; the
  POSITIVE part accrues a tax reserve at ``short_term_tax_rate`` (losses bank an
  offset against the running realized-gain total, never a refund below 0). The
  reserve is a LEDGER line — gross NAV is unaffected, but ``after_tax_nav =
  nav - tax_reserve`` and compounding/CAGR are reported after tax too. Average
  cost basis is tracked per symbol (lots merged at cost) and reset to the new
  fill price when a position is re-established.

PURE / DETERMINISTIC: no LLM, no MCP, no network in this module. Data comes from
the DuckDB ``bars`` table (``timeframe='1d'`` ADJUSTED) via
:func:`load_daily_bars`, or can be injected directly as a price panel for tests.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from backtest.daily.result import DailyResult
from data.schema import DEFAULT_DB_PATH, connect

# Default short-term capital-gains reserve rate (ordinary income proxy). The
# real number is the user's marginal bracket; 0.30 is a deliberately blunt,
# conservative placeholder the caller overrides.
DEFAULT_SHORT_TERM_TAX_RATE = 0.30


# --------------------------------------------------------------------------- #
# The strategy interface (the Phase-2 contract)
# --------------------------------------------------------------------------- #
@runtime_checkable
class DailyStrategy(Protocol):
    """A daily, long-only, point-in-time target-weight allocator.

    The single method the engine calls each decision day. See the module
    docstring for the full contract (no-lookahead, long-only, sum <= 1).
    """

    def target_weights(
        self, asof_date: date, history: "DailyHistory"
    ) -> dict[str, float]:
        ...


# --------------------------------------------------------------------------- #
# Point-in-time history access helper (no lookahead by construction)
# --------------------------------------------------------------------------- #
class DailyHistory:
    """Point-in-time ADJUSTED daily price views, sliced to ``<= asof_date``.

    Constructed by the engine each decision day from the full universe price
    panel. Every frame this hands back is already clipped to dates
    ``<= asof_date``, so a strategy cannot reach a future bar.

    Parameters
    ----------
    panel
        The FULL wide ADJUSTED-close DataFrame (index = trading dates ascending,
        columns = universe symbols). Stored by reference; never mutated.
    asof_date
        The current decision date. The visible window is ``index <= asof_date``.
    """

    def __init__(self, panel: pd.DataFrame, asof_date: date):
        self._panel = panel
        self.asof_date = asof_date
        # Slice ONCE to the visible window; all views derive from this.
        self._visible = panel.loc[panel.index <= asof_date]

    # -- metadata --------------------------------------------------------- #
    @property
    def dates(self) -> list:
        """All trading dates ``<= asof_date`` (ascending)."""
        return list(self._visible.index)

    @property
    def universe(self) -> list[str]:
        """All symbols in the panel (columns)."""
        return list(self._panel.columns)

    # -- price / return views (already sliced; no lookahead) -------------- #
    def prices(
        self, symbols: Sequence[str] | None = None, lookback: int | None = None
    ) -> pd.DataFrame:
        """Wide ADJUSTED-close frame for ``symbols`` over the visible window.

        ``symbols=None`` -> the whole universe. ``lookback=N`` -> only the last N
        rows (the most recent N visible trading days). The result is a copy, safe
        for the strategy to mutate.
        """
        df = self._visible if symbols is None else self._visible[list(symbols)]
        if lookback is not None:
            df = df.iloc[-int(lookback):]
        return df.copy()

    def returns(
        self, symbols: Sequence[str] | None = None, lookback: int | None = None
    ) -> pd.DataFrame:
        """Simple daily returns of :meth:`prices` (``pct_change``, head dropped).

        ``lookback`` here is the number of RETURN rows kept (so it needs N+1
        price rows under the hood); the engine fetches the full window then
        trims, so the most recent ``lookback`` returns are exact.
        """
        px = self._visible if symbols is None else self._visible[list(symbols)]
        rets = px.pct_change().iloc[1:]
        if lookback is not None:
            rets = rets.iloc[-int(lookback):]
        return rets.copy()

    def asof_prices(self, symbols: Sequence[str] | None = None) -> pd.Series:
        """The ADJUSTED-close row AT ``asof_date`` (the rebalance mark).

        Returns a Series indexed by symbol. Symbols with no bar on ``asof_date``
        come back as NaN (the engine drops them — a missing mark is untradeable).
        """
        if len(self._visible) == 0:
            cols = self.universe if symbols is None else list(symbols)
            return pd.Series(index=cols, dtype="float64")
        row = self._visible.iloc[-1]
        if symbols is not None:
            row = row.reindex(list(symbols))
        return row.astype("float64")


# --------------------------------------------------------------------------- #
# Data loading — daily ADJUSTED bars from the DuckDB bars table
# --------------------------------------------------------------------------- #
def _as_date(d) -> date | None:
    """Coerce date / datetime / 'YYYY-MM-DD' / None -> ``date`` | None."""
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


def load_daily_bars(
    universe: Sequence[str],
    start=None,
    end=None,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Load a wide ADJUSTED daily-close panel for ``universe`` from the DB.

    Pulls ``timeframe='1d'`` bars from the ``bars`` table for each symbol, takes
    the ADJUSTED close, and pivots to a wide frame (index = ``datetime.date``
    ascending, columns = symbols). Optionally clipped to an inclusive
    ``[start, end]`` date window. Missing symbols simply have no column.

    Point-in-time note: daily bars are stored split-adjusted (``adjusted=True``);
    this loader does NOT forward-fill, so a symbol that did not trade a given day
    is NaN and the engine carries the position at its last good mark (no
    fabricated price). Pure / read-only.

    Returns an empty DataFrame if nothing is found.
    """
    own_con = con is None
    if own_con:
        con = connect(db_path)
    try:
        placeholders = ",".join("?" for _ in universe)
        rows = con.execute(
            f"""
            SELECT symbol, ts_utc, close
            FROM bars
            WHERE timeframe = ? AND symbol IN ({placeholders})
            ORDER BY symbol, ts_utc
            """,
            [timeframe, *list(universe)],
        ).df()
    finally:
        if own_con:
            con.close()

    if len(rows) == 0:
        return pd.DataFrame()

    rows["date"] = pd.to_datetime(rows["ts_utc"]).dt.date
    panel = rows.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()

    s = _as_date(start)
    e = _as_date(end)
    if s is not None:
        panel = panel.loc[[d for d in panel.index if d >= s]]
    if e is not None:
        panel = panel.loc[[d for d in panel.index if d <= e]]
    # Keep the universe column order stable (only those actually present).
    cols = [sym for sym in universe if sym in panel.columns]
    return panel[cols]


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
def _rebalance_flags(dates: Sequence[date], rebalance: str) -> list[bool]:
    """Mark which trading dates are rebalance days for an M / W schedule.

    A date is a rebalance day if it is the LAST trading date in its calendar
    period (ISO week for 'W', year-month for 'M') within the sample — i.e. the
    last session before the period rolls over. The first date is always a
    rebalance day (initial allocation). This is computed purely from the date
    list, no exchange calendar needed.
    """
    rb = rebalance.upper()
    if rb not in ("M", "W"):
        raise ValueError(f"rebalance must be 'M' or 'W', got {rebalance!r}")

    def _key(d: date):
        if rb == "M":
            return (d.year, d.month)
        iso = d.isocalendar()
        return (iso[0], iso[1])  # (iso_year, iso_week)

    n = len(dates)
    flags = [False] * n
    if n == 0:
        return flags
    flags[0] = True  # initial allocation
    for i in range(n):
        # last date of its period == next date is in a different period (or end).
        if i == n - 1 or _key(dates[i]) != _key(dates[i + 1]):
            flags[i] = True
    return flags


# --------------------------------------------------------------------------- #
# Weight validation
# --------------------------------------------------------------------------- #
def _validate_weights(weights: dict, asof_date: date, tol: float = 1e-9) -> dict:
    """Validate a strategy's target weights against the long-only / sum<=1 contract.

    Raises ``ValueError`` on a negative (short) weight or a sum exceeding 1 by
    more than ``tol``. Returns a clean ``{symbol: float}`` dict (drops zeros and
    non-finite/None values).
    """
    if weights is None:
        return {}
    clean: dict[str, float] = {}
    total = 0.0
    for sym, w in weights.items():
        if w is None:
            continue
        wf = float(w)
        if not np.isfinite(wf):
            continue
        if wf < -tol:
            raise ValueError(
                f"negative (short) weight {wf:.6f} for {sym!r} on {asof_date} — "
                "the daily interface is LONG-ONLY (use a positive weight on an "
                "inverse ETF for inverse exposure)"
            )
        if wf <= tol:
            continue
        clean[sym] = wf
        total += wf
    if total > 1.0 + tol:
        raise ValueError(
            f"target weights sum to {total:.6f} > 1 on {asof_date} — weights are "
            "fractions of equity and must sum to <= 1 (remainder is cash)"
        )
    return clean


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def run_daily(
    strategy: DailyStrategy,
    universe: Sequence[str],
    start=None,
    end=None,
    rebalance: str = "M",
    cost_bps: float = 2.0,
    initial_equity: float = 100_000.0,
    short_term_tax_rate: float = DEFAULT_SHORT_TERM_TAX_RATE,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
    panel: pd.DataFrame | None = None,
) -> DailyResult:
    """Run a daily target-weight strategy and return its :class:`DailyResult`.

    Parameters
    ----------
    strategy
        Any object implementing :class:`DailyStrategy` (``target_weights``).
    universe
        The tradable symbols. ADJUSTED daily bars are loaded for these.
    start, end
        Inclusive date bounds (date / datetime / 'YYYY-MM-DD' / None).
    rebalance
        ``'M'`` (month-end) or ``'W'`` (week-end) rebalance cadence.
    cost_bps
        Transaction cost in basis points of the dollar notional traded
        (one-way turnover) on each rebalance. 2.0 bps default.
    initial_equity
        Starting NAV.
    short_term_tax_rate
        Reserve rate applied to realized short-term gains (CLAUDE.md §10.1).
    db_path / con
        DuckDB path or an open connection to reuse (not closed if passed in).
    panel
        OPTIONAL pre-built wide ADJUSTED-close panel (index = dates, columns =
        symbols). When given, the DB is bypassed entirely — the deterministic,
        offline path used by tests. ``universe`` then selects/orders its columns.

    Notes
    -----
    Pure / deterministic / offline. Decision and fill both use ``asof_date``'s
    close (no next-bar peeking). See the module docstring for the full cost/tax/
    no-lookahead semantics.
    """
    if panel is None:
        panel = load_daily_bars(universe, start=start, end=end, db_path=db_path, con=con)
    else:
        panel = panel.copy()
        panel.index = [_as_date(d) for d in panel.index]
        panel = panel.sort_index()
        s, e = _as_date(start), _as_date(end)
        if s is not None:
            panel = panel.loc[[d for d in panel.index if d >= s]]
        if e is not None:
            panel = panel.loc[[d for d in panel.index if d <= e]]
        cols = [sym for sym in universe if sym in panel.columns]
        panel = panel[cols]

    cost_rate = float(cost_bps) * 1e-4
    dates = list(panel.index)

    if len(dates) == 0:
        empty = pd.Series(dtype="float64")
        return DailyResult(
            nav=empty, after_tax_nav=empty, initial_equity=float(initial_equity),
            total_costs=0.0, tax_reserve=0.0, realized_gains=0.0,
            rebalance_turnover=pd.Series(dtype="float64"),
            short_term_tax_rate=float(short_term_tax_rate),
        )

    flags = _rebalance_flags(dates, rebalance)

    # ---- portfolio state ----
    cash = float(initial_equity)
    shares: dict[str, float] = {}        # symbol -> shares held
    avg_cost: dict[str, float] = {}      # symbol -> average cost basis / share
    last_mark: dict[str, float] = {}     # symbol -> last good price (carry NaN)

    total_costs = 0.0
    tax_reserve = 0.0
    realized_gains = 0.0                  # cumulative net realized short-term gain
    nav_records: list[float] = []
    nav_index: list[date] = []
    reserve_records: list[float] = []     # running tax reserve at each day's close
    turnover_records: dict[date, float] = {}

    def _price(sym: str, row: pd.Series) -> float | None:
        """Today's mark for ``sym``: today's close, else the last good mark."""
        px = row.get(sym, np.nan)
        if px is not None and np.isfinite(px):
            last_mark[sym] = float(px)
            return float(px)
        return last_mark.get(sym)  # carry the stale mark (None if never seen)

    for i, d in enumerate(dates):
        row = panel.loc[d]

        # Refresh marks for everything we can price today (updates last_mark).
        for sym in panel.columns:
            _price(sym, row)

        # ---- mark-to-market NAV (gross, pre-rebalance) ----
        position_value = sum(
            sh * last_mark[sym]
            for sym, sh in shares.items()
            if sym in last_mark and last_mark[sym] is not None
        )
        nav = cash + position_value

        # ---- rebalance at today's close (decision & fill same day) ----
        if flags[i] and nav > 0:
            history = DailyHistory(panel, d)
            raw = strategy.target_weights(d, history)
            weights = _validate_weights(raw, d)

            # Target $ per symbol (only symbols we can mark TODAY are tradable).
            target_dollars: dict[str, float] = {}
            for sym, w in weights.items():
                px = _price(sym, row)
                if px is None or px <= 0:
                    continue  # untradeable today -> weight forfeited to cash
                target_dollars[sym] = w * nav

            # Symbols to consider: current holdings + new targets.
            syms = set(shares) | set(target_dollars)
            day_turnover = 0.0
            for sym in syms:
                px = last_mark.get(sym)
                if px is None or px <= 0:
                    continue
                cur_sh = shares.get(sym, 0.0)
                cur_val = cur_sh * px
                tgt_val = target_dollars.get(sym, 0.0)
                tgt_sh = tgt_val / px
                d_sh = tgt_sh - cur_sh
                if abs(d_sh) < 1e-12:
                    continue
                trade_notional = abs(d_sh) * px
                day_turnover += trade_notional

                if d_sh < 0:  # SELL -> realize gain/loss on the shares sold
                    sold = -d_sh
                    basis = avg_cost.get(sym, px)
                    gain = (px - basis) * sold
                    realized_gains += gain
                    if gain > 0:
                        tax_reserve += gain * float(short_term_tax_rate)
                    new_sh = cur_sh - sold
                    if new_sh <= 1e-12:
                        shares.pop(sym, None)
                        avg_cost.pop(sym, None)
                    else:
                        shares[sym] = new_sh
                        # avg cost unchanged on a partial sell
                    cash += sold * px
                else:          # BUY -> merge lots at cost
                    bought = d_sh
                    cash -= bought * px
                    prev_sh = cur_sh
                    prev_basis = avg_cost.get(sym, px)
                    new_sh = prev_sh + bought
                    avg_cost[sym] = (prev_sh * prev_basis + bought * px) / new_sh
                    shares[sym] = new_sh

            # Transaction cost on the one-way dollar turnover, paid from cash.
            cost = day_turnover * cost_rate
            cash -= cost
            total_costs += cost
            turnover_records[d] = day_turnover / nav if nav > 0 else 0.0

            # Recompute NAV after the rebalance + cost so the curve reflects it.
            position_value = sum(
                sh * last_mark[sym]
                for sym, sh in shares.items()
                if sym in last_mark and last_mark[sym] is not None
            )
            nav = cash + position_value

        nav_records.append(nav)
        nav_index.append(d)
        reserve_records.append(tax_reserve)  # running reserve AS OF this close

    nav_series = pd.Series(nav_records, index=nav_index, name="nav")
    nav_series.index.name = "date"
    # After-tax NAV subtracts the tax reserve AS IT ACCRUED, not the final total —
    # so an early day is not retro-penalized for a gain realized later. Subtracting
    # the running reserve also means the after-tax curve only steps down on a
    # gain-realizing rebalance, matching how the reserve ledger actually fills.
    reserve_series = pd.Series(reserve_records, index=nav_index, dtype="float64")
    after_tax = (nav_series - reserve_series).rename("after_tax_nav")
    after_tax.index.name = "date"
    turnover_series = pd.Series(turnover_records, dtype="float64").sort_index()
    turnover_series.name = "rebalance_turnover"

    return DailyResult(
        nav=nav_series,
        after_tax_nav=after_tax,
        initial_equity=float(initial_equity),
        total_costs=total_costs,
        tax_reserve=tax_reserve,
        realized_gains=realized_gains,
        rebalance_turnover=turnover_series,
        short_term_tax_rate=float(short_term_tax_rate),
    )


# --------------------------------------------------------------------------- #
# Robustness helper — the GEM-fragility guard (dispersion, not peak)
# --------------------------------------------------------------------------- #
def run_param_grid(
    strategy_factory: Callable[[object], DailyStrategy],
    param_name: str,
    values: Sequence,
    universe: Sequence[str],
    start=None,
    end=None,
    rebalance: str = "M",
    cost_bps: float = 2.0,
    initial_equity: float = 100_000.0,
    short_term_tax_rate: float = DEFAULT_SHORT_TERM_TAX_RATE,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
    panel: pd.DataFrame | None = None,
) -> dict:
    """Sweep one parameter and report the DISPERSION of CAGR / Sharpe / maxDD.

    The robustness / GEM-fragility guard (MASTER_PLAN §0/§5): a real edge is a
    *plateau*, not a *spike*. We re-run the strategy across ``values`` of one
    knob and report how stable the headline metrics are — a strategy whose CAGR
    swings wildly with a small parameter nudge is overfit, however good its peak
    backtest looks. ``strategy_factory(value)`` must build a fresh strategy for
    each ``value`` of ``param_name``.

    To avoid re-loading the DB on every grid point, the panel is loaded ONCE here
    (unless ``panel`` is supplied) and threaded into each :func:`run_daily`.

    Returns
    -------
    dict with:
      ``param``        the swept parameter name
      ``values``       the swept values (list)
      ``grid``         list of ``{value, CAGR, sharpe, max_drawdown,
                       after_tax_CAGR, turnover_per_year}`` rows
      ``dispersion``   ``{metric: {min, max, mean, std, spread, cv}}`` for
                       CAGR / sharpe / max_drawdown (spread = max-min;
                       cv = std/|mean|, the scale-free fragility number)
    """
    if panel is None:
        panel = load_daily_bars(universe, start=start, end=end, db_path=db_path, con=con)

    grid: list[dict] = []
    for v in values:
        strat = strategy_factory(v)
        res = run_daily(
            strat, universe, start=start, end=end, rebalance=rebalance,
            cost_bps=cost_bps, initial_equity=initial_equity,
            short_term_tax_rate=short_term_tax_rate, panel=panel,
        )
        s = res.summary()
        grid.append({
            "value": v,
            "CAGR": s["CAGR"],
            "sharpe": s["sharpe"],
            "max_drawdown": s["max_drawdown"],
            "after_tax_CAGR": s["after_tax_CAGR"],
            "turnover_per_year": s["turnover_per_year"],
        })

    def _disp(metric: str) -> dict:
        arr = np.asarray([g[metric] for g in grid], dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan"),
                    "std": float("nan"), "spread": float("nan"), "cv": float("nan")}
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": mean,
            "std": std,
            "spread": float(arr.max() - arr.min()),
            "cv": (std / abs(mean)) if mean != 0 else float("nan"),
        }

    return {
        "param": param_name,
        "values": list(values),
        "grid": grid,
        "dispersion": {
            "CAGR": _disp("CAGR"),
            "sharpe": _disp("sharpe"),
            "max_drawdown": _disp("max_drawdown"),
        },
    }
