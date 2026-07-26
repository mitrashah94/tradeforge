"""backtest/daily/bracket_engine.py — the DAILY BRACKETED-SWING PORTFOLIO backtester.

WHY A THIRD ENGINE (not ``backtest/engine`` and not ``backtest/daily/engine``)
------------------------------------------------------------------------------
* ``backtest/engine`` is a path-dependent *single-symbol intraday* round-trip
  machine (break -> retest -> OCO bracket -> EOD-flat). It never holds overnight
  and never holds more than one position.
* ``backtest/daily/engine`` is a *cross-sectional long-only weight allocator*:
  it rebalances the whole book to a target-weight vector on an M/W schedule and
  has no per-position stops at all — a position only changes when the next
  rebalance retargets it.

This engine is the third animal the operator asked for: an **actively-managed
daily swing portfolio** that holds MULTIPLE concurrent positions, where EVERY
position carries a per-position ATR STOP-LOSS and a profit-taking exit, and the
book is re-evaluated EVERY trading day (cut losers fast, let winners run). It
reuses the live fast loop's LIFECYCLE SHAPE
(``orchestrator/fast_loop/lifecycle.py``: entry -> TP1 partial -> stop-to-
breakeven -> trail runner -> stop/target/close) and the RISK-SIZING SPIRIT of
``orchestrator/fast_loop/sizing.py`` (``shares = $risk / stop_distance``, so the
$-risk per trade is constant across vol), but it is a *portfolio backtester over
daily ADJUSTED bars*, not the live loop and not a single-symbol simulator.

THE SWING-STRATEGY INTERFACE (the Phase-2 contract — implement against THIS)
===========================================================================
A swing strategy is any object with an ``entry_score`` method (and an OPTIONAL
``exit_signal``)::

    class SwingStrategy(Protocol):
        def entry_score(
            self, symbol: str, asof_date: date, history: DailyHistory
        ) -> float | None:
            '''Strength of a NEW entry in ``symbol`` AS OF ``asof_date``'s close.

            Return ``None`` (or a non-finite number) for "no entry today". A
            higher score is a STRONGER candidate; when more symbols pass than
            there are free slots, the engine RANKS by score (desc) and opens the
            best. The score is opaque to the engine — any monotone signal works
            (momentum, z-score, breakout distance, ...).'''
            ...

        def exit_signal(  # OPTIONAL — omit for bracket-only management
            self, symbol: str, asof_date: date, history: DailyHistory, position
        ) -> bool:
            '''Discretionary exit BEYOND the brackets. Return True to close the
            whole remaining position at ``asof_date``'s close. ``position`` is
            the engine's :class:`OpenPosition` (read-only view of the live
            bracket state). Omit the method entirely for pure bracket exits.'''
            ...

CONTRACT (enforced / relied on by the engine):
  * ``history`` is the SAME point-in-time :class:`DailyHistory` the daily engine
    uses — ADJUSTED daily closes sliced to ``<= asof_date``. NO LOOKAHEAD: a bar
    dated after ``asof_date`` is not present in any frame the strategy can touch.
    (Brackets need O/H/L, which the strategy does not — it decides on the CLOSE.
    The engine itself reads the next day's O/H/L to RESOLVE the brackets, never
    the strategy.)
  * ``entry_score`` is asked only for symbols NOT currently held (you cannot
    re-enter a name you already hold). Decisions use ``asof_date``'s close; the
    entry FILLS at that same close (no next-bar peeking on entry).
  * LONG-ONLY. There is no short side — inverse exposure is a positive position
    in an inverse ETF (the universe must include it).

THE BRACKET (ATR-based, configurable) — see :class:`BracketConfig`
------------------------------------------------------------------
On entry at the close ``E`` with ``ATR = atr(atr_window)`` as of the entry day:
  * initial stop  ``S0 = E - stop_atr_mult * ATR``      (risk per share R = E-S0)
  * TP1           ``E + tp1_R * R``    -> sell ``tp1_fraction``, move stop to BE
  * runner trail  chandelier: ``stop = highest_high_since_entry - trail_atr_mult
                  * ATR_entry`` (never loosens) once ``use_trail`` and past TP1
  * hard target   if ``hard_target_R`` is set, a FULL take-profit at
                  ``E + hard_target_R * R`` that caps the upside (the "max gain
                  sell" variant). With ``use_trail=False`` and a ``hard_target_R``
                  you get a classic fixed bracket; the default
                  (partial-TP + breakeven + trail, no hard cap) lets winners run.

DAILY SEMANTICS (point-in-time, NO lookahead, GAP-AWARE)
--------------------------------------------------------
Each trading day ``d``, in this order:

  (1) MANAGE every open position against ``d``'s ``open / high / low``:
      * GAP FIRST: if ``d`` OPENS at/through the current stop, the stop fills at
        the OPEN (worse than the resting stop) — a gap-down below the stop is not
        magically filled at the stop. Symmetrically a favorable gap THROUGH the
        TP1/hard-target limit fills at the OPEN (better).
      * Then intrabar by ``[low, high]``: STOP-FIRST on ambiguity (if one bar can
        reach both the stop and a profit target, assume the stop — bar data hides
        intrabar order; conservative, matches the intraday engine).
      * STOP hit -> exit the remaining shares at the stop (or gap open).
      * TP1 (``tp1_R``) hit (pre-partial only) -> sell ``tp1_fraction`` at TP1
        (or the gap open if better), move the stop to BREAKEVEN (entry price).
      * HARD TARGET (``hard_target_R``) hit -> full exit at the target (or gap).
      * After a non-exiting bar, TRAIL the runner's chandelier stop up.
      * ``exit_signal`` (if the strategy implements it) -> close at the close.
      Realize each partial/exit: book net PnL, charge ``cost_bps`` on the traded
      notional, accrue the short-term tax reserve on realized GAINS only.

  (2) ENTRIES: for every symbol NOT held, ask ``entry_score``; keep the finite
      scores, rank DESC, and open the best up to the free slots
      (``max_concurrent`` minus the count still open after step 1). Size each by
      RISK: ``shares = (risk_pct_per_trade * equity) / (E - S0)`` (FRACTIONAL),
      enter at the close ``E`` with stop ``S0 = E - stop_atr_mult * ATR``. A
      degenerate stop distance / missing ATR / unpriced symbol is skipped.

  (3) MARK every open position to ``d``'s close -> the daily equity point; record
      the day's exposure (invested fraction) and turnover.

COSTS & SHORT-TERM TAX (CLAUDE.md §10.1: after-tax equity is first-class)
-------------------------------------------------------------------------
* Transaction cost = ``cost_bps * 1e-4 * |traded notional|`` on EVERY fill
  (entries and exits), paid from cash. 2 bps default.
* Short-term tax: realized gains are short-term/ordinary income (taxable
  wrapper). Each closing/partial fill realizes ``(exit - entry) * shares_sold``;
  the POSITIVE part accrues a reserve at ``short_term_tax_rate`` (a loss banks an
  offset against the running realized total, never a refund below 0). The reserve
  is a LEDGER line — gross NAV is unaffected, ``after_tax_nav = nav - reserve``,
  and CAGR is reported after tax too.

PURE / DETERMINISTIC / OFFLINE: no LLM, no MCP, no network, no clock. Data is the
DuckDB ``bars`` table (``timeframe='1d'`` ADJUSTED OHLC) via
:func:`load_daily_ohlc`, or injected directly as an OHLC panel for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from backtest.daily.engine import DailyHistory, _as_date, load_daily_bars
from backtest.daily.result import BracketResult, BracketTrade
from data.schema import DEFAULT_DB_PATH, connect

DEFAULT_SHORT_TERM_TAX_RATE = 0.30


# --------------------------------------------------------------------------- #
# The swing-strategy interface (the Phase-2 contract)
# --------------------------------------------------------------------------- #
@runtime_checkable
class SwingStrategy(Protocol):
    """A daily, long-only swing strategy: rank entries, optionally exit early.

    ``entry_score`` is the only required method (see the module docstring for the
    full contract). ``exit_signal`` is OPTIONAL — when absent, positions are
    managed purely by the ATR bracket.
    """

    def entry_score(
        self, symbol: str, asof_date: date, history: "DailyHistory"
    ) -> float | None:
        ...


# --------------------------------------------------------------------------- #
# The bracket configuration (ATR-based, configurable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BracketConfig:
    """ATR-based bracket parameters (the per-position protection plan).

    Defaults are the positive-skew "let winners run" plan: scale a partial at
    +``tp1_R``R, ratchet the stop to breakeven, then trail the runner by a
    chandelier — NO hard cap on the upside.

    Parameters
    ----------
    atr_window
        Lookback (trading days) for the ATR used to set the stop distance and the
        chandelier band. ATR is the mean TRUE RANGE over the window, computed
        point-in-time on bars up to AND including the entry day.
    stop_atr_mult
        Initial stop = ``entry - stop_atr_mult * ATR``. The per-share risk
        ``R = entry - stop`` is the sizing + R-multiple basis.
    tp1_R
        First (partial) profit target in R multiples of the initial risk. TP1
        price = ``entry + tp1_R * R``. Set ``tp1_fraction`` to 0 to disable the
        partial entirely (pure trail / hard-target management).
    tp1_fraction
        Fraction of the position scaled out at TP1 (e.g. 0.5 -> sell half). On
        the TP1 fill the remaining runner's stop moves to BREAKEVEN (entry).
    trail_atr_mult
        Chandelier multiple for the runner: ``stop = highest_high_since_entry -
        trail_atr_mult * ATR_entry`` (never loosens). Only active once
        ``use_trail`` is True and the position is past TP1 (a runner).
    use_trail
        Master toggle for the chandelier trail. False -> the runner keeps its
        breakeven (or initial) stop and rides to a hard target / stop / exit
        signal without trailing.
    hard_target_R
        Optional FULL take-profit cap in R multiples (the "max gain sell"
        variant): a full exit at ``entry + hard_target_R * R``. ``None`` (default)
        -> no cap, the runner is free. Set it (e.g. with ``use_trail=False``) for
        a classic fixed bracket that caps upside.
    """

    atr_window: int = 14
    stop_atr_mult: float = 2.5
    tp1_R: float = 1.5
    tp1_fraction: float = 0.5
    trail_atr_mult: float = 3.0
    use_trail: bool = True
    hard_target_R: float | None = None


# --------------------------------------------------------------------------- #
# Open-position state (the engine's bracket bookkeeping)
# --------------------------------------------------------------------------- #
@dataclass
class OpenPosition:
    """One live bracketed long position the engine is managing.

    This is the read-only view handed to ``exit_signal``. It carries the bracket
    plan (initial stop, current stop, TP1 / hard-target prices), the
    chandelier-trail state (highest high since entry, ATR at entry), and the
    realized-so-far accumulators a partial scale-out feeds (so the final
    closed-trade record aggregates over every piece with a SIZE-WEIGHTED R).
    """

    symbol: str
    entry_date: date
    entry_price: float                 # fill = the entry day's adjusted close
    initial_stop: float                # E - stop_atr_mult*ATR (risk basis)
    initial_shares: float              # qty at entry
    atr_entry: float                   # ATR as of the entry day (trail band)

    shares: float                      # CURRENT live shares (shrinks after TP1)
    stop: float                        # CURRENT protective stop (BE, then trails)
    tp1_price: float | None            # +tp1_R*R partial target (None if no TP1)
    hard_target: float | None          # +hard_target_R*R full cap (None if off)

    # ---- runtime ----
    tp1_done: bool = False             # True once the partial has scaled out
    highest_high: float = 0.0          # highest high seen since entry (chandelier)
    bars_held: int = 0                 # trading days since entry

    # ---- realized-so-far accumulators (the partial folds into the final trade)
    realized_pnl: float = 0.0          # net $ booked from partials so far
    realized_gross: float = 0.0        # gross $ booked from partials so far
    realized_costs: float = 0.0        # cost $ booked from partials so far
    realized_r: float = 0.0            # size-weighted R booked from partials

    @property
    def risk_per_share(self) -> float:
        """The INITIAL per-share risk ``entry - initial_stop`` (R basis)."""
        return self.entry_price - self.initial_stop


# --------------------------------------------------------------------------- #
# Data loading — daily ADJUSTED OHLC panels (brackets need O/H/L, not just C)
# --------------------------------------------------------------------------- #
def load_daily_ohlc(
    universe: Sequence[str],
    start=None,
    end=None,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
    timeframe: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Load wide ADJUSTED daily O/H/L/C panels for ``universe`` from the DB.

    The daily target-weight engine only needs the close, but a bracket engine
    must resolve stops/targets against the intraday RANGE, so this loads the full
    OHLC. Returns a dict ``{"open": df, "high": df, "low": df, "close": df}``
    where each ``df`` is a wide frame (index = ``datetime.date`` ascending,
    columns = the symbols actually present, in ``universe`` order). The OHLC are
    total-return ADJUSTED (gap logic uses the ADJUSTED open). Pure / read-only.

    Returns four EMPTY frames if nothing is found.
    """
    own_con = con is None
    if own_con:
        con = connect(db_path)
    try:
        placeholders = ",".join("?" for _ in universe)
        rows = con.execute(
            f"""
            SELECT symbol, ts_utc, open, high, low, close
            FROM bars
            WHERE timeframe = ? AND symbol IN ({placeholders})
            ORDER BY symbol, ts_utc
            """,
            [timeframe, *list(universe)],
        ).df()
    finally:
        if own_con:
            con.close()

    empty = pd.DataFrame()
    if len(rows) == 0:
        return {"open": empty, "high": empty, "low": empty, "close": empty.copy()}

    rows["date"] = pd.to_datetime(rows["ts_utc"]).dt.date
    out: dict[str, pd.DataFrame] = {}
    s, e = _as_date(start), _as_date(end)
    for field_name in ("open", "high", "low", "close"):
        panel = rows.pivot_table(
            index="date", columns="symbol", values=field_name, aggfunc="last"
        ).sort_index()
        if s is not None:
            panel = panel.loc[[d for d in panel.index if d >= s]]
        if e is not None:
            panel = panel.loc[[d for d in panel.index if d <= e]]
        cols = [sym for sym in universe if sym in panel.columns]
        out[field_name] = panel[cols]
    return out


# --------------------------------------------------------------------------- #
# ATR (true-range) — point-in-time, computed on a rolling OHLC window
# --------------------------------------------------------------------------- #
def _atr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Mean-true-range ATR over ``window`` bars, indexed like the inputs.

    True range = max(h-l, |h-prev_close|, |l-prev_close|). The ATR at index i is
    the simple mean of the last ``window`` true ranges ending at i (so it uses
    only bars up to and including i — point-in-time by construction). Leading
    rows with fewer than ``window`` observations are NaN.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=int(window), min_periods=int(window)).mean()


# --------------------------------------------------------------------------- #
# Panel normalization (shared with the daily engine's panel-injection path)
# --------------------------------------------------------------------------- #
def _normalize_ohlc(
    panel: dict | None,
    universe: Sequence[str],
    start,
    end,
    db_path: str,
    con,
) -> dict[str, pd.DataFrame]:
    """Resolve the OHLC panels: load from the DB, or sanitize an injected dict.

    ``panel`` may be ``None`` (load from the DB), or a dict with at least
    ``open/high/low/close`` wide frames (the offline/test path). When injected,
    every frame is re-dated, sorted, window-clipped to ``[start, end]`` and
    column-selected to ``universe`` order, exactly like the DB path.
    """
    if panel is None:
        return load_daily_ohlc(universe, start=start, end=end, db_path=db_path, con=con)

    s, e = _as_date(start), _as_date(end)
    out: dict[str, pd.DataFrame] = {}
    # The close frame defines the master date index / column set.
    for field_name in ("open", "high", "low", "close"):
        df = panel[field_name].copy()
        df.index = [_as_date(d) for d in df.index]
        df = df.sort_index()
        if s is not None:
            df = df.loc[[d for d in df.index if d >= s]]
        if e is not None:
            df = df.loc[[d for d in df.index if d <= e]]
        cols = [sym for sym in universe if sym in df.columns]
        out[field_name] = df[cols]
    return out


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def run_bracket_portfolio(
    strategy: SwingStrategy,
    universe: Sequence[str],
    start=None,
    end=None,
    *,
    risk_pct_per_trade: float = 0.01,
    max_concurrent: int = 8,
    bracket: BracketConfig = BracketConfig(),
    cost_bps: float = 2.0,
    initial_equity: float = 100_000.0,
    short_term_tax_rate: float = DEFAULT_SHORT_TERM_TAX_RATE,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
    panel: dict | None = None,
) -> BracketResult:
    """Run an actively-managed daily bracketed-swing portfolio.

    Parameters
    ----------
    strategy
        Any object implementing :class:`SwingStrategy` (``entry_score`` required,
        ``exit_signal`` optional).
    universe
        The tradable symbols. ADJUSTED daily OHLC bars are loaded for these.
    start, end
        Inclusive date bounds (date / datetime / 'YYYY-MM-DD' / None).
    risk_pct_per_trade
        Fraction of CURRENT equity risked per new entry. ``shares = (risk_pct *
        equity) / (entry - stop)`` (the vol-target identity: $-risk per trade is
        constant across vol). 0.01 == 1% (RI 5 floor).
    max_concurrent
        Maximum number of simultaneously-open positions (free-slot budget).
    bracket
        The :class:`BracketConfig` (ATR window + stop/TP1/trail/target plan).
    cost_bps
        Transaction cost in basis points of the traded notional, on every fill.
    initial_equity
        Starting NAV.
    short_term_tax_rate
        Reserve rate on realized short-term gains (CLAUDE.md §10.1).
    db_path / con
        DuckDB path or an open connection to reuse (not closed if passed in).
    panel
        OPTIONAL pre-built OHLC dict ``{"open","high","low","close": wide df}``
        (index = dates, columns = symbols). When given, the DB is bypassed — the
        deterministic, offline path used by tests.

    Returns
    -------
    BracketResult
        Daily NAV + after-tax curve, the closed-trade ledger, the per-day
        exposure series, and the accrued cost / tax / turnover totals.

    Notes
    -----
    Pure / deterministic / offline. Manage-then-enter ordering, gap-aware fills,
    no lookahead (entry decisions and fills both use ``asof_date``'s close; the
    engine reads the NEXT day's range only to RESOLVE brackets, never to decide).
    """
    panels = _normalize_ohlc(panel, universe, start, end, db_path, con)
    close = panels["close"]
    opn = panels["open"]
    high = panels["high"]
    low = panels["low"]

    cost_rate = float(cost_bps) * 1e-4
    dates = list(close.index)
    syms = list(close.columns)

    if len(dates) == 0 or len(syms) == 0:
        empty = pd.Series(dtype="float64")
        return BracketResult(
            nav=empty, after_tax_nav=empty, initial_equity=float(initial_equity),
            total_costs=0.0, tax_reserve=0.0, realized_gains=0.0,
            rebalance_turnover=pd.Series(dtype="float64"),
            short_term_tax_rate=float(short_term_tax_rate),
            trades=[], exposure=pd.Series(dtype="float64"),
        )

    # Precompute the ATR for every symbol once (point-in-time rolling mean TR).
    atr = pd.DataFrame(index=close.index, columns=syms, dtype="float64")
    for sym in syms:
        atr[sym] = _atr_series(high[sym], low[sym], close[sym], bracket.atr_window)

    # The CLOSE-only panel feeds the point-in-time DailyHistory the strategy sees
    # (it decides on closes; the engine alone reads O/H/L to resolve brackets).
    history_panel = close

    # ---- portfolio state ----
    cash = float(initial_equity)
    positions: dict[str, OpenPosition] = {}

    total_costs = 0.0
    tax_reserve = 0.0
    realized_gains = 0.0
    closed_trades: list[BracketTrade] = []

    nav_records: list[float] = []
    after_tax_records: list[float] = []
    exposure_records: list[float] = []
    turnover_records: dict[date, float] = {}
    nav_index: list[date] = []

    def _accrue(gross: float) -> None:
        """Book a realized gain/loss into the running tax reserve (gains only)."""
        nonlocal realized_gains, tax_reserve
        realized_gains += gross
        if gross > 0:
            tax_reserve += gross * float(short_term_tax_rate)

    def _sell(pos: OpenPosition, qty: float, fill_price: float, day_turnover_box):
        """Sell ``qty`` shares of ``pos`` at ``fill_price``: book PnL/cost/tax.

        Updates cash, the position's realized accumulators, and the running
        tax reserve. ``day_turnover_box`` is a 1-element list accumulating the
        day's traded notional (for the turnover series). Returns the net PnL.
        """
        nonlocal cash, total_costs
        notional = qty * fill_price
        cost = notional * cost_rate
        gross = (fill_price - pos.entry_price) * qty
        net = gross - cost
        cash += notional - cost
        total_costs += cost
        day_turnover_box[0] += notional
        _accrue(gross)
        pos.realized_gross += gross
        pos.realized_pnl += net
        pos.realized_costs += cost
        # Size-weighted R against the INITIAL risk.
        rps = pos.risk_per_share
        if rps > 0:
            pos.realized_r += ((fill_price - pos.entry_price) / rps) * (
                qty / pos.initial_shares
            )
        return net

    def _close_trade(pos: OpenPosition, exit_date: date, reason: str) -> None:
        """Finalize ``pos`` into a closed :class:`BracketTrade` and drop it."""
        gross = pos.realized_gross
        net = pos.realized_pnl
        costs = pos.realized_costs
        # Size-weighted average exit price over all pieces (entry + gross / shares
        # recovers it: gross = sum (exit_i - entry) * qty_i).
        if pos.initial_shares > 0:
            avg_exit = pos.entry_price + gross / pos.initial_shares
        else:
            avg_exit = pos.entry_price
        closed_trades.append(
            BracketTrade(
                symbol=pos.symbol,
                entry_date=pos.entry_date,
                exit_date=exit_date,
                entry_price=pos.entry_price,
                avg_exit_price=avg_exit,
                shares=pos.initial_shares,
                initial_stop=pos.initial_stop,
                pnl=net,
                gross_pnl=gross,
                costs=costs,
                r_multiple=pos.realized_r,
                bars_held=pos.bars_held,
                exit_reason=reason,
            )
        )
        positions.pop(pos.symbol, None)

    has_exit_signal = hasattr(strategy, "exit_signal") and callable(
        getattr(strategy, "exit_signal")
    )

    for d in dates:
        o_row = opn.loc[d]
        h_row = high.loc[d]
        l_row = low.loc[d]
        c_row = close.loc[d]
        day_turnover_box = [0.0]

        # ---------------- (1) MANAGE open positions against today's range ----
        # Snapshot the symbols to manage (we mutate ``positions`` while iterating).
        for sym in list(positions.keys()):
            pos = positions.get(sym)
            if pos is None:
                continue
            pos.bars_held += 1

            o = o_row.get(sym, np.nan)
            h = h_row.get(sym, np.nan)
            l = l_row.get(sym, np.nan)
            if not (np.isfinite(o) and np.isfinite(h) and np.isfinite(l)):
                continue  # untradeable today; carry the position untouched

            o, h, l = float(o), float(h), float(l)
            pos.highest_high = max(pos.highest_high, h)

            # --- GAP THROUGH the stop on the open: fill at the (worse) open. ---
            if o <= pos.stop:
                _sell(pos, pos.shares, o, day_turnover_box)
                _close_trade(pos, d, "trail_stop_gap" if pos.tp1_done else "stop_gap")
                continue

            # --- GAP THROUGH a profit target on the open: fill at the (better)
            #     open. Hard target wins over TP1 (it is the full-exit cap). ---
            if pos.hard_target is not None and o >= pos.hard_target:
                _sell(pos, pos.shares, o, day_turnover_box)
                _close_trade(pos, d, "hard_target_gap")
                continue
            if (
                not pos.tp1_done
                and pos.tp1_price is not None
                and o >= pos.tp1_price
                and bracket.tp1_fraction > 0
            ):
                _do_tp1(pos, o, day_turnover_box, _sell, bracket)
                # The runner survives; fall through to manage the remainder this
                # same bar (its stop is now breakeven, may still be hit below).

            pos = positions.get(sym)
            if pos is None:
                continue

            # --- INTRABAR by [low, high], STOP-FIRST on ambiguity. ---
            if l <= pos.stop:
                _sell(pos, pos.shares, pos.stop, day_turnover_box)
                _close_trade(pos, d, "trail_stop" if pos.tp1_done else "stop")
                continue

            if pos.hard_target is not None and h >= pos.hard_target:
                _sell(pos, pos.shares, pos.hard_target, day_turnover_box)
                _close_trade(pos, d, "hard_target")
                continue

            if (
                not pos.tp1_done
                and pos.tp1_price is not None
                and h >= pos.tp1_price
                and bracket.tp1_fraction > 0
            ):
                _do_tp1(pos, pos.tp1_price, day_turnover_box, _sell, bracket)
                pos = positions.get(sym)
                if pos is None:
                    continue

            # --- TRAIL the runner's chandelier stop (never loosening). ---
            if bracket.use_trail and pos.tp1_done:
                cand = pos.highest_high - bracket.trail_atr_mult * pos.atr_entry
                if cand > pos.stop:
                    pos.stop = cand

            # --- Discretionary exit_signal -> close at today's CLOSE. ---
            if has_exit_signal:
                c = c_row.get(sym, np.nan)
                if np.isfinite(c):
                    hist = DailyHistory(history_panel, d)
                    try:
                        want_exit = bool(
                            strategy.exit_signal(sym, d, hist, pos)
                        )
                    except TypeError:
                        want_exit = bool(strategy.exit_signal(sym, d, hist))
                    if want_exit:
                        _sell(pos, pos.shares, float(c), day_turnover_box)
                        _close_trade(pos, d, "exit_signal")
                        continue

        # ---------------- (2) ENTRIES: rank free-slot candidates -------------
        free_slots = max_concurrent - len(positions)
        if free_slots > 0:
            hist = DailyHistory(history_panel, d)
            candidates: list[tuple[float, str]] = []
            for sym in syms:
                if sym in positions:
                    continue
                c = c_row.get(sym, np.nan)
                a = atr.at[d, sym] if sym in atr.columns else np.nan
                if not (np.isfinite(c) and np.isfinite(a)) or c <= 0 or a <= 0:
                    continue
                score = strategy.entry_score(sym, d, hist)
                if score is None:
                    continue
                sf = float(score)
                if not np.isfinite(sf):
                    continue
                candidates.append((sf, sym))

            # Rank by score DESC (stable on ties by symbol for determinism).
            candidates.sort(key=lambda t: (-t[0], t[1]))

            # Current equity (mark-to-close) is the sizing base, recomputed each
            # entry so concurrent fills size off the same snapshot consistently.
            pos_value = _portfolio_value(positions, c_row)
            equity = cash + pos_value

            for _score, sym in candidates[:free_slots]:
                c = float(c_row[sym])
                a = float(atr.at[d, sym])
                stop = c - bracket.stop_atr_mult * a
                risk_per_share = c - stop
                if risk_per_share <= 0:
                    continue
                dollar_risk = risk_pct_per_trade * equity
                if dollar_risk <= 0:
                    continue
                shares = dollar_risk / risk_per_share
                if shares <= 0:
                    continue
                notional = shares * c
                # Cap by available cash (no leverage / no margin in this book).
                if notional > cash:
                    if cash <= 0:
                        continue
                    shares = cash / c
                    notional = shares * c
                cost = notional * cost_rate
                cash -= notional + cost
                total_costs += cost
                day_turnover_box[0] += notional

                rps = c - stop
                tp1_price = (
                    c + bracket.tp1_R * rps
                    if bracket.tp1_fraction > 0 and bracket.tp1_R is not None
                    else None
                )
                hard_target = (
                    c + bracket.hard_target_R * rps
                    if bracket.hard_target_R is not None
                    else None
                )
                positions[sym] = OpenPosition(
                    symbol=sym,
                    entry_date=d,
                    entry_price=c,
                    initial_stop=stop,
                    initial_shares=shares,
                    atr_entry=a,
                    shares=shares,
                    stop=stop,
                    tp1_price=tp1_price,
                    hard_target=hard_target,
                    highest_high=float(h_row.get(sym, c)) if np.isfinite(h_row.get(sym, np.nan)) else c,
                    # Seed the realized-cost accumulator with the ENTRY cost so the
                    # closed-trade ``costs`` field reflects BOTH sides (the exit
                    # cost is added as pieces are sold). Net PnL excludes the entry
                    # cost from per-piece accounting, so subtract it here too.
                    realized_costs=cost,
                    realized_pnl=-cost,
                )

        # ---------------- (3) MARK to close -> daily equity point ------------
        pos_value = _portfolio_value(positions, c_row)
        nav = cash + pos_value
        nav_records.append(nav)
        after_tax_records.append(nav - tax_reserve)
        exposure_records.append((pos_value / nav) if nav > 0 else 0.0)
        nav_index.append(d)
        if day_turnover_box[0] > 0:
            turnover_records[d] = day_turnover_box[0] / nav if nav > 0 else 0.0

    nav_series = pd.Series(nav_records, index=nav_index, name="nav")
    nav_series.index.name = "date"
    after_tax = pd.Series(after_tax_records, index=nav_index, name="after_tax_nav")
    after_tax.index.name = "date"
    exposure_series = pd.Series(exposure_records, index=nav_index, name="exposure")
    exposure_series.index.name = "date"
    turnover_series = pd.Series(turnover_records, dtype="float64").sort_index()
    turnover_series.name = "turnover"

    return BracketResult(
        nav=nav_series,
        after_tax_nav=after_tax,
        initial_equity=float(initial_equity),
        total_costs=total_costs,
        tax_reserve=tax_reserve,
        realized_gains=realized_gains,
        rebalance_turnover=turnover_series,
        short_term_tax_rate=float(short_term_tax_rate),
        trades=closed_trades,
        exposure=exposure_series,
    )


# --------------------------------------------------------------------------- #
# Internal helpers (module-level so the closures above stay readable)
# --------------------------------------------------------------------------- #
def _portfolio_value(positions: dict, close_row: pd.Series) -> float:
    """Mark-to-close dollar value of all open positions on a given day.

    A position whose symbol has no finite close today is carried at its entry
    price (no fabricated mark; a missing close is rare for a held name and the
    entry is the most recent KNOWN good price for it).
    """
    total = 0.0
    for sym, pos in positions.items():
        px = close_row.get(sym, np.nan)
        mark = float(px) if (px is not None and np.isfinite(px)) else pos.entry_price
        total += pos.shares * mark
    return total


def _do_tp1(pos: OpenPosition, fill_price: float, day_turnover_box, sell_fn, bracket: BracketConfig) -> None:
    """Scale out ``tp1_fraction`` of ``pos`` at ``fill_price`` and move to BE.

    Mirrors the live lifecycle's ``on_partial_filled``: sell the partial, shrink
    the live qty to the runner, and ratchet the stop to BREAKEVEN (the entry
    price), making the runner a free option (MASTER_PLAN §1.B). Idempotent — a
    no-op if the partial already happened or the fraction is degenerate.
    """
    if pos.tp1_done:
        return
    part = pos.initial_shares * bracket.tp1_fraction
    if part <= 0 or part >= pos.shares:
        return
    sell_fn(pos, part, fill_price, day_turnover_box)
    pos.shares -= part
    pos.tp1_done = True
    # Move the runner's stop to breakeven (the entry price).
    pos.stop = max(pos.stop, pos.entry_price)
