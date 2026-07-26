"""backtest/daily/portfolio_backtester.py — drive the cross-strategy PORTFOLIO ENGINE.

The fourth backtester: where ``bracket_engine`` runs ONE bracketed-swing strategy
and ``engine`` runs ONE target-weight sleeve, ``run_portfolio`` runs the WHOLE
BOOK — every daily sleeve (rotation + swing_meanrev + swing_breakout) funded by a
single :class:`~portfolio.engine.PortfolioEngine` under one risk budget. It owns
exactly the machinery the per-sleeve engines own (the OHLC panel, the ATR cache,
NAV / after-tax / exposure / turnover marking, the short-term-tax reserve) and
DELEGATES every per-day decision — what to open, close, resize, and reject — to
the engine. On top of the per-sleeve world it adds the DCA deposit stream and the
deposit-vs-edge split (TWR excludes flows; MWR/IRR includes them).

Each trading day ``d``:
  (0) inject the day's contribution into the book's cash (a DCA deposit);
  (1) set the engine's halt reference levels (peak / month-start / week-start /
      prior-close NAV) from the book's history;
  (2) call ``engine.step(d, history, day_bars, book)`` — the engine evolves the
      book (cash, lots, costs, tax, closed trades, attribution) and returns the
      Allocation;
  (3) mark the book to ``d``'s close → the day's NAV / after-tax / exposure /
      turnover points, and the flow-free TWR return.

PURE / DETERMINISTIC / OFFLINE: data is the DuckDB ``bars`` table (1d ADJUSTED
OHLC) via ``load_daily_ohlc``, or an injected panel for tests. No LLM / MCP / clock.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.daily.bracket_engine import _atr_series, _normalize_ohlc
from backtest.daily.engine import DailyHistory, _as_date
from backtest.daily.result import PortfolioResult
from data.schema import DEFAULT_DB_PATH
from portfolio.config import PortfolioConfig, load_portfolio_config
from portfolio.engine import PortfolioEngine
from portfolio.ledger import ContributionLedger
from portfolio.model import BookState, DayBars, SleeveSpec
from orchestrator.agents.regime_reader import SleeveArmConfig


def _atr_windows(sleeves: Sequence[SleeveSpec], pcfg: PortfolioConfig) -> set:
    """Every ATR window the engine will look up: synthetic band + each bracket."""
    windows = {int(pcfg.synthetic_stop.window)}
    for s in sleeves:
        if s.kind == "score" and s.bracket is not None:
            windows.add(int(getattr(s.bracket, "atr_window", 14)))
    return windows


def run_portfolio(
    sleeves: Sequence[SleeveSpec],
    universe: Sequence[str],
    start=None,
    end=None,
    *,
    initial_equity: float = 100_000.0,
    contributions: Optional[Mapping] = None,
    limits=None,
    pcfg: Optional[PortfolioConfig] = None,
    arm_cfg: Optional[SleeveArmConfig] = None,
    correlation_matrix=None,
    corr_threshold: float = 0.8,
    market_proxy: Optional[str] = None,
    forecast_provider=None,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
    panel: Optional[dict] = None,
) -> PortfolioResult:
    """Run the blended daily book over ``universe`` and return a :class:`PortfolioResult`.

    Parameters mirror the per-sleeve engines plus the cross-strategy pieces:
    ``sleeves`` (the admitted :class:`SleeveSpec` list), ``contributions``
    (``{date: cash_flow}`` DCA deposits), and ``correlation_matrix`` (for cluster
    dedup). ``panel`` is the offline OHLC-dict injection used by tests (bypasses
    the DB). See the module docstring for the per-day sequence.
    """
    pcfg = pcfg if pcfg is not None else load_portfolio_config()
    panels = _normalize_ohlc(panel, universe, start, end, db_path, con)
    close = panels["close"]
    dates = list(close.index)
    syms = list(close.columns)

    ledger = ContributionLedger(contributions)

    if len(dates) == 0 or len(syms) == 0:
        empty = pd.Series(dtype="float64")
        return PortfolioResult(
            nav=empty, after_tax_nav=empty, initial_equity=float(initial_equity),
            total_costs=0.0, tax_reserve=0.0, realized_gains=0.0,
            rebalance_turnover=pd.Series(dtype="float64"),
            short_term_tax_rate=float(pcfg.short_term_tax_rate),
            trades=[], exposure=pd.Series(dtype="float64"),
            contributions=dict(ledger.contributions),
            total_deposited=ledger.total_deposited(),
        )

    # Precompute ATR for every needed window once (point-in-time rolling mean TR).
    windows = _atr_windows(sleeves, pcfg)
    atr_by_window: dict = {}
    for w in windows:
        frame = pd.DataFrame(index=close.index, columns=syms, dtype="float64")
        for sym in syms:
            frame[sym] = _atr_series(panels["high"][sym], panels["low"][sym], close[sym], w)
        atr_by_window[w] = frame

    engine = PortfolioEngine(
        sleeves,
        limits=limits,
        pcfg=pcfg,
        arm_cfg=arm_cfg,
        correlation_matrix=correlation_matrix,
        corr_threshold=corr_threshold,
        market_proxy=market_proxy,
        forecast_provider=forecast_provider,
    )

    book = BookState(
        cash=float(initial_equity),
        peak_equity=float(initial_equity),
        month_start_equity=float(initial_equity),
        week_start_equity=float(initial_equity),
        prev_nav=float(initial_equity),
    )

    nav_index: list = []
    nav_records: list = []
    after_tax_records: list = []
    exposure_records: list = []
    turnover_records: dict = {}

    def _period_keys(d: date) -> tuple:
        iso = d.isocalendar()
        return (d.year, d.month), (iso[0], iso[1])

    for d in dates:
        d = _as_date(d)
        month_key, week_key = _period_keys(d)
        # Roll the halt reference levels on a period change (before today's flow).
        if book.month_key != month_key:
            book.month_key = month_key
            book.month_start_equity = book.prev_nav
        if book.week_key != week_key:
            book.week_key = week_key
            book.week_start_equity = book.prev_nav

        # (0) inject the day's DCA contribution into cash.
        flow = ledger.flow_on(d)
        if flow:
            book.cash += flow

        # (2) build the per-day market view + history; delegate the decision.
        bars = DayBars(
            asof=d,
            open=panels["open"].loc[d],
            high=panels["high"].loc[d],
            low=panels["low"].loc[d],
            close=close.loc[d],
            atr={w: atr_by_window[w].loc[d] for w in windows},
        )
        history = DailyHistory(close, d)
        engine.step(d, history, bars, book)

        # (3) mark to today's close -> NAV / after-tax / exposure / turnover.
        pos_value = 0.0
        c_row = close.loc[d]
        for sym, lot in book.lots.items():
            px = c_row.get(sym, np.nan)
            mark = float(px) if (px is not None and np.isfinite(px)) else lot.entry_price
            pos_value += lot.shares * mark
        nav = book.cash + pos_value
        nav_index.append(d)
        nav_records.append(nav)
        after_tax_records.append(nav - book.tax_reserve)
        exposure_records.append((pos_value / nav) if nav > 0 else 0.0)
        if book.turnover_today > 0:
            turnover_records[d] = book.turnover_today / nav if nav > 0 else 0.0

        book.prev_nav = nav
        book.peak_equity = max(book.peak_equity, nav)

    nav_series = pd.Series(nav_records, index=nav_index, name="nav")
    nav_series.index.name = "date"
    after_tax = pd.Series(after_tax_records, index=nav_index, name="after_tax_nav")
    after_tax.index.name = "date"
    exposure_series = pd.Series(exposure_records, index=nav_index, name="exposure")
    exposure_series.index.name = "date"
    turnover_series = pd.Series(turnover_records, dtype="float64").sort_index()
    turnover_series.name = "turnover"

    # Flow-free TWR returns + money-weighted IRR.
    twr = ContributionLedger.twr_daily_returns(nav_series, ledger.contributions)
    mwr = ledger.mwr_irr(nav_series, initial_equity)

    # Per-sleeve attribution: realized (already in sleeve_realized) + terminal
    # unrealized mark-to-market of the still-open lots.
    attribution = dict(book.sleeve_realized)
    last_close = close.loc[dates[-1]]
    for sym, lot in book.lots.items():
        px = last_close.get(sym, np.nan)
        mark = float(px) if (px is not None and np.isfinite(px)) else lot.entry_price
        unrealized = (mark - lot.entry_price) * lot.shares
        attribution[lot.sleeve] = attribution.get(lot.sleeve, 0.0) + unrealized

    return PortfolioResult(
        nav=nav_series,
        after_tax_nav=after_tax,
        initial_equity=float(initial_equity),
        total_costs=book.total_costs,
        tax_reserve=book.tax_reserve,
        realized_gains=book.realized_gains,
        rebalance_turnover=turnover_series,
        short_term_tax_rate=float(pcfg.short_term_tax_rate),
        trades=list(book.closed_trades),
        exposure=exposure_series,
        contributions=dict(ledger.contributions),
        twr_returns=twr,
        sleeve_attribution=attribution,
        total_deposited=ledger.total_deposited(),
        mwr_irr=mwr,
    )
