"""orchestrator/workflows/premarket.py — the premarket sequence (MASTER_PLAN §4).

Runs once before the open, in the SLOW loop. The full sequence is: refresh levels
-> regime read + arm -> WATCHLIST TIERING -> (optional) Kronos forecast batch ->
premarket digest. This module implements the deterministic UNIVERSE step that ties
Phase 2 (the ticker-universe engine) to Phase 1 (the portfolio engine): it runs
the weekly tiering and hands the CORE/ACTIVE set to the portfolio engine as its
tradable ``universe``. The regime/levels/digest steps are owned by their agents
(regime_reader, journalist) and wired in around this.

Determinism: the universe selection here is pure over the weekly
:class:`~watchlist.weekly.WatchlistResult` (which itself is offline over
``market.duckdb`` + the JOINed ``equity_metadata`` table). No MCP in this path —
the Robinhood metadata refresh runs separately, off the hot path
(``watchlist.fetch_metadata.refresh_metadata``).
"""

from __future__ import annotations

from typing import Optional, Sequence

from watchlist.weekly import WatchlistResult, run_weekly


def select_universe(result: WatchlistResult, *, include_active: bool = True) -> list[str]:
    """The tradable set from a weekly run: CORE, plus ACTIVE when ``include_active``.

    CORE is the live-grade, diversified, cluster-deduped few; ACTIVE is the paper
    bench. The portfolio engine "holds only the best few", so it trades the CORE
    (+ ACTIVE) names — the deliberately narrow, liquid, decorrelated slice the
    universe engine curated. Order: CORE first, then ACTIVE (priority order).
    """
    syms = list(result.core())
    if include_active:
        syms.extend(result.active())
    # de-dup, preserve order
    return list(dict.fromkeys(syms))


def _sleeve_symbols(sleeve) -> list[str]:
    """Every symbol a sleeve's strategy can reference (so the engine can price it).

    A weight sleeve must be able to mark every name in its target vector — the GEM
    legs, sectors, safe/inverse/leveraged ETFs (``extra_symbols``) or its own
    ``universe``. A score sleeve trades the watchlist selection itself, so it adds
    nothing here.
    """
    strat = getattr(sleeve, "strategy", sleeve)
    out: list[str] = []
    if hasattr(strat, "extra_symbols") and callable(strat.extra_symbols):
        try:
            out.extend(strat.extra_symbols())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(strat, "universe"):
        try:
            out.extend(list(strat.universe))
        except Exception:  # noqa: BLE001
            pass
    return out


def portfolio_universe(
    result: WatchlistResult,
    sleeves: Sequence,
    *,
    include_active: bool = True,
) -> list[str]:
    """The engine ``universe``: the watchlist selection UNION every sleeve's symbols.

    The score sleeve hunts breakouts across the curated CORE/ACTIVE names; the
    weight sleeves additionally need their structural tickers priceable (rotation's
    sectors / safe assets, mean-reversion's index ETFs). The union of the two is
    exactly what ``run_portfolio`` must load OHLC for. Sorted for determinism.
    """
    syms = set(select_universe(result, include_active=include_active))
    for sleeve in sleeves:
        syms.update(_sleeve_symbols(sleeve))
    return sorted(syms)


def run_kronos_batch(
    con,
    symbols: Sequence[str],
    asof,
    forecaster,
    *,
    timeframe: str = "1d",
    horizon: int = 5,
    n_paths: int = 32,
    lookback: int = 128,
) -> int:
    """Forecast each armed CORE/ACTIVE symbol and write ``kronos_forecasts``.

    The premarket Kronos step (SLOW loop, AFTER regime arm + universe tiering). For
    each symbol it loads the post-cutoff OHLC slice, calls
    ``forecaster.forecast_distribution`` (which enforces the leakage guard), and
    upserts the row. A symbol whose history is contaminated (any pre-cutoff bar) or
    too short is SKIPPED — the guard refuses it rather than producing a leaky
    forecast. Returns the number of forecasts written. ``forecaster`` is injected
    (a :class:`forecast.kronos.predictor.KronosForecaster`, possibly with a
    deterministic sampler), so this is testable without torch.
    """
    from datetime import datetime

    from forecast.kronos.leakage import LeakageError
    from forecast.kronos.store import write_forecast

    asof_d = asof.date() if isinstance(asof, datetime) else asof
    written = 0
    for sym in symbols:
        hist = _load_recent_ohlc(con, sym, asof_d, timeframe, lookback)
        if hist is None or len(hist) == 0:
            continue
        try:
            fc = forecaster.forecast_distribution(
                sym, asof_d, hist, horizon=horizon, n_paths=n_paths
            )
        except LeakageError:
            continue  # contaminated / too-short history -> skip (never leak)
        write_forecast(con, sym, asof_d, fc, timeframe=timeframe,
                       horizon=horizon, n_paths=n_paths,
                       model_id=fc.get("model_id", "kronos-mini"))
        written += 1
    return written


def _load_recent_ohlc(con, symbol, asof, timeframe, lookback):
    """The last ``lookback`` OHLCV bars for ``symbol`` up to ``asof`` (or None)."""
    try:
        df = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars WHERE symbol = ? AND timeframe = ? AND ts_utc <= ?
            ORDER BY ts_utc DESC LIMIT ?
            """,
            [symbol, timeframe, asof, int(lookback)],
        ).df()
    except Exception:  # noqa: BLE001
        return None
    if len(df) == 0:
        return None
    return df.iloc[::-1].reset_index(drop=True)  # back to ascending


def premarket_universe(
    con=None,
    *,
    sleeves: Optional[Sequence] = None,
    criteria: Optional[dict] = None,
    include_active: bool = True,
    persist_result: bool = False,
    **run_weekly_kwargs,
) -> tuple[WatchlistResult, list[str]]:
    """Run the weekly tiering and return ``(WatchlistResult, engine_universe)``.

    The premarket-sequence entry point for the universe step: it executes
    :func:`watchlist.weekly.run_weekly` (which JOINs the off-hot-path metadata) and
    then resolves the engine's tradable universe via :func:`portfolio_universe`
    (or :func:`select_universe` when no ``sleeves`` are supplied). ``persist_result``
    defaults to False so a premarket dry-run doesn't churn the universe DB.
    """
    result = run_weekly(con=con, criteria=criteria, persist_result=persist_result,
                        **run_weekly_kwargs)
    if sleeves:
        universe = portfolio_universe(result, sleeves, include_active=include_active)
    else:
        universe = select_universe(result, include_active=include_active)
    return result, universe
