"""backtest/runner.py — the canonical backtest run harness.

Generalizes ``backtest/run_gate.py`` (which is hard-wired to QQQ 5m + V0) into a
reusable runner the rest of Stage 2 (ablation, walk-forward, edge-portfolio,
correlation/blend) calls. It owns three jobs:

1. :func:`load_bars_levels` — load RTH-filtered bars + a ``levels_by_session``
   map for any (symbol, timeframe), optionally clipped to a [start, end] date
   window. Mirrors ``run_gate.load_qqq_5m`` but parameterized.
2. :func:`run_strategy` — run an *instantiated* engine ``Strategy`` through the
   :class:`~backtest.engine.engine.BacktestEngine` under a named cost profile,
   returning a :class:`~backtest.engine.result.BacktestResult`. Fill realism is
   tied to the cost profile exactly as the gate does (``tv_style`` => optimistic
   TV-parity fills; anything else => model adverse stop gaps).
3. :func:`daily_returns` — the KEY output for downstream correlation/blend:
   a pandas Series of per-session **realized $ PnL**, indexed by the trade's
   exit session_date. Equity-only here (crypto ignored for this stage), so $ PnL
   is the natural unit; a returns variant is also provided.

Strategy argument contract
--------------------------
The engine takes an *already-instantiated* ``Strategy`` (see
``run_gate.run_v0``: ``BreakoutRetestStrategy(params=...)`` is built then handed
to ``BacktestEngine``). ``run_strategy`` accepts EITHER:
  * an instantiated ``Strategy`` (the common case), or
  * a ``(factory, params)`` tuple, where ``factory(params)`` -> ``Strategy``
    (handy for walk-forward, which re-instantiates per fold).
This mirrors what the engine expects and keeps the caller flexible.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Callable

import pandas as pd

from backtest.engine.cost import CostModel
from backtest.engine.engine import BacktestEngine, Strategy, bars_from_df
from backtest.engine.result import BacktestResult
from data.schema import DEFAULT_DB_PATH, connect
from data.sessions import et_session_date, is_rth

# Asset class inferred from the symbol shape ('BTC/USD' => crypto).
def _infer_asset_class(symbol: str) -> str:
    return "crypto" if "/" in symbol else "equity"


def _tick_for(symbol: str, asset_class: str) -> float:
    # Equities trade in pennies; crypto we leave at a fine default (ignored this
    # stage, but kept sensible so the engine's risk-skip logic behaves).
    return 0.01 if asset_class == "equity" else 0.01


def _as_date(d) -> date | None:
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
# Data loading
# --------------------------------------------------------------------------- #
def load_bars_levels(
    symbol: str,
    timeframe: str,
    start=None,
    end=None,
    db_path: str = DEFAULT_DB_PATH,
    con=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (bars_df, levels_df) for one (symbol, timeframe), RTH-filtered.

    Parameters
    ----------
    symbol, timeframe
        e.g. ``"QQQ"``, ``"5m"``.
    start, end
        Optional inclusive date bounds (date / datetime / 'YYYY-MM-DD'); filter
        on the bar's ET session date for equities. ``None`` means unbounded.
    db_path / con
        DB path, or an open connection to reuse (the connection is not closed if
        you pass it in).

    Returns
    -------
    ``(bars_df, levels_df)`` where ``bars_df`` has ``ts_utc, open, high, low,
    close, volume`` (RTH only for equities) and ``levels_df`` has the level
    columns. To get the engine's ``levels_by_session`` map use
    :func:`levels_map`.
    """
    asset_class = _infer_asset_class(symbol)
    own_con = con is None
    if own_con:
        con = connect(db_path)
    try:
        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND timeframe = ?
            ORDER BY ts_utc
            """,
            [symbol, timeframe],
        ).df()
        levels = con.execute(
            """
            SELECT session_date, pdh, pdl, pmh, pml, ntz_low, ntz_high,
                   ntz_valid, atr14
            FROM levels
            WHERE symbol = ?
            ORDER BY session_date
            """,
            [symbol],
        ).df()
    finally:
        if own_con:
            con.close()

    if len(bars) > 0:
        # Equities: keep only RTH bars (crypto has no RTH concept; pass through).
        if asset_class == "equity":
            bars = bars[bars["ts_utc"].apply(is_rth)].reset_index(drop=True)

        s = _as_date(start)
        e = _as_date(end)
        if s is not None or e is not None:
            sess = bars["ts_utc"].apply(et_session_date)
            mask = pd.Series(True, index=bars.index)
            if s is not None:
                mask &= sess >= s
            if e is not None:
                mask &= sess <= e
            bars = bars[mask].reset_index(drop=True)

    return bars, levels


def levels_map(levels_df: pd.DataFrame) -> dict:
    """Convert a levels DataFrame into the engine's ``levels_by_session`` dict.

    Keys are normalized to ``datetime.date`` to match ``et_session_date``.
    """
    out: dict = {}
    for row in levels_df.itertuples(index=False):
        sd = row.session_date
        if hasattr(sd, "date"):
            sd = sd.date()
        out[sd] = {
            "pdh": None if pd.isna(row.pdh) else float(row.pdh),
            "pdl": None if pd.isna(row.pdl) else float(row.pdl),
            "pmh": None if pd.isna(row.pmh) else float(row.pmh),
            "pml": None if pd.isna(row.pml) else float(row.pml),
            "ntz_low": None if pd.isna(row.ntz_low) else float(row.ntz_low),
            "ntz_high": None if pd.isna(row.ntz_high) else float(row.ntz_high),
            "ntz_valid": bool(row.ntz_valid),
            "atr14": None if pd.isna(row.atr14) else float(row.atr14),
        }
    return out


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _resolve_strategy(strategy) -> Strategy:
    """Accept an instantiated Strategy or a ``(factory, params)`` tuple."""
    if isinstance(strategy, Strategy):
        return strategy
    if isinstance(strategy, tuple) and len(strategy) == 2:
        factory, params = strategy
        if not callable(factory):
            raise TypeError("(factory, params): factory must be callable")
        return factory(params)
    raise TypeError(
        "strategy must be an engine Strategy instance or a (factory, params) "
        f"tuple; got {type(strategy)!r}"
    )


def run_strategy(
    strategy,
    symbol: str,
    timeframe: str,
    start=None,
    end=None,
    cost_profile: str = "tv_style",
    db_path: str = DEFAULT_DB_PATH,
    initial_equity: float = 100_000.0,
    percent_of_equity: float = 1.0,
    con=None,
) -> BacktestResult:
    """Run an engine ``Strategy`` and return its :class:`BacktestResult`.

    ``strategy`` is an instantiated engine Strategy (or a ``(factory, params)``
    tuple — see module docstring). Fill realism follows the cost profile, exactly
    like the gate: ``tv_style`` uses TradingView-parity optimistic fills
    (``model_stop_gaps=False``); any other profile models adverse stop gaps.
    """
    strat = _resolve_strategy(strategy)
    asset_class = _infer_asset_class(symbol)

    bars_df, levels_df = load_bars_levels(
        symbol, timeframe, start=start, end=end, db_path=db_path, con=con
    )
    lv_map = levels_map(levels_df)

    cost = CostModel.from_profile(cost_profile)
    engine = BacktestEngine(
        strat,
        cost,
        symbol=symbol,
        asset_class=asset_class,
        tick=_tick_for(symbol, asset_class),
        initial_equity=initial_equity,
        percent_of_equity=percent_of_equity,
        model_stop_gaps=(cost_profile != "tv_style"),
    )
    bars = bars_from_df(bars_df)
    return engine.run(bars, lv_map, lambda bar: et_session_date(bar.ts))


# --------------------------------------------------------------------------- #
# Daily returns — the key downstream output
# --------------------------------------------------------------------------- #
def daily_returns(result: BacktestResult) -> pd.Series:
    """Per-session realized $ PnL, indexed by the trade's EXIT session_date.

    This is the canonical series the correlation/blend stage consumes: each
    trade's net pnl is bucketed onto the ET session date it EXITED, and same-day
    trades are summed. Sessions with no closed trade do not appear (callers that
    need a dense calendar can reindex against the session list).

    Equity-only this stage (crypto ignored), so dollars are the unit. The index
    is ``datetime.date`` and the series is sorted ascending. Empty result ->
    empty float Series named 'daily_pnl'.
    """
    t = result.trades
    if t is None or len(t) == 0:
        return pd.Series(dtype="float64", name="daily_pnl")

    exit_dates = t["exit_ts"].apply(et_session_date)
    s = (
        pd.Series(t["pnl"].astype("float64").to_numpy(), index=exit_dates.to_numpy())
        .groupby(level=0)
        .sum()
        .sort_index()
    )
    s.name = "daily_pnl"
    s.index.name = "session_date"
    return s


def daily_returns_pct(
    result: BacktestResult, base_equity: float | None = None
) -> pd.Series:
    """Per-session realized return as a FRACTION of equity (for Sharpe/regime).

    Divides each session's $ PnL by ``base_equity`` (defaults to the result's
    ``initial_equity``) to give a simple daily return series suitable for
    ``metrics.sharpe`` and ``regime.by_regime``. Same index/empty behavior as
    :func:`daily_returns`.
    """
    pnl = daily_returns(result)
    if len(pnl) == 0:
        return pd.Series(dtype="float64", name="daily_return")
    base = base_equity if base_equity is not None else result.initial_equity
    if not base or base <= 0:
        base = 1.0
    out = pnl / float(base)
    out.name = "daily_return"
    out.index.name = "session_date"
    return out
