"""watchlist/level_respect.py — the per-strategy FIT score (a mini-backtest).

MASTER_PLAN.md §4: "Universe tiers ... gated by a per-strategy **level-respect
score** (replay 90 days; screening *is* a backtest)." This module IS that
replay. For a given ``(symbol, strategy)`` it runs the strategy through the real
:class:`~backtest.engine.engine.BacktestEngine` over a bounded recent window
(~90 sessions by default) and reduces the result to a single bounded **[0, 1]**
score: how well does this symbol *respect* the strategy's levels — do breaks lead
to clean retests and follow-through (high score), or does it whipsaw and stop you
out (low score)?

WHY A BACKTEST AND NOT A HAND-ROLLED HEURISTIC
----------------------------------------------
The plan is explicit that screening IS a backtest — the same engine that decides
whether an edge is real should decide whether a *symbol* expresses that edge.
Reusing :func:`backtest.runner.run_strategy` means the score reflects the exact
break/retest/stop/target semantics the live strategy uses (Pine-parity fills,
the OCO bracket, EOD-flat), not an approximation that could drift from
production. It is kept bounded and fast by clipping to ~90 sessions and a single
cost profile.

THE SCORE (0 = noise / whipsaw, 1 = clean respect)
--------------------------------------------------
We blend three normalized, sign-meaningful signals from the replay, each squashed
to [0, 1], so a clearly trending / level-respecting series scores high and pure
noise scores low:

  * **win_rate**       — fraction of trades that closed green. A symbol whose
                         breaks follow through wins more retests.
  * **expectancy_R**   — mean R per trade, squashed through a logistic centered at
                         0 (so 0R -> 0.5, positive -> >0.5). This is the core
                         "did respecting the level pay" signal.
  * **profit_factor**  — gross win / gross loss, squashed via PF/(PF+1) (so PF 1
                         -> 0.5, PF 2 -> 0.67, PF→∞ -> 1). Robust to sizing.

A symbol that produces **no trades** over the window cannot be said to respect or
disrespect the levels — it returns ``score = 0.0`` with ``n_trades = 0`` (a clean
"no evidence" the tiering treats as ungradeable, not as a pass). The blend
weights and logistic slope are module constants so they are easy to tune later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from backtest.runner import load_bars_levels, run_strategy
from backtest.stats.metrics import (
    expectancy_r,
    profit_factor,
    trade_pnls,
    trade_r_multiples,
    win_rate,
)
from data.schema import DEFAULT_DB_PATH
from data.sessions import et_session_date

# ---- score blend (sums to 1.0) ----
_W_WIN_RATE = 0.30
_W_EXPECTANCY = 0.45
_W_PROFIT_FACTOR = 0.25

# Logistic slope for the expectancy-R squash (R units). A slope of ~1.5 maps
# +0.5R -> ~0.68 and -0.5R -> ~0.32, a reasonable spread for retest expectancy.
_EXP_R_SLOPE = 1.5

# Default replay window length in sessions (~90 calendar / trading days).
DEFAULT_LOOKBACK_SESSIONS = 90


@dataclass(frozen=True)
class LevelRespectScore:
    """The bounded fit score plus the diagnostics it was built from."""

    symbol: str
    strategy: str
    score: float          # [0, 1]
    n_trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    n_sessions: int       # sessions actually replayed

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "score": self.score,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "expectancy_r": self.expectancy_r,
            "profit_factor": self.profit_factor,
            "n_sessions": self.n_sessions,
        }


def _logistic(x: float, slope: float = 1.0) -> float:
    """Numerically-stable logistic squash to (0, 1)."""
    z = slope * x
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _squash_pf(pf: float) -> float:
    """Map a profit factor to [0, 1] via PF/(PF+1). PF 1 -> 0.5, inf -> 1."""
    if pf != pf:           # nan (no trades)
        return 0.0
    if math.isinf(pf):
        return 1.0
    if pf <= 0:
        return 0.0
    return pf / (pf + 1.0)


def score_from_arrays(pnls, r_multiples) -> tuple[float, dict]:
    """Compute the bounded [0,1] score directly from per-trade arrays.

    Exposed separately from the replay so the math is unit-testable without a DB
    (feed synthetic trade arrays). Returns ``(score, diagnostics)``. With no
    trades, returns ``(0.0, {...})``.
    """
    n = int(len(list(pnls)))
    if n == 0:
        return 0.0, {
            "n_trades": 0,
            "win_rate": float("nan"),
            "expectancy_r": float("nan"),
            "profit_factor": float("nan"),
        }
    wr = win_rate(pnls)
    er = expectancy_r(r_multiples)
    pf = profit_factor(pnls)

    s_wr = float(wr) if wr == wr else 0.0
    s_er = _logistic(float(er) if er == er else 0.0, _EXP_R_SLOPE)
    s_pf = _squash_pf(pf)

    score = _W_WIN_RATE * s_wr + _W_EXPECTANCY * s_er + _W_PROFIT_FACTOR * s_pf
    # Clamp defensively into [0, 1].
    score = max(0.0, min(1.0, float(score)))
    return score, {
        "n_trades": n,
        "win_rate": float(wr),
        "expectancy_r": float(er),
        "profit_factor": float(pf),
    }


def _last_n_sessions_start(levels_df: pd.DataFrame, lookback: int):
    """Return the session_date that starts the last ``lookback`` sessions.

    Uses the levels table's session_date list (one row per session). Returns
    ``None`` (unbounded) if there are fewer than ``lookback`` sessions.
    """
    if levels_df is None or len(levels_df) == 0:
        return None
    sds = sorted(
        (d.date() if hasattr(d, "date") else d)
        for d in levels_df["session_date"].tolist()
    )
    if len(sds) <= lookback:
        return None
    return sds[-lookback]


def level_respect_score(
    symbol: str,
    strategy,
    timeframe: str = "5m",
    strategy_name: str | None = None,
    lookback_sessions: int = DEFAULT_LOOKBACK_SESSIONS,
    cost_profile: str = "realistic",
    db_path: str = DEFAULT_DB_PATH,
    con=None,
) -> LevelRespectScore:
    """Replay ~``lookback_sessions`` of ``symbol`` through ``strategy`` and score it.

    Parameters
    ----------
    symbol, timeframe
        e.g. ``"SPY"``, ``"5m"``.
    strategy
        An instantiated engine ``Strategy`` OR a ``(factory, params)`` tuple —
        exactly what :func:`backtest.runner.run_strategy` accepts.
    strategy_name
        Label recorded on the score (defaults to the strategy's class name).
    lookback_sessions
        Window length in sessions; the replay is clipped to the last N sessions
        for speed/recency (bounded mini-backtest, MASTER_PLAN §4).
    cost_profile
        ``"realistic"`` by default — the honest small-account picture (the score
        should reward symbols that respect levels *net of costs*).

    Returns a :class:`LevelRespectScore` in [0, 1]. A symbol with no levels/bars
    or no trades scores 0.0 (ungradeable / no evidence).
    """
    name = strategy_name or (
        strategy.__class__.__name__
        if not isinstance(strategy, tuple)
        else "strategy"
    )

    # Find the start date that bounds the replay to the last N sessions.
    _, levels_df = load_bars_levels(
        symbol, timeframe, db_path=db_path, con=con
    )
    start = _last_n_sessions_start(levels_df, lookback_sessions)

    result = run_strategy(
        strategy,
        symbol,
        timeframe,
        start=start,
        cost_profile=cost_profile,
        db_path=db_path,
        con=con,
    )

    pnls = trade_pnls(result)
    rmults = trade_r_multiples(result)
    score, diag = score_from_arrays(pnls, rmults)

    # Sessions actually replayed (distinct exit session dates is a fine proxy;
    # fall back to the levels window length).
    if result.trades is not None and len(result.trades) > 0:
        n_sessions = int(
            result.trades["exit_ts"].apply(et_session_date).nunique()
        )
    else:
        n_sessions = (
            min(lookback_sessions, len(levels_df)) if levels_df is not None else 0
        )

    return LevelRespectScore(
        symbol=symbol,
        strategy=name,
        score=score,
        n_trades=diag["n_trades"],
        win_rate=diag["win_rate"],
        expectancy_r=diag["expectancy_r"],
        profit_factor=diag["profit_factor"],
        n_sessions=n_sessions,
    )


def score_synthetic_bars(
    bars: pd.DataFrame,
    levels_by_session: dict,
    strategy,
    symbol: str = "SYN",
    asset_class: str = "equity",
    strategy_name: str | None = None,
    cost_profile: str = "tv_style",
    session_of=None,
) -> LevelRespectScore:
    """Score an in-memory synthetic series (no DB) for unit tests.

    Runs the engine directly on caller-supplied bars + a ``levels_by_session``
    map, so a test can feed a clean trending/level-respecting frame and a noisy
    one and assert the former scores higher. ``session_of`` defaults to the ET
    session date of each bar.
    """
    from backtest.engine.cost import CostModel
    from backtest.engine.engine import BacktestEngine, bars_from_df
    from backtest.runner import _resolve_strategy

    strat = _resolve_strategy(strategy)
    name = strategy_name or strat.__class__.__name__
    session_of = session_of or (lambda bar: et_session_date(bar.ts))

    cost = CostModel.from_profile(cost_profile)
    engine = BacktestEngine(
        strat,
        cost,
        symbol=symbol,
        asset_class=asset_class,
        tick=0.01,
        model_stop_gaps=(cost_profile != "tv_style"),
    )
    result = engine.run(bars_from_df(bars), levels_by_session, session_of)

    pnls = trade_pnls(result)
    rmults = trade_r_multiples(result)
    score, diag = score_from_arrays(pnls, rmults)
    n_sessions = len(levels_by_session) if levels_by_session else 0
    return LevelRespectScore(
        symbol=symbol,
        strategy=name,
        score=score,
        n_trades=diag["n_trades"],
        win_rate=diag["win_rate"],
        expectancy_r=diag["expectancy_r"],
        profit_factor=diag["profit_factor"],
        n_sessions=n_sessions,
    )
