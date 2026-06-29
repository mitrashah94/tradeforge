"""orchestrator/agents/performance_analyst.py — the deterministic compute behind
the performance-analyst LLM agent (MASTER_PLAN.md §9 "Metrics That Matter").

The performance-analyst answers one question above all others (§0, §9, §11):

    *Is the system beating SPY after costs, and compounding the GEOMETRIC growth
    rate safely?* — "the system's first job is not dying; its second is beating
    SPY after costs."

This module is the **computable** part of that policy: pure, deterministic,
network-free functions plus a thin :class:`PerformanceAnalyst` orchestrator that
subscribes to the bus, tracks realized trades, and EMITS growth/risk events. It
NEVER touches money and NEVER writes live config.

Research firewall (MASTER_PLAN.md §6, CLAUDE.md "Self-improvement"):
-------------------------------------------------------------------
The analyst is a *research* agent: it **reads everything, writes nothing live.**
Concretely it:
  * READS the paper/backtest ledgers, ``risk/limits.yaml`` (via
    :func:`risk.config.load_limits`), the strategy registry, and SPY bars.
  * EMITS events — ``STRATEGY_DEMOTED`` (decay), ``MILESTONE_REACHED`` +
    ``RATCHET_SWEEP`` (milestone ratchet). A human / the deterministic gate
    *applies* a demotion or a sweep; the analyst only proposes it on the bus.
  * WRITES, at most, report artifacts under the agent-writable ``reports/``
    prefix (see :mod:`orchestrator.agents.firewall`). It NEVER writes
    ``risk/limits.yaml`` or live ``strategies/registry.yaml`` fields.

Reuse (do not re-derive):
  * :mod:`backtest.stats.metrics` — profit_factor, expectancy, max_drawdown.
  * :func:`risk.ratchet.check_ratchet` — the milestone sweep math from P0.
  * :func:`risk.config.load_limits` — ratchet milestones / sweep fraction.

Public API (stable for the journalist's EOD digest + tests)
-----------------------------------------------------------
Pure functions:
    equity_curve(trade_pnls, starting_capital)              -> np.ndarray
    alpha_vs_spy(equity_or_returns, spy_bars, start, end)   -> dict
    geometric_growth(daily_returns)                          -> dict
    after_tax_equity(trade_pnls, starting_capital, rate)     -> dict
    reconcile(live_pnls, backtest_expectation)               -> dict
    rolling_profit_factor(trade_pnls, window=30)             -> list[float]
    risk_of_ruin(win_rate, avg_win_r, avg_loss_r,
                 risk_fraction, ruin_threshold)              -> float

Orchestrator (bus-driven, stateful, emits events):
    PerformanceAnalyst(bus=None, limits=None, ...)
        .on_position_closed(event)   # bus handler
        .check_demotion(strategy)    # -> emits STRATEGY_DEMOTED once on decay
        .check_milestone()           # -> emits MILESTONE_REACHED + RATCHET_SWEEP
        .summary() / .eod_report()   # the headline lines for the digest
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Mapping, Sequence

import numpy as np

from backtest.stats.metrics import (
    _to_array,
    expectancy_dollar,
    max_drawdown,
    profit_factor,
)
from orchestrator.events import Event, EventType
from risk.config import Limits, load_limits
from risk.ratchet import check_ratchet

# Default short-term-gains tax-reserve rate. Taxable account (CLAUDE.md P0
# decision #1): realized gains are ordinary income, so reserve a fraction and
# compound on AFTER-TAX dollars. This is a *reserve* rate, NOT tax advice — the
# real number is the user's marginal bracket. 0.30 is a reasonable default.
DEFAULT_TAX_RESERVE_RATE = 0.30

# Rolling profit-factor window for decay detection (§6 "auto-demote decayed
# strategies LIVE->PAPER (deterministic)"). 30 trades matches the paper->live
# trade-count floor in CLAUDE.md.
DEFAULT_PF_WINDOW = 30

# Default decay threshold. The paper->live gate requires PF >= 1.3; a live PF
# that decays below 1.0 (losing money gross) is unambiguous decay. We demote at
# 1.0 by default — a strategy that has slipped to net-losing over the last
# window is no longer earning its live slot.
DEFAULT_PF_DEMOTE_THRESHOLD = 1.0

TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# 1. Equity curve from realized trades
# --------------------------------------------------------------------------- #
def equity_curve(trade_pnls, starting_capital: float = 1000.0) -> np.ndarray:
    """Cumulative equity curve from a sequence of realized per-trade pnls ($).

    Returns an array of equity *levels* including the opening point, so a curve
    of ``n`` trades has ``n + 1`` points: ``[start, start+pnl0, ...]``. This is
    the level series :func:`backtest.stats.metrics.max_drawdown` consumes.
    """
    pnls = _to_array(trade_pnls)
    curve = np.empty(pnls.size + 1, dtype="float64")
    curve[0] = float(starting_capital)
    if pnls.size:
        curve[1:] = float(starting_capital) + np.cumsum(pnls)
    return curve


# --------------------------------------------------------------------------- #
# helpers: returns <-> levels, SPY buy-and-hold
# --------------------------------------------------------------------------- #
def _total_return_from_levels(levels) -> float:
    """Total (cumulative) simple return from an equity-level series."""
    eq = _to_array(levels)
    if eq.size < 2:
        return 0.0
    first, last = float(eq[0]), float(eq[-1])
    if first == 0.0:
        return 0.0
    return last / first - 1.0


def _looks_like_returns(series) -> bool:
    """Heuristic: a series of *returns* (small, can be negative, near 0) vs a
    series of equity *levels* (positive, typically >> 1).

    We treat the input as a return series when every value is "small" in
    magnitude (|x| < 1, i.e. < ±100% per step) — the natural shape of per-period
    fractional returns — and as levels otherwise. Callers who want to be explicit
    can pass an equity-level array (e.g. from :func:`equity_curve`).
    """
    arr = _to_array(series)
    if arr.size == 0:
        return False
    return bool(np.all(np.abs(arr) < 1.0))


def _compound(returns) -> float:
    """Total return from a per-period simple-return series: prod(1+r) - 1."""
    arr = _to_array(returns)
    if arr.size == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def _coerce_ts(x) -> datetime | None:
    """Best-effort coercion of a date/datetime/ISO-string to a datetime."""
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if isinstance(x, str):
        try:
            return datetime.fromisoformat(x)
        except ValueError:
            return datetime.fromisoformat(x[:10])
    raise TypeError(f"cannot coerce {x!r} to a datetime")


def spy_buy_and_hold_return(spy_bars, start=None, end=None) -> float:
    """SPY buy-&-hold total return over [start, end], close-to-close.

    ``spy_bars`` is any of:
      * a sequence of ``(ts, close)`` pairs or dicts with ``ts_utc``/``close``
        (or ``ts``/``c``) keys — the shape a DuckDB ``bars`` fetch yields;
      * a plain sequence of closing prices (no timestamps; ``start``/``end``
        windowing is then skipped and the full series is used).

    The return is ``last_close / first_close - 1`` over the windowed bars. With
    fewer than two qualifying bars the return is ``0.0`` (no benchmark move).

    SPY buy-&-hold is the §9 headline benchmark; we measure the strategy AFTER
    its own costs against this passive, cost-free baseline, which is the honest
    bar (the alternative use of the same capital).
    """
    closes = _extract_windowed_closes(spy_bars, start, end)
    if len(closes) < 2:
        return 0.0
    first, last = float(closes[0]), float(closes[-1])
    if first == 0.0:
        return 0.0
    return last / first - 1.0


def _extract_windowed_closes(spy_bars, start, end) -> list[float]:
    """Pull (ts-filtered) closing prices out of the flexible ``spy_bars`` input."""
    start_dt = _coerce_ts(start)
    end_dt = _coerce_ts(end)

    rows: list[tuple[datetime | None, float]] = []
    for b in spy_bars:
        ts: datetime | None = None
        close: float
        if isinstance(b, Mapping):
            close = float(b.get("close", b.get("c")))
            raw_ts = b.get("ts_utc", b.get("ts", b.get("timestamp")))
            ts = _coerce_ts(raw_ts) if raw_ts is not None else None
        elif isinstance(b, (tuple, list)) and len(b) >= 2:
            ts = _coerce_ts(b[0])
            close = float(b[1])
        else:  # a bare price
            close = float(b)
        rows.append((ts, close))

    # If timestamps are present, sort + window; otherwise keep input order.
    if rows and rows[0][0] is not None:
        rows.sort(key=lambda r: r[0])
        if start_dt is not None:
            rows = [r for r in rows if r[0] >= start_dt]
        if end_dt is not None:
            rows = [r for r in rows if r[0] <= end_dt]
    return [c for _, c in rows]


# --------------------------------------------------------------------------- #
# 2. ALPHA VS SPY AFTER COSTS — the headline metric (§9)
# --------------------------------------------------------------------------- #
def alpha_vs_spy(equity_or_returns, spy_bars, start=None, end=None) -> dict:
    """Strategy net return minus SPY buy-&-hold over the same window (§9 headline).

    Parameters
    ----------
    equity_or_returns:
        Either an equity-LEVEL series (e.g. from :func:`equity_curve`) or a
        per-period fractional-RETURN series. Auto-detected: an all-|x|<1 series
        is treated as returns and compounded; otherwise it is treated as levels
        and the total return is ``last/first - 1``. The strategy figure is
        assumed to already be NET OF COSTS (the ledger books costs into pnl).
    spy_bars, start, end:
        Passed to :func:`spy_buy_and_hold_return` for the passive benchmark over
        the SAME window.

    Returns
    -------
    dict with ``strategy_return``, ``spy_return``, ``alpha`` (strategy − SPY).
    Positive alpha clears the bar; **alpha ≤ 0 over 6 months live → stop** (§9).
    """
    if _looks_like_returns(equity_or_returns):
        strat = _compound(equity_or_returns)
    else:
        strat = _total_return_from_levels(equity_or_returns)

    spy = spy_buy_and_hold_return(spy_bars, start, end)
    return {
        "strategy_return": float(strat),
        "spy_return": float(spy),
        "alpha": float(strat - spy),
    }


# --------------------------------------------------------------------------- #
# 3. GEOMETRIC growth rate + curve volatility (the §0 maximization target)
# --------------------------------------------------------------------------- #
def geometric_growth(daily_returns, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Geometric (compounded) growth rate + curve volatility from a return series.

    The §0 thesis: compounded growth ``g ≈ mean − variance/2``. We report:
      * ``geo_mean_daily``  — true geometric mean per period:
        ``exp(mean(log(1+r))) - 1``; equivalently ``(prod(1+r))**(1/n) - 1``.
      * ``arith_mean_daily`` — arithmetic mean per period (the numerator term).
      * ``vol_daily``        — stdev of per-period returns (the variance penalty
        source; the "curve volatility" of §9).
      * ``g_approx_daily``   — the ``mean − var/2`` approximation, the quantity
        the system MAXIMIZES (cut variance for a given edge → higher g).
      * ``cagr``             — annualized geometric growth:
        ``(1 + geo_mean_daily)**periods_per_year - 1``.
      * ``vol_annual``       — annualized volatility (``vol_daily * sqrt(P)``).
      * ``n``                — observation count.

    Empty / single-point series → all-zero (``n`` reflects the count). A daily
    return of ``-1`` (total wipeout) yields ``-inf`` log; we guard by clipping
    ``1+r`` at a tiny positive floor so the geometric mean stays finite.
    """
    arr = _to_array(daily_returns)
    n = int(arr.size)
    if n == 0:
        return {
            "geo_mean_daily": 0.0,
            "arith_mean_daily": 0.0,
            "vol_daily": 0.0,
            "g_approx_daily": 0.0,
            "cagr": 0.0,
            "vol_annual": 0.0,
            "n": 0,
        }

    arith = float(np.mean(arr))
    # ddof=1 sample stdev when we have >=2 points; 0.0 dispersion on a single pt.
    vol = float(np.std(arr, ddof=1)) if n >= 2 else 0.0

    growth = np.clip(1.0 + arr, 1e-12, None)  # guard log of non-positive growth
    geo_mean = float(np.exp(np.mean(np.log(growth))) - 1.0)

    g_approx = arith - 0.5 * (vol ** 2)
    cagr = float((1.0 + geo_mean) ** periods_per_year - 1.0)
    vol_annual = float(vol * math.sqrt(periods_per_year))

    return {
        "geo_mean_daily": geo_mean,
        "arith_mean_daily": arith,
        "vol_daily": vol,
        "g_approx_daily": g_approx,
        "cagr": cagr,
        "vol_annual": vol_annual,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# 4. After-tax equity (taxable account; first-class metric per CLAUDE.md §10.1)
# --------------------------------------------------------------------------- #
def after_tax_equity(
    trade_pnls,
    starting_capital: float = 1000.0,
    tax_reserve_rate: float = DEFAULT_TAX_RESERVE_RATE,
) -> dict:
    """Apply a short-term-gains tax reserve to realized GAINS → after-tax equity.

    Taxable account (CLAUDE.md P0 #1): realized gains are short-term/ordinary
    income, so we reserve ``tax_reserve_rate`` of NET realized gains and report
    the equity you actually get to compound. Losses do not generate a reserve
    (and we do not model loss carry-forwards / wash sales — this is a reserve
    line, not a tax return).

    Reserve is computed on the NET realized gain over the period
    (``max(0, sum(pnls))``): if the account is net-down for the period there is
    no gain to tax. Returns a dict with:
      * ``pretax_equity``   — start + sum(pnls).
      * ``realized_gain``   — sum(pnls) (may be negative).
      * ``tax_reserve``     — ``rate * max(0, realized_gain)``.
      * ``aftertax_equity`` — ``pretax_equity - tax_reserve``.
      * ``tax_reserve_rate``.

    CONTRACT — consistency with the daily backtester (backtest/daily/engine.py).
    -----------------------------------------------------------------------------
    Both reservers model the SAME tax policy and must stay aligned:
      * Same default rate: ``DEFAULT_TAX_RESERVE_RATE`` here ==
        ``backtest.daily.engine.DEFAULT_SHORT_TERM_TAX_RATE`` (0.30). The real
        number is the user's marginal bracket; this is a blunt reserve, not advice.
      * Same sign rule: a reserve accrues only on POSITIVE net realized gains and
        is FLOORED AT 0 (a net-loss period reserves nothing; there is never a
        refund / negative reserve).
      * Same headline: ``after-tax = pretax_nav − running_tax_reserve``, and
        compounding/CAGR are reported on after-tax dollars.
    The one DELIBERATE difference is the offset granularity. This trade-ledger
    view nets ALL pnls in the period first, then reserves on the single net
    figure (``max(0, Σpnl)``) — so within a call a loss fully offsets an earlier
    gain. The daily engine accrues the reserve PER REBALANCE on the running
    realized-gain total (losses bank an offset against that running total, never
    below 0), so an interim gain followed by a later loss can leave a residual
    reserve the engine never refunds — the conservative ledger behavior. Over a
    full period that ends net-up by the same total, both converge; the engine is
    simply more conservative on the path. Neither models wash sales or carry-
    forwards. Keep the rate and the gains-only/floor-at-0 rule identical here and
    in the engine if either is ever changed.
    """
    pnls = _to_array(trade_pnls)
    realized = float(pnls.sum()) if pnls.size else 0.0
    pretax = float(starting_capital) + realized
    reserve = float(tax_reserve_rate) * max(0.0, realized)
    return {
        "pretax_equity": pretax,
        "realized_gain": realized,
        "tax_reserve": reserve,
        "aftertax_equity": pretax - reserve,
        "tax_reserve_rate": float(tax_reserve_rate),
    }


def after_tax_equity_curve(
    trade_pnls,
    starting_capital: float = 1000.0,
    tax_reserve_rate: float = DEFAULT_TAX_RESERVE_RATE,
) -> np.ndarray:
    """After-tax equity *curve*: a running tax reserve on the high-water gain.

    Reserves ``rate`` of the running net gain over the starting capital at each
    point (reserve floored at 0 so a drawdown below start carries no reserve).
    ``n`` trades → ``n + 1`` level points, like :func:`equity_curve`.
    """
    pre = equity_curve(trade_pnls, starting_capital)
    gains = np.clip(pre - float(starting_capital), 0.0, None)
    return pre - float(tax_reserve_rate) * gains


# --------------------------------------------------------------------------- #
# 5. Paper-vs-backtest reconciliation — the sim-to-real gap (§5 / §9)
# --------------------------------------------------------------------------- #
def reconcile(
    live_pnls,
    backtest_expectation: Mapping[str, float],
    *,
    pf_drift_tol: float = 0.30,
    expectancy_drift_tol: float = 0.50,
) -> dict:
    """Compare LIVE/paper PF & expectancy to the backtest expectation; flag drift.

    The sim-to-real gap (§5 "backtest = live (no train/serve skew)", §9
    "paper-vs-backtest reconciliation"). A live edge materially below its
    backtest is the early warning that costs/slippage/regime are eating the edge.

    Parameters
    ----------
    live_pnls:
        Sequence of realized per-trade pnls ($) from the paper/live ledger.
    backtest_expectation:
        Mapping with at least ``profit_factor`` and ``expectancy_dollar`` (the
        keys :func:`backtest.stats.metrics.compute_metrics` emits). Either may be
        absent / NaN, in which case that leg's drift is reported as ``None``.
    pf_drift_tol, expectancy_drift_tol:
        Relative shortfall tolerances. ``pf_drift`` is the *relative* shortfall
        ``(expected - live) / expected``; values above the tolerance set the
        corresponding ``*_drift_flag``.

    Returns
    -------
    dict with live/expected PF & expectancy, the relative drifts, per-metric
    flags, an overall ``drift`` boolean, and the live trade count ``n``.
    """
    live = _to_array(live_pnls)
    n = int(live.size)
    live_pf = profit_factor(live)
    live_exp = expectancy_dollar(live)

    exp_pf = backtest_expectation.get("profit_factor")
    exp_exp = backtest_expectation.get("expectancy_dollar")

    def _rel_shortfall(expected, live_val) -> float | None:
        if expected is None or live_val is None:
            return None
        if not np.isfinite(expected) or not np.isfinite(live_val):
            return None
        if expected == 0:
            return None
        return float((expected - live_val) / abs(expected))

    pf_drift = _rel_shortfall(exp_pf, live_pf)
    exp_drift = _rel_shortfall(exp_exp, live_exp)

    pf_flag = pf_drift is not None and pf_drift > pf_drift_tol
    exp_flag = exp_drift is not None and exp_drift > expectancy_drift_tol

    return {
        "n": n,
        "live_pf": live_pf,
        "expected_pf": exp_pf,
        "pf_drift": pf_drift,
        "pf_drift_flag": pf_flag,
        "live_expectancy": live_exp,
        "expected_expectancy": exp_exp,
        "expectancy_drift": exp_drift,
        "expectancy_drift_flag": exp_flag,
        "drift": bool(pf_flag or exp_flag),
    }


# --------------------------------------------------------------------------- #
# 6. Rolling PF(30) decay detection
# --------------------------------------------------------------------------- #
def rolling_profit_factor(trade_pnls, window: int = DEFAULT_PF_WINDOW) -> list[float]:
    """Rolling profit factor over a trailing ``window`` of trades.

    Returns one PF per position where a full ``window`` of trades is available
    (i.e. ``len(pnls) - window + 1`` values; empty if fewer than ``window``
    trades). Each value is :func:`backtest.stats.metrics.profit_factor` over the
    trailing slice — ``inf`` for an all-winners window, ``nan`` for an empty one.
    """
    pnls = _to_array(trade_pnls)
    if pnls.size < window:
        return []
    out: list[float] = []
    for i in range(window, pnls.size + 1):
        out.append(profit_factor(pnls[i - window : i]))
    return out


# --------------------------------------------------------------------------- #
# 8. Estimated RISK OF RUIN at the current RI + measured edge
# --------------------------------------------------------------------------- #
def risk_of_ruin(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    risk_fraction: float,
    ruin_threshold: float = 1.0,
) -> float:
    """Estimated probability of ruin given the measured edge and current risk.

    A standard gambler's-ruin / risk-of-ruin estimate (§9 "estimated risk of
    ruin at current RI and measured edge"). We model the equity process as a
    sequence of i.i.d. per-trade returns expressed as a fraction of equity:
      * a win moves equity by ``+risk_fraction * avg_win_r``  (prob ``win_rate``),
      * a loss moves it by ``-risk_fraction * avg_loss_r``    (prob ``1-win_rate``),
    where ``avg_win_r`` / ``avg_loss_r`` are the average win / loss in R units
    (so ``avg_loss_r = 1`` means the average loser equals the planned 1R risk).

    "Ruin" is drawing the log-equity down by ``ruin_threshold`` in log-units from
    the start (``ruin_threshold = 1`` ≈ losing ~63% of capital; the classic
    "lose it all" framing in a continuous model). Using the log-return random
    walk, the ruin probability is ``exp(-2 * mu * a / sigma^2)`` clipped to
    [0, 1], where ``mu`` / ``sigma`` are the per-trade log-return mean / stdev
    and ``a = ruin_threshold`` is the log-distance to the ruin barrier — the
    standard diffusion approximation to gambler's ruin.

    Behavior (the contract the tests pin):
      * **Non-positive edge** (``mu <= 0``) → ``1.0``: ruin is (asymptotically)
        certain for a fair-or-worse game pressed indefinitely.
      * **Strong edge + small risk fraction** → ``→ 0``.
      * Always in ``[0, 1]``.

    Degenerate inputs (no risk taken, no dispersion, empty edge) collapse
    sensibly: ``risk_fraction <= 0`` → ``0.0`` (you cannot lose what you do not
    risk); a measured loss leg of zero with positive expectancy → ``0.0``.
    """
    p = float(win_rate)
    rf = float(risk_fraction)
    a = float(ruin_threshold)

    # No risk taken or no barrier distance -> cannot be ruined.
    if rf <= 0.0 or a <= 0.0:
        return 0.0
    # Degenerate win-rate.
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0

    win_step = rf * float(avg_win_r)   # fractional equity gain on a win
    loss_step = rf * float(avg_loss_r)  # fractional equity loss on a loss

    # Per-trade log-returns of the two outcomes (multiplicative compounding).
    # Guard a >=100% loss step (log of <=0) by clamping just above wipeout.
    g_win = math.log(max(1e-12, 1.0 + win_step))
    g_loss = math.log(max(1e-12, 1.0 - loss_step))

    mu = p * g_win + (1.0 - p) * g_loss             # mean per-trade log return
    var = (
        p * (g_win - mu) ** 2 + (1.0 - p) * (g_loss - mu) ** 2
    )  # variance of per-trade log return
    sigma2 = var

    # Non-positive drift -> ruin asymptotically certain under indefinite play.
    if mu <= 0.0:
        return 1.0
    # No dispersion with positive drift -> never ruined.
    if sigma2 <= 0.0:
        return 0.0

    # Diffusion approximation to gambler's ruin over a log-distance ``a``.
    ror = math.exp(-2.0 * mu * a / sigma2)
    return float(min(1.0, max(0.0, ror)))


# --------------------------------------------------------------------------- #
# Strategy-level rolling-edge state (for the orchestrator)
# --------------------------------------------------------------------------- #
@dataclass
class StrategyEdge:
    """Per-strategy realized-trade accumulator for rolling decay detection."""

    name: str
    pnls: list[float] = field(default_factory=list)
    r_multiples: list[float] = field(default_factory=list)
    demoted: bool = False

    def record(self, pnl: float, r_multiple: float | None = None) -> None:
        self.pnls.append(float(pnl))
        if r_multiple is not None:
            self.r_multiples.append(float(r_multiple))

    def rolling_pf(self, window: int = DEFAULT_PF_WINDOW) -> float:
        if len(self.pnls) < window:
            return float("nan")
        return profit_factor(self.pnls[-window:])


# --------------------------------------------------------------------------- #
# 9. PerformanceAnalyst — bus-driven orchestrator (emits events, never writes live)
# --------------------------------------------------------------------------- #
class PerformanceAnalyst:
    """Tracks realized trades off the bus and emits growth/risk events.

    Subscribes to ``POSITION_CLOSED`` (realized pnl per strategy) and, optionally,
    ``ORDER_FILLED`` (informational). On each closed position it:
      * accumulates the strategy's realized pnls and the account equity,
      * checks rolling-PF(window) decay → EMITS ``STRATEGY_DEMOTED`` once,
      * checks milestone crossing → EMITS ``MILESTONE_REACHED`` + ``RATCHET_SWEEP``.

    FIREWALL (§6): this object EMITS events and may write ``reports/`` artifacts.
    It NEVER writes ``risk/limits.yaml`` or live ``strategies/registry.yaml``
    fields — a human / the deterministic gate applies a demotion or sweep. The
    analyst only proposes them on the bus.

    The bus is duck-typed: any object with ``publish(Event)`` (and optionally
    ``subscribe``) works, so tests inject a fake recorder bus.
    """

    SOURCE = "performance_analyst"

    def __init__(
        self,
        bus=None,
        limits: Limits | None = None,
        *,
        starting_capital: float | None = None,
        pf_window: int = DEFAULT_PF_WINDOW,
        pf_demote_threshold: float = DEFAULT_PF_DEMOTE_THRESHOLD,
        tax_reserve_rate: float = DEFAULT_TAX_RESERVE_RATE,
        spy_bars=None,
    ):
        self.bus = bus
        self.limits = limits if limits is not None else load_limits()
        self.pf_window = pf_window
        self.pf_demote_threshold = pf_demote_threshold
        self.tax_reserve_rate = tax_reserve_rate
        self.spy_bars = spy_bars

        self.starting_capital = (
            float(starting_capital)
            if starting_capital is not None
            else float(self.limits.ratchet.starting_capital)
        )
        # Ratchet baseline starts at the configured starting capital.
        self.baseline = float(self.limits.ratchet.starting_capital)
        self.vault_balance = 0.0

        self.equity = self.starting_capital
        self.all_pnls: list[float] = []
        self.strategies: dict[str, StrategyEdge] = {}

    # -- bus wiring -------------------------------------------------------- #
    def attach(self, bus=None) -> "PerformanceAnalyst":
        """Subscribe to the bus (POSITION_CLOSED). Returns self for chaining."""
        if bus is not None:
            self.bus = bus
        if self.bus is None or not hasattr(self.bus, "subscribe"):
            return self
        self.bus.subscribe(EventType.POSITION_CLOSED, self.on_position_closed)
        return self

    def _emit(self, type_: EventType, data: dict) -> Event | None:
        """Publish an event on the bus (no-op if no publishable bus)."""
        if self.bus is None or not hasattr(self.bus, "publish"):
            return None
        return self.bus.publish(Event(type=type_, data=data, source=self.SOURCE))

    # -- ingest ------------------------------------------------------------ #
    def _strategy(self, name: str) -> StrategyEdge:
        se = self.strategies.get(name)
        if se is None:
            se = StrategyEdge(name=name)
            self.strategies[name] = se
        return se

    def record_trade(
        self, pnl: float, strategy: str = "unknown", r_multiple: float | None = None
    ) -> None:
        """Record one realized trade and run decay + milestone checks.

        Returns nothing; side effects are equity/state updates and any emitted
        ``STRATEGY_DEMOTED`` / ``MILESTONE_REACHED`` / ``RATCHET_SWEEP`` events.
        """
        pnl = float(pnl)
        self.all_pnls.append(pnl)
        self.equity += pnl
        self._strategy(strategy).record(pnl, r_multiple)

        self.check_demotion(strategy)
        self.check_milestone()

    def on_position_closed(self, event: Event) -> None:
        """Bus handler for ``POSITION_CLOSED``.

        Reads ``realized_pnl`` (the gateway's ``close_position`` field) and the
        optional ``strategy`` / ``r_multiple`` tags from the payload.
        """
        data = event.data or {}
        if "realized_pnl" in data:
            pnl = data["realized_pnl"]
        elif "pnl" in data:
            pnl = data["pnl"]
        else:
            return  # nothing realized to book
        strategy = data.get("strategy") or data.get("strategy_id") or "unknown"
        r_multiple = data.get("r_multiple")
        self.record_trade(pnl, strategy=strategy, r_multiple=r_multiple)

    # -- (6) decay -> STRATEGY_DEMOTED ------------------------------------ #
    def check_demotion(self, strategy: str) -> Event | None:
        """Emit ``STRATEGY_DEMOTED`` exactly once when rolling PF decays.

        Deterministic (§6 allowed self-improvement #2). Fires once per strategy:
        the first time its rolling PF(window) is finite and below the demote
        threshold. The event PROPOSES LIVE→PAPER; a human / the gate applies it
        (firewall — no registry write here).
        """
        se = self.strategies.get(strategy)
        if se is None or se.demoted:
            return None
        pf = se.rolling_pf(self.pf_window)
        if not math.isfinite(pf) or pf >= self.pf_demote_threshold:
            return None
        se.demoted = True
        return self._emit(
            EventType.STRATEGY_DEMOTED,
            {
                "strategy": strategy,
                "from_status": "LIVE",
                "to_status": "PAPER",
                "reason": "rolling_pf_decay",
                "rolling_pf": pf,
                "window": self.pf_window,
                "threshold": self.pf_demote_threshold,
                "n_trades": len(se.pnls),
            },
        )

    # -- (7) milestone -> MILESTONE_REACHED + RATCHET_SWEEP --------------- #
    def check_milestone(self) -> list[Event]:
        """Emit ``MILESTONE_REACHED`` + ``RATCHET_SWEEP`` on crossing a milestone.

        Reuses :func:`risk.ratchet.check_ratchet` (P0). On a trigger:
          1. EMIT ``MILESTONE_REACHED`` ``{milestone, equity, baseline}``.
          2. EMIT ``RATCHET_SWEEP`` ``{milestone, sweep_amount, new_baseline,
             vault_balance}`` where ``sweep_amount = sweep_fraction × gains`` and
             ``new_baseline`` advances to the milestone.
        Advances the baseline + vault so the same milestone never sweeps twice.
        Returns the emitted events (possibly empty). FIREWALL: emits only; the
        actual capital move into the vault is executed by a human / the gate.
        """
        result = check_ratchet(self.equity, self.baseline, self.limits)
        if not result.triggered:
            return []

        emitted: list[Event] = []
        ev1 = self._emit(
            EventType.MILESTONE_REACHED,
            {
                "milestone": result.milestone,
                "equity": self.equity,
                "baseline": self.baseline,
            },
        )
        if ev1 is not None:
            emitted.append(ev1)

        self.vault_balance += result.sweep_amount
        ev2 = self._emit(
            EventType.RATCHET_SWEEP,
            {
                "milestone": result.milestone,
                "sweep_amount": result.sweep_amount,
                "sweep_fraction": self.limits.ratchet.sweep_fraction,
                "new_baseline": result.new_baseline,
                "vault_balance": self.vault_balance,
            },
        )
        if ev2 is not None:
            emitted.append(ev2)

        # Advance the protected baseline AFTER emitting so the same milestone
        # cannot sweep twice (mirrors check_ratchet's baseline-advance contract).
        self.baseline = result.new_baseline
        return emitted

    # -- read-only views --------------------------------------------------- #
    def current_drawdown(self) -> float:
        """Current drawdown from the realized-equity high-water, as a fraction."""
        curve = equity_curve(self.all_pnls, self.starting_capital)
        peak = float(np.max(curve))
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - self.equity) / peak)

    def measured_edge(self, strategy: str | None = None) -> dict:
        """Measured win rate / avg win-R / avg loss-R for risk-of-ruin inputs."""
        if strategy is not None:
            se = self.strategies.get(strategy)
            rs = list(se.r_multiples) if se else []
            pnls = list(se.pnls) if se else []
        else:
            rs = [r for s in self.strategies.values() for r in s.r_multiples]
            pnls = list(self.all_pnls)

        arr = _to_array(pnls)
        n = int(arr.size)
        wr = float((arr > 0).mean()) if n else 0.0

        r = _to_array(rs)
        wins = r[r > 0]
        losses = r[r < 0]
        avg_win_r = float(wins.mean()) if wins.size else 0.0
        avg_loss_r = float(-losses.mean()) if losses.size else 1.0  # 1R default
        return {
            "win_rate": wr,
            "avg_win_r": avg_win_r,
            "avg_loss_r": avg_loss_r,
            "n": n,
        }

    # -- (9) summary / EOD report ----------------------------------------- #
    def summary(self, ri: int | None = None) -> dict:
        """Assemble the headline metric lines for the journalist's EOD digest.

        Returns a plain, JSON-serializable dict with the §9 headline metrics:
        alpha-vs-SPY (if SPY bars are available), geometric growth + curve vol,
        after-tax equity, current drawdown vs the program-abort thresholds, and
        the estimated risk of ruin at the current RI's per-trade risk fraction.
        """
        ri = ri if ri is not None else self.limits.default_ri
        per_trade_pct = self.limits.level(ri).per_trade_pct
        risk_fraction = per_trade_pct / 100.0

        curve = equity_curve(self.all_pnls, self.starting_capital)
        # Per-trade returns as a fraction of the equity at the start of each
        # trade — the natural series for the geometric-growth read.
        rets = np.diff(curve) / curve[:-1] if curve.size > 1 else np.asarray([])
        geo = geometric_growth(rets)

        edge = self.measured_edge()
        ror = risk_of_ruin(
            edge["win_rate"], edge["avg_win_r"], edge["avg_loss_r"], risk_fraction
        )

        at = after_tax_equity(self.all_pnls, self.starting_capital, self.tax_reserve_rate)

        dd = self.current_drawdown()
        abort = self.limits.program_abort
        out = {
            "equity": self.equity,
            "starting_capital": self.starting_capital,
            "n_trades": len(self.all_pnls),
            "baseline": self.baseline,
            "vault_balance": self.vault_balance,
            "ri": ri,
            "risk_fraction": risk_fraction,
            "profit_factor": profit_factor(self.all_pnls),
            "expectancy_dollar": expectancy_dollar(self.all_pnls),
            "geometric_growth": geo,
            "curve_vol_daily": geo["vol_daily"],
            "after_tax": at,
            "current_drawdown": dd,
            "max_drawdown": max_drawdown(curve),
            "abort_monthly_review_pct": abort.monthly_review_drawdown_pct,
            "abort_peak_halt_pct": abort.peak_halt_drawdown_pct,
            "drawdown_breaches_peak_halt": dd * 100.0 >= abort.peak_halt_drawdown_pct,
            "risk_of_ruin": ror,
            "measured_edge": edge,
        }
        if self.spy_bars is not None:
            out["alpha_vs_spy"] = alpha_vs_spy(curve, self.spy_bars)
        return out

    def eod_report(self, ri: int | None = None) -> dict:
        """Alias for :meth:`summary` — the journalist's EOD digest entry point."""
        return self.summary(ri=ri)
