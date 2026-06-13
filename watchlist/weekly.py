"""watchlist/weekly.py — the WEEKLY watchlist runner (a deferred-agent stand-in).

MASTER_PLAN.md §4 defers the ``watchlist-curator`` LLM agent and says it "starts
as a weekly script" — this is that script. Once a week it:

  1. **Builds the candidate universe** from the screeners
     (:mod:`watchlist.screeners`): every symbol present in ``market.duckdb`` is
     reduced to a liquidity snapshot and filtered against the per-tier thresholds
     in ``criteria.yaml`` (min dollar-volume, min price). Unusual-volume flags are
     attached as a priority hint.
  2. **Scores each (symbol, strategy)** with the per-strategy level-respect score
     (:mod:`watchlist.level_respect`) — a bounded [0, 1] mini-backtest of how well
     the symbol respects that strategy's levels over ~90 sessions. A symbol's
     headline fit score is the MAX across the configured strategies (it earns a
     tier if it fits *any* live strategy well).
  3. **Assigns CORE / ACTIVE / SCOUT tiers** (CORE 2–4 live-grade, ACTIVE ≤8
     paper, SCOUT ~25 observe), each gated by the level-respect score and capped
     by the tier's ``max_symbols``.
  4. **Runs a CORRELATION CHECK on CORE** so it stays genuinely diversified: a
     candidate is admitted to CORE only if its daily-return correlation to every
     current CORE member is below ``core_max_correlation`` — otherwise it is
     demoted to ACTIVE (don't fill CORE with three SPY proxies, §4). Correlation
     uses daily returns computed from each symbol's own bars (price returns), not
     strategy PnL, so it measures genuine market co-movement.
  5. **Persists** the run to ``watchlist/universe.duckdb`` (tables: ``symbols``,
     ``tiers``, ``scores``, ``correlation``) and **emits** ``WATCHLIST_UPDATED`` +
     per-symbol ``SYMBOL_PROMOTED`` / ``SYMBOL_DEMOTED`` events on an injected bus
     (optional — the script runs fine with ``bus=None``).

Determinism / offline: everything reads from the supplied DuckDB connection; no
network. The screeners and scorer are pure over the loaded bars.

CLI:  PYTHONPATH=. .venv/bin/python -m watchlist.weekly
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.schema import DEFAULT_DB_PATH, connect
from data.sessions import crypto_session_date, et_session_date
from watchlist.level_respect import LevelRespectScore, level_respect_score
from watchlist.screeners import (
    SymbolStats,
    screen_crypto,
    screen_stocks,
    summarize_symbol,
    unusual_volume_flags,
)

CRITERIA_PATH = Path(__file__).resolve().parent / "criteria.yaml"
UNIVERSE_DB_PATH = "watchlist/universe.duckdb"

TIER_ORDER = ["CORE", "ACTIVE", "SCOUT"]
_TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}  # CORE=0 (highest)


# --------------------------------------------------------------------------- #
# Default strategy set to score against
# --------------------------------------------------------------------------- #
def default_strategies() -> dict:
    """The strategies the weekly score replays each symbol against.

    Returns ``{name -> (factory, params)}`` — the ``(factory, params)`` tuple
    form the runner/level_respect accept. Uses each strategy's headline variant
    (breakout_retest -> v0_atr_stop, the realistic-cost survivor; the complements
    -> DEFAULT), matching ``strategies/registry.yaml``.
    """
    from strategies.breakout_retest.strategy import (
        BreakoutRetestStrategy,
        load_params as br_load,
    )
    from strategies.level_meanrev.strategy import (
        LevelMeanRevStrategy,
        load_params as lmr_load,
    )
    from strategies.momentum_thrust.strategy import (
        MomentumThrustStrategy,
        load_params as mt_load,
    )

    return {
        "breakout_retest": (
            lambda p: BreakoutRetestStrategy(params=p),
            br_load("v0_atr_stop"),
        ),
        "level_meanrev": (
            lambda p: LevelMeanRevStrategy(params=p),
            lmr_load("DEFAULT"),
        ),
        "momentum_thrust": (
            lambda p: MomentumThrustStrategy(params=p),
            mt_load("DEFAULT"),
        ),
    }


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_criteria(path: str | Path = CRITERIA_PATH) -> dict:
    """Load the tier thresholds + gates from ``criteria.yaml``."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class SymbolEntry:
    """One symbol's full weekly read: liquidity, fit, tier, rvol."""

    symbol: str
    asset_class: str
    tier: str | None                       # CORE | ACTIVE | SCOUT | None
    fit_score: float                       # max level-respect score across strats
    best_strategy: str | None
    avg_dollar_volume: float
    last_price: float
    rvol: float
    scores: dict = field(default_factory=dict)   # {strategy -> score float}
    reason: str = ""                       # why this tier (audit trail)


@dataclass
class WatchlistResult:
    """The full weekly run, returned by :func:`run_weekly`."""

    run_ts: datetime
    entries: list[SymbolEntry]
    correlation: pd.DataFrame              # daily-return corr of selected symbols
    promoted: list[tuple[str, str]] = field(default_factory=list)  # (sym, tier)
    demoted: list[tuple[str, str]] = field(default_factory=list)
    criteria: dict = field(default_factory=dict)

    def by_tier(self, tier: str) -> list[str]:
        return [e.symbol for e in self.entries if e.tier == tier]

    def core(self) -> list[str]:
        return self.by_tier("CORE")

    def active(self) -> list[str]:
        return self.by_tier("ACTIVE")

    def scout(self) -> list[str]:
        return self.by_tier("SCOUT")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _infer_asset_class(symbol: str) -> str:
    return "crypto" if "/" in symbol else "equity"


def load_universe_bars(con, timeframe: str = "5m") -> dict[str, pd.DataFrame]:
    """Load ``{symbol -> OHLCV bars}`` for every symbol at ``timeframe``.

    The candidate universe is "whatever is in ``market.duckdb``" (no network).
    """
    symbols = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT symbol FROM bars WHERE timeframe = ? ORDER BY symbol",
            [timeframe],
        ).fetchall()
    ]
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars WHERE symbol = ? AND timeframe = ? ORDER BY ts_utc
            """,
            [sym, timeframe],
        ).df()
        if len(df) > 0:
            out[sym] = df
    return out


# --------------------------------------------------------------------------- #
# Daily price returns (for the correlation check)
# --------------------------------------------------------------------------- #
def daily_price_returns(bars: pd.DataFrame, asset_class: str) -> pd.Series:
    """Per-session close-to-close fractional return, indexed by session date.

    Built from the symbol's own bars (the session's last close), so the
    correlation check measures genuine market co-movement between candidates —
    the thing that decides whether two CORE names are really diversified.
    """
    if bars is None or len(bars) == 0:
        return pd.Series(dtype="float64")
    df = bars.copy()
    sess_fn = crypto_session_date if asset_class == "crypto" else et_session_date
    df["_sd"] = df["ts_utc"].apply(sess_fn)
    closes = df.groupby("_sd")["close"].last().astype("float64").sort_index()
    rets = closes.pct_change().dropna()
    rets.index.name = "session_date"
    rets.name = "ret"
    return rets


def returns_correlation(
    symbols: list[str], bars_by_symbol: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily price returns over the union calendar.

    Aligns each symbol's daily-return series on the union of session dates (inner
    join via pandas DataFrame, so only sessions both traded contribute — the
    natural definition of co-movement). Empty / single-symbol input returns a
    trivial frame.
    """
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        bars = bars_by_symbol.get(sym)
        if bars is None:
            continue
        r = daily_price_returns(bars, _infer_asset_class(sym))
        if len(r) > 0:
            r.index = [d for d in r.index]
            series[sym] = r
    if not series:
        return pd.DataFrame()
    mat = pd.DataFrame(series)
    return mat.corr()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_symbol_all_strategies(
    symbol: str,
    strategies: dict,
    timeframe: str,
    lookback_sessions: int,
    cost_profile: str,
    con,
) -> dict[str, LevelRespectScore]:
    """Score one symbol against every strategy; return ``{name -> score}``.

    Equity-only level-respect (the engine/runner is RTH/equity this stage); a
    crypto symbol short-circuits to an empty dict so it can still be SCOUT/ACTIVE
    on liquidity but is not scored against the equity strategies.
    """
    out: dict[str, LevelRespectScore] = {}
    if _infer_asset_class(symbol) == "crypto":
        return out
    for name, strat in strategies.items():
        try:
            sc = level_respect_score(
                symbol,
                strat,
                timeframe=timeframe,
                strategy_name=name,
                lookback_sessions=lookback_sessions,
                cost_profile=cost_profile,
                con=con,
            )
        except Exception:  # noqa: BLE001 — a degenerate symbol must not kill the run
            continue
        out[name] = sc
    return out


# --------------------------------------------------------------------------- #
# Tiering + correlation check
# --------------------------------------------------------------------------- #
def assign_tiers(
    candidates: list[SymbolEntry],
    criteria: dict,
    bars_by_symbol: dict[str, pd.DataFrame],
) -> tuple[list[SymbolEntry], pd.DataFrame]:
    """Assign CORE/ACTIVE/SCOUT tiers with the CORE correlation check.

    Candidates are processed best-fit-first. For each tier (CORE, then ACTIVE,
    then SCOUT) we admit symbols that:
      * clear the tier's liquidity/price thresholds, AND
      * clear the tier's ``require_score`` level-respect gate, AND
      * for CORE only: have daily-return correlation BELOW
        ``core_max_correlation`` to every already-admitted CORE member.
    A CORE candidate rejected by the correlation check (or by a full CORE) falls
    through to ACTIVE. ACTIVE overflow falls through to SCOUT. Each tier is capped
    at its ``max_symbols``.

    Returns ``(entries, core_corr)`` where ``core_corr`` is the correlation
    matrix of the symbols that ended up in CORE (for persistence/inspection).
    """
    tiers_cfg = criteria.get("tiers", {})
    core_max_corr = float(criteria.get("core_max_correlation", 0.85))
    default_gate = float(criteria.get("require_level_respect_score", 0.0))
    do_corr_check = bool(criteria.get("require_correlation_check", True))

    # Best fit first; tie-break by liquidity so a more-liquid name wins a slot.
    ordered = sorted(
        candidates,
        key=lambda e: (e.fit_score, e.avg_dollar_volume),
        reverse=True,
    )

    assigned: dict[str, SymbolEntry] = {}
    core_members: list[str] = []

    def _clears(entry: SymbolEntry, cfg: dict) -> bool:
        return (
            entry.avg_dollar_volume >= float(cfg.get("min_dollar_volume", 0))
            and entry.last_price >= float(cfg.get("min_price", 0))
        )

    def _gate_for(cfg: dict) -> float:
        return float(cfg.get("require_score", default_gate))

    # ---- CORE: gated, capped, AND correlation-checked ----
    core_cfg = tiers_cfg.get("CORE", {})
    core_cap = int(core_cfg.get("max_symbols", 4))
    core_gate = _gate_for(core_cfg)
    for entry in ordered:
        if len(core_members) >= core_cap:
            break
        if entry.symbol in assigned:
            continue
        if not _clears(entry, core_cfg):
            continue
        if entry.fit_score < core_gate:
            continue
        # Correlation check vs current CORE members.
        if do_corr_check and core_members:
            max_corr = _max_corr_to(entry.symbol, core_members, bars_by_symbol)
            if max_corr is not None and max_corr > core_max_corr:
                entry.reason = (
                    f"CORE rejected: corr {max_corr:.2f} > "
                    f"{core_max_corr:.2f} to existing CORE"
                )
                continue  # falls through to ACTIVE below
        entry.tier = "CORE"
        entry.reason = entry.reason or f"CORE: fit {entry.fit_score:.2f}"
        assigned[entry.symbol] = entry
        core_members.append(entry.symbol)

    # ---- ACTIVE: gated, capped (CORE rejects land here first) ----
    active_cfg = tiers_cfg.get("ACTIVE", {})
    active_cap = int(active_cfg.get("max_symbols", 8))
    active_gate = _gate_for(active_cfg)
    n_active = 0
    for entry in ordered:
        if n_active >= active_cap:
            break
        if entry.symbol in assigned:
            continue
        if not _clears(entry, active_cfg):
            continue
        if entry.fit_score < active_gate:
            continue
        entry.tier = "ACTIVE"
        if not entry.reason or entry.reason.startswith("CORE rejected"):
            entry.reason = (
                (entry.reason + "; " if entry.reason else "")
                + f"ACTIVE: fit {entry.fit_score:.2f}"
            )
        assigned[entry.symbol] = entry
        n_active += 1

    # ---- SCOUT: gated, capped (observe-only tail) ----
    scout_cfg = tiers_cfg.get("SCOUT", {})
    scout_cap = int(scout_cfg.get("max_symbols", 25))
    scout_gate = _gate_for(scout_cfg)
    n_scout = 0
    for entry in ordered:
        if n_scout >= scout_cap:
            break
        if entry.symbol in assigned:
            continue
        if not _clears(entry, scout_cfg):
            continue
        if entry.fit_score < scout_gate:
            continue
        entry.tier = "SCOUT"
        entry.reason = entry.reason or f"SCOUT: fit {entry.fit_score:.2f}"
        assigned[entry.symbol] = entry
        n_scout += 1

    # Symbols that found no tier keep tier=None.
    core_corr = returns_correlation(core_members, bars_by_symbol)
    return ordered, core_corr


def _max_corr_to(
    symbol: str, others: list[str], bars_by_symbol: dict[str, pd.DataFrame]
) -> float | None:
    """Max absolute-but-signed daily-return correlation of ``symbol`` to ``others``.

    Returns the maximum (most positive) pairwise correlation, or ``None`` if it
    cannot be computed (no overlapping returns). We gate on the most-positive
    correlation because two CORE names moving together is the concentration risk
    we are trying to avoid (a strongly NEGATIVE pair is *good* diversification).
    """
    corr = returns_correlation([symbol] + list(others), bars_by_symbol)
    if corr.empty or symbol not in corr.columns:
        return None
    vals = [
        float(corr.loc[symbol, o])
        for o in others
        if o in corr.columns and not pd.isna(corr.loc[symbol, o])
    ]
    if not vals:
        return None
    return max(vals)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def init_universe_schema(con) -> None:
    """Create the universe.duckdb tables if absent."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            run_ts            TIMESTAMP,
            symbol            VARCHAR,
            asset_class       VARCHAR,
            avg_dollar_volume DOUBLE,
            last_price        DOUBLE,
            rvol              DOUBLE,
            PRIMARY KEY (run_ts, symbol)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tiers (
            run_ts     TIMESTAMP,
            symbol     VARCHAR,
            tier       VARCHAR,
            fit_score  DOUBLE,
            best_strategy VARCHAR,
            reason     VARCHAR,
            PRIMARY KEY (run_ts, symbol)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            run_ts    TIMESTAMP,
            symbol    VARCHAR,
            strategy  VARCHAR,
            score     DOUBLE,
            n_trades  BIGINT,
            win_rate  DOUBLE,
            expectancy_r DOUBLE,
            profit_factor DOUBLE,
            PRIMARY KEY (run_ts, symbol, strategy)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS correlation (
            run_ts  TIMESTAMP,
            symbol_a VARCHAR,
            symbol_b VARCHAR,
            corr    DOUBLE,
            PRIMARY KEY (run_ts, symbol_a, symbol_b)
        )
        """
    )


def persist(con, result: WatchlistResult, raw_scores: dict) -> None:
    """Write the run to universe.duckdb (symbols, tiers, scores, correlation)."""
    init_universe_schema(con)
    rt = result.run_ts

    for e in result.entries:
        con.execute(
            "INSERT OR REPLACE INTO symbols VALUES (?, ?, ?, ?, ?, ?)",
            [rt, e.symbol, e.asset_class, e.avg_dollar_volume, e.last_price, e.rvol],
        )
        con.execute(
            "INSERT OR REPLACE INTO tiers VALUES (?, ?, ?, ?, ?, ?)",
            [rt, e.symbol, e.tier, e.fit_score, e.best_strategy, e.reason],
        )

    for sym, by_strat in raw_scores.items():
        for name, sc in by_strat.items():
            con.execute(
                "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    rt, sym, name, sc.score, sc.n_trades,
                    _nan_to_none(sc.win_rate), _nan_to_none(sc.expectancy_r),
                    _nan_to_none(sc.profit_factor),
                ],
            )

    corr = result.correlation
    if corr is not None and not corr.empty:
        for a in corr.columns:
            for b in corr.columns:
                v = corr.loc[a, b]
                con.execute(
                    "INSERT OR REPLACE INTO correlation VALUES (?, ?, ?, ?)",
                    [rt, a, b, _nan_to_none(float(v))],
                )


def _nan_to_none(x):
    try:
        return None if x != x else float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Event emission
# --------------------------------------------------------------------------- #
def _prior_tiers(con) -> dict[str, str]:
    """Most-recent prior tier assignment per symbol (for promote/demote diff)."""
    try:
        rows = con.execute(
            """
            SELECT symbol, tier FROM tiers
            WHERE run_ts = (SELECT MAX(run_ts) FROM tiers)
            """
        ).fetchall()
    except Exception:  # noqa: BLE001 — table may not exist yet
        return {}
    return {sym: tier for sym, tier in rows if tier is not None}


def _emit_events(bus, result: WatchlistResult) -> None:
    """Emit WATCHLIST_UPDATED + per-symbol SYMBOL_PROMOTED/DEMOTED on ``bus``."""
    if bus is None:
        return
    from orchestrator.events import Event, EventType

    bus.publish(
        Event(
            type=EventType.WATCHLIST_UPDATED,
            source="watchlist.weekly",
            data={
                "run_ts": result.run_ts.isoformat(),
                "core": result.core(),
                "active": result.active(),
                "scout": result.scout(),
                "n_symbols": len(result.entries),
            },
        )
    )
    for sym, tier in result.promoted:
        bus.publish(
            Event(
                type=EventType.SYMBOL_PROMOTED,
                source="watchlist.weekly",
                data={"symbol": sym, "tier": tier},
            )
        )
    for sym, tier in result.demoted:
        bus.publish(
            Event(
                type=EventType.SYMBOL_DEMOTED,
                source="watchlist.weekly",
                data={"symbol": sym, "tier": tier},
            )
        )


# --------------------------------------------------------------------------- #
# The weekly runner
# --------------------------------------------------------------------------- #
def run_weekly(
    con=None,
    bus=None,
    strategies: dict | None = None,
    timeframe: str = "5m",
    lookback_sessions: int = 90,
    cost_profile: str = "realistic",
    criteria: dict | None = None,
    db_path: str = DEFAULT_DB_PATH,
    universe_db_path: str = UNIVERSE_DB_PATH,
    persist_result: bool = True,
    run_ts: datetime | None = None,
) -> WatchlistResult:
    """Build the universe, score it, tier it (with the CORE correlation check),
    persist, and emit events. Returns a :class:`WatchlistResult`.

    Parameters
    ----------
    con
        Open ``market.duckdb`` connection (reused for screening + scoring). If
        ``None``, one is opened on ``db_path`` and closed before returning.
    bus
        Optional event bus (anything with ``publish``). If given, emits
        WATCHLIST_UPDATED + SYMBOL_PROMOTED/DEMOTED.
    strategies
        ``{name -> (factory, params)}`` to score against (defaults to
        :func:`default_strategies`).
    persist_result
        If True (default), writes to ``universe_db_path``. Promote/demote events
        diff against the most-recent prior tiers stored there.
    """
    own_con = con is None
    if own_con:
        con = connect(db_path)
    criteria = criteria if criteria is not None else load_criteria()
    strategies = strategies if strategies is not None else default_strategies()
    run_ts = run_ts or datetime.utcnow()

    try:
        # --- 1. candidate universe (screened) ---
        bars_by_symbol = load_universe_bars(con, timeframe=timeframe)

        eq_bars = {s: b for s, b in bars_by_symbol.items()
                   if _infer_asset_class(s) == "equity"}
        cx_bars = {s: b for s, b in bars_by_symbol.items()
                   if _infer_asset_class(s) == "crypto"}

        # Screen against the WIDEST tier (SCOUT) thresholds to form the candidate
        # pool; per-tier gates are applied again in assign_tiers.
        scout_cfg = criteria.get("tiers", {}).get("SCOUT", {})
        sc_dv = float(scout_cfg.get("min_dollar_volume", 0))
        sc_px = float(scout_cfg.get("min_price", 0))
        eq_pass = {s.symbol for s in screen_stocks(eq_bars, sc_dv, sc_px)}
        cx_pass = {s.symbol for s in screen_crypto(cx_bars, sc_dv, sc_px)}
        candidate_syms = sorted(eq_pass | cx_pass)

        # rvol hints
        rvol_map: dict[str, float] = {}
        for flag in unusual_volume_flags(eq_bars, "equity"):
            rvol_map[flag.symbol] = flag.rvol
        for flag in unusual_volume_flags(cx_bars, "crypto"):
            rvol_map[flag.symbol] = flag.rvol

        # --- 2. score each candidate against every strategy ---
        raw_scores: dict[str, dict] = {}
        candidates: list[SymbolEntry] = []
        for sym in candidate_syms:
            stats = summarize_symbol(
                sym, bars_by_symbol[sym], _infer_asset_class(sym)
            )
            by_strat = score_symbol_all_strategies(
                sym, strategies, timeframe, lookback_sessions, cost_profile, con
            )
            raw_scores[sym] = by_strat
            if by_strat:
                best_name = max(by_strat, key=lambda n: by_strat[n].score)
                best_score = by_strat[best_name].score
            else:
                best_name, best_score = None, 0.0
            candidates.append(
                SymbolEntry(
                    symbol=sym,
                    asset_class=_infer_asset_class(sym),
                    tier=None,
                    fit_score=best_score,
                    best_strategy=best_name,
                    avg_dollar_volume=stats.avg_dollar_volume if stats else 0.0,
                    last_price=stats.last_price if stats else 0.0,
                    rvol=rvol_map.get(sym, float("nan")),
                    scores={n: s.score for n, s in by_strat.items()},
                )
            )

        # --- 3+4. tier assignment with the CORE correlation check ---
        entries, core_corr = assign_tiers(candidates, criteria, bars_by_symbol)

        # --- diff vs prior run for promote/demote ---
        prior = _prior_tiers(connect(universe_db_path)) if (
            persist_result and os.path.exists(universe_db_path)
        ) else {}
        promoted, demoted = _diff_tiers(prior, entries)

        result = WatchlistResult(
            run_ts=run_ts,
            entries=entries,
            correlation=core_corr,
            promoted=promoted,
            demoted=demoted,
            criteria=criteria,
        )

        # --- 5. persist + emit ---
        if persist_result:
            ucon = connect(universe_db_path)
            try:
                persist(ucon, result, raw_scores)
            finally:
                ucon.close()
        _emit_events(bus, result)

        return result
    finally:
        if own_con:
            con.close()


def _diff_tiers(
    prior: dict[str, str], entries: list[SymbolEntry]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Compare prior vs new tiers -> (promoted, demoted) lists of (symbol, tier).

    A symbol is PROMOTED if its new tier is strictly higher (CORE>ACTIVE>SCOUT>
    none) than its prior tier, DEMOTED if strictly lower (including dropping out
    of all tiers). New symbols entering any tier count as promotions.
    """
    promoted: list[tuple[str, str]] = []
    demoted: list[tuple[str, str]] = []

    def rank(tier: str | None) -> int:
        # Lower number = higher tier; None (untiered) is worst.
        return _TIER_RANK.get(tier, len(TIER_ORDER)) if tier else len(TIER_ORDER)

    new_tiers = {e.symbol: e.tier for e in entries}
    all_syms = set(prior) | set(new_tiers)
    for sym in sorted(all_syms):
        old = prior.get(sym)
        new = new_tiers.get(sym)
        if rank(new) < rank(old):
            promoted.append((sym, new or "NONE"))
        elif rank(new) > rank(old):
            demoted.append((sym, new or "NONE"))
    return promoted, demoted


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #
def format_report(result: WatchlistResult) -> str:
    lines = []
    lines.append("TradeForge — WEEKLY WATCHLIST")
    lines.append(f"run_ts: {result.run_ts.isoformat()}")
    lines.append("")
    for tier in TIER_ORDER:
        members = [e for e in result.entries if e.tier == tier]
        cap = result.criteria.get("tiers", {}).get(tier, {}).get("max_symbols", "?")
        lines.append(f"=== {tier} ({len(members)}/{cap}) ===")
        for e in members:
            lines.append(
                f"  {e.symbol:<10s} fit={e.fit_score:.3f} "
                f"strat={e.best_strategy or '-':<16s} "
                f"avgDV={e.avg_dollar_volume/1e6:8.1f}M "
                f"px={e.last_price:8.2f}  {e.reason}"
            )
        lines.append("")

    if result.correlation is not None and not result.correlation.empty:
        lines.append("=== CORE daily-return correlation ===")
        corr = result.correlation
        names = list(corr.columns)
        lines.append("  " + " " * 10 + "".join(f"{n:>10s}" for n in names))
        for rn in names:
            row = "  " + f"{rn:<10s}" + "".join(
                f"{corr.loc[rn, cn]:>+10.3f}" for cn in names
            )
            lines.append(row)
        lines.append("")

    if result.promoted:
        lines.append("PROMOTED: " + ", ".join(f"{s}->{t}" for s, t in result.promoted))
    if result.demoted:
        lines.append("DEMOTED:  " + ", ".join(f"{s}->{t}" for s, t in result.demoted))
    return "\n".join(lines)


def main() -> int:
    con = connect(DEFAULT_DB_PATH)
    try:
        result = run_weekly(con=con)
        print(format_report(result))
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
