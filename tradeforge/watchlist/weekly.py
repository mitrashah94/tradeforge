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
    # ---- universe-engine metadata (Phase 2) ----
    sector: str | None = None              # GICS sector / category (ETF map or MCP)
    spread_bps: float = float("nan")       # bid/ask spread in bps (MCP quote)
    fractional_enabled: bool = True        # fractional-share eligible (MCP tradability)
    price_history_sessions: int = 0        # traded sessions of history (from bars)
    cluster_id: str | None = None          # correlation-cluster representative


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


def price_history_sessions(bars: pd.DataFrame, asset_class: str) -> int:
    """Number of distinct traded SESSIONS in a symbol's bars (the history gate).

    Counted from the symbol's own bars' session dates, so a young ticker (a recent
    listing / a fresh crypto-ETF) is correctly flagged as thin-history regardless
    of how many intraday bars it has.
    """
    if bars is None or len(bars) == 0:
        return 0
    sess_fn = crypto_session_date if asset_class == "crypto" else et_session_date
    return int(bars["ts_utc"].apply(sess_fn).nunique())


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
    """Assign CORE/ACTIVE/SCOUT tiers with the full universe-engine gates.

    Candidates are processed best-fit-first. A symbol earns a tier only if it
    clears EVERY gate of that tier:
      * liquidity / price (``min_dollar_volume`` / ``min_price``);
      * the ``require_score`` level-respect gate;
      * the metadata gates — ``max_spread_bps`` (tight enough to trade),
        ``min_price_history_sessions`` (enough history), ``fractional_required``
        (a $1k book can take a slice). Each gate is NON-BINDING when its metadata
        is absent for the symbol, so the run works before any MCP refresh;
      * a per-tier SECTOR-CONCENTRATION cap (``sector_max_concentration`` of the
        tier's slots may share one sector);
      * for CORE only: at most ONE name per correlation CLUSTER (single-linkage
        union-find on daily-return correlation — replaces the old pairwise reject;
        ``corr_cluster_threshold`` sets the collapse |rho|).
    A CORE candidate rejected by the cluster / sector check falls through to
    ACTIVE; ACTIVE overflow falls through to SCOUT. Each tier is capped at its
    ``max_symbols``.

    Returns ``(entries, core_corr)`` where ``core_corr`` is the correlation matrix
    of the CORE symbols (persistence/inspection).
    """
    from watchlist.clustering import cluster_symbols

    tiers_cfg = criteria.get("tiers", {})
    default_gate = float(criteria.get("require_level_respect_score", 0.0))
    do_corr_check = bool(criteria.get("require_correlation_check", True))
    cluster_thr = float(
        criteria.get("corr_cluster_threshold", criteria.get("core_max_correlation", 0.85))
    )

    # Best fit first; tie-break by liquidity so a more-liquid name wins a slot.
    ordered = sorted(
        candidates,
        key=lambda e: (e.fit_score, e.avg_dollar_volume),
        reverse=True,
    )

    # Cluster every candidate ONCE off the daily-return correlation matrix; the
    # CORE loop then admits at most one per cluster (the shared union-find).
    cand_syms = [e.symbol for e in ordered]
    corr_all = returns_correlation(cand_syms, bars_by_symbol)
    clusters = cluster_symbols(cand_syms, corr_all, threshold=cluster_thr)
    for e in ordered:
        e.cluster_id = clusters.get(e.symbol, e.symbol)

    assigned: dict[str, SymbolEntry] = {}
    core_members: list[str] = []

    def _g(cfg: dict, key: str, default):
        """Tier-level override of a global criteria key (tier wins if present)."""
        if key in cfg:
            return cfg[key]
        return criteria.get(key, default)

    def _clears(entry: SymbolEntry, cfg: dict) -> str | None:
        """Return a rejection reason if a hard gate fails, else None."""
        if entry.avg_dollar_volume < float(cfg.get("min_dollar_volume", 0)):
            return "below min_dollar_volume"
        if entry.last_price < float(cfg.get("min_price", 0)):
            return "below min_price"
        # spread gate (non-binding when spread unknown / NaN)
        max_spread = _g(cfg, "max_spread_bps", None)
        if max_spread is not None and np.isfinite(entry.spread_bps):
            if entry.spread_bps > float(max_spread):
                return f"spread {entry.spread_bps:.0f}bps > {float(max_spread):.0f}"
        # price-history gate
        min_hist = _g(cfg, "min_price_history_sessions", None)
        if min_hist is not None and entry.price_history_sessions > 0:
            if entry.price_history_sessions < int(min_hist):
                return (f"history {entry.price_history_sessions} "
                        f"< {int(min_hist)} sessions")
        # fractional-eligibility gate
        if bool(_g(cfg, "fractional_required", False)) and not entry.fractional_enabled:
            return "not fractional-eligible"
        return None

    def _gate_for(cfg: dict) -> float:
        return float(cfg.get("require_score", default_gate))

    def _sector_cap(cfg: dict, cap: int, tier_sectors: dict, sector: str | None) -> bool:
        """True if admitting ``sector`` would breach the per-tier sector cap."""
        frac = _g(cfg, "sector_max_concentration", None)
        if frac is None or sector is None:
            return False
        max_per_sector = max(1, int(float(frac) * cap))
        return tier_sectors.get(sector, 0) >= max_per_sector

    def _fill_tier(tier: str, cfg: dict, cap: int, gate: float, *, cluster_check: bool):
        tier_sectors: dict = {}
        taken_clusters: set = set()
        n = 0
        for entry in ordered:
            if n >= cap:
                break
            if entry.symbol in assigned:
                continue
            why = _clears(entry, cfg)
            if why is not None:
                if not entry.reason:
                    entry.reason = f"{tier} rejected: {why}"
                continue
            if entry.fit_score < gate:
                continue
            if _sector_cap(cfg, cap, tier_sectors, entry.sector):
                entry.reason = f"{tier} rejected: sector '{entry.sector}' cap reached"
                continue
            if cluster_check and do_corr_check and entry.cluster_id in taken_clusters:
                entry.reason = (
                    f"{tier} rejected: cluster {entry.cluster_id} already in {tier}"
                )
                continue
            entry.tier = tier
            entry.reason = (
                f"{tier}: fit {entry.fit_score:.2f}"
                if (not entry.reason or entry.reason.endswith("rejected"))
                else entry.reason
            )
            assigned[entry.symbol] = entry
            taken_clusters.add(entry.cluster_id)
            if entry.sector:
                tier_sectors[entry.sector] = tier_sectors.get(entry.sector, 0) + 1
            if tier == "CORE":
                core_members.append(entry.symbol)
            n += 1

    core_cfg = tiers_cfg.get("CORE", {})
    _fill_tier("CORE", core_cfg, int(core_cfg.get("max_symbols", 4)),
               _gate_for(core_cfg), cluster_check=True)
    active_cfg = tiers_cfg.get("ACTIVE", {})
    _fill_tier("ACTIVE", active_cfg, int(active_cfg.get("max_symbols", 8)),
               _gate_for(active_cfg), cluster_check=False)
    scout_cfg = tiers_cfg.get("SCOUT", {})
    _fill_tier("SCOUT", scout_cfg, int(scout_cfg.get("max_symbols", 25)),
               _gate_for(scout_cfg), cluster_check=False)

    # Symbols that found no tier keep tier=None.
    core_corr = returns_correlation(core_members, bars_by_symbol)
    return ordered, core_corr


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _load_metadata(universe_db_path: str) -> dict:
    """JOIN-side read of equity_metadata from universe.duckdb (empty if absent).

    The deterministic path: it only READS a table a separate, off-hot-path MCP
    refresh (``watchlist.fetch_metadata.refresh_metadata``) populated. Never calls
    MCP itself, so ``run_weekly`` stays offline/deterministic.
    """
    if not os.path.exists(universe_db_path):
        return {}
    from watchlist.fetch_metadata import read_metadata
    mcon = connect(universe_db_path)
    try:
        return read_metadata(mcon)
    except Exception:  # noqa: BLE001 — metadata is optional; never block the run
        return {}
    finally:
        mcon.close()


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

        # Metadata (sector / spread / fractional) is fetched OFF the hot path into
        # universe.duckdb.equity_metadata; here we just JOIN it (never call MCP).
        metadata = _load_metadata(universe_db_path)

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
        from watchlist.fetch_metadata import etf_sector
        for sym in candidate_syms:
            asset_class = _infer_asset_class(sym)
            stats = summarize_symbol(sym, bars_by_symbol[sym], asset_class)
            by_strat = score_symbol_all_strategies(
                sym, strategies, timeframe, lookback_sessions, cost_profile, con
            )
            raw_scores[sym] = by_strat
            if by_strat:
                best_name = max(by_strat, key=lambda n: by_strat[n].score)
                best_score = by_strat[best_name].score
            else:
                best_name, best_score = None, 0.0
            m = metadata.get(sym, {})
            candidates.append(
                SymbolEntry(
                    symbol=sym,
                    asset_class=asset_class,
                    tier=None,
                    fit_score=best_score,
                    best_strategy=best_name,
                    avg_dollar_volume=stats.avg_dollar_volume if stats else 0.0,
                    last_price=stats.last_price if stats else 0.0,
                    rvol=rvol_map.get(sym, float("nan")),
                    scores={n: s.score for n, s in by_strat.items()},
                    sector=m.get("sector") or etf_sector(sym),
                    spread_bps=m.get("spread_bps", float("nan")) if m.get("spread_bps") is not None else float("nan"),
                    fractional_enabled=bool(m.get("fractional_enabled", True)),
                    price_history_sessions=price_history_sessions(bars_by_symbol[sym], asset_class),
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
