"""portfolio/engine.py — the cross-strategy book allocator (PortfolioEngine.step).

ONE deterministic loop turns every sleeve's signals into a single risk-budgeted
book. Each capability of the plan is a tagged step of :meth:`PortfolioEngine.step`:

  (0) ARM       regime arm/disarm + the vol-target exposure scalar; a disarmed
                weight sleeve parks in ``risk_off_weights``, a disarmed score
                sleeve simply stops opening.
  (1) MANAGE    resolve the ATR brackets for ALL score lots (gap-first, stop-first,
                TP1 → breakeven, chandelier trail, optional exit_signal) →
                partials / closes. Weight lots are reconciled in (6).
  (2) COLLECT   each armed sleeve's adapter emits Candidates (score sleeves skip
                names they already hold; weight sleeves emit their full vector).  (#1)
  (3) RANK      normalize within sleeve → one cross-sleeve priority order.          (#2)
  (4) RESOLVE   same-symbol conflicts → one candidate per symbol (grade then score).(#6)
  (5) DEDUP     one position per correlation cluster (seeded with held clusters).   (#3)
  (6) BUDGET    program-abort + daily/weekly halts (opens off, closes run); weight
                reconcile (trim/close/open standing targets); heat-admission walk
                that sizes + admits new opens under the book heat / family / cash /
                concurrency caps.                                              (#4,#5)
  (7) EMIT      an :class:`Allocation` (opens / closes / resizes / rejected) and a
                fully-evolved ``book`` (cash, lots, costs, tax, closed trades,
                per-sleeve attribution).

The engine is PURE (no DB / LLM / MCP / clock). In a backtest the driver evolves
the book day by day; in the live slow loop the same Allocation is rendered to
ORDER_INTENTs (``portfolio.intents``) while the broker holds the real cash.
``risk/limits.yaml`` is read-only single-source-of-truth; exits / risk-reducers
always execute even on a halt day (the F2 bypass).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from backtest.daily.result import PortfolioTrade
from portfolio.budget import (
    admit_candidates,
    dial_halt,
    kronos_veto,
    program_abort_halt,
    synthetic_stop,
)
from portfolio.conflicts import cluster_dedup, resolve_conflicts
from portfolio.config import PortfolioConfig, load_portfolio_config
from portfolio.model import (
    Allocation,
    BookState,
    Candidate,
    DayBars,
    OpenLot,
    PlannedClose,
    PlannedOpen,
    PlannedResize,
    SleeveSpec,
)
from portfolio.normalize import make_adapter
from portfolio.rank import rank_candidates
from orchestrator.agents.regime_reader import (
    SleeveArmConfig,
    arm_signals,
    realized_vol_annualized,
    vol_target_scalar,
)
from risk.config import Limits, load_limits


def _as_date(d):
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    return d


class PortfolioEngine:
    """The deterministic cross-sleeve allocator. One ``step`` per trading day.

    Parameters
    ----------
    sleeves
        The admitted :class:`SleeveSpec` list (rotation / swing_meanrev /
        swing_breakout). Mixed weight + score kinds.
    limits
        The validated ``risk/limits.yaml`` (READ-ONLY). Defaults to ``load_limits``.
    pcfg
        The engine policy (synthetic-stop band, family caps, cost/tax, Kronos
        flags). Defaults to ``load_portfolio_config``.
    arm_cfg
        The ``regime_reader`` arm / vol-target tunables.
    correlation_matrix
        A measured cross-symbol correlation matrix (DataFrame or nested mapping,
        shaped like ``backtest.portfolio.correlation_matrix``) for the cluster
        dedup. ``None`` → every symbol is its own cluster (no dedup).
    corr_threshold
        ``|rho|`` at which two symbols collapse into one cluster.
    market_proxy
        The broad-market close column used for the exposure-scalar vol estimate
        (defaults to ``arm_cfg.market_proxy``).
    """

    def __init__(
        self,
        sleeves: Sequence[SleeveSpec],
        *,
        limits: Optional[Limits] = None,
        pcfg: Optional[PortfolioConfig] = None,
        arm_cfg: Optional[SleeveArmConfig] = None,
        correlation_matrix=None,
        corr_threshold: float = 0.8,
        market_proxy: Optional[str] = None,
        forecast_provider=None,
    ):
        self.sleeves = list(sleeves)
        # Kronos overlay (Phase 3): a callable(asof, symbol) -> forecast dict|None,
        # READ from the kronos_forecasts table the slow loop wrote. Off unless the
        # config's use_kronos flag is set AND a provider is supplied (so the engine
        # runs identically — no torch, no forecasts — by default).
        self.forecast_provider = forecast_provider
        self.limits = limits if limits is not None else load_limits()
        self.pcfg = pcfg if pcfg is not None else load_portfolio_config()
        self.arm_cfg = arm_cfg or SleeveArmConfig()
        self.corr = correlation_matrix
        self.corr_threshold = float(corr_threshold)
        self.market_proxy = market_proxy or self.arm_cfg.market_proxy

        self._spec_by_name = {s.name: s for s in self.sleeves}
        self._adapters = {s.name: make_adapter(s) for s in self.sleeves}
        self._sleeve_benchmarks = {
            s.name: (s.benchmark or self.market_proxy) for s in self.sleeves
        }
        self._alloc = {s.name: float(s.allocation) for s in self.sleeves}
        self.cost_rate = float(self.pcfg.cost_bps) * 1e-4
        self.tax_rate = float(self.pcfg.short_term_tax_rate)
        self.synth = self.pcfg.synthetic_stop
        self.use_kronos = bool(self.pcfg.kronos.use_kronos) and forecast_provider is not None

    # ------------------------------------------------------------------ #
    # The per-day loop                                                    #
    # ------------------------------------------------------------------ #
    def step(self, asof, history, ohlc_today: DayBars, book: BookState) -> Allocation:
        """Run one decision day; evolve ``book`` in place and return the Allocation."""
        asof = _as_date(asof)
        alloc = Allocation()
        book.turnover_today = 0.0

        # ---- (0) ARM + exposure scalar ----
        close_panel = history._panel if hasattr(history, "_panel") else None
        armed = self._arm(close_panel, asof)
        scalar = self._exposure_scalar(close_panel, asof)

        # ---- (1) MANAGE score lots (brackets) ----
        self._manage_score_lots(asof, history, ohlc_today, book, alloc)

        # ---- (2) COLLECT ----
        weight_cands, score_cands, weight_targets = self._collect(
            asof, history, ohlc_today, book, armed
        )

        # ---- (6a/b) BOOK-LEVEL HALTS (computed on the post-manage equity) ----
        equity = self._equity(book, ohlc_today)
        halt_reason = program_abort_halt(equity, book, self.limits) or dial_halt(
            equity, book, self.limits
        )
        if halt_reason:
            alloc.halted = True
            alloc.halt_reason = halt_reason

        # ---- (6) WEIGHT RECONCILE (trim/close standing targets; closes run even on a halt) ----
        self._reconcile_weight_lots(
            asof, ohlc_today, book, alloc, weight_targets, scalar, halted=alloc.halted
        )

        # ---- new-open competition: only when NOT halted ----
        if not alloc.halted:
            open_pool = self._open_pool(weight_cands, score_cands, book, alloc)
            open_pool = self._kronos_filter(open_pool, alloc)
            ranked = self._rank_resolve_dedup(open_pool, book, alloc)
            self._admit_and_open(asof, ohlc_today, book, alloc, ranked, equity, scalar)
        else:
            # Record the suppressed opens as rejected for transparency.
            for c in [*weight_cands, *score_cands]:
                if c.symbol not in book.held_symbols():
                    alloc.rejected.append((c, f"halted: {alloc.halt_reason}"))

        return alloc

    # ------------------------------------------------------------------ #
    # (0) ARM + exposure                                                  #
    # ------------------------------------------------------------------ #
    def _arm(self, close_panel, asof) -> dict:
        """Per-sleeve arm flags from the close panel.

        The regime arm is a SECOND trend gate ON TOP of each sleeve's own internal
        gate, so when it cannot be evaluated (the market proxy / a sleeve's
        benchmark is not in this universe) we default that sleeve to ARMED rather
        than silently flatten the whole book — the sleeve's own 200d gate still
        protects it. It only ever DISARMS on a benchmark it can actually read.
        """
        cols = set(getattr(close_panel, "columns", []) if close_panel is not None else [])
        if close_panel is None or self.market_proxy not in cols:
            return {s.name: True for s in self.sleeves}
        try:
            armed = arm_signals(close_panel, asof, self._sleeve_benchmarks, cfg=self.arm_cfg)
        except Exception:
            return {s.name: True for s in self.sleeves}
        # Override any sleeve whose benchmark is absent (undecidable -> armed).
        for s in self.sleeves:
            bench = self._sleeve_benchmarks.get(s.name)
            if bench not in cols:
                armed[s.name] = True
        return armed

    def _exposure_scalar(self, close_panel, asof) -> float:
        """The vol-target exposure scalar from the market proxy's realized vol."""
        if close_panel is None or self.market_proxy not in getattr(close_panel, "columns", []):
            return 1.0
        visible = close_panel.loc[[d for d in close_panel.index if _as_date(d) <= asof]]
        proxy = visible[self.market_proxy] if self.market_proxy in visible.columns else None
        rv = realized_vol_annualized(proxy, self.arm_cfg.vol_lookback, self.arm_cfg.trading_days)
        return vol_target_scalar(rv, self.arm_cfg)

    # ------------------------------------------------------------------ #
    # equity / marking helpers                                            #
    # ------------------------------------------------------------------ #
    def _equity(self, book: BookState, bars: DayBars) -> float:
        """Mark-to-close equity = cash + Σ lot.shares * today's close (carry stale)."""
        total = float(book.cash)
        for sym, lot in book.lots.items():
            px = bars.close_of(sym)
            mark = px if px is not None else lot.entry_price
            total += lot.shares * mark
        return total

    # ------------------------------------------------------------------ #
    # (1) MANAGE score lots — the ATR bracket (ported from bracket_engine) #
    # ------------------------------------------------------------------ #
    def _manage_score_lots(self, asof, history, bars: DayBars, book: BookState, alloc: Allocation) -> None:
        for sym in list(book.lots.keys()):
            lot = book.lots.get(sym)
            if lot is None or lot.kind != "score":
                continue
            lot.bars_held += 1
            o, h, l = bars.open_of(sym), bars.high_of(sym), bars.low_of(sym)
            if o is None or h is None or l is None:
                continue  # untradeable today; carry untouched
            lot.highest_high = max(lot.highest_high, h)
            spec = self._spec_by_name.get(lot.sleeve)
            br = spec.bracket if spec is not None else None
            tp1_fraction = float(getattr(br, "tp1_fraction", 0.0) or 0.0)
            trail_mult = float(getattr(br, "trail_atr_mult", 3.0))
            use_trail = bool(getattr(br, "use_trail", True))

            # --- GAP through the stop on the open -> fill at the (worse) open ---
            if o <= lot.stop:
                self._close_lot(book, alloc, lot, o, asof,
                                "trail_stop_gap" if lot.tp1_done else "stop_gap")
                continue
            # --- GAP through hard target (full cap wins over TP1) ---
            if lot.hard_target is not None and o >= lot.hard_target:
                self._close_lot(book, alloc, lot, o, asof, "hard_target_gap")
                continue
            if (not lot.tp1_done and lot.tp1_price is not None
                    and o >= lot.tp1_price and tp1_fraction > 0):
                self._do_tp1(book, alloc, lot, o, asof, tp1_fraction)
                lot = book.lots.get(sym)
                if lot is None:
                    continue

            # --- INTRABAR by [low, high], STOP-FIRST on ambiguity ---
            if l <= lot.stop:
                self._close_lot(book, alloc, lot, lot.stop, asof,
                                "trail_stop" if lot.tp1_done else "stop")
                continue
            if lot.hard_target is not None and h >= lot.hard_target:
                self._close_lot(book, alloc, lot, lot.hard_target, asof, "hard_target")
                continue
            if (not lot.tp1_done and lot.tp1_price is not None
                    and h >= lot.tp1_price and tp1_fraction > 0):
                self._do_tp1(book, alloc, lot, lot.tp1_price, asof, tp1_fraction)
                lot = book.lots.get(sym)
                if lot is None:
                    continue

            # --- TRAIL the runner's chandelier stop (never loosening) ---
            if use_trail and lot.tp1_done and lot.atr_entry:
                cand = lot.highest_high - trail_mult * lot.atr_entry
                if cand > lot.stop:
                    lot.stop = cand

            # --- discretionary exit_signal -> close at today's close ---
            strat = spec.strategy if spec is not None else None
            if strat is not None and hasattr(strat, "exit_signal") and callable(strat.exit_signal):
                c = bars.close_of(sym)
                if c is not None:
                    try:
                        want = bool(strat.exit_signal(sym, asof, history, lot))
                    except TypeError:
                        want = bool(strat.exit_signal(sym, asof, history))
                    if want:
                        self._close_lot(book, alloc, lot, c, asof, "exit_signal")

    # ------------------------------------------------------------------ #
    # cash / accounting primitives (the engine owns the book)             #
    # ------------------------------------------------------------------ #
    def _sell(self, book: BookState, lot: OpenLot, qty: float, fill: float) -> float:
        """Sell ``qty`` of ``lot`` at ``fill``: book cash/cost/tax/attribution. Net."""
        notional = qty * fill
        cost = notional * self.cost_rate
        gross = (fill - lot.entry_price) * qty
        net = gross - cost
        book.cash += notional - cost
        book.total_costs += cost
        book.turnover_today += notional
        book.realized_gains += gross
        if gross > 0:
            book.tax_reserve += gross * self.tax_rate
        lot.realized_gross += gross
        lot.realized_pnl += net
        lot.realized_costs += cost
        rps = lot.risk_per_share
        if rps > 0 and lot.initial_shares:
            lot.realized_r += ((fill - lot.entry_price) / rps) * (qty / lot.initial_shares)
        book.sleeve_realized[lot.sleeve] = book.sleeve_realized.get(lot.sleeve, 0.0) + net
        return net

    def _do_tp1(self, book, alloc, lot: OpenLot, fill: float, asof, tp1_fraction: float) -> None:
        """Scale out ``tp1_fraction`` of ``lot`` at ``fill`` and ratchet to breakeven."""
        if lot.tp1_done or not lot.initial_shares:
            return
        part = lot.initial_shares * tp1_fraction
        if part <= 0 or part >= lot.shares:
            return
        self._sell(book, lot, part, fill)
        lot.shares -= part
        lot.tp1_done = True
        lot.stop = max(lot.stop, lot.entry_price)
        alloc.resizes.append(PlannedResize(
            sleeve=lot.sleeve, symbol=lot.symbol, delta_shares=-part,
            fill_price=fill, reason="tp1_partial",
        ))

    def _close_lot(self, book, alloc, lot: OpenLot, fill: float, asof, reason: str) -> None:
        """Fully close ``lot`` at ``fill``: sell the remainder, record the trade, drop it."""
        qty = lot.shares
        self._sell(book, lot, qty, fill)
        gross = lot.realized_gross
        if lot.initial_shares:
            avg_exit = lot.entry_price + gross / lot.initial_shares
        else:
            avg_exit = lot.entry_price
        book.closed_trades.append(PortfolioTrade(
            symbol=lot.symbol,
            entry_date=lot.entry_date,
            exit_date=asof,
            entry_price=lot.entry_price,
            avg_exit_price=avg_exit,
            shares=lot.initial_shares if lot.initial_shares else qty,
            initial_stop=lot.initial_stop if lot.initial_stop is not None else lot.entry_price,
            pnl=lot.realized_pnl,
            gross_pnl=lot.realized_gross,
            costs=lot.realized_costs,
            r_multiple=lot.realized_r,
            bars_held=lot.bars_held,
            exit_reason=reason,
            sleeve=lot.sleeve,
            kind=lot.kind,
        ))
        alloc.closes.append(PlannedClose(
            sleeve=lot.sleeve, symbol=lot.symbol, shares=qty, fill_price=fill, reason=reason,
        ))
        book.lots.pop(lot.symbol, None)

    def _open_lot(self, asof, book, alloc, planned: PlannedOpen) -> None:
        """Apply a :class:`PlannedOpen`: charge cash/cost, create the lot, log it."""
        notional = planned.shares * planned.entry_price
        cost = notional * self.cost_rate
        book.cash -= notional + cost
        book.total_costs += cost
        book.turnover_today += notional
        book.sleeve_realized[planned.sleeve] = (
            book.sleeve_realized.get(planned.sleeve, 0.0) - cost
        )
        lot = OpenLot(
            sleeve=planned.sleeve,
            symbol=planned.symbol,
            kind=planned.kind,
            entry_date=asof,
            entry_price=planned.entry_price,
            shares=planned.shares,
            grade=planned.grade,
            family=planned.family,
            initial_stop=planned.stop,
            initial_shares=planned.shares,
            atr_entry=planned.atr,
            stop=planned.stop,
            tp1_price=planned.tp1_price,
            hard_target=planned.hard_target,
            highest_high=planned.entry_price,
            target_weight=planned.target_weight,
            realized_costs=cost,
            realized_pnl=-cost,
        )
        book.lots[planned.symbol] = lot
        alloc.opens.append(planned)

    # ------------------------------------------------------------------ #
    # (2) COLLECT                                                          #
    # ------------------------------------------------------------------ #
    def _collect(self, asof, history, bars: DayBars, book: BookState, armed: dict):
        """Gather weight + score candidates and the per-sleeve weight target vectors."""
        weight_cands: list = []
        score_cands: list = []
        weight_targets: dict = {}      # sleeve -> {symbol: weight}
        for spec in self.sleeves:
            adapter = self._adapters[spec.name]
            is_armed = bool(armed.get(spec.name, True))
            if spec.kind == "weight":
                cands = adapter.candidates(
                    asof, history, bars,
                    armed=is_armed,
                    risk_off_weights=self._risk_off_weights(),
                    synthetic_window=self.synth.window,
                )
                weight_cands.extend(cands)
                weight_targets[spec.name] = {c.symbol: c.target_weight for c in cands}
            else:  # score
                if not is_armed:
                    continue  # disarmed score sleeve simply stops opening
                cands = adapter.candidates(
                    asof, history, bars,
                    held_by_sleeve=book.held_by_sleeve(spec.name),
                )
                score_cands.extend(cands)
        # Kronos overlay: attach each candidate's forecast (point-in-time lookup).
        if self.use_kronos and self.forecast_provider is not None:
            for c in [*weight_cands, *score_cands]:
                try:
                    c.forecast = self.forecast_provider(asof, c.symbol)
                except Exception:  # noqa: BLE001 — a missing forecast must not crash
                    c.forecast = None
        return weight_cands, score_cands, weight_targets

    def _risk_off_weights(self) -> dict:
        """The defensive parking target for a disarmed weight sleeve."""
        return {self.arm_cfg.risk_off_symbol: 1.0}

    # ------------------------------------------------------------------ #
    # (6) WEIGHT RECONCILE — trim/close/hold standing targets             #
    # ------------------------------------------------------------------ #
    def _reconcile_weight_lots(
        self, asof, bars: DayBars, book: BookState, alloc: Allocation,
        weight_targets: dict, scalar: float, halted: bool,
    ) -> None:
        """Reconcile each held WEIGHT lot toward its sleeve's fresh standing target.

        Trims / closes (risk-reducing) ALWAYS execute — even on a halt day. Buys
        (increasing toward a higher target) are skipped on a halt day (no new risk)
        and otherwise clipped to available cash (no leverage). A held name dropped
        from the target is closed.
        """
        equity = self._equity(book, bars)
        for sym in list(book.lots.keys()):
            lot = book.lots.get(sym)
            if lot is None or lot.kind != "weight":
                continue
            targets = weight_targets.get(lot.sleeve, {})
            tw = targets.get(sym)
            mark = bars.close_of(sym)
            if mark is None:
                continue  # can't reconcile an unpriced name; carry it
            if tw is None or tw <= 0:
                # dropped from the target -> close the standing position.
                self._close_lot(book, alloc, lot, mark, asof, "weight_exit")
                continue
            sleeve_alloc = self._alloc.get(lot.sleeve, 1.0)
            target_dollars = float(tw) * equity * scalar * sleeve_alloc
            target_shares = target_dollars / mark
            delta = target_shares - lot.shares
            if abs(delta) * mark < 1e-9:
                continue
            if delta < 0:
                self._reconcile_sell(asof, book, alloc, lot, -delta, mark)
            elif not halted:
                self._reconcile_buy(book, alloc, lot, delta, mark)

    def _reconcile_sell(self, asof, book, alloc, lot: OpenLot, qty: float, fill: float) -> None:
        """Trim ``qty`` of a weight lot (risk-reducing); close it if fully exited."""
        qty = min(qty, lot.shares)
        if qty <= 0:
            return
        if qty >= lot.shares - 1e-12:
            self._close_lot(book, alloc, lot, fill, asof, "weight_exit")
            return
        self._sell(book, lot, qty, fill)
        lot.shares -= qty
        alloc.resizes.append(PlannedResize(
            sleeve=lot.sleeve, symbol=lot.symbol, delta_shares=-qty,
            fill_price=fill, reason="weight_trim",
        ))

    def _reconcile_buy(self, book, alloc, lot: OpenLot, qty: float, fill: float) -> None:
        """Add ``qty`` to a weight lot, clipped to cash (no leverage); merge avg cost."""
        if qty <= 0:
            return
        notional = qty * fill
        cost = notional * self.cost_rate
        if notional + cost > book.cash:
            if book.cash <= 0:
                return
            qty = book.cash / (fill * (1.0 + self.cost_rate))
            notional = qty * fill
            cost = notional * self.cost_rate
            if qty <= 0:
                return
        book.cash -= notional + cost
        book.total_costs += cost
        book.turnover_today += notional
        book.sleeve_realized[lot.sleeve] = book.sleeve_realized.get(lot.sleeve, 0.0) - cost
        # merge the cost basis (avg cost) so a later trim realizes the right gain.
        new_shares = lot.shares + qty
        lot.entry_price = (lot.shares * lot.entry_price + qty * fill) / new_shares
        lot.shares = new_shares
        alloc.resizes.append(PlannedResize(
            sleeve=lot.sleeve, symbol=lot.symbol, delta_shares=qty,
            fill_price=fill, reason="weight_add",
        ))

    # ------------------------------------------------------------------ #
    # open-competition pool: drop held symbols + already-held weight names #
    # ------------------------------------------------------------------ #
    def _open_pool(self, weight_cands, score_cands, book: BookState, alloc: Allocation) -> list:
        """New-open candidates only (symbols not held by the book), with reject logging."""
        held = book.held_symbols()
        pool: list = []
        for c in [*weight_cands, *score_cands]:
            if c.symbol in held:
                # weight name already held -> handled by reconcile (silent);
                # cross-sleeve duplicate -> reject for transparency.
                owner = book.lots[c.symbol].sleeve
                if owner != c.sleeve:
                    alloc.rejected.append((c, f"already held by {owner}"))
                continue
            pool.append(c)
        return pool

    # ------------------------------------------------------------------ #
    # Kronos veto (filter weak signals) — off unless use_kronos            #
    # ------------------------------------------------------------------ #
    def _kronos_filter(self, pool: list, alloc: Allocation) -> list:
        """Drop candidates Kronos vetoes (negative exp_return / excess downside)."""
        if not self.use_kronos:
            return pool
        kept: list = []
        for c in pool:
            why = kronos_veto(c, self.pcfg.kronos)
            if why is not None:
                alloc.rejected.append((c, why))
            else:
                kept.append(c)
        return kept

    # ------------------------------------------------------------------ #
    # (3/4/5) RANK -> RESOLVE same-symbol -> DEDUP cluster                 #
    # ------------------------------------------------------------------ #
    def _rank_resolve_dedup(self, pool: list, book: BookState, alloc: Allocation) -> list:
        """Normalize+rank, resolve same-symbol (grade-aware), then cluster-dedup.

        Same-symbol RESOLVE runs BEFORE cluster DEDUP so the conviction-grade
        winner (not merely the rank winner) survives a duplicate ticker; cluster
        dedup then collapses distinct-but-correlated names, seeded with the
        clusters the book already holds (no doubling an existing bet).
        """
        if not pool:
            return []
        ranked = rank_candidates(pool, kronos_blend=(self.pcfg.kronos.rank_blend if self.use_kronos else 0.0))
        # (4) same-symbol resolve (grade then score)
        resolved, conflict_rejects = resolve_conflicts(ranked)
        for c, why in conflict_rejects:
            alloc.rejected.append((c, why))
        # re-apply the cross-sleeve order after the collapse
        resolved = rank_candidates(resolved, kronos_blend=(self.pcfg.kronos.rank_blend if self.use_kronos else 0.0))
        # (5) cluster dedup, seeded with the book's held names (no doubling a bet)
        kept, dedup_rejects, _clusters = cluster_dedup(
            resolved, self.corr, threshold=self.corr_threshold,
            held_symbols=book.held_symbols(),
        )
        for c, why in dedup_rejects:
            alloc.rejected.append((c, why))
        return kept

    # ------------------------------------------------------------------ #
    # (6) heat-admission walk + apply opens                               #
    # ------------------------------------------------------------------ #
    def _admit_and_open(self, asof, bars: DayBars, book: BookState, alloc: Allocation,
                        ranked: list, equity: float, scalar: float) -> None:
        if not ranked:
            return
        current_heat, family_heat = self._current_heat(book, bars)
        sleeve_scalars = {name: scalar * self._alloc.get(name, 1.0) for name in self._alloc}
        opens, rejected = admit_candidates(
            ranked, equity, self.limits, self.pcfg,
            scalar=scalar,
            sleeve_scalars=sleeve_scalars,
            current_book_heat=current_heat,
            current_family_heat=family_heat,
            available_cash=max(book.cash, 0.0),
            n_held=len(book.lots),
        )
        for c, why in rejected:
            alloc.rejected.append((c, why))
        for planned in opens:
            self._open_lot(asof, book, alloc, planned)

    def _current_heat(self, book: BookState, bars: DayBars) -> tuple:
        """(total, {family: heat}) dollar-risk of the EXISTING lots (post-reconcile).

        Score lots use their live bracket stop; weight lots use the synthetic band
        recomputed on today's mark — the same convention the admission walk sizes
        new candidates with, so the headroom is apples-to-apples.
        """
        total = 0.0
        fam: dict = {}
        for sym, lot in book.lots.items():
            mark = bars.close_of(sym)
            if mark is None:
                mark = lot.entry_price
            if lot.kind == "score" and lot.stop is not None:
                risk = lot.shares * max(mark - lot.stop, 0.0)
            else:
                atr = bars.atr_of(sym, self.synth.window)
                stop = synthetic_stop(mark, atr, self.synth.k, self.synth.fallback_band_frac)
                risk = lot.shares * max(mark - stop, 0.0)
            total += risk
            fam[lot.family] = fam.get(lot.family, 0.0) + risk
        return total, fam
