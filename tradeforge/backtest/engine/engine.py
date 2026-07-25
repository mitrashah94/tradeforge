"""backtest/engine/engine.py — a thin, custom, event-driven bar simulator.

WHY CUSTOM (not vectorbt / backtesting.py)
------------------------------------------
The breakout_retest edge is a *path-dependent intraday state machine*:

    break (close beyond a level) -> wait 2..7 bars -> retest (touch + close
    back beyond) -> place a MARKET order that fills at the NEXT bar's open ->
    manage an OCO bracket intrabar with a specific stop-first priority -> flat
    by EOD -> reset all daily state at the next session.

That control flow, plus *exact TradingView Pine fill parity*
(``process_orders_on_close = false`` -> market orders fill at the next bar
open while stop/target prices are pinned to the SIGNAL bar's close) and a
bespoke spread/slippage cost model applied to every fill, are first-class
needs here. vectorbt's vectorized signal model expresses entries/exits as
boolean arrays evaluated against a single price series — it cannot natively
express "fill at next-bar open but compute the bracket off this bar's close,
then resolve stop-vs-target intrabar with a gap-aware priority". backtesting.py
is event-driven but bakes in its own fill timing and a fixed commission model
and does not expose the intrabar OCO priority we must replicate. Bending either
to our semantics costs more than a small purpose-built loop, and the loop is
also the cleanest place to plug the order-ledger slippage feedback (§4) later.

So: a deterministic, single-pass, bar-by-bar simulator that we fully control.

THE STRATEGY INTERFACE (the Stage-2 contract)
---------------------------------------------
Strategies subclass :class:`Strategy` and implement two callbacks. The engine
drives them; they never touch bars or fills directly — they only inspect the
:class:`Context` and request orders through it.

    class Strategy:
        def on_session_start(self, ctx: Context) -> None: ...
        def on_bar(self, ctx: Context) -> None: ...

``Context`` (read by the strategy on each ``on_bar``) exposes:
    ctx.bar            -> current Bar (ts, open, high, low, close, volume)
    ctx.prev_bars      -> list[Bar] of this session's bars BEFORE the current
                          one (index -1 is the immediately prior bar)
    ctx.bar_index      -> 0-based index of the current bar within the session
    ctx.levels         -> dict of this session's levels (pdh, pdl, pmh, pml,
                          ntz_low, ntz_high, ntz_valid, atr14)
    ctx.position       -> the open Position or None
    ctx.in_session     -> True while inside RTH for the session
    ctx.is_eod         -> True on the final managed bar of the session
    ctx.symbol, ctx.asset_class, ctx.tick

Order methods (called from on_bar; at most one entry per bar):
    ctx.enter_long(stop, target=None)   -> MARKET long, fills NEXT bar open;
                                           stop/target pinned to THIS close
    ctx.enter_short(stop, target=None)  -> symmetric
    ctx.close()                         -> request a market close of the
                                           current position at the next bar open

The engine guarantees: entries fill at the next bar's open (Pine parity); the
OCO bracket is managed intrabar each subsequent bar with the documented
priority; the CostModel is applied to every fill; EOD-flat closes any open
position on the final bar's close; one entry attempt per side per day and no
re-entry that side after a stop-out that day.

INTRABAR OCO PRIORITY (documented reproduction-delta source)
------------------------------------------------------------
On each managed bar, with stop S and target T for the open position:
  1. GAP THROUGH on the open wins first: if the bar OPENS beyond the target,
     fill the target at the open; if it OPENS beyond the stop, fill the stop at
     the open (a gap fills at the open, not the resting price).
  2. Otherwise, if the bar's [low, high] spans BOTH S and T, assume the STOP
     fills first (conservative — we cannot see intrabar order on bar data).
  3. Otherwise fill whichever of S/T the bar reaches.
This stop-first-on-ambiguity rule is a known source of small deltas vs the
TradingView consolidated-feed reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Sequence

import pandas as pd

from backtest.engine.cost import (
    FILL_LIMIT,
    FILL_MARKET,
    FILL_STOP,
    CostModel,
)
from backtest.engine.result import BacktestResult, TradeRecord


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. ``ts`` is the bar OPEN time (tz-naive UTC)."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PartialPlan:
    """Optional partial-exit + trailing-runner plan attached to an entry (V3/V4).

    When present on a :class:`Position`, the engine scales out ``fraction`` of
    the position at ``tp1`` (a resting limit, ~+``tp1_r``R), then moves the stop
    to breakeven (the entry's ``signal_close``) and trails the REMAINING shares
    by the prior managed bar's low (long) / high (short) until the runner's stop
    is hit or EOD-flat. This produces the positive-skew structure of MASTER_PLAN
    §1.B (cut losers fast, let winners run).

    V0/V1/V2 entries carry no PartialPlan, so the engine's exit path is
    unchanged for them (fixed-2R or no target).
    """

    tp1: float                # price of the first (partial) scale-out limit
    tp1_r: float              # R multiple of tp1 (diagnostic)
    fraction: float           # fraction of shares taken at tp1 (e.g. 0.5)
    trail_mode: str = "prior_bar"  # how the runner stop trails


@dataclass
class Position:
    """An open position with its (signal-close-derived) bracket."""

    side: str                 # 'long' | 'short'
    shares: float
    entry_ts: datetime
    entry_price: float        # cost-adjusted fill
    ref_entry_price: float    # raw next-bar open before costs (for diagnostics)
    stop: float               # raw stop price (pre-cost)
    target: float | None      # raw target price (pre-cost) or None
    entry_commission: float
    signal_close: float       # the signal bar's close (bracket reference)
    entry_bar_index: int = -1  # session bar index the entry filled on
    bars_held: int = 0
    mfe: float = 0.0          # max favorable excursion in price (>=0)
    mae: float = 0.0          # max adverse excursion in price (>=0)
    # ---- partial + runner state (V3/V4); None/0 keeps the V0 path intact ----
    partial: "PartialPlan | None" = None
    initial_shares: float = 0.0   # shares at entry (for partial accounting)
    partial_done: bool = False    # True once the TP1 scale-out has filled
    realized_partial_pnl: float = 0.0  # net P&L booked from the partial fill
    realized_partial_r: float = 0.0    # R booked from the partial (size-weighted)
    ref_stop: float = 0.0         # original entry stop (runner risk basis; pre-trail)


@dataclass
class _PendingOrder:
    """A market order requested on the signal bar, to fill at next bar open."""

    kind: str                 # 'enter' | 'close'
    side: str | None          # for 'enter'
    stop: float | None
    target: float | None
    signal_close: float | None
    signal_ts: datetime | None
    partial: "PartialPlan | None" = None  # V3/V4 only; None keeps V0 path


# --------------------------------------------------------------------------- #
# Context — the strategy's view of the world on each bar
# --------------------------------------------------------------------------- #
class Context:
    """Read-only-ish view passed to the strategy each bar.

    The strategy reads state and requests orders; the engine owns execution.
    """

    def __init__(self, engine: "BacktestEngine"):
        self._engine = engine
        # Per-bar fields, refreshed by the engine before each on_bar.
        self.bar: Bar | None = None
        self.bar_index: int = -1
        self.prev_bars: list[Bar] = []
        self.levels: dict = {}
        self.in_session: bool = True
        self.is_eod: bool = False
        # Per-symbol static.
        self.symbol: str = engine.symbol
        self.asset_class: str = engine.asset_class
        self.tick: float = engine.tick

    @property
    def position(self) -> Position | None:
        return self._engine.position

    # ----- order requests (queued; executed at next bar open) -----
    def enter_long(
        self,
        stop: float,
        target: float | None = None,
        partial: "PartialPlan | None" = None,
    ) -> None:
        self._engine.request_entry("long", stop, target, partial)

    def enter_short(
        self,
        stop: float,
        target: float | None = None,
        partial: "PartialPlan | None" = None,
    ) -> None:
        self._engine.request_entry("short", stop, target, partial)

    def close(self) -> None:
        self._engine.request_close()


# --------------------------------------------------------------------------- #
# Strategy interface
# --------------------------------------------------------------------------- #
class Strategy:
    """Base class for all engine strategies (the Stage-2 contract).

    Subclasses override :meth:`on_session_start` (reset daily state, read the
    session's levels) and :meth:`on_bar` (inspect ``ctx`` and request orders).
    Both default to no-ops so a strategy can implement only what it needs.
    """

    def on_session_start(self, ctx: Context) -> None:  # noqa: D401
        """Called once at the start of each RTH session, before any bars."""

    def on_bar(self, ctx: Context) -> None:
        """Called for each in-session bar after the engine refreshes ctx."""


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class BacktestEngine:
    """Deterministic, single-pass, bar-by-bar simulator.

    Groups bars by session, drives a :class:`Strategy`, fills market entries at
    the next bar's open, manages OCO brackets intrabar, applies a
    :class:`CostModel` to every fill, enforces EOD-flat and one-attempt-per-
    side-per-day, and records every trade into a :class:`BacktestResult`.
    """

    def __init__(
        self,
        strategy: Strategy,
        cost_model: CostModel,
        *,
        symbol: str,
        asset_class: str = "equity",
        tick: float = 0.01,
        initial_equity: float = 100_000.0,
        percent_of_equity: float = 1.0,
        model_stop_gaps: bool = False,
    ):
        self.strategy = strategy
        self.cost_model = cost_model
        self.symbol = symbol
        self.asset_class = asset_class
        self.tick = tick
        self.initial_equity = float(initial_equity)
        self.percent_of_equity = float(percent_of_equity)
        # When True, a STOP whose bar OPENS beyond it fills at the OPEN (an
        # adverse gap-through becomes a market fill) — the realistic behavior.
        # When False (TradingView-parity), stops fill at the stop price even on
        # a gap-through (TV's standard optimistic backtest assumption). Targets
        # are resting limits and ALWAYS fill at the limit price under both.
        self.model_stop_gaps = bool(model_stop_gaps)

        # Mutable run state.
        self.equity = float(initial_equity)
        self.position: Position | None = None
        self._pending: _PendingOrder | None = None
        self._trades: list[TradeRecord] = []
        self._equity_points: list[tuple[datetime, float]] = []
        self._ctx = Context(self)

    # ----------------------------------------------------------- public API
    def run(
        self,
        bars: Sequence[Bar],
        levels_by_session,
        session_of,
    ) -> BacktestResult:
        """Run the simulation.

        Parameters
        ----------
        bars
            In-session (e.g. RTH) bars, ascending by ts. Only bars that should
            be simulated should be passed in (the caller pre-filters to RTH).
        levels_by_session
            Mapping ``session_date -> levels dict`` (pdh, pdl, ...).
        session_of
            Callable ``Bar -> session_date`` (e.g. ``et_session_date(bar.ts)``).
        """
        # Group bars by session, preserving order.
        sessions: list[tuple[object, list[Bar]]] = []
        cur_key = object()
        cur_list: list[Bar] = []
        for b in bars:
            k = session_of(b)
            if k != cur_key:
                if cur_list:
                    sessions.append((cur_key, cur_list))
                cur_key = k
                cur_list = []
            cur_list.append(b)
        if cur_list:
            sessions.append((cur_key, cur_list))

        for session_date, session_bars in sessions:
            self._run_session(session_date, session_bars, levels_by_session)

        equity_curve = pd.Series(
            [v for _, v in self._equity_points],
            index=pd.DatetimeIndex([t for t, _ in self._equity_points]),
            dtype="float64",
            name="equity",
        )
        trades_df = TradeRecord.to_frame(self._trades)
        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_curve,
            initial_equity=self.initial_equity,
            symbol=self.symbol,
        )

    # ----- order requests routed from the Context -----
    def request_entry(
        self,
        side: str,
        stop: float,
        target: float | None,
        partial: "PartialPlan | None" = None,
    ) -> None:
        """Queue a market entry to fill at the next bar's open (Pine parity)."""
        if self.position is not None:
            return  # already in a position; ignore (one position at a time)
        bar = self._ctx.bar
        self._pending = _PendingOrder(
            kind="enter",
            side=side,
            stop=stop,
            target=target,
            signal_close=bar.close,
            signal_ts=bar.ts,
            partial=partial,
        )

    def request_close(self) -> None:
        """Queue a market close to fill at the next bar's open."""
        if self.position is None:
            return
        self._pending = _PendingOrder(
            kind="close", side=None, stop=None, target=None,
            signal_close=None, signal_ts=None,
        )

    # ----------------------------------------------------------- per session
    def _run_session(self, session_date, session_bars, levels_by_session) -> None:
        # Force-flat any leftover position at the session boundary (defensive;
        # EOD-flat should have already closed it on the prior session).
        if self.position is not None:
            last = session_bars[0]
            self._exit_position(last, last.open, FILL_MARKET, "session_reset")

        self._pending = None
        levels = levels_by_session.get(session_date, {}) or {}

        ctx = self._ctx
        ctx.levels = levels
        ctx.in_session = True
        ctx.is_eod = False
        self.strategy.on_session_start(ctx)

        n = len(session_bars)
        for i, bar in enumerate(session_bars):
            is_eod = i == n - 1

            # 1) Execute any order pending from the PRIOR bar at THIS bar's open.
            self._execute_pending(bar, i)

            # 2) Manage an open OCO bracket against THIS bar's range — but only
            #    on bars AFTER the entry bar. The entry filled at this bar's
            #    open; per the OCO spec the bracket is managed "each subsequent
            #    bar", matching Pine where a market entry and its protective
            #    stop/target are not both resolved on the same bar's range.
            pos = self.position
            if pos is not None and pos.entry_bar_index != i:
                pos.bars_held += 1
                self._update_excursions(bar)
                self._manage_bracket(bar)

            # 3) Strategy sees the bar (may queue an order for the next open).
            ctx.bar = bar
            ctx.bar_index = i
            ctx.prev_bars = list(session_bars[:i])
            ctx.is_eod = is_eod
            ctx.in_session = True
            # Do not let the strategy open a new position on the EOD bar — it
            # could never be managed (and Pine would flatten it immediately).
            if not (is_eod and self.position is None):
                self.strategy.on_bar(ctx)

            # 4) EOD-flat: close any open position at this bar's close.
            if is_eod and self.position is not None:
                self._exit_position(bar, bar.close, FILL_MARKET, "eod_flat")
                self._pending = None

            # Mark equity at each bar close (mark-to-close while in a position).
            self._record_equity(bar)

        # Any order queued on the EOD bar never fills (no next bar this session).
        self._pending = None

    # --------------------------------------------------------- order exec
    def _execute_pending(self, bar: Bar, bar_index: int) -> None:
        pend = self._pending
        self._pending = None
        if pend is None:
            return

        if pend.kind == "close":
            if self.position is not None:
                self._exit_position(bar, bar.open, FILL_MARKET, "strategy_close")
            return

        # kind == 'enter'
        if self.position is not None:
            return
        side = pend.side
        ref_entry = bar.open  # market order fills at the next bar's open
        risk = abs(pend.signal_close - pend.stop)
        if risk <= 0:
            return  # skip degenerate risk (per V0 rules)

        fill = self.cost_model.apply_entry(side, ref_entry, self.asset_class)
        shares = self._size(fill)
        if shares <= 0:
            return
        notional = shares * fill
        entry_comm = self.cost_model.commission(shares, notional, self.asset_class)

        self.position = Position(
            side=side,
            shares=shares,
            entry_ts=bar.ts,
            entry_price=fill,
            ref_entry_price=ref_entry,
            stop=pend.stop,
            target=pend.target,
            entry_commission=entry_comm,
            signal_close=pend.signal_close,
            entry_bar_index=bar_index,
            partial=pend.partial,
            initial_shares=shares,
            ref_stop=pend.stop,
        )

    def _size(self, fill_price: float) -> float:
        """Percent-of-equity sizing (Pine parity: 100% of equity, fractional)."""
        if fill_price <= 0:
            return 0.0
        return (self.equity * self.percent_of_equity) / fill_price

    # ------------------------------------------------------ bracket mgmt
    def _manage_bracket(self, bar: Bar) -> None:
        """Resolve the OCO bracket against ``bar``'s range (Pine-faithful).

        Fill semantics (the documented reproduction-delta policy):
          - TARGET is a resting LIMIT: it fills at the TARGET PRICE — even when
            the bar gaps through it (TradingView assumes limit orders fill at
            the limit price). This is what keeps a 2R fixed target capped at ~2R
            and is required for parity with the Pine reference.
          - STOP fills at the STOP PRICE normally, but if the bar OPENS beyond
            the stop (an adverse gap), the stop becomes a market order and fills
            at the OPEN (the realistic, worse price). Tagged ``stop_gap``.
          - If a single bar's range spans BOTH stop and target, the STOP fills
            first (conservative — bar data hides intrabar order). A simultaneous
            adverse open-gap through the stop still wins (worst case first).
        """
        pos = self.position
        if pos is None:
            return
        if pos.partial is not None:
            self._manage_partial_bracket(bar)
            return
        stop = pos.stop
        target = pos.target

        if pos.side == "long":
            # Adverse gap: bar opens at/below the stop -> stop fills at the open
            # (only when gap modeling is on; TV-parity fills at the stop price).
            if self.model_stop_gaps and bar.open <= stop:
                self._exit_position(bar, bar.open, FILL_STOP, "stop_gap")
                return
            hit_stop = bar.low <= stop
            hit_target = target is not None and bar.high >= target
            if hit_stop:                       # stop-first on any ambiguity
                self._exit_position(bar, stop, FILL_STOP, "stop")
            elif hit_target:                   # limit fills at the target price
                self._exit_position(bar, target, FILL_LIMIT, "target")
        else:  # short
            if self.model_stop_gaps and bar.open >= stop:
                self._exit_position(bar, bar.open, FILL_STOP, "stop_gap")
                return
            hit_stop = bar.high >= stop
            hit_target = target is not None and bar.low <= target
            if hit_stop:
                self._exit_position(bar, stop, FILL_STOP, "stop")
            elif hit_target:
                self._exit_position(bar, target, FILL_LIMIT, "target")

    # ------------------------------------------------ partial + runner mgmt
    def _manage_partial_bracket(self, bar: Bar) -> None:
        """Resolve a partial-exit + trailing-runner position against ``bar``.

        Two phases (V3/V4 positive-skew structure, MASTER_PLAN §1.B):

        PHASE 1 (before TP1): the full position rests with a protective stop and
        a partial-limit at ``tp1``. Stop-first on ambiguity, exactly like the
        fixed bracket. If TP1 is reached, scale out ``fraction`` at the tp1
        limit, book that partial P&L on the position, and move the runner's stop
        to breakeven (the entry signal close). The runner survives to phase 2.

        PHASE 2 (runner): no fixed target — the stop trails by the PRIOR managed
        bar's low (long) / high (short), never loosening. A hit closes the
        runner (reason ``trail_stop``); otherwise EOD-flat closes it at the
        session close, folding the booked partial P&L into the final record.
        """
        pos = self.position
        if pos is None:
            return
        pl = pos.partial

        if not pos.partial_done:
            # ---- PHASE 1: original stop + partial limit at tp1 -------------
            stop = pos.stop
            if pos.side == "long":
                if self.model_stop_gaps and bar.open <= stop:
                    self._exit_position(bar, bar.open, FILL_STOP, "stop_gap")
                    return
                hit_stop = bar.low <= stop
                hit_tp1 = bar.high >= pl.tp1
                if hit_stop:                   # stop-first on ambiguity
                    self._exit_position(bar, stop, FILL_STOP, "stop")
                    return
                if hit_tp1:
                    self._scale_out_partial(bar, pl.tp1)
            else:  # short
                if self.model_stop_gaps and bar.open >= stop:
                    self._exit_position(bar, bar.open, FILL_STOP, "stop_gap")
                    return
                hit_stop = bar.high >= stop
                hit_tp1 = bar.low <= pl.tp1
                if hit_stop:
                    self._exit_position(bar, stop, FILL_STOP, "stop")
                    return
                if hit_tp1:
                    self._scale_out_partial(bar, pl.tp1)
            return

        # ---- PHASE 2: runner with a trailing stop (no fixed target) -------
        stop = pos.stop
        if pos.side == "long":
            if self.model_stop_gaps and bar.open <= stop:
                self._exit_position(bar, bar.open, FILL_STOP, "stop_gap")
                return
            if bar.low <= stop:
                self._exit_position(bar, stop, FILL_STOP, "trail_stop")
                return
            # Trail up by this bar's low (becomes the next bar's stop floor).
            if pl.trail_mode == "prior_bar":
                pos.stop = max(pos.stop, bar.low)
        else:  # short
            if self.model_stop_gaps and bar.open >= stop:
                self._exit_position(bar, bar.open, FILL_STOP, "stop_gap")
                return
            if bar.high >= stop:
                self._exit_position(bar, stop, FILL_STOP, "trail_stop")
                return
            if pl.trail_mode == "prior_bar":
                pos.stop = min(pos.stop, bar.high)

    def _scale_out_partial(self, bar: Bar, tp1: float) -> None:
        """Sell ``fraction`` of the position at the ``tp1`` resting limit.

        Books the partial's net P&L and size-weighted R on the position, shrinks
        the share count to the runner, and moves the runner's stop to breakeven
        (the entry signal close). The remaining shares ride until the trailing
        stop or EOD-flat, at which point :meth:`_exit_position` folds in the
        booked partial.
        """
        pos = self.position
        if pos is None or pos.partial_done:
            return
        pl = pos.partial
        part_shares = pos.initial_shares * pl.fraction
        if part_shares <= 0 or part_shares >= pos.shares:
            return  # degenerate; leave the position whole

        exit_fill = self.cost_model.apply_exit(
            pos.side, tp1, self.asset_class, FILL_LIMIT
        )
        exit_comm = self.cost_model.commission(
            part_shares, part_shares * exit_fill, self.asset_class
        )
        if pos.side == "long":
            gross = (exit_fill - pos.entry_price) * part_shares
        else:
            gross = (pos.entry_price - exit_fill) * part_shares
        net = gross - exit_comm
        self.equity += net

        risk_per_share = abs(pos.signal_close - pos.stop)
        if pos.side == "long":
            move = exit_fill - pos.entry_price
        else:
            move = pos.entry_price - exit_fill
        r = (move / risk_per_share) if risk_per_share > 0 else 0.0

        pos.realized_partial_pnl += net
        pos.realized_partial_r += r * pl.fraction  # size-weight the booked R
        pos.shares -= part_shares
        pos.partial_done = True
        # Move the runner's stop to breakeven (the entry signal close).
        pos.stop = pos.signal_close

    def _update_excursions(self, bar: Bar) -> None:
        pos = self.position
        if pos is None:
            return
        ref = pos.signal_close
        if pos.side == "long":
            pos.mfe = max(pos.mfe, bar.high - ref)
            pos.mae = max(pos.mae, ref - bar.low)
        else:
            pos.mfe = max(pos.mfe, ref - bar.low)
            pos.mae = max(pos.mae, bar.high - ref)

    # ----------------------------------------------------------- exit
    def _exit_position(
        self, bar: Bar, ref_price: float, fill_kind: str, reason: str
    ) -> None:
        pos = self.position
        if pos is None:
            return
        exit_fill = self.cost_model.apply_exit(
            pos.side, ref_price, self.asset_class, fill_kind
        )
        exit_comm = self.cost_model.commission(
            pos.shares, pos.shares * exit_fill, self.asset_class
        )

        if pos.side == "long":
            gross = (exit_fill - pos.entry_price) * pos.shares
        else:
            gross = (pos.entry_price - exit_fill) * pos.shares
        costs = pos.entry_commission + exit_comm
        net = gross - costs

        # Realized R is measured against the signal-bar risk (Pine parity:
        # bracket is pinned to the signal close, entry filled at next open).
        # For a runner, the risk is the ORIGINAL stop distance (the runner's own
        # ``stop`` has trailed / moved to breakeven), so use the entry stop kept
        # implicit in signal_close-vs-original-risk via the partial's R bookkeep.
        if pos.partial is not None and pos.initial_shares > 0:
            risk_per_share = abs(pos.signal_close - pos.ref_stop)
        else:
            risk_per_share = abs(pos.signal_close - pos.stop)
        if pos.side == "long":
            price_move = exit_fill - pos.entry_price
        else:
            price_move = pos.entry_price - exit_fill
        runner_r = price_move / risk_per_share if risk_per_share > 0 else 0.0

        # Fold in any booked partial P&L and size-weight the runner's R.
        if pos.partial is not None and pos.initial_shares > 0:
            runner_fraction = pos.shares / pos.initial_shares
            gross += pos.realized_partial_pnl  # partial was net-of-comm already
            net += pos.realized_partial_pnl
            r_multiple = pos.realized_partial_r + runner_r * runner_fraction
        else:
            r_multiple = runner_r

        self.equity += net

        self._trades.append(
            TradeRecord(
                symbol=self.symbol,
                side=pos.side,
                entry_ts=pos.entry_ts,
                exit_ts=bar.ts,
                entry_price=pos.entry_price,
                exit_price=exit_fill,
                ref_entry_price=pos.ref_entry_price,
                signal_close=pos.signal_close,
                stop=(pos.ref_stop if pos.partial is not None else pos.stop),
                target=pos.target,
                shares=pos.shares,
                gross_pnl=gross,
                costs=costs,
                pnl=net,
                r_multiple=r_multiple,
                exit_reason=reason,
                bars_held=pos.bars_held,
                mfe=pos.mfe,
                mae=pos.mae,
                equity_after=self.equity,
            )
        )
        self.position = None

    # ----------------------------------------------------------- equity
    def _record_equity(self, bar: Bar) -> None:
        """Mark equity to the bar close (unrealized while in a position)."""
        mark = self.equity
        if self.position is not None:
            pos = self.position
            if pos.side == "long":
                unreal = (bar.close - pos.entry_price) * pos.shares
            else:
                unreal = (pos.entry_price - bar.close) * pos.shares
            mark = self.equity + unreal
        self._equity_points.append((bar.ts, mark))


# --------------------------------------------------------------------------- #
# Convenience: build Bar objects from a DataFrame
# --------------------------------------------------------------------------- #
def bars_from_df(df: pd.DataFrame) -> list[Bar]:
    """Build a list of :class:`Bar` from a DataFrame with OHLCV + ts_utc."""
    out: list[Bar] = []
    for row in df.itertuples(index=False):
        out.append(
            Bar(
                ts=row.ts_utc,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(getattr(row, "volume", 0.0) or 0.0),
            )
        )
    return out
