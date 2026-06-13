"""orchestrator/fast_loop/lifecycle.py — the deterministic open-position state machine (flow F2).

Flow F2 (MASTER_PLAN §4): ``FILLED -> manage (TP1 50%, stop->BE, trail) ->
close``. This module owns the *management* half of that — once the fast loop has
an open position, a pure, deterministic state machine drives it to flat:

    OPEN_PENDING  -- entry ORDER_INTENT published, awaiting fill
       | ORDER_FILLED
    OPEN          -- full position live; watching for TP1 / stop / time / EOD
       | bar.high|low reaches TP1                 -> publish partial-exit intent
    TP1_PENDING   -- partial-exit intent published, awaiting partial fill
       | ORDER_PARTIAL                            -> publish stop->BE move intent
    RUNNER        -- breakeven stop, trailing the remainder (prior-bar / ATR)
       | trail stop hit | time-stop | EOD-flat    -> publish close intent
    CLOSING       -- close intent published, awaiting POSITION_CLOSED
       | POSITION_CLOSED
    FLAT          -- terminal

Every state transition that touches the broker PUBLISHES an ``ORDER_INTENT``
(partial exit / stop-move / market close) and emits the matching lifecycle event
(``TP1_HIT`` / ``STOP_HIT`` / ``POSITION_CLOSED`` are surfaced by the engine when
fills arrive). The machine itself performs NO I/O beyond calling an injected
``emit(intent_data, reason)`` callback — it is pure and unit-testable.

This produces the positive-skew structure of MASTER_PLAN §1.B: scale a partial
at +1R (cap the loser potential), ratchet the stop to breakeven (make the runner
free), and trail to let the winner run. Cut losers fast, let winners run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LifecycleState(str, Enum):
    """States of the open-position management state machine."""

    OPEN_PENDING = "OPEN_PENDING"   # entry intent out, no fill yet
    OPEN = "OPEN"                   # full position live (pre-TP1)
    TP1_PENDING = "TP1_PENDING"     # partial-exit intent out, awaiting partial
    RUNNER = "RUNNER"               # breakeven + trailing the remainder
    CLOSING = "CLOSING"             # close intent out, awaiting POSITION_CLOSED
    FLAT = "FLAT"                   # terminal


@dataclass
class ManagedPosition:
    """The fast loop's deterministic view of one open position and its plan.

    Holds the bracket plan (stop / TP1 / trailing config) and the runtime state
    the state machine advances each bar and on each fill event. Prices are raw
    (pre-cost) intents; the order gateway / broker owns actual fills.
    """

    client_id: str
    symbol: str
    side: str                       # 'long' | 'short'
    strategy: str

    entry_price: float              # signal-close reference (bracket basis)
    stop_price: float               # current protective stop (moves to BE, trails)
    initial_stop: float             # original stop (runner risk basis)
    qty: float                      # current live qty (shrinks after TP1)
    initial_qty: float              # qty at entry

    # --- TP1 / partial / trailing plan ---
    tp1_price: float | None = None  # +1R partial-exit price (None -> no partial)
    tp1_fraction: float = 0.5       # fraction scaled out at TP1
    trail_mode: str = "prior_bar"   # 'prior_bar' | 'atr_chandelier' | 'none'
    trail_atr_mult: float = 3.0     # chandelier multiple (atr_chandelier mode)

    # --- time-stop ---
    time_stop_bars: int | None = None  # exit after N managed bars if not progressed
    target_price: float | None = None  # fixed target (non-partial path); optional

    # --- runtime state ---
    state: LifecycleState = LifecycleState.OPEN_PENDING
    bars_held: int = 0
    filled: bool = False
    tp1_done: bool = False
    prior_bar_extreme: float | None = None  # prior managed bar low(long)/high(short)

    def remaining_qty(self) -> float:
        return self.qty


@dataclass
class LifecycleAction:
    """One emitted intent + its lifecycle reason (returned for the engine/tests)."""

    intent: dict
    reason: str


class PositionLifecycle:
    """Deterministic per-position manager. Pure: emits intents via a callback.

    Construction takes a ``ManagedPosition`` and an ``emit`` callable. The fast
    loop calls :meth:`on_bar` for each managed bar and the ``on_*`` fill hooks
    when the bus delivers fill events. Each method returns the list of
    :class:`LifecycleAction` it produced (also passed to ``emit``) so tests can
    assert directly without a bus.
    """

    def __init__(self, pos: ManagedPosition, emit):
        self.pos = pos
        self._emit = emit

    # ------------------------------------------------------------ fill hooks
    def on_entry_filled(self) -> list[LifecycleAction]:
        """ORDER_FILLED for the entry: position is now live; start managing."""
        if self.pos.state is not LifecycleState.OPEN_PENDING:
            return []
        self.pos.filled = True
        self.pos.state = LifecycleState.OPEN
        return []

    def on_partial_filled(self) -> list[LifecycleAction]:
        """ORDER_PARTIAL for the TP1 scale-out: move the stop to BREAKEVEN.

        On TP1 fill, the remaining runner's stop ratchets to the entry price
        (breakeven), making the runner a free option (MASTER_PLAN §1.B). The
        position transitions OPEN/TP1_PENDING -> RUNNER and we publish a
        stop-move ORDER_INTENT.
        """
        if self.pos.tp1_done or self.pos.state not in (
            LifecycleState.TP1_PENDING,
            LifecycleState.OPEN,
        ):
            return []
        self.pos.tp1_done = True
        # Shrink the live qty to the runner.
        self.pos.qty = self.pos.initial_qty * (1.0 - self.pos.tp1_fraction)
        # Move stop to breakeven (the entry reference).
        self.pos.stop_price = self.pos.entry_price
        self.pos.state = LifecycleState.RUNNER
        action = self._make_stop_move(self.pos.entry_price, reason="tp1_breakeven")
        return [action]

    def on_position_closed(self) -> list[LifecycleAction]:
        """POSITION_CLOSED: terminal. No further intents."""
        self.pos.state = LifecycleState.FLAT
        self.pos.qty = 0.0
        return []

    # ------------------------------------------------------------ per-bar
    def on_bar(self, bar) -> list[LifecycleAction]:
        """Advance the state machine against one managed bar.

        ``bar`` is duck-typed: it must expose ``high``, ``low``, ``close`` (and
        an optional ``is_eod`` flag set by the engine). Returns the intents this
        bar produced. Order of checks is deterministic and conservative:

          1. EOD session-flatten wins (always be flat by RTH close).
          2. Stop / trail-stop hit -> close.
          3. (pre-TP1 only) TP1 reached -> partial-exit intent.
          4. (pre-TP1 only, if a fixed target exists) target reached -> close.
          5. Time-stop: after N managed bars, close.
          6. Trail the runner's stop (RUNNER state).
        """
        pos = self.pos
        if pos.state in (LifecycleState.FLAT, LifecycleState.OPEN_PENDING, LifecycleState.CLOSING):
            # Not actively manageable (no fill yet, already closing, or done).
            return []

        actions: list[LifecycleAction] = []
        pos.bars_held += 1
        is_eod = bool(getattr(bar, "is_eod", False))

        # 1) Session-flatten at EOD (RTH close): always be flat overnight.
        if is_eod:
            actions.append(self._close(reason="session_flatten", fill_ref=bar.close))
            return actions

        # 2) Protective / trailing stop hit -> close the (remaining) position.
        #    A stop exit fills at the STOP LEVEL (the breached price), not the bar
        #    close — this is what makes the runner's realized PnL correct.
        if self._stop_hit(bar):
            reason = "trail_stop" if pos.state is LifecycleState.RUNNER else "stop"
            actions.append(self._close(reason=reason, fill_ref=pos.stop_price))
            return actions

        # Pre-TP1 management.
        if pos.state is LifecycleState.OPEN and not pos.tp1_done:
            # 3) TP1 reached -> publish partial-exit intent, await ORDER_PARTIAL.
            if pos.tp1_price is not None and self._tp1_hit(bar):
                # Move to TP1_PENDING BEFORE emitting: on the real (synchronous)
                # bus the partial fill round-trips DURING the emit and calls
                # ``on_partial_filled`` (TP1_PENDING -> RUNNER). With a deferred
                # fake-bus partial, the state is still TP1_PENDING here, which
                # ``on_partial_filled`` accepts too. Either way we never clobber a
                # RUNNER set by a synchronous partial.
                pos.state = LifecycleState.TP1_PENDING
                actions.append(self._scale_out_tp1())
                # Still record prior-bar extreme for later trailing.
                self._update_prior_extreme(bar)
                return actions
            # 4) Fixed target (non-partial path) reached -> close at target.
            if pos.target_price is not None and self._target_hit(bar):
                actions.append(self._close(reason="target", fill_ref=pos.target_price))
                return actions

        # 5) Time-stop: exit if held >= N managed bars and not yet progressed.
        if pos.time_stop_bars is not None and pos.bars_held >= pos.time_stop_bars:
            actions.append(self._close(reason="time_stop", fill_ref=bar.close))
            return actions

        # 6) Trail the runner's stop (never loosening).
        if pos.state is LifecycleState.RUNNER:
            moved = self._trail(bar)
            if moved is not None:
                actions.append(moved)

        self._update_prior_extreme(bar)
        return actions

    # ------------------------------------------------------------ predicates
    def _stop_hit(self, bar) -> bool:
        pos = self.pos
        if pos.side == "long":
            return bar.low <= pos.stop_price
        return bar.high >= pos.stop_price

    def _tp1_hit(self, bar) -> bool:
        pos = self.pos
        if pos.side == "long":
            return bar.high >= pos.tp1_price
        return bar.low <= pos.tp1_price

    def _target_hit(self, bar) -> bool:
        pos = self.pos
        if pos.side == "long":
            return bar.high >= pos.target_price
        return bar.low <= pos.target_price

    # ------------------------------------------------------------ trailing
    def _trail(self, bar) -> LifecycleAction | None:
        """Trail the runner's stop; return a stop-move action if it tightened."""
        pos = self.pos
        new_stop = None
        if pos.trail_mode == "prior_bar":
            ref = pos.prior_bar_extreme
            if ref is None:
                return None
            if pos.side == "long":
                cand = ref
                if cand > pos.stop_price:
                    new_stop = cand
            else:
                cand = ref
                if cand < pos.stop_price:
                    new_stop = cand
        elif pos.trail_mode == "atr_chandelier":
            # Chandelier: stop = extreme -/+ mult*ATR. Here we approximate the
            # ATR band off the bar range when no ATR is supplied; engine may pass
            # a precomputed band via bar.trail_stop for exactness.
            band = getattr(bar, "trail_stop", None)
            if band is not None:
                if pos.side == "long" and band > pos.stop_price:
                    new_stop = band
                elif pos.side == "short" and band < pos.stop_price:
                    new_stop = band
        if new_stop is None:
            return None
        pos.stop_price = new_stop
        return self._make_stop_move(new_stop, reason="trail")

    def _update_prior_extreme(self, bar) -> None:
        self.pos.prior_bar_extreme = bar.low if self.pos.side == "long" else bar.high

    # ------------------------------------------------------------ intent emit
    def _opposite(self) -> str:
        return "sell" if self.pos.side == "long" else "buy"

    def _scale_out_tp1(self) -> LifecycleAction:
        pos = self.pos
        part_qty = pos.initial_qty * pos.tp1_fraction
        intent = {
            "client_id": f"{pos.client_id}:tp1",
            "symbol": pos.symbol,
            "side": self._opposite(),
            "qty": part_qty,
            "order_type": "limit",
            "limit_price": pos.tp1_price,
            "strategy": pos.strategy,
            "reason": "tp1_partial_exit",
        }
        return self._dispatch(intent, "tp1_partial_exit")

    def _make_stop_move(self, new_stop: float, reason: str) -> LifecycleAction:
        pos = self.pos
        intent = {
            "client_id": f"{pos.client_id}:stop",
            "symbol": pos.symbol,
            "side": self._opposite(),
            "qty": pos.remaining_qty(),
            "order_type": "stop",
            "stop_price": new_stop,
            "strategy": pos.strategy,
            "reason": reason,
        }
        return self._dispatch(intent, reason)

    def _close(self, reason: str, fill_ref: float | None = None) -> LifecycleAction:
        pos = self.pos
        pos.state = LifecycleState.CLOSING
        intent = {
            "client_id": f"{pos.client_id}:close",
            "symbol": pos.symbol,
            "side": self._opposite(),
            "qty": pos.remaining_qty(),
            "order_type": "market",
            "strategy": pos.strategy,
            "reason": reason,
        }
        # The realistic exit fill reference: the stop level for a stop exit, the
        # target for a target exit, the bar close for a mark-out (EOD/time). The
        # gateway/venue fills the market close against this so realized PnL is
        # correct end-to-end (it falls back to the entry only if none is known).
        if fill_ref is not None:
            intent["ref_price"] = fill_ref
            intent["intended_price"] = fill_ref
        return self._dispatch(intent, reason)

    def _dispatch(self, intent: dict, reason: str) -> LifecycleAction:
        action = LifecycleAction(intent=intent, reason=reason)
        if self._emit is not None:
            self._emit(intent, reason)
        return action
