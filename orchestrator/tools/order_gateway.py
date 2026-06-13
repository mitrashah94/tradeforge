"""orchestrator/tools/order_gateway.py — the deterministic order gateway.

Subscribes to ``ORDER_INTENT`` and drives the ONE event path that paper and live
share (MASTER_PLAN.md §4, flows F1/F2; CLAUDE.md "one event path"):

    ORDER_INTENT
      -> halt gate (refuse while halted)
      -> risk caps (per-trade %, portfolio heat) via risk.config
      -> LIVE only: P0 hook (orchestrator.hooks.evaluate)
      -> ORDER_APPROVED | ORDER_VETOED
      -> venue adapter.submit()
      -> ORDER_SUBMITTED -> ORDER_WORKING -> ORDER_FILLED | ORDER_PARTIAL | ORDER_REJECTED
      -> drives the order FSM + OCO brackets in orderbook.state_machine

HOT PATH PURITY: this module performs NO LLM and NO MCP calls. The only "live"
surface is the gated :class:`LiveBroker` stub (raises unless explicitly enabled),
reachable solely behind the P0 hook AND a LIVE flag. Default venue = paper.

ORDER_INTENT ``data`` schema — THE FINAL RECONCILED CONTRACT (P3 seam #2).
The FastLoop (orchestrator/fast_loop/engine.py) emits exactly these fields; the
gateway requires only ``symbol``/``side``/``qty`` and defaults gracefully for
every optional field. The risk cap uses qty x stop_distance when ``order_notional``
is absent; ``portfolio_heat_pct`` defaults to 0; ``client_id`` (the loop's
correlation handle) is preserved through every emitted lifecycle event.
    {
      # --- core order (FastLoop always emits; gateway requires symbol/side/qty) -
      "symbol": str,                      # required
      "side": "buy"|"sell"|"long"|"short",# required
      "qty": float,                       # required (already vol-target sized)
      "order_type": "market"|"limit"|"stop"|"stop_market"|"stop_limit",  # default "market"
      "client_id": str,                   # loop correlation id (preserved in all events)
      # --- pricing: FastLoop sets BOTH to the signal close ---------------------
      "intended_price": float|None,       # slippage base / expected price
      "ref_price": float|None,            # price to fill against (paper venue)
      "limit_price": float|None,          # for limit legs (TP1)
      "stop_price": float|None,           # for stop legs
      # --- risk-cap inputs (FastLoop supplies equity + stop_distance_pct) ------
      "equity": float|None,               # current equity for per-trade cap
      "stop_distance_pct": float|None,    # |entry-stop|/entry; cap = qty*notional*this
      "portfolio_heat_pct": float|None,   # open heat for the heat cap (default 0)
      "order_notional": float|None,       # optional; else qty*intended_price
      "grade": "B"|"A"|"A+",              # conviction tier -> RI (default "A")
      "route": "paper"|"live",            # default "paper"
      "strategy": str,                    # strategy id (for ledger/journal)
      "reason": str,                      # provenance; ALSO selects the F2 path:
                                          #   entry_*           -> new entry (gated)
                                          #   tp1_partial_exit  -> ORDER_PARTIAL
                                          #   tp1_breakeven|trail-> stop relocation
                                          #   stop|trail_stop|target|time_stop|
                                          #   session_flatten|strategy_close|close
                                          #                     -> exit + POSITION_CLOSED
      # --- OCO bracket spec (entry only) --------------------------------------
      "bracket": {"stop_price": float, "target_price": float} | None,
      # --- live-only gate inputs (forwarded to orchestrator.hooks.evaluate) ----
      "strategy_status", "risk_token", "live_confirm", "buying_power", "tool_name"
    }

F2 management intents (tp1/stop-move/exit ``reason``s) BYPASS the entry halt+risk
gate — exits/risk-reducers must always execute, even while halted — and are
translated to the lifecycle events the FastLoop's state machine consumes.

Events emitted (all carry the originating intent's ``client_intent_id`` in data
so subscribers can correlate):
    ORDER_APPROVED, ORDER_VETOED, ORDER_SUBMITTED, ORDER_WORKING,
    ORDER_FILLED, ORDER_PARTIAL, ORDER_REJECTED, ORDER_CANCELLED
"""

from __future__ import annotations

import uuid

from orchestrator.events import Event, EventType
from orderbook.state_machine import OrderBook, OrderState

# Halt-inducing events the gateway watches. While any of these is "on" (and not
# cleared), the gateway VETOES new intents — refusing approvals while halted.
_HALT_ON = {
    EventType.CIRCUIT_BREAKER_TRIPPED,
    EventType.COOLDOWN_STARTED,
    EventType.NO_TRADE_WINDOW,
}
# Events that clear a halt (a sibling breaker agent may emit these; we honor a
# ``data["cleared"]`` flag or a RECONCILED resume).
_HALT_CLEAR = {EventType.RECONCILED}

# --- F2 lifecycle intent classification (MASTER_PLAN §4 flow F2) -------------- #
# The FastLoop publishes management actions as fresh ORDER_INTENT events whose
# ``reason`` names the management step. The gateway recognises these so it can
# translate the venue fill into the lifecycle event the loop is waiting on
# (ORDER_PARTIAL after a TP1 scale-out; POSITION_CLOSED after an exit), and so a
# stop-MOVE relocates the bracket's stop leg instead of opening a new position.
_TP1_REASONS = {"tp1_partial_exit"}
_STOP_MOVE_REASONS = {"tp1_breakeven", "trail"}
_EXIT_REASONS = {
    "stop", "trail_stop", "target", "time_stop", "session_flatten",
    "strategy_close", "close",
}


class OrderGateway:
    """Deterministic ORDER_INTENT consumer: risk/halt/hook gate + venue routing.

    Construct with an injectable ``bus`` (anything exposing ``publish`` /
    ``subscribe``), an :class:`OrderBook`, a default ``paper_broker`` and an
    optional ``live_broker`` (the gated stub). Call :meth:`register` to wire the
    subscriptions, or drive :meth:`on_intent` directly in tests.
    """

    def __init__(
        self,
        bus,
        orderbook: OrderBook,
        paper_broker,
        live_broker=None,
        limits=None,
        source: str = "order_gateway",
    ):
        self.bus = bus
        self.ob = orderbook
        self.paper_broker = paper_broker
        self.live_broker = live_broker
        self._limits = limits
        self.source = source
        self._halted = False
        self._halt_reason = ""

    # ------------------------------------------------------------------ #
    # wiring                                                             #
    # ------------------------------------------------------------------ #
    def register(self) -> None:
        """Subscribe to ORDER_INTENT and the halt-state events."""
        self.bus.subscribe(EventType.ORDER_INTENT, self.on_intent)
        self.bus.subscribe(list(_HALT_ON), self.on_halt_event)
        self.bus.subscribe(list(_HALT_CLEAR), self.on_clear_event)

    @property
    def halted(self) -> bool:
        return self._halted

    def on_halt_event(self, event: Event) -> None:
        """Set the halt latch when a breaker/cooldown/no-trade event arrives."""
        # A breaker may explicitly signal clearance with data["cleared"] True.
        if event.data.get("cleared"):
            self._halted = False
            self._halt_reason = ""
            return
        self._halted = True
        self._halt_reason = event.data.get("reason", event.type.value)

    def on_clear_event(self, event: Event) -> None:
        """Clear the halt latch on a RECONCILED resume (or explicit resume)."""
        if event.data.get("resume", True):
            self._halted = False
            self._halt_reason = ""

    def _limits_obj(self):
        if self._limits is None:
            from risk.config import load_limits  # lazy

            self._limits = load_limits()
        return self._limits

    # ------------------------------------------------------------------ #
    # the gate + route                                                   #
    # ------------------------------------------------------------------ #
    def on_intent(self, event: Event) -> dict:
        """Handle one ORDER_INTENT: gate, approve/veto, route, drive FSM.

        Returns a result dict for direct callers/tests; emits the lifecycle
        events on the bus regardless.
        """
        d = dict(event.data or {})
        intent_id = d.get("client_intent_id") or f"intent_{uuid.uuid4().hex[:10]}"
        d["client_intent_id"] = intent_id
        route = (d.get("route") or "paper").lower()

        # --- F2 management intents bypass the entry gate. ---
        # TP1 scale-outs, stop-moves and exits REDUCE risk; they must execute
        # even while halted (you must always be able to flatten). They are routed
        # to the lifecycle path, which advances the loop's F2 state machine.
        reason = d.get("reason", "")
        if reason in _TP1_REASONS:
            return self._on_tp1_partial(d, route)
        if reason in _STOP_MOVE_REASONS:
            return self._on_stop_move(d, route)
        if reason in _EXIT_REASONS:
            return self._on_exit(d, route)

        # --- (a) halt gate: refuse NEW-ENTRY approvals while halted. ---
        if self._halted:
            return self._veto(d, f"halted: {self._halt_reason}")

        # --- (b) risk caps (per-trade %, portfolio heat). ---
        risk_fail = self._check_risk(d)
        if risk_fail is not None:
            return self._veto(d, risk_fail)

        # --- (c) LIVE only: the P0 hook. ---
        if route == "live":
            hook_fail = self._check_live_hook(d)
            if hook_fail is not None:
                return self._veto(d, hook_fail)

        # --- approved: stage + drive the order through the FSM + venue. ---
        return self._approve_and_route(d, route)

    # ---- gate helpers ----
    def _check_risk(self, d: dict) -> str | None:
        """Return a veto reason string if a risk cap is breached, else None.

        Caps are only enforced when their inputs are present (mirrors the hook).
        """
        try:
            limits = self._limits_obj()
            from risk.sizing import per_trade_dollar_risk, resolve_ri  # lazy

            grade = d.get("grade", "A") or "A"
            ri = resolve_ri(grade, limits)

            equity = d.get("equity")
            stop_distance_pct = d.get("stop_distance_pct")
            order_notional = d.get("order_notional")
            if order_notional is None and equity is not None:
                # Fall back to qty * intended_price when notional not supplied.
                px = d.get("intended_price") or d.get("ref_price")
                if px is not None and d.get("qty") is not None:
                    order_notional = float(px) * float(d["qty"])

            if equity is not None and order_notional is not None and stop_distance_pct is not None:
                trade_risk = float(order_notional) * float(stop_distance_pct)
                cap = per_trade_dollar_risk(float(equity), ri, limits)
                if trade_risk > cap:
                    return (f"per-trade risk {trade_risk:.2f} exceeds cap "
                            f"{cap:.2f} at RI {ri}")

            heat = d.get("portfolio_heat_pct")
            if heat is not None:
                heat_cap = limits.level(ri).portfolio_heat_pct
                if float(heat) > heat_cap:
                    return f"portfolio heat {heat} exceeds cap {heat_cap} at RI {ri}"
        except Exception as exc:  # noqa: BLE001 — FAIL CLOSED on risk-load error.
            return f"cannot verify risk caps ({exc}); failing closed"
        return None

    def _check_live_hook(self, d: dict) -> str | None:
        """Return a veto reason if the P0 hook blocks the live order, else None."""
        from orchestrator.hooks import evaluate  # lazy

        payload = {
            "tool_name": d.get("tool_name", "order_gateway_live"),
            "tool_input": {},
            "context": {
                "strategy_status": d.get("strategy_status"),
                "risk_token": d.get("risk_token"),
                "live_confirm": d.get("live_confirm"),
                "equity": d.get("equity"),
                "order_notional": d.get("order_notional"),
                "stop_distance_pct": d.get("stop_distance_pct"),
                "portfolio_heat_pct": d.get("portfolio_heat_pct"),
                "buying_power": d.get("buying_power"),
                "grade": d.get("grade", "A"),
            },
        }
        decision = evaluate(payload, limits=self._limits)
        if not decision.allow or decision.route != "live":
            return f"live hook blocked: {decision.reason}"
        return None

    # ---- emit helpers ----
    def _emit(self, etype: EventType, data: dict) -> Event:
        return self.bus.publish(Event(type=etype, data=data, source=self.source))

    def _veto(self, d: dict, reason: str) -> dict:
        data = {**d, "reason": reason}
        self._emit(EventType.ORDER_VETOED, data)
        return {"approved": False, "reason": reason, "intent_id": d["client_intent_id"]}

    # ------------------------------------------------------------------ #
    # approve + route through the FSM + venue                            #
    # ------------------------------------------------------------------ #
    def _approve_and_route(self, d: dict, route: str) -> dict:
        broker = self._broker_for(route)

        # Stage the entry order in the ledger, then APPROVE it.
        entry_id = self.ob.create_order(
            symbol=d["symbol"], side=d["side"], qty=float(d["qty"]),
            order_type=d.get("order_type", "market"),
            intended_price=d.get("intended_price"),
            limit_price=d.get("limit_price"), stop_price=d.get("stop_price"),
            strategy=d.get("strategy", ""), route=route,
            state=OrderState.STAGED,
        )
        self.ob.transition(entry_id, OrderState.APPROVED)

        # Optionally stage an OCO bracket (stop + target protective legs).
        bracket_id = None
        bspec = d.get("bracket")
        if bspec:
            bracket_id = self._stage_bracket(d, entry_id, route, bspec)

        approved_data = {**d, "order_id": entry_id, "bracket_id": bracket_id, "route": route}
        self._emit(EventType.ORDER_APPROVED, approved_data)

        # Submit the entry to the venue.
        self.ob.transition(entry_id, OrderState.SUBMITTED)
        self._emit(EventType.ORDER_SUBMITTED, {**approved_data})

        entry_order = self.ob.get_order(entry_id)
        result = broker.submit(entry_order, ref_price=d.get("ref_price"))

        return self._handle_venue_result(d, entry_id, bracket_id, result, route)

    def _stage_bracket(self, d: dict, entry_id: str, route: str, bspec: dict) -> str:
        """Create the stop + target protective legs and the OCO group.

        # REQUIRES dead-man's switch (watchdog.py, P6) — local OCO brackets.
        """
        exit_side = "sell" if d["side"].lower() in ("buy", "long") else "buy"
        stop_id = self.ob.create_order(
            symbol=d["symbol"], side=exit_side, qty=float(d["qty"]),
            order_type="stop_market", stop_price=bspec.get("stop_price"),
            intended_price=bspec.get("stop_price"),
            strategy=d.get("strategy", ""), route=route, parent_id=entry_id,
            state=OrderState.STAGED,
        )
        target_id = self.ob.create_order(
            symbol=d["symbol"], side=exit_side, qty=float(d["qty"]),
            order_type="limit", limit_price=bspec.get("target_price"),
            intended_price=bspec.get("target_price"),
            strategy=d.get("strategy", ""), route=route, parent_id=entry_id,
            state=OrderState.STAGED,
        )
        return self.ob.create_bracket(d["symbol"], entry_id, stop_id, target_id)

    def _broker_for(self, route: str):
        if route == "live":
            if self.live_broker is None:
                raise RuntimeError("no live broker configured; live route refused")
            return self.live_broker
        return self.paper_broker

    def _handle_venue_result(
        self, d: dict, entry_id: str, bracket_id: str | None, result: dict, route: str
    ) -> dict:
        """Translate the venue result into FSM transitions + lifecycle events."""
        status = result.get("status")
        base = {**d, "order_id": entry_id, "bracket_id": bracket_id,
                "route": route, "venue_order_id": result.get("venue_order_id")}

        if status == "rejected":
            self.ob.transition(entry_id, OrderState.REJECTED)
            self._emit(EventType.ORDER_REJECTED, {**base, "reason": result.get("reason", "")})
            return {"approved": True, "filled": False, "order_id": entry_id,
                    "status": "rejected"}

        # Order accepted by the venue -> WORKING.
        self.ob.transition(entry_id, OrderState.WORKING)
        self._emit(EventType.ORDER_WORKING, base)
        if bracket_id is not None:
            # Protective legs go live (SUBMITTED->WORKING) once the entry works.
            self._work_bracket_legs(bracket_id)

        if status == "working":
            return {"approved": True, "filled": False, "order_id": entry_id,
                    "status": "working"}

        # Fill (or partial).
        fill_price = result.get("fill_price")
        filled_qty = result.get("filled_qty", d.get("qty"))
        fill = self.ob.record_fill(entry_id, price=fill_price, qty=filled_qty)

        if status == "partial":
            self.ob.transition(entry_id, OrderState.PARTIAL)
            self._emit(EventType.ORDER_PARTIAL, {**base, **fill})
            return {"approved": True, "filled": False, "order_id": entry_id,
                    "status": "partial", "fill": fill}

        # Full fill: FILLED, open the position, activate the bracket.
        self.ob.transition(entry_id, OrderState.FILLED)
        pos_id = self.ob.open_position(
            symbol=d["symbol"], side=d["side"], qty=float(filled_qty),
            avg_price=fill_price,
        )
        if bracket_id is not None:
            self.ob.activate_bracket(bracket_id)
        self._emit(EventType.ORDER_FILLED, {**base, **fill, "position_id": pos_id})
        return {"approved": True, "filled": True, "order_id": entry_id,
                "status": "filled", "fill": fill, "position_id": pos_id,
                "bracket_id": bracket_id}

    def _work_bracket_legs(self, bracket_id: str) -> None:
        """Move the protective legs APPROVED->SUBMITTED->WORKING (local OCO)."""
        bracket = self.ob.get_bracket(bracket_id)
        for leg_id in (bracket["stop_id"], bracket["target_id"]):
            state = self.ob.get_state(leg_id)
            if state == OrderState.STAGED:
                self.ob.transition(leg_id, OrderState.APPROVED)
            if self.ob.get_state(leg_id) == OrderState.APPROVED:
                self.ob.transition(leg_id, OrderState.SUBMITTED)
            if self.ob.get_state(leg_id) == OrderState.SUBMITTED:
                self.ob.transition(leg_id, OrderState.WORKING)

    # ------------------------------------------------------------------ #
    # F2 lifecycle management intents (TP1 / stop-move / exit)            #
    # ------------------------------------------------------------------ #
    def _on_tp1_partial(self, d: dict, route: str) -> dict:
        """A TP1 scale-out intent: fill the partial at the venue, REDUCE the
        position, and emit ORDER_PARTIAL (which the FastLoop reads to ratchet the
        stop to breakeven). Carries the loop's ``client_id`` so the loop can
        correlate the partial with its managed position.
        """
        symbol = d["symbol"]
        qty = float(d["qty"])
        price = d.get("limit_price") or d.get("ref_price") or d.get("intended_price")
        broker = self._broker_for(route)

        # Stage->fill a single-leg exit order at the venue (slippage capture).
        oid = self.ob.create_order(
            symbol=symbol, side=d["side"], qty=qty, order_type="limit",
            limit_price=d.get("limit_price"), intended_price=price,
            strategy=d.get("strategy", ""), route=route, state=OrderState.STAGED,
        )
        self.ob.transition(oid, OrderState.APPROVED)
        self.ob.transition(oid, OrderState.SUBMITTED)
        result = broker.submit(self.ob.get_order(oid), ref_price=price)
        fill_price = result.get("fill_price", price)
        self.ob.transition(oid, OrderState.WORKING)
        fill = self.ob.record_fill(oid, price=fill_price, qty=qty)
        self.ob.transition(oid, OrderState.FILLED)

        pos = self.ob.position_for_symbol(symbol)
        reduced = None
        if pos is not None:
            reduced = self.ob.reduce_position(pos.position_id, exit_price=fill_price, qty=qty)

        out = {**d, "order_id": oid, "route": route, **fill}
        if reduced is not None:
            out["position_id"] = pos.position_id
            out["partial_realized_pnl"] = reduced["partial_realized_pnl"]
            out["realized_pnl"] = reduced["partial_realized_pnl"]
        self._emit(EventType.ORDER_PARTIAL, out)
        return {"approved": True, "filled": True, "order_id": oid,
                "status": "partial", "fill": fill, "reduced": reduced}

    def _on_stop_move(self, d: dict, route: str) -> dict:
        """A stop relocation (breakeven / trail): move the bracket's stop leg
        price in place. No fill, no position change — the leg stays WORKING.
        """
        symbol = d["symbol"]
        new_stop = d.get("stop_price")
        bracket = self.ob.active_bracket_for_symbol(symbol)
        moved_leg = None
        if bracket is not None and new_stop is not None:
            moved_leg = bracket["stop_id"]
            self.ob.update_order_stop(moved_leg, float(new_stop))
        out = {**d, "route": route, "bracket_id": bracket["bracket_id"] if bracket else None,
               "order_id": moved_leg, "new_stop": new_stop}
        # Surface as an ORDER_WORKING ack (deterministic record of the relocation).
        self._emit(EventType.ORDER_WORKING, out)
        return {"approved": True, "filled": False, "status": "stop_moved",
                "order_id": moved_leg, "new_stop": new_stop}

    def _on_exit(self, d: dict, route: str) -> dict:
        """An exit intent (stop/trail/target/time/EOD/strategy close): fill the
        exit at the venue, CLOSE the position, cancel any remaining bracket legs,
        and emit ORDER_FILLED + POSITION_CLOSED (which frees the loop's slot).
        """
        symbol = d["symbol"]
        qty = float(d["qty"]) if d.get("qty") is not None else None
        price = d.get("ref_price") or d.get("intended_price") or d.get("stop_price") \
            or d.get("limit_price")
        broker = self._broker_for(route)

        otype = d.get("order_type", "market")
        oid = self.ob.create_order(
            symbol=symbol, side=d["side"], qty=qty or 0.0, order_type=otype,
            stop_price=d.get("stop_price"), limit_price=d.get("limit_price"),
            intended_price=price, strategy=d.get("strategy", ""), route=route,
            state=OrderState.STAGED,
        )
        self.ob.transition(oid, OrderState.APPROVED)
        self.ob.transition(oid, OrderState.SUBMITTED)
        result = broker.submit(self.ob.get_order(oid), ref_price=price)
        fill_price = result.get("fill_price", price)
        self.ob.transition(oid, OrderState.WORKING)
        fill = self.ob.record_fill(oid, price=fill_price, qty=qty)
        self.ob.transition(oid, OrderState.FILLED)
        self._emit(EventType.ORDER_FILLED, {**d, "order_id": oid, "route": route, **fill})

        # Cancel any still-working protective legs (local OCO teardown).
        bracket = self.ob.active_bracket_for_symbol(symbol)
        if bracket is not None:
            for leg_id in (bracket["stop_id"], bracket["target_id"]):
                st = self.ob.get_state(leg_id)
                if st is not None and st not in (OrderState.FILLED, OrderState.CANCELLED,
                                                 OrderState.REJECTED, OrderState.EXPIRED):
                    self.ob._force_cancel(leg_id)
                    self._emit(EventType.ORDER_CANCELLED,
                               {"order_id": leg_id, "symbol": symbol,
                                "bracket_id": bracket["bracket_id"],
                                "reason": "exit closed bracket"})

        pos = self.ob.position_for_symbol(symbol)
        closed = None
        if pos is not None:
            closed = self.ob.close_position(pos.position_id, exit_price=fill_price)
            self._emit(EventType.POSITION_CLOSED, {"symbol": symbol, **d, **closed})
        return {"approved": True, "filled": True, "order_id": oid,
                "status": "closed", "closed": closed}

    # ------------------------------------------------------------------ #
    # bracket leg fill -> OCO sibling cancel (driven by the fast loop)   #
    # ------------------------------------------------------------------ #
    def fill_bracket_leg(self, leg_order_id: str, price: float) -> dict:
        """Record a protective-leg fill and cancel its OCO sibling.

        Called by the fast loop / venue callback when a stop or target leg
        triggers. Records the fill, transitions the leg to FILLED, cancels the
        sibling, closes the position, and emits ORDER_FILLED + ORDER_CANCELLED.
        Returns a summary dict.
        """
        leg = self.ob.get_order(leg_order_id)
        if leg is None:
            raise KeyError(f"unknown leg {leg_order_id}")
        fill = self.ob.record_fill(leg_order_id, price=price)
        self.ob.transition(leg_order_id, OrderState.FILLED)
        cancelled = self.ob.on_leg_fill(leg_order_id)

        base = {"order_id": leg_order_id, "symbol": leg.symbol, "side": leg.side,
                "leg": leg.leg, "bracket_id": leg.bracket_id, "route": leg.route}
        self._emit(EventType.ORDER_FILLED, {**base, **fill})

        # Close the position on the symbol (protective leg fill = exit).
        pos = self.ob.position_for_symbol(leg.symbol)
        closed = None
        if pos is not None:
            closed = self.ob.close_position(pos.position_id, exit_price=price)
            self._emit(EventType.POSITION_CLOSED,
                       {"symbol": leg.symbol, **closed})

        for cid in cancelled:
            c = self.ob.get_order(cid)
            self._emit(EventType.ORDER_CANCELLED,
                       {"order_id": cid, "symbol": c.symbol if c else leg.symbol,
                        "bracket_id": leg.bracket_id, "reason": "OCO sibling filled"})
        return {"filled_leg": leg_order_id, "cancelled": cancelled, "closed": closed}
