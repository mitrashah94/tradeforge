"""orchestrator/hooks.py — live-order safety gate (single source of truth).

This module is the *testable core* of the TradeForge live-order safety hook. The
Claude Code PreToolUse shim (`.claude/hooks/block_live_orders.py`) is a thin,
stdlib-only wrapper that delegates here; all real decision logic lives in this
file so it can be unit-tested without the Claude Code runtime.

Design contract (see MASTER_PLAN §3/§4 and CLAUDE.md):
  - The brokerage is Robinhood via an agentic MCP. Live order-placement
    endpoints are MCP tools whose names contain `place_equity_order` (and the
    future `place_crypto_order` / `place_option_order`). `review_equity_order`
    is a *simulation* and must NOT be blocked.
  - No live order may pass unless THREE hard gates are satisfied: strategy is
    LIVE, a risk-approval token is present, and an interactive live-confirm is
    set. On top of that, numeric caps (per-trade $ risk, portfolio heat, live
    buying power) are enforced whenever the relevant inputs are present.
  - FAIL CLOSED: if the risk limits cannot be loaded while evaluating a
    live-order tool that already cleared the three hard gates, the order is
    blocked. A live order is never allowed when its caps cannot be verified.

`risk.config` / `risk.sizing` are imported LAZILY (inside functions) so that
merely importing this module never hard-requires pydantic/yaml — the shim may
run before project dependencies are installed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

# Matches live order-PLACEMENT endpoints: place_equity_order, place_crypto_order,
# place_option_order (and harmless plural variants). MCP tools are namespaced
# (e.g. "mcp__rh__place_equity_order"), so we search for the substring rather
# than anchoring. `review_equity_order`, `get_equity_quotes`, etc. do not match.
_LIVE_ORDER_RE = re.compile(r"place_(equity|crypto|option)s?_order")

# Truthy string values used for env-var / context coercion.
_TRUTHY = {"1", "true", "yes", "y", "on"}


def is_live_order_tool(tool_name: str) -> bool:
    """Return True if `tool_name` is a live order-PLACEMENT endpoint.

    A name matches when it contains a `place_(equity|crypto|option)_order`
    fragment OR the literal `order_gateway_live`. Read-only / simulation tools
    such as `review_equity_order`, `get_equity_quotes`, or any name containing
    "review" / "quote" / "get_" are NOT live-order tools and return False.
    """
    name = tool_name or ""
    return bool(_LIVE_ORDER_RE.search(name)) or "order_gateway_live" in name


def _truthy(value) -> bool:
    """Coerce a context/env value to a boolean.

    Real booleans pass through; strings are compared case-insensitively against
    the truthy set; everything else falls back to Python truthiness.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


def _first(*values):
    """Return the first value that is not None and not an empty string."""
    for v in values:
        if v is not None and v != "":
            return v
    return None


def _maybe_float(value):
    """Best-effort float coercion; return None on missing/unparseable input."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Decision:
    """Outcome of evaluating a tool call against the live-order safety gate.

    Attributes:
        allow:  True if the call may proceed (exit 0); False blocks it (exit 2).
        route:  "live" when a live order is permitted, otherwise "paper" — a
                blocked live order falls back to the paper path.
        reason: Human-readable explanation (joined fail reasons on a block).
        checks: Per-check booleans / values for observability.
    """

    allow: bool
    route: str
    reason: str
    checks: dict = field(default_factory=dict)


def evaluate(payload: dict, limits=None) -> Decision:
    """Evaluate a tool-call `payload` and decide whether a live order may pass.

    `payload` may contain `tool_name` (str), `tool_input` (dict) and `context`
    (dict). Values are gathered from `context` first, then `tool_input`, then OS
    environment fallbacks. Non-live-order tools are always allowed (routed to
    paper, since they never touch the live order path).

    For a live-order tool, ALL THREE hard gates must pass:
        (a) strategy_status == "LIVE"
        (b) risk_token present / non-empty
        (c) live_confirm truthy
    Then numeric caps are enforced whenever their inputs are present: per-trade
    $ risk vs the RI cap, portfolio heat vs the RI heat cap, and order notional
    vs live buying power.

    FAIL CLOSED: if the three hard gates pass but risk limits cannot be loaded,
    the order is blocked rather than allowed.
    """
    payload = payload or {}
    tool_name = payload.get("tool_name", "") or ""
    tool_input = payload.get("tool_input") or {}
    context = payload.get("context") or {}

    # --- Non-live-order endpoints: always allowed, never touch the live path. ---
    if not is_live_order_tool(tool_name):
        return Decision(
            allow=True,
            route="paper",
            reason="not a live-order endpoint",
            checks={"is_live_order_tool": False, "tool_name": tool_name},
        )

    # --- Gather inputs: context -> tool_input -> OS env fallback. ---
    def _ctx(key):
        return _first(context.get(key), tool_input.get(key))

    strategy_status = _first(_ctx("strategy_status"), os.environ.get("TRADEFORGE_STRATEGY_STATUS"))
    risk_token = _first(_ctx("risk_token"), os.environ.get("TRADEFORGE_RISK_TOKEN"))

    live_confirm_raw = _ctx("live_confirm")
    if live_confirm_raw is not None:
        live_confirm = _truthy(live_confirm_raw)
    else:
        live_confirm = os.environ.get("TRADEFORGE_LIVE_CONFIRM", "").strip().lower() in {"1", "true", "yes"}

    equity = _maybe_float(_ctx("equity"))
    order_notional = _maybe_float(_ctx("order_notional"))
    stop_distance_pct = _maybe_float(_ctx("stop_distance_pct"))
    portfolio_heat_pct = _maybe_float(_ctx("portfolio_heat_pct"))
    buying_power = _maybe_float(_first(_ctx("buying_power"), os.environ.get("TRADEFORGE_BUYING_POWER")))
    grade = context.get("grade", tool_input.get("grade", "A")) or "A"

    fail_reasons: list[str] = []
    checks: dict = {
        "is_live_order_tool": True,
        "tool_name": tool_name,
        "strategy_status": strategy_status,
        "grade": grade,
    }

    # --- Hard gate (a): strategy status must be LIVE. ---
    gate_status = (strategy_status or "").upper() == "LIVE"
    checks["gate_strategy_live"] = gate_status
    if not gate_status:
        fail_reasons.append("strategy status is not LIVE")

    # --- Hard gate (b): a risk-approval token must be present. ---
    gate_token = bool(risk_token)
    checks["gate_risk_token"] = gate_token
    if not gate_token:
        fail_reasons.append("missing risk-approval token")

    # --- Hard gate (c): interactive live-confirm must be set. ---
    checks["gate_live_confirm"] = bool(live_confirm)
    if not live_confirm:
        fail_reasons.append("interactive live-confirm not set")

    # --- Numeric caps. Only verified once the three hard gates pass, so that a
    #     limits-load failure does not mask the simpler, always-relevant gate
    #     failures. If the gates pass we MUST be able to verify the caps. ---
    if not fail_reasons:
        try:
            if limits is None:
                from risk.config import load_limits  # lazy import

                limits = load_limits()
            from risk.sizing import per_trade_dollar_risk, resolve_ri  # lazy import

            ri = resolve_ri(grade, limits)
            checks["ri"] = ri

            # Per-trade $ risk cap.
            if equity is not None and order_notional is not None and stop_distance_pct is not None:
                trade_risk = order_notional * stop_distance_pct
                cap = per_trade_dollar_risk(equity, ri, limits)
                checks["trade_risk"] = trade_risk
                checks["per_trade_cap"] = cap
                checks["gate_per_trade_cap"] = trade_risk <= cap
                if trade_risk > cap:
                    fail_reasons.append(
                        f"order risk {trade_risk:.2f} exceeds per-trade cap {cap:.2f} at RI {ri}"
                    )

            # Portfolio heat cap.
            if portfolio_heat_pct is not None:
                heat_cap = limits.level(ri).portfolio_heat_pct
                checks["portfolio_heat_pct"] = portfolio_heat_pct
                checks["heat_cap"] = heat_cap
                checks["gate_portfolio_heat"] = portfolio_heat_pct <= heat_cap
                if portfolio_heat_pct > heat_cap:
                    fail_reasons.append("would breach portfolio heat cap")

            # Live buying-power check (risk-based intraday margin is dynamic).
            if buying_power is not None and order_notional is not None:
                checks["buying_power"] = buying_power
                checks["order_notional"] = order_notional
                checks["gate_buying_power"] = order_notional <= buying_power
                if order_notional > buying_power:
                    fail_reasons.append("order notional exceeds live buying power")

        except Exception as exc:  # noqa: BLE001 — any failure here FAILS CLOSED.
            checks["limits_error"] = repr(exc)
            return Decision(
                allow=False,
                route="paper",
                reason="cannot load risk limits; failing closed",
                checks=checks,
            )

    # --- Decision. Any failure routes the order to paper (blocked). ---
    if fail_reasons:
        return Decision(
            allow=False,
            route="paper",
            reason="; ".join(fail_reasons),
            checks=checks,
        )

    return Decision(
        allow=True,
        route="live",
        reason="all live gates passed",
        checks=checks,
    )


def main(stdin_text: str | None = None) -> int:
    """Read a JSON payload, evaluate it, and return a process exit code.

    Returns 0 to allow the tool call and 2 to block it. On a block the reason is
    written to stderr. Everything is wrapped so that for a live-order tool any
    unexpected error returns 2 (fail closed); for non-order tools it returns 0.
    """
    raw = stdin_text if stdin_text is not None else sys.stdin.read()
    tool_name = ""
    try:
        payload = json.loads(raw) if raw and raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
        tool_name = payload.get("tool_name", "") or ""
        decision = evaluate(payload)
        if decision.allow:
            return 0
        sys.stderr.write(
            f"[TradeForge hook] BLOCKED live order -> {decision.reason}. Use the paper path.\n"
        )
        return 2
    except Exception as exc:  # noqa: BLE001 — fail closed for live-order tools.
        if is_live_order_tool(tool_name):
            sys.stderr.write(
                f"[TradeForge hook] FAIL-CLOSED blocking live order ({exc}).\n"
            )
            return 2
        return 0


if __name__ == "__main__":
    sys.exit(main())
