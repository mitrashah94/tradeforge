"""portfolio/intents.py — render an Allocation to ORDER_INTENTs (the live seam).

The backtester applies an :class:`~portfolio.model.Allocation` to a simulated
book; the LIVE slow loop instead renders it to the ``ORDER_INTENT`` dicts the
deterministic ``order_gateway`` consumes. This module is that mapping — designed
and unit-tested here, but NOT wired into the live bus (the plan: "design +
unit-test the mapping; do not wire live"). Keeping the seam pure and tested means
the day the fast loop plugs in, the contract is already proven.

THE MAPPING
-----------
  * a :class:`PlannedOpen` → a NEW-ENTRY intent (``reason="entry"``) that carries
    everything the gateway's risk gate checks (``grade`` / ``equity`` /
    ``order_notional`` / ``stop_distance_pct``), so a correctly-sized open passes
    the same per-trade cap the gateway enforces;
  * a :class:`PlannedClose` → an EXIT intent whose ``reason`` is one of the
    gateway's ``_EXIT_REASONS`` so it BYPASSES the halt gate (you must always be
    able to flatten — the F2 risk-reducer bypass);
  * a :class:`PlannedResize` → a TP1 partial (``reason="tp1_partial_exit"``), a
    risk-reducing weight trim (mapped to ``strategy_close``, also a bypass), or a
    risk-adding weight add (``reason="entry"``, gated like a new position).

Every intent carries ``bypasses_halt`` so a test (and the future live wiring) can
assert which actions must execute even while the book is halted.
"""

from __future__ import annotations

from typing import Optional

from portfolio.model import Allocation, PlannedClose, PlannedOpen, PlannedResize

# Map a portfolio close reason -> the gateway's F2 exit reason (a halt bypass).
_CLOSE_REASON_MAP = {
    "stop": "stop",
    "stop_gap": "stop",
    "trail_stop": "trail_stop",
    "trail_stop_gap": "trail_stop",
    "hard_target": "target",
    "hard_target_gap": "target",
    "exit_signal": "strategy_close",
    "weight_exit": "strategy_close",
}
# The gateway reasons that bypass the entry/halt gate (mirror order_gateway).
_BYPASS_REASONS = {
    "stop", "trail_stop", "target", "time_stop", "session_flatten",
    "strategy_close", "close", "tp1_partial_exit", "tp1_breakeven", "trail",
}


def _stop_distance_pct(entry: float, stop: Optional[float]) -> Optional[float]:
    """``(entry - stop) / entry`` — the gateway's per-trade-risk denominator."""
    if stop is None or entry is None or entry <= 0:
        return None
    d = (float(entry) - float(stop)) / float(entry)
    return d if d > 0 else None


def open_to_intent(p: PlannedOpen, equity: float, route: str = "paper") -> dict:
    """Render a :class:`PlannedOpen` to a NEW-ENTRY ORDER_INTENT dict."""
    notional = p.shares * p.entry_price
    return {
        "symbol": p.symbol,
        "side": "buy",
        "qty": float(p.shares),
        "order_type": "limit",
        "intended_price": float(p.entry_price),
        "ref_price": float(p.entry_price),
        "grade": p.grade,
        "strategy": p.sleeve,
        "reason": "entry",
        "equity": float(equity),
        "order_notional": float(notional),
        "stop_distance_pct": _stop_distance_pct(p.entry_price, p.stop),
        "route": route,
        "bypasses_halt": False,
    }


def close_to_intent(p: PlannedClose, route: str = "paper") -> dict:
    """Render a :class:`PlannedClose` to an EXIT intent (a halt bypass)."""
    reason = _CLOSE_REASON_MAP.get(p.reason, "close")
    return {
        "symbol": p.symbol,
        "side": "sell",
        "qty": float(p.shares),
        "order_type": "market",
        "intended_price": float(p.fill_price),
        "ref_price": float(p.fill_price),
        "strategy": p.sleeve,
        "reason": reason,
        "route": route,
        "bypasses_halt": reason in _BYPASS_REASONS,
    }


def resize_to_intent(p: PlannedResize, equity: float, route: str = "paper") -> dict:
    """Render a :class:`PlannedResize` to a partial-fill intent.

    A negative ``delta_shares`` (TP1 partial / weight trim) is a risk-reducing SELL
    that bypasses the halt gate; a positive delta (weight add) is a risk-ADDING BUY
    gated like a new entry.
    """
    selling = p.delta_shares < 0
    qty = abs(float(p.delta_shares))
    if selling:
        reason = "tp1_partial_exit" if p.reason == "tp1_partial" else "strategy_close"
        return {
            "symbol": p.symbol,
            "side": "sell",
            "qty": qty,
            "order_type": "market",
            "intended_price": float(p.fill_price),
            "ref_price": float(p.fill_price),
            "strategy": p.sleeve,
            "reason": reason,
            "route": route,
            "bypasses_halt": reason in _BYPASS_REASONS,
        }
    return {
        "symbol": p.symbol,
        "side": "buy",
        "qty": qty,
        "order_type": "limit",
        "intended_price": float(p.fill_price),
        "ref_price": float(p.fill_price),
        "strategy": p.sleeve,
        "reason": "entry",
        "equity": float(equity),
        "order_notional": qty * float(p.fill_price),
        "route": route,
        "bypasses_halt": False,
    }


def allocation_to_intents(
    allocation: Allocation, equity: float, route: str = "paper"
) -> list:
    """Render a whole :class:`Allocation` to ordered ORDER_INTENT dicts.

    Order matters for a live book: CLOSES and risk-reducing RESIZES first (free up
    cash + cut risk), THEN new OPENS. Returns plain dicts (the gateway's
    ``ORDER_INTENT`` payload shape); this function never touches the bus.
    """
    intents: list = []
    for c in allocation.closes:
        intents.append(close_to_intent(c, route=route))
    # risk-reducing resizes (sells) before risk-adding ones (buys)
    sells = [r for r in allocation.resizes if r.delta_shares < 0]
    buys = [r for r in allocation.resizes if r.delta_shares > 0]
    for r in sells:
        intents.append(resize_to_intent(r, equity, route=route))
    for o in allocation.opens:
        intents.append(open_to_intent(o, equity, route=route))
    for r in buys:
        intents.append(resize_to_intent(r, equity, route=route))
    return intents
