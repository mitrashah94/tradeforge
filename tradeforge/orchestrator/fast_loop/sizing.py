"""orchestrator/fast_loop/sizing.py — volatility-target position sizing (MASTER_PLAN §1.B).

The fast loop's one job here is to make the **$-risk per trade CONSTANT across
volatility regimes**. We do that by sizing inversely to the per-share stop
distance (which is ATR-derived): a wide (high-vol) stop buys fewer shares, a
tight (low-vol) stop buys more — so ``qty * stop_distance == $risk`` regardless
of vol. This smooths the equity curve and stabilizes the geometric compounding
rate (MASTER_PLAN §1.B "Volatility targeting").

The dollar-risk budget itself is the equity-scaled per-trade risk from the risk
index (``risk.sizing.per_trade_dollar_risk``), recomputed off CURRENT equity so
profits and deposits auto-compound (MASTER_PLAN §1.D). This module only converts
that $-risk + a stop distance into a share count.

Everything here is a PURE function — no I/O, no clock, no network — so it is
trivially unit-testable and safe in the deterministic hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

from risk.config import Limits
from risk.sizing import per_trade_dollar_risk


@dataclass(frozen=True)
class SizingResult:
    """The outcome of a vol-target sizing computation.

    ``qty`` is the share/contract count to trade; ``dollar_risk`` is the budgeted
    $-risk (constant across vol); ``stop_distance`` is the per-share risk used.
    ``skipped`` is True (and ``qty == 0``) when sizing is degenerate — a
    non-positive stop distance, equity, or $-risk — which the fast loop treats as
    "do not enter".
    """

    qty: float
    dollar_risk: float
    stop_distance: float
    skipped: bool = False
    reason: str = ""


def vol_target_qty(
    *,
    equity: float,
    ri: int,
    limits: Limits,
    entry_price: float,
    stop_price: float,
    allow_fractional: bool = True,
    min_qty: float = 0.0,
) -> SizingResult:
    """Volatility-target position size: keep $-risk per trade constant.

    Parameters
    ----------
    equity
        Current account equity (the sizing base; profits/deposits compound it).
    ri
        Risk index (1-10). Selects the per-trade % from ``limits`` (RI 5 -> 1.0%).
    limits
        Validated ``risk.config.Limits`` (single source of truth).
    entry_price, stop_price
        The intended entry and protective stop. ``stop_distance`` (per-share
        risk) is ``abs(entry_price - stop_price)`` — ATR-derived upstream, so a
        higher-vol regime yields a wider stop and therefore a SMALLER qty.
    allow_fractional
        If True, return a fractional qty (crypto / fractional shares). If False,
        floor to a whole number of shares/contracts.
    min_qty
        If the computed qty is below this floor (e.g. < 1 whole share when
        fractional is off), the trade is skipped.

    Returns
    -------
    SizingResult
        ``qty = dollar_risk / stop_distance`` (the vol-target identity, so
        ``qty * stop_distance == dollar_risk`` holds exactly before rounding).
        ``qty == 0`` and ``skipped == True`` on any degenerate input.
    """
    dollar_risk = per_trade_dollar_risk(equity, ri, limits)
    stop_distance = abs(entry_price - stop_price)

    if equity <= 0:
        return SizingResult(0.0, dollar_risk, stop_distance, True, "non_positive_equity")
    if dollar_risk <= 0:
        return SizingResult(0.0, dollar_risk, stop_distance, True, "non_positive_risk_budget")
    if stop_distance <= 0:
        # No stop distance -> infinite/undefined size. Refuse to trade.
        return SizingResult(0.0, dollar_risk, stop_distance, True, "zero_stop_distance")

    qty = dollar_risk / stop_distance
    if not allow_fractional:
        qty = float(int(qty))  # floor to whole shares/contracts

    if qty <= 0 or qty < min_qty:
        return SizingResult(0.0, dollar_risk, stop_distance, True, "below_min_qty")

    return SizingResult(
        qty=qty,
        dollar_risk=dollar_risk,
        stop_distance=stop_distance,
        skipped=False,
    )


def atr_stop_distance(atr: float, atr_mult: float) -> float:
    """Per-share stop distance from ATR: ``atr_mult * atr`` (>= 0).

    A convenience for callers that derive the stop purely from ATR rather than a
    structural level. The fast loop usually takes the stop straight from the
    strategy's role-reversal level, but this keeps the ATR-derivation explicit
    and testable for the vol-target invariant.
    """
    if atr <= 0 or atr_mult <= 0:
        return 0.0
    return atr_mult * atr
