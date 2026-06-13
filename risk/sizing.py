"""risk/sizing.py — conviction->RI resolution and equity-scaled position sizing.

Per-trade $ risk is ALWAYS recomputed against CURRENT equity (see CLAUDE.md),
so every realized profit and every deposit auto-compounds the next position's
size. Conviction tiering flexes the dial inside [current floor, band_high]:
B -> 5, A -> 6-7, A+ -> 8, clamped to the live floor.
"""

from __future__ import annotations

from risk.config import Limits


def resolve_ri(grade: str, limits: Limits, floor: int | None = None) -> int:
    """Map a setup grade to a risk index, clamped into [floor, band_high].

    The base RI comes from `limits.conviction_tiers[grade]` (KeyError on an
    unknown grade is acceptable — callers should pass a known grade). The result
    is clamped UP to the live floor and DOWN to the band high (8).
    """
    floor = limits.default_ri if floor is None else floor
    base = limits.conviction_tiers[grade]
    return min(max(base, floor), limits.band_high)


def per_trade_dollar_risk(equity: float, ri: int, limits: Limits) -> float:
    """Return the per-trade $ risk = (per_trade_pct / 100) * current equity."""
    return limits.level(ri).per_trade_pct / 100.0 * equity


def size_for_setup(
    equity: float, grade: str, limits: Limits, floor: int | None = None
) -> dict:
    """Resolve RI for a setup grade and return its sizing breakdown.

    Returns a dict with the resolved risk index, the per-trade percentage for
    that index, and the dollar risk against current equity.
    """
    ri = resolve_ri(grade, limits, floor)
    pct = limits.level(ri).per_trade_pct
    return {
        "ri": ri,
        "per_trade_pct": pct,
        "dollar_risk": per_trade_dollar_risk(equity, ri, limits),
    }
