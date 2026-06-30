"""Tests for conviction-grade -> risk-index resolution and clamping."""

from risk.config import load_limits
from risk.sizing import resolve_ri


def test_grade_maps_to_base_ri():
    limits = load_limits()
    # The live floor is risk_index.default (now 6 — the operator's "start at RI 6"
    # decision), and resolve_ri clamps the base tier UP to that floor. So a B
    # setup, whose BASE RI is 5, now resolves to 6; pass an explicit floor=5 to
    # observe the raw base mapping. A (6) and A+ (8) already sit at/above the floor.
    assert limits.conviction_tiers["B"] == 5         # base tier value unchanged
    assert resolve_ri("B", limits, floor=5) == 5     # raw base mapping (floor permitting)
    assert resolve_ri("B", limits) == 6              # RI-6 floor clamps B up
    assert resolve_ri("A", limits) == 6
    assert resolve_ri("A+", limits) == 8


def test_floor_clamps_up():
    limits = load_limits()
    # B's base RI is 5, but a floor of 7 clamps it UP to 7.
    assert resolve_ri("B", limits, floor=7) == 7


def test_band_high_clamp_is_respected():
    limits = load_limits()
    # Even the strongest grade and a high floor never exceed band_high (8).
    assert resolve_ri("A+", limits) == 8
    assert resolve_ri("A+", limits, floor=8) == 8
    assert limits.band_high == 8
    for grade in limits.conviction_tiers:
        assert resolve_ri(grade, limits) <= 8
        assert resolve_ri(grade, limits, floor=8) <= 8
