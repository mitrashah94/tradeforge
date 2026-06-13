"""Tests for conviction-grade -> risk-index resolution and clamping."""

from risk.config import load_limits
from risk.sizing import resolve_ri


def test_grade_maps_to_base_ri():
    limits = load_limits()
    assert resolve_ri("B", limits) == 5
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
