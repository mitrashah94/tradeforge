"""Tests for the milestone gain-ratchet."""

import pytest

from risk.config import load_limits
from risk.ratchet import check_ratchet


def test_fresh_baseline_matches_starting_capital():
    limits = load_limits()
    assert limits.ratchet.starting_capital == 1000


def test_milestone_triggers_sweep():
    limits = load_limits()
    result = check_ratchet(2600, 1000, limits)
    assert result.triggered is True
    assert result.milestone == 2500
    # Sweep 25% of the gain over baseline: 0.25 * (2600 - 1000) == 400.0.
    assert result.sweep_amount == pytest.approx(400.0)
    # Baseline advances to the milestone — the new protected high-water floor.
    assert result.new_baseline == 2500


def test_below_milestone_no_sweep():
    limits = load_limits()
    result = check_ratchet(2000, 1000, limits)
    assert result.triggered is False
    assert result.sweep_amount == pytest.approx(0.0)
    assert result.new_baseline == 1000


def test_no_double_sweep_after_baseline_advances():
    limits = load_limits()
    # After the baseline advanced to 2500, 2600 is still below the next
    # milestone (5000), so it must not trigger a second sweep.
    result = check_ratchet(2600, 2500, limits)
    assert result.triggered is False
    assert result.sweep_amount == pytest.approx(0.0)
    assert result.new_baseline == 2500
