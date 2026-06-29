"""Tests for the milestone gain-ratchet.

Covers both ratchet regimes:
  * unconditional sweep (no ``sweep_threshold`` key, default 0.0) — IDENTICAL to
    the original behavior: every crossed milestone sweeps ``sweep_fraction × gain``.
  * early-game no-sweep gate (``sweep_threshold > 0``) — below the threshold a
    crossed milestone is a CHECKPOINT (baseline advances, sweep is 0.0); at/above
    the threshold the normal sweep applies.

The threshold cases build synthetic :class:`Ratchet`/:class:`Limits` objects so
they pin the CODE contract regardless of what ``risk/limits.yaml`` currently
holds (the operator may tune the YAML by hand).
"""

import pytest

from risk.config import Limits, Ratchet, load_limits
from risk.ratchet import check_ratchet


def _limits_with_ratchet(ratchet: Ratchet) -> Limits:
    """A minimal real ``Limits`` (from the YAML) with its ratchet swapped out."""
    base = load_limits()
    return base.model_copy(update={"ratchet": ratchet})


# --------------------------------------------------------------------------- #
# Live YAML sanity
# --------------------------------------------------------------------------- #
def test_fresh_baseline_matches_starting_capital():
    limits = load_limits()
    # Starting capital is operator-tuned in risk/limits.yaml; just assert it is a
    # sane positive amount within the documented $500–$100k range.
    assert 500 <= limits.ratchet.starting_capital <= 100_000


def test_optional_threshold_keys_default_to_zero():
    # A ratchet block with NEITHER optional key parses fine; defaults are 0.0,
    # which means the no-sweep gate is OFF (unconditional-sweep behavior).
    r = Ratchet(
        starting_capital=1000,
        sweep_fraction=0.25,
        milestones=[2500, 5000],
        vault_sleeve="vault",
    )
    assert r.sweep_threshold == 0.0
    assert r.vault_below_threshold == 0.0


# --------------------------------------------------------------------------- #
# Unconditional sweep (no threshold) — original behavior preserved
# --------------------------------------------------------------------------- #
def _unconditional_limits() -> Limits:
    return _limits_with_ratchet(
        Ratchet(
            starting_capital=1000,
            sweep_fraction=0.25,
            milestones=[2500, 5000, 10000, 25000, 50000, 100000],
            vault_sleeve="vault",
        )
    )


def test_milestone_triggers_sweep():
    limits = _unconditional_limits()
    result = check_ratchet(2600, 1000, limits)
    assert result.triggered is True
    assert result.milestone == 2500
    # Sweep 25% of the gain over baseline: 0.25 * (2600 - 1000) == 400.0.
    assert result.sweep_amount == pytest.approx(400.0)
    # Baseline advances to the milestone — the new protected high-water floor.
    assert result.new_baseline == 2500


def test_below_milestone_no_sweep():
    limits = _unconditional_limits()
    result = check_ratchet(2000, 1000, limits)
    assert result.triggered is False
    assert result.sweep_amount == pytest.approx(0.0)
    assert result.new_baseline == 1000


def test_no_double_sweep_after_baseline_advances():
    limits = _unconditional_limits()
    # After the baseline advanced to 2500, 2600 is still below the next
    # milestone (5000), so it must not trigger a second sweep.
    result = check_ratchet(2600, 2500, limits)
    assert result.triggered is False
    assert result.sweep_amount == pytest.approx(0.0)
    assert result.new_baseline == 2500


# --------------------------------------------------------------------------- #
# Early-game no-sweep gate (sweep_threshold > 0)
# --------------------------------------------------------------------------- #
def _gated_limits() -> Limits:
    # No sweep below $10k: early gains compound fully; the 25% vault sweep only
    # starts at/above $10k — the activated form of the limits.yaml ratchet block.
    return _limits_with_ratchet(
        Ratchet(
            starting_capital=500,
            sweep_fraction=0.25,
            milestones=[1000, 2500, 5000, 10000, 25000, 50000, 100000],
            vault_sleeve="vault",
            sweep_threshold=10000,
            vault_below_threshold=0,
        )
    )


def test_below_threshold_is_a_checkpoint_zero_sweep_baseline_advances():
    limits = _gated_limits()
    # Equity 2600 < 10000 threshold: crossing the 2500 milestone is a CHECKPOINT.
    result = check_ratchet(2600, 500, limits)
    assert result.triggered is True
    assert result.milestone == 2500
    # Sweep NOTHING below the threshold — early gains fully compound.
    assert result.sweep_amount == pytest.approx(0.0)
    # Baseline still advances to the milestone (a protected high-water floor).
    assert result.new_baseline == 2500


def test_below_threshold_checkpoint_does_not_double_trigger():
    limits = _gated_limits()
    # After the baseline advanced to 2500, 2600 is below the next milestone
    # (5000) AND below the threshold — no re-trigger, no sweep.
    result = check_ratchet(2600, 2500, limits)
    assert result.triggered is False
    assert result.sweep_amount == pytest.approx(0.0)
    assert result.new_baseline == 2500


def test_at_or_above_threshold_sweeps_normally():
    limits = _gated_limits()
    # Equity 11000 >= 10000 threshold: crossing 10000 sweeps normally.
    # Baseline 5000 (last checkpoint below the threshold); gain over baseline is
    # 11000 - 5000 = 6000; sweep 25% => 1500.
    result = check_ratchet(11000, 5000, limits)
    assert result.triggered is True
    assert result.milestone == 10000
    assert result.sweep_amount == pytest.approx(1500.0)
    assert result.new_baseline == 10000


def test_exactly_at_threshold_sweeps_not_checkpoints():
    limits = _gated_limits()
    # Equity == threshold (10000) is NOT below it, so the normal sweep applies.
    result = check_ratchet(10000, 5000, limits)
    assert result.triggered is True
    assert result.milestone == 10000
    # gain = 10000 - 5000 = 5000; sweep 25% => 1250.
    assert result.sweep_amount == pytest.approx(1250.0)
    assert result.new_baseline == 10000
