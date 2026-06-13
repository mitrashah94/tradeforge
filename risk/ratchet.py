"""risk/ratchet.py — milestone gain-ratchet logic.

At each milestone reached, sweep `sweep_fraction` of gains (measured against the
last protected baseline) into a vault sleeve the aggressive engine cannot touch,
then advance the baseline to the milestone — a new protected high-water floor.
This protects each milestone from giveback while still pressing with the rest.

A fresh account starts with `baseline = limits.ratchet.starting_capital`.
"""

from __future__ import annotations

from dataclasses import dataclass

from risk.config import Limits


@dataclass
class RatchetResult:
    """Outcome of a single ratchet check."""

    triggered: bool
    milestone: float | None
    sweep_amount: float
    new_baseline: float


def check_ratchet(equity: float, baseline: float, limits: Limits) -> RatchetResult:
    """Check whether crossing a milestone triggers a sweep.

    Considers only milestones strictly above the current baseline that the
    equity has reached. If one or more qualify, sweeps `sweep_fraction` of the
    gain over baseline and advances the baseline to the highest such milestone.
    Because the baseline advances to the milestone, the same milestone cannot
    trigger a second sweep on a later call.
    """
    candidates = [m for m in limits.ratchet.milestones if m > baseline and equity >= m]
    if not candidates:
        return RatchetResult(
            triggered=False,
            milestone=None,
            sweep_amount=0.0,
            new_baseline=baseline,
        )

    milestone = max(candidates)
    gain = equity - baseline
    sweep_amount = limits.ratchet.sweep_fraction * gain
    return RatchetResult(
        triggered=True,
        milestone=milestone,
        sweep_amount=sweep_amount,
        new_baseline=milestone,
    )
