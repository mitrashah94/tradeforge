"""risk/ratchet.py — milestone gain-ratchet logic.

At each milestone reached, sweep `sweep_fraction` of gains (measured against the
last protected baseline) into a vault sleeve the aggressive engine cannot touch,
then advance the baseline to the milestone — a new protected high-water floor.
This protects each milestone from giveback while still pressing with the rest.

A fresh account starts with `baseline = limits.ratchet.starting_capital`.

Early-game no-sweep regime (P0 ratchet block — `sweep_threshold`):
--------------------------------------------------------------------
While `sweep_threshold > 0` and equity is BELOW it, a crossed milestone is a
CHECKPOINT, not a sweep: the baseline still advances to the milestone (so the
milestone cannot re-trigger) but `sweep_amount = 0.0`, letting the small account
fully compound its early gains instead of vaulting 25% of them. At or above the
threshold the normal `sweep_fraction × gain` sweep applies. With no
`sweep_threshold` key (default 0.0) the gate is off and behavior is identical to
the original unconditional sweep.
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
    """Check whether crossing a milestone triggers a sweep (or a checkpoint).

    Considers only milestones strictly above the current baseline that the
    equity has reached. If one or more qualify, the highest such milestone fires
    and the baseline advances to it. Because the baseline advances to the
    milestone, the same milestone cannot trigger a second sweep on a later call.

    Sweep amount depends on the early-game gate ``sweep_threshold``:
      * ``sweep_threshold > 0`` and ``equity < sweep_threshold`` → CHECKPOINT:
        ``sweep_amount = 0.0`` (advance the baseline, sweep nothing — early gains
        compound fully). ``triggered`` is still True so the caller advances state.
      * otherwise → normal sweep: ``sweep_amount = sweep_fraction × gain`` over the
        baseline. With ``sweep_threshold == 0.0`` (no key) this is the only path,
        i.e. identical to the original unconditional-sweep behavior.
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
    threshold = limits.ratchet.sweep_threshold
    if threshold > 0.0 and equity < threshold:
        # Below-threshold checkpoint: advance the baseline, sweep nothing.
        sweep_amount = 0.0
    else:
        gain = equity - baseline
        sweep_amount = limits.ratchet.sweep_fraction * gain
    return RatchetResult(
        triggered=True,
        milestone=milestone,
        sweep_amount=sweep_amount,
        new_baseline=milestone,
    )
