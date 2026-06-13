"""momentum_thrust — deliberately low-correlation THRUST-continuation complement.

The trend/expansion-regime FOLLOW counterpart to level_meanrev (and
breakout_retest) (MASTER_PLAN.md §1.A/B, §5). Where level_meanrev FADES
extensions back to the mean, momentum_thrust FOLLOWS a strong directional
thrust with the trend and rides it on a trailing stop (no fixed target) — the
opposite reaction to the same price action, decorrelated by construction.
"""

from strategies.momentum_thrust.strategy import (
    DEFAULT_VARIANT,
    VARIANTS,
    MomentumThrustStrategy,
    load_params,
)

__all__ = [
    "MomentumThrustStrategy",
    "load_params",
    "VARIANTS",
    "DEFAULT_VARIANT",
]
