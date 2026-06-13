"""level_meanrev — deliberately low-correlation MEAN-REVERSION complement.

The chop/range-regime FADE counterpart to breakout_retest (MASTER_PLAN.md
§1.A/B, §5). Where breakout_retest BUYS a clean level break (continuation),
level_meanrev FADES a level tag-and-rejection back toward the mean — the
opposite reaction to the same level event, decorrelated by construction.
"""

from strategies.level_meanrev.strategy import (
    DEFAULT_VARIANT,
    VARIANTS,
    LevelMeanRevStrategy,
    load_params,
)

__all__ = [
    "LevelMeanRevStrategy",
    "load_params",
    "VARIANTS",
    "DEFAULT_VARIANT",
]
