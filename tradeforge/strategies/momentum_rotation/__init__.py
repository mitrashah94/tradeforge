"""momentum_rotation — trend-filtered dual-momentum + sector-RS rotation sleeve.

The slow, cross-asset, LONG-ONLY daily allocation edge (MASTER_PLAN.md §1.A/B
edge-stacking, §5): a GEM-lite dual-momentum core (US vs ex-US, gated by absolute
momentum vs cash / a 200d SMA, else bonds) stacked with a sector RS rotation
(top-N of the 11 SPDR sectors, each gated by its own 200d trend), the combined
book scaled inversely to realized vol toward a vol target. Two RESEARCH growth
levers (inverse-ETF risk-off; 2x/3x leveraged long) are wired but DEFAULT OFF.

Implements the :class:`~backtest.daily.engine.DailyStrategy` interface
(``target_weights(asof_date, history) -> {symbol: fraction}``); decorrelated from
the intraday breakout/fade edges by both horizon (months vs minutes) and
mechanism (trend vs break/fade).
"""

from strategies.momentum_rotation.strategy import (
    DEFAULT_VARIANT,
    VARIANTS,
    MomentumRotationStrategy,
    above_sma,
    blended_momentum,
    gem_select,
    load_params,
    rank_sectors,
    realized_vol,
    total_return,
    vol_scalar,
)

__all__ = [
    "MomentumRotationStrategy",
    "load_params",
    "VARIANTS",
    "DEFAULT_VARIANT",
    "total_return",
    "blended_momentum",
    "above_sma",
    "rank_sectors",
    "realized_vol",
    "vol_scalar",
    "gem_select",
]
