"""swing_breakout strategy: the DAILY DONCHIAN-BREAKOUT TREND sleeve — long-only,
multi-day breakout-in-uptrend (fresh N-day high ABOVE a trend SMA), ranked by
momentum strength, expressed against the bracketed-swing engine's
:class:`~backtest.daily.bracket_engine.SwingStrategy` interface (``entry_score``
+ optional ``exit_signal``). The engine's ATR bracket (partial-TP + breakeven +
chandelier trail) then cuts losers fast and lets winners run. The MIRROR of
``swing_meanrev`` (it buys the strength that sleeve never touches), built for
decorrelation as well as raw trend alpha (MASTER_PLAN.md §1.A/B). Parameters +
variant deltas live in params.yaml.
"""

from strategies.swing_breakout.strategy import (
    SwingBreakoutStrategy,
    load_params,
)

__all__ = ["SwingBreakoutStrategy", "load_params"]
