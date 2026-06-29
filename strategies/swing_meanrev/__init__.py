"""swing_meanrev strategy: the SWING MEAN-REVERSION sleeve — Connors RSI(2)
oversold dip-buy on broad/sector index ETFs, gated by a 200d uptrend, LONG-only
multi-day holds expressed as daily target weights against the daily engine's
:class:`~backtest.daily.engine.DailyStrategy` interface (MASTER_PLAN.md §1.A/B).
A deliberately low-correlation complement to the momentum/breakout sleeves (it
BUYS the dips they sell). Parameters + variant deltas live in params.yaml.
"""

from strategies.swing_meanrev.strategy import (
    SwingMeanRevStrategy,
    load_params,
    wilder_rsi,
)

__all__ = ["SwingMeanRevStrategy", "load_params", "wilder_rsi"]
