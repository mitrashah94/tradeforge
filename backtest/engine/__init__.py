"""Backtest engine: a thin, custom, event-driven bar simulator with a realistic
small-account cost model (commission + spread + slippage), exact TradingView
Pine fill parity, and a documented Strategy interface (MASTER_PLAN.md §4, §5).

Why custom over vectorbt/backtesting.py is justified in engine.py's docstring.

Public API (the Stage-2 contract — later strategies implement ``Strategy``):
    from backtest.engine import (
        BacktestEngine, Strategy, Context, Bar, Position, bars_from_df,
        CostModel, BacktestResult, TradeRecord,
    )
"""

from backtest.engine.cost import CostModel
from backtest.engine.engine import (
    Bar,
    BacktestEngine,
    Context,
    Position,
    Strategy,
    bars_from_df,
)
from backtest.engine.result import BacktestResult, TradeRecord

__all__ = [
    "BacktestEngine",
    "Strategy",
    "Context",
    "Bar",
    "Position",
    "bars_from_df",
    "CostModel",
    "BacktestResult",
    "TradeRecord",
]
