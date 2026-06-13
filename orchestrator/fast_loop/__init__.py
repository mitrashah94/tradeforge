"""Deterministic fast loop package: entry/exit/stop/TP1/breakeven/trail/time-stop engine with no LLM and no MCP in the hot path (MASTER_PLAN.md §4).

Public API:
    FastLoop        — the deterministic real-time engine (construct with a bus,
                      armed strategies, an equity source, and risk limits).
    ArmedStrategy   — a pre-armed strategy instance + its live context config.
    LiveBar         — normalized OHLCV bar delivered on a bar event.
    LiveContext     — the Context adapter that drives the Strategy interface.
    vol_target_qty  — pure volatility-target position sizing (§1.B).
    SizingResult    — result of a sizing computation.
    PositionLifecycle / ManagedPosition / LifecycleState — the F2 state machine.
"""

from orchestrator.fast_loop.engine import (
    ArmedStrategy,
    FastLoop,
    LatencyStats,
    LiveBar,
    LiveContext,
)
from orchestrator.fast_loop.lifecycle import (
    LifecycleAction,
    LifecycleState,
    ManagedPosition,
    PositionLifecycle,
)
from orchestrator.fast_loop.sizing import (
    SizingResult,
    atr_stop_distance,
    vol_target_qty,
)

__all__ = [
    "FastLoop",
    "ArmedStrategy",
    "LiveBar",
    "LiveContext",
    "LatencyStats",
    "vol_target_qty",
    "atr_stop_distance",
    "SizingResult",
    "PositionLifecycle",
    "ManagedPosition",
    "LifecycleState",
    "LifecycleAction",
]
