"""Deterministic execution engine: evaluates pre-armed rules against the live feed and manages entry, stop, TP1 partial, breakeven move, trailing runner, time-stop, and session-flatten — no LLM, no MCP in the hot path (MASTER_PLAN.md §4)."""

# TODO: implement the deterministic state-driven manager and latency-budget logging; keep all decision logic pure Python.
