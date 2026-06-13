"""tests/test_hook.py — unit tests for the live-order safety gate.

These exercise `orchestrator.hooks.evaluate` (the single source of truth that the
PreToolUse shim delegates to). TRADEFORGE_* env vars are cleared per-test so the
payload `context` drives behavior deterministically.
"""

from __future__ import annotations

import pytest

from orchestrator.hooks import Decision, evaluate, is_live_order_tool


def _clear_env(monkeypatch):
    """Clear all TRADEFORGE_* env vars so context fully drives the decision."""
    for name in (
        "TRADEFORGE_STRATEGY_STATUS",
        "TRADEFORGE_RISK_TOKEN",
        "TRADEFORGE_LIVE_CONFIRM",
        "TRADEFORGE_BUYING_POWER",
    ):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# is_live_order_tool                                                           #
# --------------------------------------------------------------------------- #


def test_is_live_order_tool_place_equity_order():
    assert is_live_order_tool("mcp__abc__place_equity_order") is True


def test_is_live_order_tool_review_is_not_live():
    assert is_live_order_tool("mcp__abc__review_equity_order") is False


def test_is_live_order_tool_quotes_is_not_live():
    assert is_live_order_tool("mcp__abc__get_equity_quotes") is False


def test_is_live_order_tool_future_endpoints():
    assert is_live_order_tool("mcp__rh__place_crypto_order") is True
    assert is_live_order_tool("mcp__rh__place_option_order") is True
    assert is_live_order_tool("order_gateway_live") is True


# --------------------------------------------------------------------------- #
# Hard gates                                                                   #
# --------------------------------------------------------------------------- #


def test_block_when_status_not_live(monkeypatch):
    _clear_env(monkeypatch)
    payload = {
        "tool_name": "mcp__rh__place_equity_order",
        "context": {"strategy_status": "PAPER", "risk_token": "t", "live_confirm": True},
    }
    decision = evaluate(payload)
    assert decision.allow is False
    assert decision.route == "paper"
    assert "LIVE" in decision.reason


def test_block_when_missing_token(monkeypatch):
    _clear_env(monkeypatch)
    payload = {
        "tool_name": "mcp__rh__place_equity_order",
        "context": {"strategy_status": "LIVE", "live_confirm": True},
    }
    decision = evaluate(payload)
    assert decision.allow is False
    assert "token" in decision.reason.lower()


def test_block_when_no_live_confirm(monkeypatch):
    _clear_env(monkeypatch)
    payload = {
        "tool_name": "mcp__rh__place_equity_order",
        "context": {"strategy_status": "LIVE", "risk_token": "ok", "live_confirm": False},
    }
    decision = evaluate(payload)
    assert decision.allow is False
    assert "confirm" in decision.reason.lower()


def test_non_order_tool_always_allowed(monkeypatch):
    _clear_env(monkeypatch)
    payload = {"tool_name": "mcp__rh__get_equity_quotes"}
    decision = evaluate(payload)
    assert decision.allow is True


# --------------------------------------------------------------------------- #
# Numeric caps (uses the real limits.yaml)                                     #
# --------------------------------------------------------------------------- #


def test_all_gates_pass_within_caps_routes_live(monkeypatch):
    _clear_env(monkeypatch)
    from risk.config import load_limits

    lim = load_limits()
    payload = {
        "tool_name": "mcp__rh__place_equity_order",
        "context": {
            "strategy_status": "LIVE",
            "risk_token": "ok",
            "live_confirm": True,
            "grade": "A",
            "equity": 10000,
            "order_notional": 1000,
            "stop_distance_pct": 0.01,  # trade_risk = 10 <= cap
            "portfolio_heat_pct": 1.0,
            "buying_power": 5000,
        },
    }
    decision = evaluate(payload, limits=lim)
    assert decision.allow is True
    assert decision.route == "live"


def test_per_trade_cap_blocks(monkeypatch):
    _clear_env(monkeypatch)
    from risk.config import load_limits

    lim = load_limits()
    payload = {
        "tool_name": "mcp__rh__place_equity_order",
        "context": {
            "strategy_status": "LIVE",
            "risk_token": "ok",
            "live_confirm": True,
            "grade": "A",
            "equity": 10000,
            "order_notional": 1000,
            "stop_distance_pct": 0.50,  # trade_risk = 500 >> cap
            "portfolio_heat_pct": 1.0,
            "buying_power": 5000,
        },
    }
    decision = evaluate(payload, limits=lim)
    assert decision.allow is False
    assert "per-trade cap" in decision.reason


def test_decision_dataclass_shape():
    """Decision exposes the documented public fields."""
    d = Decision(allow=True, route="live", reason="ok", checks={})
    assert d.allow is True
    assert d.route == "live"
    assert d.reason == "ok"
    assert d.checks == {}
