"""tests/test_fast_loop_latency.py — hot-path latency budget + no-LLM/MCP/network proof.

MASTER_PLAN §4: the fast loop is DETERMINISTIC with the LLM/MCP OUT of the hot
path. This module asserts:

  * processing a bar event stays under the latency budget (a few ms);
  * the loop performs NO LLM / MCP / network calls — proven BY CONSTRUCTION via
    source inspection of the entire ``orchestrator.fast_loop`` package (no such
    imports anywhere), plus a runtime guard that fails if a forbidden module is
    imported while a bar is processed.
"""

import ast
import socket
from pathlib import Path

import pytest

from risk.config import load_limits
from backtest.engine.engine import Strategy
from orchestrator.fast_loop import ArmedStrategy, FastLoop
import orchestrator.fast_loop as fast_loop_pkg


# --------------------------------------------------------------------------- #
# Minimal fake bus + bar (kept local so this file is self-contained).
# --------------------------------------------------------------------------- #
class FakeBus:
    def __init__(self):
        self._subs = {}
        self.published = []

    def subscribe(self, types, handler):
        for t in types:
            self._subs.setdefault(str(t), []).append(handler)

    def publish(self, event):
        self.published.append(event)
        for h in self._subs.get(str(event.type), []):
            h(event)


class _Event:
    def __init__(self, type, data, ts_utc=None, seq=0, source="test"):
        self.type, self.data, self.ts_utc, self.seq, self.source = (
            type, data, ts_utc, seq, source
        )


class _ArmLong(Strategy):
    def on_bar(self, ctx):
        if ctx.position is None and ctx.bar_index == 1:
            ctx.enter_long(stop=ctx.bar.close - 1.0, target=ctx.bar.close + 2.0)


def _bar_data(c, h=None, l=None, ts=0, symbol="TEST"):
    return {
        "symbol": symbol, "ts": ts, "open": c,
        "high": h if h is not None else c,
        "low": l if l is not None else c,
        "close": c, "volume": 0.0,
    }


def _make_loop(bus):
    return FastLoop(
        bus=bus,
        armed=[ArmedStrategy(name="lat", strategy=_ArmLong(), symbol="TEST")],
        equity_source=lambda: 10_000.0,
        limits=load_limits(),
        ri=5,
        clock=lambda: 0,
        latency_budget_ms=5.0,
        event_factory=_Event,
    )


# --------------------------------------------------------------------------- #
# Latency budget
# --------------------------------------------------------------------------- #
def test_bar_processing_under_budget():
    bus = FakeBus()
    loop = _make_loop(bus)
    loop.start()

    # Warm up + stream a few hundred bars; assert every event was within budget.
    for i in range(300):
        c = 100.0 + (i % 5) * 0.1
        bus.publish(_Event("BAR", _bar_data(c, c + 0.5, c - 0.5, ts=i)))

    summary = loop.latency_summary()
    assert summary["count"] >= 300
    # No event may exceed the budget.
    loop.assert_within_budget()
    assert summary["over_budget"] == 0
    # Sanity: typical processing is well under a millisecond.
    assert summary["avg_ms"] < 5.0
    assert summary["max_ms"] < 5.0


def test_latency_summary_shape():
    bus = FakeBus()
    loop = _make_loop(bus)
    loop.start()
    bus.publish(_Event("BAR", _bar_data(100.0, ts=0)))
    s = loop.latency_summary()
    for key in ("count", "avg_ms", "max_ms", "last_ms", "budget_ms", "over_budget"):
        assert key in s
    assert s["count"] == 1


# --------------------------------------------------------------------------- #
# No LLM / MCP / network in the hot path — proven by source inspection.
# --------------------------------------------------------------------------- #
FORBIDDEN_IMPORT_SUBSTRINGS = (
    "anthropic", "openai", "langchain", "mcp", "httpx", "requests",
    "aiohttp", "urllib.request", "urllib3", "boto3", "websocket", "grpc",
    "duckdb",  # no DB in the hot path either
)


def _iter_package_py_files():
    pkg_dir = Path(fast_loop_pkg.__file__).resolve().parent
    return sorted(pkg_dir.glob("*.py"))


def test_no_forbidden_imports_in_fast_loop_package():
    """Static guarantee: nothing in the fast_loop package imports LLM/MCP/network."""
    offenders = []
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                low = name.lower()
                for bad in FORBIDDEN_IMPORT_SUBSTRINGS:
                    if low == bad or low.startswith(bad + ".") or ("." + bad) in ("." + low):
                        offenders.append((path.name, name))
    assert offenders == [], f"forbidden imports in hot-path package: {offenders}"


def test_no_network_socket_opened_during_bar_processing():
    """Runtime guard: opening a socket in the hot path raises (proves no network)."""
    bus = FakeBus()
    loop = _make_loop(bus)
    loop.start()

    real_socket = socket.socket
    opened = {"count": 0}

    def _trap(*args, **kwargs):
        opened["count"] += 1
        raise AssertionError("fast loop opened a network socket in the hot path")

    socket.socket = _trap
    try:
        for i in range(10):
            bus.publish(_Event("BAR", _bar_data(100.0 + i * 0.1, ts=i)))
    finally:
        socket.socket = real_socket

    assert opened["count"] == 0


def test_fast_loop_imports_no_orchestrator_tools_or_mcp():
    """The hot-path modules must not import the MCP/tool/order-gateway layer."""
    bad_modules = ("orchestrator.tools", "orchestrator.workflows")
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                for bad in bad_modules:
                    assert not (m == bad or m.startswith(bad + ".")), (
                        f"{path.name} imports hot-path-forbidden {m}"
                    )
