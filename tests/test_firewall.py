"""tests/test_firewall.py — the research-firewall write guard (MASTER_PLAN.md §6).

The cultural rule ("agents read everything, write nothing live") lives in the
.md files; this is the programmatic backstop. Verifies that writes to protected
live-config paths RAISE, that research/journal/report paths are allowed, and that
the matching is robust to path style (relative, ./, absolute-in-repo, ..).
"""

from __future__ import annotations

import os

import pytest

from orchestrator.agents.firewall import (
    FirewallViolation,
    agent_writable,
    assert_not_live_config,
)


# --- protected live config raises ---------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        "risk/limits.yaml",
        "strategies/registry.yaml",
        ".claude/settings.json",
        ".claude/settings.local.json",
        "backtest/stats/oos_vault.yaml",
        "CLAUDE.md",
        "risk/breakers.py",        # under the protected risk/ prefix
        ".claude/agents/regime-reader.md",  # under the protected .claude/ prefix
        "orderbook/orderbook.duckdb",
        "paper/ledger.duckdb",
    ],
)
def test_protected_paths_raise(path):
    with pytest.raises(FirewallViolation):
        assert_not_live_config(path)
    assert agent_writable(path) is False


# --- research / journal / report paths are allowed ----------------------------
@pytest.mark.parametrize(
    "path",
    [
        "research/proposals/breakout_v5.yaml",
        "research/hypotheses.log",
        "journal/2026-06-13/trade_001.md",
        "journal/digests/eod_2026-06-13.json",
        "reports/weekly_alpha.md",
        "reporting/out/equity_curve.png",
        "backtest/reports/run_2026-06-13.html",
    ],
)
def test_research_paths_allowed(path):
    # No raise, and explicitly writable.
    assert_not_live_config(path)
    assert agent_writable(path) is True


# --- robustness to path style -------------------------------------------------
def _repo_root() -> str:
    # tests/ is one level under the repo root.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_relative_dotslash_and_absolute_all_match():
    here = _repo_root()
    abs_limits = os.path.join(here, "risk", "limits.yaml")
    for variant in ("risk/limits.yaml", "./risk/limits.yaml", abs_limits,
                    "risk/../risk/limits.yaml"):
        with pytest.raises(FirewallViolation):
            assert_not_live_config(variant)
        assert agent_writable(variant) is False


def test_absolute_research_path_allowed():
    here = _repo_root()
    abs_journal = os.path.join(here, "journal", "x.md")
    assert_not_live_config(abs_journal)  # no raise
    assert agent_writable(abs_journal) is True


# --- deny by default ----------------------------------------------------------
def test_unknown_path_is_not_writable_but_does_not_raise():
    # Neither protected nor on the writable allowlist: assert_not_live_config does
    # not raise (it only guards LIVE config), but agent_writable denies by default.
    p = "some/unknown/area/file.txt"
    assert_not_live_config(p)  # no raise
    assert agent_writable(p) is False


def test_firewall_violation_is_permission_error():
    # Subclasses PermissionError so generic write-error handlers also catch it.
    assert issubclass(FirewallViolation, PermissionError)
