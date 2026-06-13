"""tests/test_go_live_checklist.py — tests for the §8 GO-LIVE GATE runner.

Deterministic, offline, no network. Crucially, these tests do NOT depend on the
sibling-built watchdog (orchestrator/watchdog.py + tests/test_dead_mans_switch.py)
actually existing: the FAIL-FAST behaviour is exercised with INJECTED/mocked
checks, and the strategy-status check is exercised against TEMP registry files.
A few checks are also run against the REAL repo files (cost model, ratchet/abort)
to confirm they pass today.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.go_live_checklist import (
    CheckResult,
    DEFAULT_CHECKS,
    check_cost_model_realistic,
    check_ratchet_and_abort_live,
    check_strategy_past_paper_to_live,
    main,
    run_checklist,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Injected-check helpers (no dependency on any sibling-built module)          #
# --------------------------------------------------------------------------- #
def _passing(name: str = "ok") -> CheckResult:
    return CheckResult(name, True, "fine")


def _failing(name: str = "bad", detail: str = "broke") -> CheckResult:
    return CheckResult(name, False, detail)


def passing_check() -> CheckResult:
    return _passing("passing_check")


def failing_check() -> CheckResult:
    return _failing("failing_check", "deliberate failure")


# --------------------------------------------------------------------------- #
# (a) Mixed pass/fail injected checks -> overall not-ready, exit code != 0,    #
#     failing item is named / reported.                                        #
# --------------------------------------------------------------------------- #
def test_mixed_checks_overall_not_ready_and_names_the_failure(capsys):
    code = main(checks=[passing_check, failing_check, passing_check])
    assert code != 0  # any failure -> non-zero exit
    out = capsys.readouterr().out
    assert "NOT-READY" in out
    assert "failing_check" in out          # the failing item is reported
    assert "deliberate failure" in out     # ...with its detail
    assert "[FAIL]" in out and "[PASS]" in out


def test_run_checklist_preserves_order_and_results():
    results = run_checklist(checks=[passing_check, failing_check])
    assert [r.name for r in results] == ["passing_check", "failing_check"]
    assert results[0].passed is True
    assert results[1].passed is False


def test_a_single_failing_check_fails_the_gate():
    code = main(checks=[failing_check])
    assert code == 1


def test_a_crashing_check_is_treated_as_failed_not_silent():
    def boom() -> CheckResult:
        raise RuntimeError("kaboom")

    results = run_checklist(checks=[boom])
    assert results[0].passed is False
    assert "kaboom" in results[0].detail
    assert main(checks=[boom]) == 1


# --------------------------------------------------------------------------- #
# (b) strategy-status check: fails on no-LIVE registry, passes on a LIVE one.  #
#     Uses TEMP files so no dependency on the real registry's current state.   #
# --------------------------------------------------------------------------- #
_NO_LIVE_REGISTRY = textwrap.dedent(
    """
    strategies:
      breakout_retest:
        status: PAPER
      level_meanrev:
        status: RESEARCH
    """
)

_HAS_LIVE_REGISTRY = textwrap.dedent(
    """
    strategies:
      breakout_retest:
        status: LIVE
      level_meanrev:
        status: RESEARCH
    """
)


def test_strategy_status_check_fails_on_no_live_registry(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text(_NO_LIVE_REGISTRY, encoding="utf-8")
    result = check_strategy_past_paper_to_live(registry_path=reg)
    assert result.passed is False
    assert "no LIVE" in result.detail


def test_strategy_status_check_passes_on_live_registry(tmp_path):
    reg = tmp_path / "registry.yaml"
    reg.write_text(_HAS_LIVE_REGISTRY, encoding="utf-8")
    result = check_strategy_past_paper_to_live(registry_path=reg)
    assert result.passed is True
    assert "breakout_retest" in result.detail


def test_strategy_status_check_fails_on_missing_registry(tmp_path):
    result = check_strategy_past_paper_to_live(registry_path=tmp_path / "nope.yaml")
    assert result.passed is False
    assert "missing" in result.detail


# --------------------------------------------------------------------------- #
# (c) cost-model and ratchet/abort checks pass against the REAL repo files.    #
# --------------------------------------------------------------------------- #
def test_cost_model_check_passes_against_real_repo():
    result = check_cost_model_realistic()
    assert result.passed is True, result.detail
    assert "realistic profile present" in result.detail


def test_cost_model_check_fails_when_crypto_spread_is_zero(tmp_path):
    bad = tmp_path / "costs.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            profiles:
              realistic:
                equity: {commission_per_order: 0.0, half_spread_price: 0.005, slippage_price: 0.01}
                crypto: {commission_per_order: 0.0, half_spread_bps: 0.0, slippage_bps: 0.0}
                option: {half_spread_price: 0.05}
            """
        ),
        encoding="utf-8",
    )
    result = check_cost_model_realistic(costs_path=bad)
    assert result.passed is False
    assert "crypto" in result.detail


def test_ratchet_and_abort_check_passes_against_real_repo():
    result = check_ratchet_and_abort_live()
    assert result.passed is True, result.detail
    assert "sweep" in result.detail and "abort" in result.detail


def test_ratchet_check_fails_when_program_abort_missing(tmp_path):
    bad = tmp_path / "limits.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            ratchet:
              sweep_fraction: 0.25
              milestones: [2500, 5000]
            """
        ),
        encoding="utf-8",
    )
    result = check_ratchet_and_abort_live(limits_path=bad)
    assert result.passed is False
    assert "program_abort" in result.detail


# --------------------------------------------------------------------------- #
# (d) all checks pass -> exit zero.                                            #
# --------------------------------------------------------------------------- #
def test_all_passing_checks_exit_zero(capsys):
    code = main(checks=[passing_check, passing_check, passing_check])
    assert code == 0
    out = capsys.readouterr().out
    assert "READY for live" in out
    assert "NOT-READY" not in out


def test_empty_checklist_is_vacuously_ready():
    # No checks -> all() over empty is True -> exit 0 (degenerate but defined).
    assert main(checks=[]) == 0


# --------------------------------------------------------------------------- #
# Sanity: the real DEFAULT_CHECKS registry is wired and reports NOT-READY      #
# today (paper-only). We assert the structure + the EXPECTED failing items,    #
# without depending on the sibling watchdog: those checks may legitimately     #
# flip to PASS once the sibling lands, so we only assert the paper-only ones.  #
# --------------------------------------------------------------------------- #
def test_default_checks_has_nine_items():
    assert len(DEFAULT_CHECKS) == 9


def test_real_repo_is_not_ready_today_paper_only():
    results = run_checklist()  # real DEFAULT_CHECKS against the real repo
    by_name = {r.name: r for r in results}
    # Paper-only gate: these two MUST fail today regardless of the sibling.
    assert by_name[">=1 strategy past paper->live"].passed is False
    assert by_name["API+infra cost < expected edge"].passed is False
    # Overall verdict is NOT-READY while any check fails.
    assert main() == 1
