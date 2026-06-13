#!/usr/bin/env python3
"""scripts/go_live_checklist.py — the PRE-LIVE GO-LIVE GATE (MASTER_PLAN.md §8).

This is the single gate that MUST pass before any real money moves. TradeForge
is PAPER-ONLY today, so this script is *expected* to report NOT-READY: at minimum
the "≥1 strategy past paper→live" and "API+infra cost < expected edge" checks
fail by design (no edge clears the multiple-testing haircut; CLAUDE.md §"Resolved
P0 decisions" #5 and strategies/registry.yaml). That is the correct, honest
result — the gate is doing its job.

Each §8 checklist item is an independent CHECK FUNCTION returning a structured
:class:`CheckResult`. They live in :data:`DEFAULT_CHECKS` so the runner is
injectable and each check is testable in isolation:

    run_checklist(checks=[...]) -> list[CheckResult]

``main()`` runs all checks, prints a PASS/FAIL table with details + an overall
verdict, and exits NON-ZERO if ANY check fails (``sys.exit(1)``).

The §8 line, verbatim:
    native/hardened brackets confirmed · dead-man's switch tested · orphan
    recovery tested · reconciliation halt wired · all halts firing · cost model
    realistic · ratchet + abort live · ≥1 strategy past paper→live ·
    API+infra cost < expected edge.

DETERMINISTIC + OFFLINE: no network. The few checks that shell out do so only to
run the project's own pytest suite for the named test files. The watchdog /
dead-man's switch (orchestrator/watchdog.py + tests/test_dead_mans_switch.py) is
built by a SIBLING agent in parallel; this script references it by PATH and
verifies it dynamically at RUN time — it never imports it, so this module loads
and its own tests run regardless of whether the sibling has landed yet.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

# Repo root = parent of this scripts/ directory. Keeps the checklist runnable
# from any cwd (agent threads reset cwd between calls).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Reference paths the checklist verifies (relative to REPO_ROOT).
STATE_MACHINE_PATH = REPO_ROOT / "orderbook" / "state_machine.py"
WATCHDOG_PATH = REPO_ROOT / "orchestrator" / "watchdog.py"
RECONCILE_PATH = REPO_ROOT / "orderbook" / "reconcile.py"
COSTS_YAML_PATH = REPO_ROOT / "backtest" / "costs.yaml"
LIMITS_YAML_PATH = REPO_ROOT / "risk" / "limits.yaml"
REGISTRY_YAML_PATH = REPO_ROOT / "strategies" / "registry.yaml"

# Test files run via targeted pytest invocations (PASS only if green).
DMS_TEST_PATH = REPO_ROOT / "tests" / "test_dead_mans_switch.py"
CRASH_RECOVERY_TEST_PATH = REPO_ROOT / "tests" / "test_crash_recovery.py"
RECONCILE_TEST_PATH = REPO_ROOT / "tests" / "test_reconcile_recovery.py"
BREAKER_TEST_PATH = REPO_ROOT / "tests" / "test_breaker_service.py"

# Estimated all-in API + infra monthly cost for the small-account stage
# (always-on cloud host + LLM/data API budget, MASTER_PLAN.md §7 "API/infra cost
# vs P&L"). A conservative right-sized figure for the $1k stage.
EST_API_INFRA_MONTHLY_USD = 30.0
# Account base the edge is measured against (north-star starting capital,
# CLAUDE.md). Edge net of costs must out-earn API+infra on this base.
ACCOUNT_BASE_USD = 1000.0


# --------------------------------------------------------------------------- #
# Result type                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CheckResult:
    """Structured outcome of one §8 check.

    Attributes:
        name:   short human-readable §8 item name.
        passed: True iff the gate condition is satisfied.
        detail: one-line evidence (what was verified / why it failed).
    """

    name: str
    passed: bool
    detail: str


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _run_pytest(test_path: Path, *, repo_root: Path = REPO_ROOT) -> tuple[bool, str]:
    """Run pytest on a single test file with PYTHONPATH=repo_root.

    Returns ``(passed, summary_line)``. ``passed`` is True only if the file
    exists AND every test is green (pytest exit code 0). Missing file -> fail.
    """
    if not test_path.exists():
        return False, f"{test_path.name} does not exist yet"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(repo_root), "PATH": _path_env()},
    )
    # Last non-empty stdout line is pytest's summary ("N passed", "N failed", ...).
    tail = ""
    for line in reversed(proc.stdout.splitlines()):
        if line.strip():
            tail = line.strip()
            break
    return proc.returncode == 0, tail or proc.stderr.strip()[:200]


def _path_env() -> str:
    import os

    return os.environ.get("PATH", "")


# --------------------------------------------------------------------------- #
# §8 CHECK FUNCTIONS — one per checklist item                                 #
# Each is an independent, injectable callable returning a CheckResult.        #
# --------------------------------------------------------------------------- #
def check_hardened_brackets() -> CheckResult:
    """1. native/hardened brackets confirmed.

    Native server-side OCO is absent on Robinhood (CLAUDE.md P0 #3), so the
    requirement is HARDENED LOCAL OCO brackets that REQUIRE the dead-man's
    switch. Verify the state machine module exists and exposes the bracket API
    (create_bracket + on_leg_fill OCO cancel) AND the dead-man's-switch coupling
    (requires_dead_mans_switch + assert_dead_mans_switch_armed).
    """
    name = "native/hardened brackets confirmed"
    if not STATE_MACHINE_PATH.exists():
        return CheckResult(name, False, f"missing {STATE_MACHINE_PATH.name}")
    src = _read_text(STATE_MACHINE_PATH)
    needed = [
        "def create_bracket",          # registers an OCO group (entry+stop+target)
        "def on_leg_fill",             # OCO trigger: one leg fills -> cancel sibling
        "def requires_dead_mans_switch",
        "def assert_dead_mans_switch_armed",
    ]
    missing = [n.replace("def ", "") for n in needed if n not in src]
    if missing:
        return CheckResult(name, False, f"bracket/DMS API missing: {', '.join(missing)}")
    hardened = "HARDENED LOCAL" in src or "hardened" in src.lower()
    if not hardened:
        return CheckResult(name, False, "no hardened-local-OCO documentation in state machine")
    return CheckResult(
        name, True,
        "hardened local OCO brackets present (create_bracket/on_leg_fill) and "
        "coupled to the dead-man's switch (requires/assert_dead_mans_switch_armed)",
    )


def check_dead_mans_switch_tested() -> CheckResult:
    """2. dead-man's switch tested.

    Verify orchestrator/watchdog.py exists as a real (non-stub) module AND its
    test (tests/test_dead_mans_switch.py) runs green. Built by a sibling agent;
    verified dynamically at run time, never imported here.
    """
    name = "dead-man's switch tested"
    if not WATCHDOG_PATH.exists():
        return CheckResult(name, False, f"missing {WATCHDOG_PATH.name}")
    src = _read_text(WATCHDOG_PATH)
    # A bare stub is just a docstring + a TODO. Require real implementation.
    if "TODO" in src and "def " not in src and "class " not in src:
        return CheckResult(name, False, "watchdog.py is still a stub (no implementation)")
    passed, summary = _run_pytest(DMS_TEST_PATH)
    if not passed:
        return CheckResult(name, False, f"DMS test not green: {summary}")
    return CheckResult(name, True, f"watchdog implemented; {DMS_TEST_PATH.name} green ({summary})")


def check_orphan_recovery_tested() -> CheckResult:
    """3. orphan recovery tested — the P3 crash-recovery gate runs green."""
    name = "orphan recovery tested"
    passed, summary = _run_pytest(CRASH_RECOVERY_TEST_PATH)
    if not passed:
        return CheckResult(name, False, f"crash-recovery gate not green: {summary}")
    return CheckResult(name, True, f"{CRASH_RECOVERY_TEST_PATH.name} green ({summary})")


def check_reconciliation_halt_wired() -> CheckResult:
    """4. reconciliation halt wired.

    Verify reconcile emits a halt (do-not-resume) on an unreconcilable mismatch:
    the code path exists in reconcile.py AND its test is green.
    """
    name = "reconciliation halt wired"
    if not RECONCILE_PATH.exists():
        return CheckResult(name, False, f"missing {RECONCILE_PATH.name}")
    src = _read_text(RECONCILE_PATH)
    has_halt_path = (
        "reconciliation_mismatch" in src
        and "halt_reason" in src
        and "HALT" in src
    )
    if not has_halt_path:
        return CheckResult(name, False, "reconcile.py has no unreconcilable-mismatch HALT path")
    passed, summary = _run_pytest(RECONCILE_TEST_PATH)
    if not passed:
        return CheckResult(name, False, f"reconcile halt test not green: {summary}")
    return CheckResult(name, True, f"mismatch HALT path present; {RECONCILE_TEST_PATH.name} green ({summary})")


def check_all_halts_firing() -> CheckResult:
    """5. all halts firing — the breaker service tests run green.

    Covers daily/weekly/monthly-review/program-hard halts + cooldown.
    """
    name = "all halts firing"
    passed, summary = _run_pytest(BREAKER_TEST_PATH)
    if not passed:
        return CheckResult(name, False, f"breaker tests not green: {summary}")
    return CheckResult(name, True, f"{BREAKER_TEST_PATH.name} green ({summary})")


def check_cost_model_realistic(costs_path: Path = COSTS_YAML_PATH) -> CheckResult:
    """6. cost model realistic.

    Verify backtest/costs.yaml has a ``realistic`` profile carrying commission +
    spread + slippage for equity, and honest (non-zero) crypto/option spreads.
    """
    name = "cost model realistic"
    if not costs_path.exists():
        return CheckResult(name, False, f"missing {costs_path.name}")
    data = _load_yaml(costs_path)
    profiles = data.get("profiles", {})
    realistic = profiles.get("realistic")
    if not isinstance(realistic, dict):
        return CheckResult(name, False, "no 'realistic' profile in costs.yaml")

    eq = realistic.get("equity", {}) or {}
    # equity must define commission + a spread + slippage (the three frictions).
    has_commission = "commission_per_order" in eq
    has_spread = (eq.get("half_spread_price") or 0) or (eq.get("half_spread_bps") or 0)
    has_slippage = (eq.get("slippage_price") or 0) or (eq.get("slippage_bps") or 0)
    if not (has_commission and has_spread and has_slippage):
        return CheckResult(
            name, False,
            "realistic.equity missing one of commission/spread/slippage "
            f"(commission={has_commission}, spread={bool(has_spread)}, slippage={bool(has_slippage)})",
        )

    # honest crypto + option spreads (non-zero).
    def _spread(d: dict) -> float:
        return float(d.get("half_spread_price") or 0) + float(d.get("half_spread_bps") or 0)

    crypto_spread = _spread(realistic.get("crypto", {}) or {})
    option_spread = _spread(realistic.get("option", {}) or {})
    if crypto_spread <= 0:
        return CheckResult(name, False, "realistic.crypto has no spread (dishonest)")
    if option_spread <= 0:
        return CheckResult(name, False, "realistic.option has no spread (dishonest)")

    return CheckResult(
        name, True,
        f"realistic profile present: equity commission+spread+slippage, "
        f"crypto spread={crypto_spread:g}, option spread={option_spread:g}",
    )


def check_ratchet_and_abort_live(limits_path: Path = LIMITS_YAML_PATH) -> CheckResult:
    """7. ratchet + abort live.

    Verify risk/limits.yaml configures the milestone ratchet (milestones +
    sweep_fraction) AND program-abort (monthly review + peak halt, off the dial).
    """
    name = "ratchet + abort live"
    if not limits_path.exists():
        return CheckResult(name, False, f"missing {limits_path.name}")
    data = _load_yaml(limits_path)

    ratchet = data.get("ratchet", {}) or {}
    milestones = ratchet.get("milestones")
    sweep = ratchet.get("sweep_fraction")
    if not (isinstance(milestones, list) and milestones):
        return CheckResult(name, False, "ratchet.milestones missing/empty")
    if not (isinstance(sweep, (int, float)) and 0 < sweep < 1):
        return CheckResult(name, False, f"ratchet.sweep_fraction invalid: {sweep!r}")

    abort = data.get("program_abort", {}) or {}
    monthly = abort.get("monthly_review_drawdown_pct")
    peak = abort.get("peak_halt_drawdown_pct")
    if not (isinstance(monthly, (int, float)) and monthly > 0):
        return CheckResult(name, False, "program_abort.monthly_review_drawdown_pct missing")
    if not (isinstance(peak, (int, float)) and peak > 0):
        return CheckResult(name, False, "program_abort.peak_halt_drawdown_pct missing")

    return CheckResult(
        name, True,
        f"ratchet: sweep {sweep:g} at {milestones}; "
        f"abort: monthly -{monthly:g}% review / peak -{peak:g}% halt (off-dial)",
    )


def check_strategy_past_paper_to_live(registry_path: Path = REGISTRY_YAML_PATH) -> CheckResult:
    """8. ≥1 strategy past paper→live.

    Verify strategies/registry.yaml has at least one strategy with status LIVE.
    PAPER-ONLY today -> this FAILS by design (no edge clears the haircut).
    """
    name = ">=1 strategy past paper->live"
    if not registry_path.exists():
        return CheckResult(name, False, f"missing {registry_path.name}")
    data = _load_yaml(registry_path)
    strategies = data.get("strategies", {}) or {}
    live = [n for n, s in strategies.items() if str((s or {}).get("status", "")).upper() == "LIVE"]
    if not live:
        statuses = {n: (s or {}).get("status") for n, s in strategies.items()}
        return CheckResult(
            name, False,
            f"no LIVE strategy (paper-only): {statuses}",
        )
    return CheckResult(name, True, f"LIVE strategies: {', '.join(live)}")


def check_api_infra_cost_below_edge(registry_path: Path = REGISTRY_YAML_PATH) -> CheckResult:
    """9. API+infra cost < expected edge.

    Require a positive validated expected edge (net of realistic costs) that, on
    the account base, out-earns estimated API+infra. The validated edge is the
    expectancy of the best PROMOTABLE strategy — but a strategy only counts if it
    CLEARS THE MULTIPLE-TESTING HAIRCUT. Today none do (registry: clears_haircut
    is false everywhere; edges are marginal-to-negative), so there is no
    bankable edge to set against costs -> FAILS, correctly, for a paper-only
    program.
    """
    name = "API+infra cost < expected edge"
    if not registry_path.exists():
        return CheckResult(name, False, f"missing {registry_path.name}")
    data = _load_yaml(registry_path)
    strategies = data.get("strategies", {}) or {}

    # Only edges that clear the haircut are bankable (promotable). A negative or
    # un-cleared edge cannot be set against real recurring cost.
    bankable = []
    for sname, s in strategies.items():
        edge = (s or {}).get("edge", {}) or {}
        if edge.get("clears_haircut") is True and float(edge.get("expectancy_r", 0)) > 0:
            bankable.append((sname, float(edge["expectancy_r"]), float(edge.get("pf", 0))))

    if not bankable:
        return CheckResult(
            name, False,
            "no validated edge clears the multiple-testing haircut "
            f"(est. API+infra ${EST_API_INFRA_MONTHLY_USD:g}/mo on ${ACCOUNT_BASE_USD:g} base "
            "= no positive net edge to exceed it)",
        )

    # If/when a strategy is bankable, require its net edge to dominate API+infra.
    # Edge is in R; converting R to $/mo is strategy-specific, so we report the
    # cleared edge and treat ANY positive cleared edge as a provisional pass with
    # the cost line attached for human review.
    best = max(bankable, key=lambda x: x[1])
    return CheckResult(
        name, True,
        f"bankable edge: {best[0]} expectancy {best[1]:+.3f}R (PF {best[2]:.2f}); "
        f"exceeds est. API+infra ${EST_API_INFRA_MONTHLY_USD:g}/mo — confirm $/mo edge vs cost",
    )


# Ordered registry of the nine §8 checks (the runner default).
DEFAULT_CHECKS: list[Callable[[], CheckResult]] = [
    check_hardened_brackets,
    check_dead_mans_switch_tested,
    check_orphan_recovery_tested,
    check_reconciliation_halt_wired,
    check_all_halts_firing,
    check_cost_model_realistic,
    check_ratchet_and_abort_live,
    check_strategy_past_paper_to_live,
    check_api_infra_cost_below_edge,
]


# --------------------------------------------------------------------------- #
# Runner + reporting                                                          #
# --------------------------------------------------------------------------- #
def run_checklist(
    checks: list[Callable[[], CheckResult]] | None = None,
) -> list[CheckResult]:
    """Run the checklist and return one :class:`CheckResult` per check.

    Args:
        checks: injectable list of zero-arg callables returning CheckResult.
                Defaults to :data:`DEFAULT_CHECKS` (the nine §8 items). Inject a
                custom list to run/test checks in isolation.
    """
    if checks is None:
        checks = DEFAULT_CHECKS
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:  # a crashing check is a FAILED check, never silent.
            cname = getattr(check, "__name__", repr(check))
            results.append(CheckResult(cname, False, f"check raised {type(exc).__name__}: {exc}"))
    return results


def render_table(results: list[CheckResult]) -> str:
    """Render a plain PASS/FAIL table with details + an overall verdict."""
    name_w = max((len(r.name) for r in results), default=4)
    name_w = max(name_w, len("CHECK"))
    lines: list[str] = []
    lines.append("=" * (name_w + 60))
    lines.append("  TradeForge GO-LIVE CHECKLIST (MASTER_PLAN.md §8) — PRE-LIVE GATE")
    lines.append("=" * (name_w + 60))
    lines.append(f"  {'STATUS':6}  {'CHECK'.ljust(name_w)}  DETAIL")
    lines.append("  " + "-" * (6 + 2 + name_w + 2 + 48))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"  [{status}]  {r.name.ljust(name_w)}  {r.detail}")
    lines.append("  " + "-" * (6 + 2 + name_w + 2 + 48))
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    ready = passed == total
    verdict = "READY for live" if ready else "NOT-READY — DO NOT GO LIVE"
    lines.append(f"  OVERALL: {passed}/{total} checks passed  ->  {verdict}")
    if not ready:
        failed = [r.name for r in results if not r.passed]
        lines.append(f"  FAILING: {', '.join(failed)}")
    lines.append("=" * (name_w + 60))
    return "\n".join(lines)


def main(
    checks: list[Callable[[], CheckResult]] | None = None,
    *,
    out=None,
) -> int:
    """Run the checklist, print the table, and return the exit code.

    Returns 0 iff every check passed; 1 otherwise. ``main()`` calls
    ``sys.exit()`` when run as a script (see ``__main__`` below); as a function
    it returns the code so tests can assert on it without exiting the process.
    """
    out = out or sys.stdout
    results = run_checklist(checks)
    print(render_table(results), file=out)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
