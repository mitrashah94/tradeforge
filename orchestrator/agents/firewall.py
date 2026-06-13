"""orchestrator/agents/firewall.py — the programmatic research-firewall backstop.

MASTER_PLAN.md §6 ("Self-Improvement Conveyor — safe by construction") states
the firewall in plain English:

    Research agents read everything, write nothing live. Production changes only
    through the deterministic gate + human confirm. ... FORBIDDEN: ... any LLM
    *writing live config or `limits.yaml`*; promoting on in-sample results.

That rule is *cultural* — it lives in the agent ``.md`` files and CLAUDE.md so the
model knows not to do it. THIS module is the *programmatic* backstop: a hard guard
that any code an agent runs can call before it writes a path, so a misbehaving (or
prompt-injected) agent cannot quietly mutate the live knobs. It enforces the
asymmetry the conveyor depends on:

    * LIVE CONFIG  (risk dial, strategy registry live fields, the locked OOS
      vault, the hook/settings file) -> READ-ONLY for agents. Edited by a human,
      by hand, deliberately, and only promoted through the deterministic gate.
    * RESEARCH / JOURNAL / REPORT areas -> agent-WRITABLE. This is where the
      strategy-researcher drops proposals, where the journalist writes journals
      and digests, and where analysts emit reports. Nothing here touches money
      until a human promotes it.

Design notes
------------
* Pure stdlib, no I/O, no network — safe to import and call anywhere, including
  inside a hook or a test. It does not *open* files; it only *judges paths*.
* Path matching is normalized and repo-root-relative so ``risk/limits.yaml``,
  ``./risk/limits.yaml``, an absolute path inside the repo, and
  ``a/b/../../risk/limits.yaml`` all resolve to the same protected target. The
  default deny posture is conservative: a path that is neither explicitly
  writable NOR explicitly protected is treated as NOT agent-writable (deny by
  default), because the safe failure mode for "can an agent write here?" is no.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath

# --------------------------------------------------------------------------- #
# Protected LIVE-CONFIG surface (agents may READ, never WRITE)                 #
# --------------------------------------------------------------------------- #
# Exact repo-root-relative files that are the live knobs / source-of-truth and
# must never be written by an agent (MASTER_PLAN.md §6, CLAUDE.md §self-improve).
PROTECTED_PATHS: frozenset[str] = frozenset(
    {
        "risk/limits.yaml",          # THE risk dial / ratchet / abort — single source of truth
        "strategies/registry.yaml",  # strategy status/allocation/edge — live promotion state
        ".claude/settings.json",     # the PreToolUse live-order gate hook config
        ".claude/settings.local.json",
        "backtest/stats/oos_vault.yaml",  # the LOCKED out-of-sample vault (never tune/reuse)
        "CLAUDE.md",                 # mission + locked-in conventions
        "MASTER_PLAN.md",            # the plan itself
    }
)

# Directory prefixes whose entire subtree is protected live config. Anything a
# human treats as a production knob lives under one of these.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "risk/",        # all risk config + deterministic breaker config
    ".claude/",     # agent defs, settings, hooks, skills — the harness wiring
    "orderbook/",   # the live order ledger / state — never agent-written
    "paper/",       # the paper ledger (identical event path to live)
)

# Directory prefixes that ARE agent-writable: research output, journals, reports,
# and digests. Nothing here is live until a human promotes it through the gate.
WRITABLE_PREFIXES: tuple[str, ...] = (
    "research/",    # strategy-researcher proposals, hypothesis logs, candidate params
    "journal/",     # per-trade journals + premarket/EOD digests (journalist)
    "reports/",     # analyst reports
    "reporting/out/",  # rendered chart PNGs / generated report artifacts
    "backtest/reports/",  # backtest run reports (NOT the locked vault)
    "tmp/",
    "scratch/",
)


class FirewallViolation(PermissionError):
    """Raised when an agent attempts to write a protected live-config path.

    Subclasses :class:`PermissionError` so existing ``except (OSError,
    PermissionError)`` handlers around file writes also catch it, while callers
    that want the specific case can ``except FirewallViolation``.
    """


def _normalize(path: str | os.PathLike) -> str:
    """Return a repo-root-relative, POSIX, ``..``-collapsed form of ``path``.

    Absolute paths inside the repo are made relative to the repo root; paths are
    lower-effort normalized with forward slashes so the matching is OS- and
    style-independent. We do NOT touch the filesystem (no realpath / no stat) so
    this stays pure and works on paths that don't exist yet.
    """
    raw = os.fspath(path)
    # Collapse separators / "." / ".." lexically and switch to POSIX slashes.
    norm = PurePosixPath(os.path.normpath(raw)).as_posix()

    repo_root = _repo_root()
    if norm.startswith(repo_root + "/"):
        norm = norm[len(repo_root) + 1 :]
    elif norm == repo_root:
        norm = ""

    # Strip a single leading "./" that normpath may leave on relative inputs.
    if norm.startswith("./"):
        norm = norm[2:]
    return norm


def _repo_root() -> str:
    """Best-effort repo root (this file is ``<root>/orchestrator/agents/firewall.py``)."""
    here = PurePosixPath(os.path.normpath(os.path.abspath(__file__))).as_posix()
    # .../orchestrator/agents/firewall.py -> strip three components.
    return str(PurePosixPath(here).parent.parent.parent)


def _is_protected(rel: str) -> bool:
    if rel in PROTECTED_PATHS:
        return True
    return any(rel.startswith(p) for p in PROTECTED_PREFIXES)


def _is_writable(rel: str) -> bool:
    return any(rel.startswith(p) for p in WRITABLE_PREFIXES)


def agent_writable(path: str | os.PathLike) -> bool:
    """Return True iff an agent is allowed to WRITE ``path``.

    Decision order (protect-first, deny-by-default):
      1. If ``path`` is a protected live-config path/prefix -> ``False``.
      2. Else if ``path`` is under an explicitly writable prefix -> ``True``.
      3. Else (unknown) -> ``False``. The safe answer to "may an agent write an
         unrecognized path?" is no; widen :data:`WRITABLE_PREFIXES` deliberately.

    Reads are never gated by this module — agents read everything (§6); this is
    purely a WRITE guard.
    """
    rel = _normalize(path)
    if _is_protected(rel):
        return False
    return _is_writable(rel)


def assert_not_live_config(path: str | os.PathLike) -> None:
    """Raise :class:`FirewallViolation` if ``path`` is protected live config.

    Call this immediately before any agent-initiated write. It enforces §6's
    "any LLM writing live config or ``limits.yaml`` is FORBIDDEN": the write of a
    protected path raises rather than proceeding. Non-protected paths return
    ``None`` (this guard does NOT, by itself, assert the path is writable — use
    :func:`agent_writable` for the positive check; a path may be neither
    protected nor on the writable allowlist).
    """
    rel = _normalize(path)
    if _is_protected(rel):
        raise FirewallViolation(
            f"research firewall: agents may not write live config {rel!r} "
            f"(MASTER_PLAN.md §6). Production changes go through the deterministic "
            f"gate + human confirm, never an LLM write."
        )
