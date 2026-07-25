"""orchestrator/agents/ — the deterministic compute behind the lean LLM roster.

MASTER_PLAN.md §4 splits the system into a deterministic fast loop and an
LLM slow loop. The LLM agents (defined under ``.claude/agents/``) decide
*policy*; the computable part of that policy lives here as plain, testable,
network-free Python so the same inputs always yield the same outputs.

Modules
-------
- :mod:`orchestrator.agents.regime_reader` — the daily policy-setter. Computes
  the session regime + realized-vol read, maps regime -> armed strategies +
  an exposure scalar, and publishes a ``REGIME_TAGGED`` event the fast-loop /
  risk gate consume. NO LLM / MCP / network in this path.
- :mod:`orchestrator.agents.firewall` — the programmatic backstop for the §6
  research firewall: agents read everything, write NOTHING live. Defines the
  protected live-config paths and a guard that refuses agent writes to them.
- :mod:`orchestrator.agents.journalist` — the journalist: auto-journals closed
  trades (frame card, slippage, MFE/MAE, deterministic plan-adherence flags, a
  3-line narrative + chart PNG), writes premarket/EOD digests, and pushes them
  through the NOTIFY tool. Writes journal/ artifacts only (FIREWALL).
"""

from orchestrator.agents.firewall import (
    PROTECTED_PREFIXES,
    PROTECTED_PATHS,
    WRITABLE_PREFIXES,
    FirewallViolation,
    agent_writable,
    assert_not_live_config,
)
from orchestrator.agents.journalist import (
    FrameCard,
    JournalEntry,
    Journalist,
    PlanAdherence,
    Trade,
)
from orchestrator.agents.regime_reader import (
    ARMED_BY_REGIME,
    RegimeAssessment,
    RegimeReaderConfig,
    assess,
    exposure_scalar_for,
    publish,
)

__all__ = [
    # firewall
    "PROTECTED_PREFIXES",
    "PROTECTED_PATHS",
    "WRITABLE_PREFIXES",
    "FirewallViolation",
    "agent_writable",
    "assert_not_live_config",
    # journalist
    "Journalist",
    "Trade",
    "JournalEntry",
    "FrameCard",
    "PlanAdherence",
    # regime_reader
    "ARMED_BY_REGIME",
    "RegimeAssessment",
    "RegimeReaderConfig",
    "assess",
    "exposure_scalar_for",
    "publish",
]
