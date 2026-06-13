"""risk/config.py — pydantic-v2 loader for risk/limits.yaml.

`risk/limits.yaml` is the SINGLE SOURCE OF TRUTH (see CLAUDE.md). This module
only *reads and validates* it; it never writes it. All downstream risk code
(sizing, ratchet, breakers) consumes the validated `Limits` object returned by
`load_limits()` rather than parsing YAML itself.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class RiskLevel(BaseModel):
    """One row of the risk-index table (a single RI setting)."""

    per_trade_pct: float
    max_concurrent: int
    daily_halt_pct: float
    weekly_halt_pct: float
    portfolio_heat_pct: float
    leverage_max: float | None
    options: str
    reference_only: bool = False


class Ratchet(BaseModel):
    """Milestone gain-ratchet configuration."""

    starting_capital: float
    sweep_fraction: float
    milestones: list[float]
    vault_sleeve: str


class ProgramAbort(BaseModel):
    """Always-on program-abort thresholds. Never on the risk dial."""

    monthly_review_drawdown_pct: float
    peak_halt_drawdown_pct: float
    on_risk_dial: bool = False


class CostViability(BaseModel):
    """Small-account cost-viability floor — raise selectivity, never widen risk."""

    min_edge_to_cost_ratio: float
    response: str
    never_widen_risk: bool = True


class Limits(BaseModel):
    """Top-level validated view of risk/limits.yaml."""

    default_ri: int
    band: list[int]
    table: dict[int, RiskLevel]
    conviction_tiers: dict[str, int]
    ratchet: Ratchet
    program_abort: ProgramAbort
    cost_viability: CostViability

    def level(self, ri: int) -> RiskLevel:
        """Return the RiskLevel row for the given risk index."""
        return self.table[ri]

    @property
    def band_low(self) -> int:
        """Low end of the operating band."""
        return self.band[0]

    @property
    def band_high(self) -> int:
        """High end of the operating band."""
        return self.band[1]


def load_limits(path: str | Path = "risk/limits.yaml") -> Limits:
    """Read, map, and validate risk/limits.yaml into a `Limits` object.

    The `meta` block in the YAML is intentionally ignored. `table` keys load as
    ints already under PyYAML, so no key coercion is needed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    risk_index = raw["risk_index"]

    return Limits(
        default_ri=risk_index["default"],
        band=risk_index["band"],
        table=risk_index["table"],
        conviction_tiers=raw["conviction_tiers"],
        ratchet=raw["ratchet"],
        program_abort=raw["program_abort"],
        cost_viability=raw["cost_viability"],
    )
