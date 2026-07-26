"""portfolio/config.py — loader for portfolio/config.yaml (engine policy knobs).

Like ``risk/config.py`` this module only READS and validates; it never writes.
The validated :class:`PortfolioConfig` is what the engine / budgeter consume, so
nothing downstream parses YAML itself. The risk dial (RI table, heat %, halts,
ratchet, program-abort) is NOT here — it stays the single source of truth in
``risk/limits.yaml``; this file is the engine's *own* policy (the synthetic-stop
band, per-family caps, backtest accounting, the Kronos overlay flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class SyntheticStop:
    """The weight-sleeve synthetic-stop band (``stop = mark - k*ATR(window)``).

    ``fallback_band_frac`` is the band width (as a fraction of the mark) used when
    a name has no computable ATR yet (a young ticker) — so a weight position still
    contributes a sane, finite dollar-risk to the heat budget rather than 0 or NaN.
    """

    k: float = 2.5
    window: int = 14
    fallback_band_frac: float = 0.15


@dataclass(frozen=True)
class KronosOverlay:
    """Phase-3 Kronos overlay flags (off by default — engine runs without torch)."""

    use_kronos: bool = False
    rank_blend: float = 0.0
    veto_negative_return: bool = False
    max_downside_cvar: Optional[float] = None


@dataclass(frozen=True)
class PortfolioConfig:
    """Validated view of portfolio/config.yaml."""

    synthetic_stop: SyntheticStop = field(default_factory=SyntheticStop)
    family_caps: dict = field(default_factory=dict)
    cost_bps: float = 2.0
    short_term_tax_rate: float = 0.30
    kronos: KronosOverlay = field(default_factory=KronosOverlay)


def load_portfolio_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PortfolioConfig:
    """Read + validate portfolio/config.yaml into a :class:`PortfolioConfig`.

    Missing keys fall back to the dataclass defaults, so a sparse (or absent) file
    still yields a usable config — the engine is runnable out of the box.
    """
    p = Path(path)
    raw = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    ss = raw.get("synthetic_stop") or {}
    synthetic = SyntheticStop(
        k=float(ss.get("k", 2.5)),
        window=int(ss.get("window", 14)),
        fallback_band_frac=float(ss.get("fallback_band_frac", 0.15)),
    )
    kr = raw.get("kronos") or {}
    kronos = KronosOverlay(
        use_kronos=bool(kr.get("use_kronos", False)),
        rank_blend=float(kr.get("rank_blend", 0.0)),
        veto_negative_return=bool(kr.get("veto_negative_return", False)),
        max_downside_cvar=(
            None if kr.get("max_downside_cvar") is None
            else float(kr.get("max_downside_cvar"))
        ),
    )
    family_caps = {str(k): float(v) for k, v in (raw.get("family_caps") or {}).items()}
    return PortfolioConfig(
        synthetic_stop=synthetic,
        family_caps=family_caps,
        cost_bps=float(raw.get("cost_bps", 2.0)),
        short_term_tax_rate=float(raw.get("short_term_tax_rate", 0.30)),
        kronos=kronos,
    )
