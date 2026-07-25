"""prop/rules.py — the prop-firm rule set (loader + validated view).

``PropRules`` is the deterministic encoding of ONE firm's evaluation + funded
rules the account state machine enforces. Loads representative presets from
``prop/firms.yaml`` (verify against the live firm before committing money). Like
``risk/config.py`` this only reads + validates; it never writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_FIRMS_PATH = Path(__file__).resolve().parent / "firms.yaml"


@dataclass(frozen=True)
class PropRules:
    """One firm's evaluation + funded-account rules (all dollar figures derived).

    The state machine reads these; the simulator reports P(pass)/payout/EV against
    them. See ``prop/firms.yaml`` for field docs. Every ``*_pct`` is a FRACTION of
    ``account_size``.
    """

    name: str
    account_size: float
    profit_target_pct: float
    max_drawdown_pct: float
    trailing: bool = True
    daily_loss_pct: Optional[float] = None
    min_trading_days: int = 0
    consistency_pct: Optional[float] = None
    profit_split: float = 0.90
    eval_fee: float = 150.0
    payout_min_profit_pct: float = 0.0

    # --- derived dollar levels ---
    @property
    def profit_target(self) -> float:
        """Dollar profit needed to pass the eval."""
        return self.account_size * self.profit_target_pct

    @property
    def max_drawdown(self) -> float:
        """Dollar overall max loss (the account-killer threshold width)."""
        return self.account_size * self.max_drawdown_pct

    @property
    def daily_loss(self) -> Optional[float]:
        """Dollar daily loss limit (``None`` if the firm has none)."""
        if self.daily_loss_pct is None:
            return None
        return self.account_size * self.daily_loss_pct

    @property
    def payout_min_profit(self) -> float:
        """Dollar profit required before a funded payout is allowed."""
        return self.account_size * self.payout_min_profit_pct

    @property
    def target_balance(self) -> float:
        """Balance at which the eval is passed."""
        return self.account_size + self.profit_target


def load_firm(name: str, path: str | Path = DEFAULT_FIRMS_PATH) -> PropRules:
    """Load one firm profile by name from ``firms.yaml`` into a :class:`PropRules`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    firms = raw.get("firms", {}) or {}
    if name not in firms:
        raise KeyError(f"unknown firm {name!r}; have {sorted(firms)}")
    cfg = dict(firms[name])
    return PropRules(
        name=name,
        account_size=float(cfg["account_size"]),
        profit_target_pct=float(cfg["profit_target_pct"]),
        max_drawdown_pct=float(cfg["max_drawdown_pct"]),
        trailing=bool(cfg.get("trailing", True)),
        daily_loss_pct=(None if cfg.get("daily_loss_pct") is None
                        else float(cfg["daily_loss_pct"])),
        min_trading_days=int(cfg.get("min_trading_days", 0)),
        consistency_pct=(None if cfg.get("consistency_pct") is None
                         else float(cfg["consistency_pct"])),
        profit_split=float(cfg.get("profit_split", 0.90)),
        eval_fee=float(cfg.get("eval_fee", 150.0)),
        payout_min_profit_pct=float(cfg.get("payout_min_profit_pct", 0.0)),
    )


def list_firms(path: str | Path = DEFAULT_FIRMS_PATH) -> list[str]:
    """Names of the available firm presets."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return sorted((raw.get("firms", {}) or {}).keys())
