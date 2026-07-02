"""strategies/lev_trend/strategy.py — the LEVERAGED TREND-GATE sleeve.

A daily :class:`~backtest.daily.engine.DailyStrategy` (weight-shaped): hold a
LEVERAGED long index ETF while the benchmark closes above its long SMA (optionally
also requiring positive 12m momentum — the dual gate), park in a safe asset below
it. See params.yaml for the thesis + honesty notes. Reuses the tested signal
primitives from the rotation sleeve (``total_return`` / ``above_sma``) so there is
one implementation of each. Point-in-time by construction (the history view is
pre-sliced); PURE / DETERMINISTIC.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from strategies.momentum_rotation.strategy import above_sma, total_return

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"


def load_params(variant: str = "DEFAULT", path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    """Load ``defaults`` merged with a named ``variant`` delta from params.yaml."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    params = dict(raw.get("defaults", {}))
    variants = raw.get("variants", {}) or {}
    if variant not in variants:
        raise KeyError(f"unknown variant {variant!r}; have {sorted(variants)}")
    params.update(variants[variant] or {})
    params["_variant"] = variant
    return params


class LevTrendStrategy:
    """Levered long above the trend gate, safe asset below it. Long-only.

    ``target_weights`` returns ``{lev_symbol: cap}`` when the gate is RISK-ON
    (benchmark close >= SMA, and — with ``require_mom`` — its 12m total return
    positive) else ``{safe_symbol: cap}``. Insufficient history for the SMA is
    risk-OFF (the conservative side, mirroring ``above_sma``'s convention).
    """

    def __init__(self, params: dict | None = None, variant: str = "DEFAULT"):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)
        p = self.params
        self.benchmark = str(p["benchmark"])
        self.sma_window = int(p["sma_window"])
        self.lev_symbol = str(p["lev_symbol"])
        self.safe_symbol = str(p["safe_symbol"])
        self.require_mom = bool(p["require_mom"])
        self.mom_lookback = int(p["mom_lookback"])
        self.weight_cap = float(p["weight_cap"])

    def extra_symbols(self) -> list[str]:
        """Every ticker this sleeve can weight or read — union into the universe."""
        return [self.benchmark, self.lev_symbol, self.safe_symbol]

    def target_weights(self, asof_date, history) -> dict:
        px = history.prices(symbols=[self.benchmark],
                            lookback=max(self.sma_window, self.mom_lookback + 1) + 5)
        if len(px) == 0:
            return {self.safe_symbol: self.weight_cap}
        series = px[self.benchmark].dropna()
        risk_on = above_sma(series, self.sma_window)
        if risk_on and self.require_mom:
            mom = total_return(series, self.mom_lookback)
            risk_on = (mom == mom) and mom > 0.0   # finite AND positive
        target = self.lev_symbol if risk_on else self.safe_symbol
        return {target: self.weight_cap}
