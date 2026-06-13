"""backtest/engine/cost.py — the CostModel driven by backtest/costs.yaml.

The cost model is a first-class part of the sim-to-real gap (MASTER_PLAN.md §5):
a PF-2.24 backtest on optimistic costs can be PF < 1 net of real frictions, so
costs are applied to *every* fill, not bolted on at the end.

Two named profiles live in ``backtest/costs.yaml``:

  - ``tv_style``  — matches the TradingView Pine strategy (commission $1/order,
    slippage 1 tick = $0.01, half_spread $0). Used for the V0 gate.
  - ``realistic`` — honest small-account frictions per asset class.

Application semantics (equity):
  - Entry (market) and STOP fills slip ADVERSELY by (half_spread + slippage).
  - LIMIT / TARGET fills (resting limit) pay half_spread but NO slippage.
  - commission() is cash per order (entry + exit = two orders per round trip).

``half_spread`` / ``slippage`` may be specified in absolute price units
(``*_price``) or basis points of the reference price (``*_bps``); exactly one
of each pair is used per asset class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_COSTS_PATH = Path(__file__).resolve().parents[1] / "costs.yaml"

# Fill kinds that determine how the cost is applied.
FILL_MARKET = "market"  # entry; crosses the spread + slips adversely
FILL_STOP = "stop"      # stop-out; crosses the spread + slips adversely
FILL_LIMIT = "limit"    # target / resting limit; pays half_spread, no slippage


@dataclass(frozen=True)
class AssetCosts:
    """Resolved per-asset-class cost parameters (absolute or bps)."""

    commission_per_order: float
    half_spread_price: float | None
    half_spread_bps: float | None
    slippage_price: float | None
    slippage_bps: float | None

    def half_spread(self, ref_price: float) -> float:
        """Absolute half-spread for ``ref_price`` (price units win over bps)."""
        if self.half_spread_price is not None:
            return float(self.half_spread_price)
        if self.half_spread_bps is not None:
            return float(self.half_spread_bps) / 10_000.0 * ref_price
        return 0.0

    def slippage(self, ref_price: float) -> float:
        """Absolute slippage for ``ref_price`` (price units win over bps)."""
        if self.slippage_price is not None:
            return float(self.slippage_price)
        if self.slippage_bps is not None:
            return float(self.slippage_bps) / 10_000.0 * ref_price
        return 0.0


class CostModel:
    """Applies a named cost profile's frictions to fills and commissions.

    Construct via :meth:`from_profile` (reads ``backtest/costs.yaml``) or pass a
    pre-built ``{asset_class: AssetCosts}`` mapping directly (used in tests).
    """

    def __init__(self, profile_name: str, per_asset: dict[str, AssetCosts]):
        self.profile_name = profile_name
        self._per_asset = per_asset

    # ------------------------------------------------------------------ load
    @classmethod
    def from_profile(
        cls, profile_name: str, path: str | Path = DEFAULT_COSTS_PATH
    ) -> "CostModel":
        """Load a named profile from ``costs.yaml`` into a CostModel."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        profiles = raw["profiles"]
        if profile_name not in profiles:
            raise KeyError(
                f"unknown cost profile {profile_name!r}; "
                f"have {sorted(profiles)}"
            )
        prof = profiles[profile_name]
        per_asset: dict[str, AssetCosts] = {}
        for asset_class in ("equity", "crypto", "option"):
            if asset_class not in prof:
                continue
            p = prof[asset_class]
            per_asset[asset_class] = AssetCosts(
                commission_per_order=float(p.get("commission_per_order", 0.0)),
                half_spread_price=p.get("half_spread_price"),
                half_spread_bps=p.get("half_spread_bps"),
                slippage_price=p.get("slippage_price"),
                slippage_bps=p.get("slippage_bps"),
            )
        return cls(profile_name, per_asset)

    def _asset(self, asset_class: str) -> AssetCosts:
        try:
            return self._per_asset[asset_class]
        except KeyError as exc:
            raise KeyError(
                f"profile {self.profile_name!r} has no params for asset_class "
                f"{asset_class!r}"
            ) from exc

    # ------------------------------------------------------------------ fills
    def apply_entry(
        self, side: str, ref_price: float, asset_class: str = "equity"
    ) -> float:
        """Adverse market entry fill price.

        ``side`` is the position direction ('long'|'short'). A long entry buys,
        so the fill is raised by (half_spread + slippage); a short entry sells,
        so the fill is lowered.
        """
        return self._adverse_fill(side, ref_price, asset_class, FILL_MARKET)

    def apply_exit(
        self,
        side: str,
        ref_price: float,
        asset_class: str = "equity",
        fill_kind: str = FILL_MARKET,
    ) -> float:
        """Exit fill price for a position of direction ``side``.

        ``fill_kind`` selects the friction model:
          - ``FILL_STOP`` / ``FILL_MARKET``: adverse (half_spread + slippage).
          - ``FILL_LIMIT``: resting limit; pays half_spread only, no slippage.

        Exiting a LONG sells (price lowered by frictions); exiting a SHORT buys
        (price raised). The adverse direction is the inverse of the entry.
        """
        if fill_kind == FILL_LIMIT:
            return self._limit_fill(side, ref_price, asset_class)
        return self._adverse_fill(_opposite(side), ref_price, asset_class, fill_kind)

    # ----------------------------------------------------------- internals
    def _adverse_fill(
        self, buy_or_sell_side: str, ref_price: float, asset_class: str, fill_kind: str
    ) -> float:
        """Move ``ref_price`` adversely for a market/stop fill.

        ``buy_or_sell_side`` == 'long' means we are BUYING (fill raised);
        'short' means we are SELLING (fill lowered).
        """
        ac = self._asset(asset_class)
        adverse = ac.half_spread(ref_price) + ac.slippage(ref_price)
        if buy_or_sell_side == "long":
            return ref_price + adverse
        return ref_price - adverse

    def _limit_fill(self, side: str, ref_price: float, asset_class: str) -> float:
        """Resting-limit fill: half_spread only, no slippage.

        Exiting a LONG via a target limit SELLS, so the realized price is
        ref_price - half_spread (you give up the half-spread, but you do not
        slip because the order was resting). Exiting a SHORT via a target limit
        BUYS, so ref_price + half_spread.
        """
        ac = self._asset(asset_class)
        hs = ac.half_spread(ref_price)
        if side == "long":  # selling to close a long
            return ref_price - hs
        return ref_price + hs  # buying to close a short

    # ------------------------------------------------------------ commission
    def commission(
        self, shares: float, notional: float, asset_class: str = "equity"
    ) -> float:
        """Cash commission for ONE order (per-order model).

        Current profiles are flat per-order; ``shares`` and ``notional`` are
        accepted for forward-compatibility with per-share / bps schedules.
        """
        ac = self._asset(asset_class)
        return float(ac.commission_per_order)


def _opposite(side: str) -> str:
    return "short" if side == "long" else "long"
