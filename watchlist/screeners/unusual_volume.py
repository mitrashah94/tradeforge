"""watchlist/screeners/unusual_volume.py — relative-volume / volume-spike flags.

Surfaces symbols whose most-recent session volume is abnormally high versus a
rolling baseline — a cheap proxy for "something is happening here" that bumps a
candidate's priority for promotion (MASTER_PLAN.md §4). Pure function over
caller-supplied bars; no network, deterministic offline.

Relative volume (RVOL) = latest session volume ÷ mean of the prior ``lookback``
sessions' volume (the latest session is EXCLUDED from its own baseline). A symbol
is "unusual" when RVOL ≥ ``threshold`` (default 2.0 — twice the typical day).
Symbols without enough history to form a baseline are reported with ``rvol=nan``
and ``unusual=False`` (we never flag on too little data).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from watchlist.screeners.common import daily_volume


@dataclass(frozen=True)
class VolumeFlag:
    """One symbol's relative-volume reading."""

    symbol: str
    asset_class: str
    latest_volume: float
    baseline_volume: float    # mean of the prior `lookback` sessions
    rvol: float               # latest / baseline (nan if no baseline)
    unusual: bool

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "latest_volume": self.latest_volume,
            "baseline_volume": self.baseline_volume,
            "rvol": self.rvol,
            "unusual": self.unusual,
        }


def relative_volume(
    bars: pd.DataFrame, asset_class: str = "equity", lookback: int = 20
) -> float:
    """Relative volume of the LATEST session vs the prior ``lookback`` sessions.

    Returns ``nan`` if there are not at least 2 sessions (no baseline) or the
    baseline is zero. The latest session is excluded from its own baseline.
    """
    vol = daily_volume(bars, asset_class)
    if len(vol) < 2:
        return float("nan")
    latest = float(vol.iloc[-1])
    prior = vol.iloc[-(lookback + 1):-1]  # up to `lookback` sessions before last
    if len(prior) == 0:
        return float("nan")
    baseline = float(prior.mean())
    if baseline <= 0:
        return float("nan")
    return latest / baseline


def unusual_volume_flags(
    bars_by_symbol: dict[str, pd.DataFrame],
    asset_class: str = "equity",
    lookback: int = 20,
    threshold: float = 2.0,
) -> list[VolumeFlag]:
    """Flag symbols whose latest-session RVOL ≥ ``threshold``.

    Returns a :class:`VolumeFlag` for every symbol (so a caller can read RVOL even
    for non-spiking names), sorted by descending RVOL with the ``unusual`` ones
    on top. Symbols without a baseline get ``rvol=nan`` / ``unusual=False`` and
    sort last.
    """
    flags: list[VolumeFlag] = []
    for symbol, bars in bars_by_symbol.items():
        vol = daily_volume(bars, asset_class)
        if len(vol) == 0:
            continue
        latest = float(vol.iloc[-1])
        prior = vol.iloc[-(lookback + 1):-1]
        baseline = float(prior.mean()) if len(prior) else 0.0
        rvol = relative_volume(bars, asset_class, lookback)
        unusual = bool(rvol == rvol and rvol >= threshold)  # rvol==rvol filters nan
        flags.append(
            VolumeFlag(
                symbol=symbol,
                asset_class=asset_class,
                latest_volume=latest,
                baseline_volume=baseline,
                rvol=rvol,
                unusual=unusual,
            )
        )

    def _sort_key(f: VolumeFlag):
        # nan rvol sorts last; otherwise descending rvol.
        r = f.rvol
        return (0 if r == r else 1, -(r if r == r else 0.0))

    flags.sort(key=_sort_key)
    return flags
