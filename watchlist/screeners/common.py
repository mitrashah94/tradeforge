"""watchlist/screeners/common.py — shared liquidity primitives for the screeners.

Pure functions over a single symbol's OHLCV bars (a DataFrame with at least
``ts_utc, open, high, low, close, volume``). They reduce intraday bars to the
per-symbol liquidity statistics the tier thresholds in ``criteria.yaml`` gate on
(average daily dollar-volume, last/median price, session count). No I/O, no
network — the caller supplies the bars (read from ``market.duckdb``), which keeps
the screeners deterministic and offline-testable (MASTER_PLAN.md §4).

Dollar-volume is computed *per session* and then averaged, so a symbol's
liquidity number is "typical dollars traded per day" rather than a total that
balloons with history length. Sessions are the ET calendar date for equities and
the UTC calendar date for crypto (matching ``data.sessions``); a caller that does
not care can pass ``asset_class`` and we pick the right calendar.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.sessions import crypto_session_date, et_session_date


@dataclass(frozen=True)
class SymbolStats:
    """Per-symbol liquidity snapshot used by the tier screeners."""

    symbol: str
    asset_class: str
    n_sessions: int
    last_price: float
    median_price: float
    avg_dollar_volume: float   # mean per-session dollar-volume
    avg_daily_volume: float    # mean per-session share/coin volume
    total_volume: float

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "n_sessions": self.n_sessions,
            "last_price": self.last_price,
            "median_price": self.median_price,
            "avg_dollar_volume": self.avg_dollar_volume,
            "avg_daily_volume": self.avg_daily_volume,
            "total_volume": self.total_volume,
        }


def _session_of(asset_class: str):
    return crypto_session_date if asset_class == "crypto" else et_session_date


def _typical_price(bars: pd.DataFrame) -> pd.Series:
    """Per-bar typical price (H+L+C)/3 — a robust proxy for traded price."""
    return (
        bars["high"].astype("float64")
        + bars["low"].astype("float64")
        + bars["close"].astype("float64")
    ) / 3.0


def daily_dollar_volume(bars: pd.DataFrame, asset_class: str = "equity") -> pd.Series:
    """Per-session dollar-volume Σ(typical_price · volume), indexed by session date.

    Dollar-volume of a bar ≈ typical price × volume; summing within a session
    gives that day's traded dollars. Returns an empty float Series for empty
    input.
    """
    if bars is None or len(bars) == 0:
        return pd.Series(dtype="float64", name="dollar_volume")
    df = bars.copy()
    df["_dv"] = _typical_price(df) * df["volume"].astype("float64")
    df["_sd"] = df["ts_utc"].apply(_session_of(asset_class))
    out = df.groupby("_sd")["_dv"].sum().sort_index()
    out.name = "dollar_volume"
    out.index.name = "session_date"
    return out


def daily_volume(bars: pd.DataFrame, asset_class: str = "equity") -> pd.Series:
    """Per-session traded volume (shares/coins), indexed by session date."""
    if bars is None or len(bars) == 0:
        return pd.Series(dtype="float64", name="volume")
    df = bars.copy()
    df["_sd"] = df["ts_utc"].apply(_session_of(asset_class))
    out = df.groupby("_sd")["volume"].sum().astype("float64").sort_index()
    out.name = "volume"
    out.index.name = "session_date"
    return out


def summarize_symbol(
    symbol: str, bars: pd.DataFrame, asset_class: str = "equity"
) -> SymbolStats | None:
    """Reduce one symbol's bars to a :class:`SymbolStats` snapshot.

    Returns ``None`` if there are no bars (nothing to screen). ``last_price`` is
    the most recent bar's close; ``median_price`` is the median typical price
    across all bars (robust to a single bad print). ``avg_dollar_volume`` /
    ``avg_daily_volume`` are the means of the per-session series.
    """
    if bars is None or len(bars) == 0:
        return None
    df = bars.sort_values("ts_utc")
    tp = _typical_price(df)
    last_price = float(df["close"].iloc[-1])
    median_price = float(tp.median())

    dv = daily_dollar_volume(df, asset_class)
    dvol = daily_volume(df, asset_class)
    n_sessions = int(len(dv))
    avg_dv = float(dv.mean()) if n_sessions else 0.0
    avg_vol = float(dvol.mean()) if len(dvol) else 0.0
    total_vol = float(df["volume"].astype("float64").sum())

    return SymbolStats(
        symbol=symbol,
        asset_class=asset_class,
        n_sessions=n_sessions,
        last_price=last_price,
        median_price=median_price,
        avg_dollar_volume=avg_dv,
        avg_daily_volume=avg_vol,
        total_volume=total_vol,
    )


def passes_thresholds(
    stats: SymbolStats, min_dollar_volume: float, min_price: float
) -> bool:
    """True if a symbol clears the tier's min dollar-volume AND min price.

    Price is gated on the LAST price (what you would trade at today); liquidity
    on the average per-session dollar-volume (typical traded dollars/day).
    """
    if stats is None:
        return False
    return (
        stats.avg_dollar_volume >= float(min_dollar_volume)
        and stats.last_price >= float(min_price)
    )
