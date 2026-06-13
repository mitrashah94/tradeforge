"""backtest/stats/regime.py — per-session market-regime classification.

MASTER_PLAN.md §5/§9: report *"performance by regime (trend / chop / vol-shock),
not just aggregate."* This module tags each equity session date as one of three
regimes from daily QQQ data, and slices a daily-return series by those tags so a
strategy's edge can be inspected per regime (a trend-follower should earn in
'trend', a mean-reverter in 'chop', and everyone should be scrutinized in
'vol_shock').

The features (all from daily OHLC + the point-in-time ATR14 in ``levels``)
--------------------------------------------------------------------------
* **true_range / ATR14** — today's true range relative to the trailing ATR. A
  ratio above ``vol_shock_tr_mult`` (default 1.8) is a volatility shock: an
  unusually wide, often gappy/news day. (Empirically ~the 92nd percentile on
  QQQ, so vol_shock is the genuinely-anomalous tail, not routine wide days.)
* **distance from the N-day SMA, in ATRs** — ``|close - SMA_N| / ATR14``. Far
  from the mean in EITHER direction = a strong directional/trending regime.
  Above ``trend_sma_atr`` (default 2.0 ATRs) tags 'trend'.

Classification (priority order)
-------------------------------
    vol_shock  if  true_range/ATR14 >= vol_shock_tr_mult        (checked first)
    trend      elif |close - SMA_N|/ATR14 >= trend_sma_atr
    chop       otherwise  (the low-range, mean-reverting default)

vol_shock wins ties because a shock day's risk character dominates whatever
trend/chop label it would otherwise get. Sessions without enough history to
compute the features (no ATR14, or fewer than N prior closes) are tagged 'chop'
(the conservative, do-nothing-special default) so EVERY session is partitioned —
the tagger never drops a date.

Thresholds are parameters on :class:`RegimeConfig`; the defaults above are
documented and were calibrated against the QQQ daily distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from data.levels import daily_ohlc_from_bars
from data.schema import DEFAULT_DB_PATH, connect

TREND = "trend"
CHOP = "chop"
VOL_SHOCK = "vol_shock"
REGIMES = (TREND, CHOP, VOL_SHOCK)


@dataclass(frozen=True)
class RegimeConfig:
    """Tunable regime thresholds (documented defaults from the QQQ daily dist)."""

    sma_window: int = 20         # lookback for the trend SMA
    trend_sma_atr: float = 2.0   # |close - SMA| / ATR14 above this => trend
    vol_shock_tr_mult: float = 1.8  # true_range / ATR14 at/above this => vol_shock


def _as_date(d) -> date:
    if hasattr(d, "date") and not isinstance(d, date):
        return d.date()
    return d


# --------------------------------------------------------------------------- #
# Feature table
# --------------------------------------------------------------------------- #
def daily_features(daily: pd.DataFrame, atr_by_date: dict, cfg: RegimeConfig):
    """Build the per-session feature frame from daily OHLC + ATR14.

    ``daily`` has columns ``session_date, open, high, low, close`` (ascending).
    ``atr_by_date`` maps ``session_date -> atr14`` (the point-in-time ATR from
    ``levels``). Returns a DataFrame with ``session_date, tr_atr, dist_sma_atr``.
    """
    df = daily.sort_values("session_date").reset_index(drop=True).copy()
    df["session_date"] = df["session_date"].apply(_as_date)
    df["atr14"] = df["session_date"].map(
        {(_as_date(k)): v for k, v in atr_by_date.items()}
    )

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    sma = close.rolling(cfg.sma_window).mean()

    atr = df["atr14"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["tr_atr"] = np.where(atr > 0, tr / atr, np.nan)
        df["dist_sma_atr"] = np.where(atr > 0, (close - sma).abs() / atr, np.nan)
    return df[["session_date", "tr_atr", "dist_sma_atr"]]


def _classify_row(tr_atr: float, dist_sma_atr: float, cfg: RegimeConfig) -> str:
    """Classify one session from its features (priority: vol_shock > trend > chop)."""
    if tr_atr is not None and np.isfinite(tr_atr) and tr_atr >= cfg.vol_shock_tr_mult:
        return VOL_SHOCK
    if (
        dist_sma_atr is not None
        and np.isfinite(dist_sma_atr)
        and dist_sma_atr >= cfg.trend_sma_atr
    ):
        return TREND
    return CHOP


# --------------------------------------------------------------------------- #
# Public: regime tags
# --------------------------------------------------------------------------- #
def regime_tags(
    symbol: str = "QQQ",
    db_path: str = DEFAULT_DB_PATH,
    cfg: RegimeConfig | None = None,
    con=None,
) -> dict:
    """Return ``{session_date: regime}`` for every session of ``symbol``.

    Reads the symbol's bars + point-in-time ATR14 from the DB, derives daily OHLC
    via :func:`data.levels.daily_ohlc_from_bars`, and tags each session. EVERY
    session that has daily data is tagged (no date is dropped); sessions lacking
    the features to decide are tagged ``chop``. Deterministic for a given DB +
    config.
    """
    cfg = cfg or RegimeConfig()
    own_con = con is None
    if own_con:
        con = connect(db_path)
    try:
        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND timeframe = '5m'
            ORDER BY ts_utc
            """,
            [symbol],
        ).df()
        lv = con.execute(
            "SELECT session_date, atr14 FROM levels WHERE symbol = ? ORDER BY session_date",
            [symbol],
        ).df()
    finally:
        if own_con:
            con.close()

    if len(bars) == 0:
        return {}

    daily = daily_ohlc_from_bars(bars, "equity")
    atr_by_date = {
        _as_date(r.session_date): (None if pd.isna(r.atr14) else float(r.atr14))
        for r in lv.itertuples(index=False)
    }
    return regime_tags_from_daily(daily, atr_by_date, cfg)


def regime_tags_from_daily(
    daily: pd.DataFrame, atr_by_date: dict, cfg: RegimeConfig | None = None
) -> dict:
    """Tag sessions directly from a daily OHLC frame + ATR map (no DB).

    Exposed for testing and for callers that already have the daily frame.
    """
    cfg = cfg or RegimeConfig()
    if daily is None or len(daily) == 0:
        return {}
    feats = daily_features(daily, atr_by_date, cfg)
    tags: dict = {}
    for row in feats.itertuples(index=False):
        tags[_as_date(row.session_date)] = _classify_row(
            row.tr_atr, row.dist_sma_atr, cfg
        )
    return tags


# --------------------------------------------------------------------------- #
# Public: slice metrics by regime
# --------------------------------------------------------------------------- #
def by_regime(daily_returns: pd.Series, tags: dict) -> dict:
    """Partition a daily-return series by regime and compute per-regime metrics.

    ``daily_returns`` is indexed by session_date (date or Timestamp);
    ``tags`` maps session_date -> regime. Returns
    ``{regime: {n_days, total, mean, std, sharpe, win_rate, ...}}`` for each of
    the three regimes (a regime with no overlapping days reports zeros / nan).

    Days in ``daily_returns`` whose date is not in ``tags`` are tagged 'chop'
    (the partition is total — nothing is silently dropped).
    """
    from backtest.stats.metrics import sharpe as _sharpe

    if daily_returns is None or len(daily_returns) == 0:
        return {r: _empty_regime_metrics() for r in REGIMES}

    s = daily_returns.copy()
    # Normalize the index to plain dates so it joins to the tag keys.
    idx_dates = [_as_date(i) for i in s.index]
    s.index = idx_dates
    regime_of = [tags.get(d, CHOP) for d in idx_dates]

    out: dict = {}
    for regime in REGIMES:
        vals = s.values[np.asarray(regime_of) == regime]
        out[regime] = _regime_metrics(np.asarray(vals, dtype="float64"), _sharpe)
    return out


def _regime_metrics(vals: np.ndarray, sharpe_fn) -> dict:
    n = int(vals.size)
    if n == 0:
        return _empty_regime_metrics()
    wins = vals[vals > 0]
    losses = vals[vals < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else float("nan"))
    return {
        "n_days": n,
        "total": float(vals.sum()),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
        "sharpe": sharpe_fn(vals) if n > 1 else float("nan"),
        "win_rate": float((vals > 0).mean()),
        "profit_factor": pf,
    }


def _empty_regime_metrics() -> dict:
    return {
        "n_days": 0,
        "total": 0.0,
        "mean": float("nan"),
        "std": float("nan"),
        "sharpe": float("nan"),
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
    }
