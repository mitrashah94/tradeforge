"""Level math + point-in-time level builder.

Pure level computation (NTZ, premarket/RTH high-low, daily OHLC rollup,
Wilder ATR(14)) plus :func:`build_levels`, which assembles a per-session
``levels`` row for each symbol from the ``bars`` table.

Point-in-time discipline (no lookahead):
- PDH/PDL come from the PRIOR completed session's RTH high/low (equity) or
  the PRIOR completed UTC day's high/low (crypto).
- PMH/PML come from THIS session's premarket (equity only; crypto has no
  premarket -> None).
- ATR14 is computed from daily bars up to AND INCLUDING the prior session,
  never the current session.
- The current session's own RTH is never used to set its own PDH/PDL.

This module deliberately imports only ``data.sessions`` and ``data.schema``
(plus pandas) so importing it never drags in ingest/broker code.
"""

from __future__ import annotations

import pandas as pd

from data import schema
from data.sessions import (
    crypto_session_date,
    et_session_date,
    is_crypto_weekend,
    is_premarket,
    is_rth,
)


def compute_ntz(pdh, pdl, pmh, pml):
    """No-Trade Zone = overlap of [pdl, pdh] and [pml, pmh].

    Returns ``(ntz_low, ntz_high, ntz_valid)``. Degenerate cases (any input
    None, inverted/zero-width range, or non-overlapping ranges) return
    ``(None, None, False)``.
    """
    if pdh is None or pdl is None or pmh is None or pml is None:
        return (None, None, False)
    if pdh <= pdl or pmh <= pml:
        return (None, None, False)
    low = max(pdl, pml)
    high = min(pdh, pmh)
    if high <= low:
        return (None, None, False)
    return (low, high, True)


def _high_low_where(session_bars: pd.DataFrame, predicate):
    """Max(high)/min(low) over rows whose ts_utc satisfies ``predicate``."""
    if session_bars is None or len(session_bars) == 0:
        return (None, None)
    mask = session_bars["ts_utc"].apply(predicate)
    sel = session_bars[mask]
    if len(sel) == 0:
        return (None, None)
    return (float(sel["high"].max()), float(sel["low"].min()))


def premarket_high_low(session_bars: pd.DataFrame):
    """(max high, min low) over premarket bars; (None, None) if none."""
    return _high_low_where(session_bars, is_premarket)


def rth_high_low(session_bars: pd.DataFrame):
    """(max high, min low) over RTH bars; (None, None) if none."""
    return _high_low_where(session_bars, is_rth)


def daily_ohlc_from_bars(bars_df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    """Roll intraday bars up into daily OHLC.

    Equities: group by ET session date over RTH bars only.
    Crypto:   group by UTC calendar date over the full UTC day.

    Returns a DataFrame sorted ascending by ``session_date`` with columns
    ``session_date, open, high, low, close`` (open = first bar's open of the
    day, close = last bar's close of the day).
    """
    cols = ["session_date", "open", "high", "low", "close"]
    if bars_df is None or len(bars_df) == 0:
        return pd.DataFrame(columns=cols)

    df = bars_df.copy()
    if asset_class == "crypto":
        df["session_date"] = df["ts_utc"].apply(crypto_session_date)
    else:
        df = df[df["ts_utc"].apply(is_rth)]
        if len(df) == 0:
            return pd.DataFrame(columns=cols)
        df["session_date"] = df["ts_utc"].apply(et_session_date)

    df = df.sort_values("ts_utc")
    grouped = df.groupby("session_date", sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    ).reset_index()
    return daily[cols]


def atr14_wilder(daily_df: pd.DataFrame):
    """Wilder RMA ATR(14) over daily OHLC. Matches TradingView's ATR(14)=RMA.

    TR = max(high-low, |high-prev_close|, |low-prev_close|). The first ATR is
    the simple mean of the first 14 true ranges; subsequent values use Wilder
    smoothing. Needs at least 15 daily rows (14 TRs + 1 for the prior close);
    returns None otherwise.
    """
    if daily_df is None or len(daily_df) < 15:
        return None

    df = daily_df.sort_values("session_date").reset_index(drop=True)
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()

    n = len(df)
    trs = []
    for i in range(1, n):
        prev_close = close[i - 1]
        tr = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )
        trs.append(tr)

    # trs has length n-1; need >= 14.
    if len(trs) < 14:
        return None

    atr = sum(trs[:14]) / 14.0
    for tr in trs[14:]:
        atr = (atr * 13.0 + tr) / 14.0
    return float(atr)


def _read_bars(con, symbol: str) -> pd.DataFrame:
    """Read all bars for a symbol, ordered by ts_utc ascending."""
    return con.execute(
        """
        SELECT symbol, asset_class, timeframe, ts_utc, open, high, low,
               close, volume, trade_count, vwap, adjusted
        FROM bars
        WHERE symbol = ?
        ORDER BY ts_utc
        """,
        [symbol],
    ).df()


def build_levels(con, symbol: str, asset_class: str) -> int:
    """Build and upsert point-in-time level rows for one symbol.

    For each session, compute PDH/PDL from the prior completed session,
    PMH/PML from this session's premarket (crypto -> None), NTZ, ATR14 from
    daily bars up to and including the prior session, and is_weekend (crypto).
    Returns the number of level rows written.
    """
    bars = _read_bars(con, symbol)
    if len(bars) == 0:
        return 0

    crypto = asset_class == "crypto"
    session_type = "crypto_utc" if crypto else "rth"
    date_fn = crypto_session_date if crypto else et_session_date

    bars = bars.copy()
    bars["_sdate"] = bars["ts_utc"].apply(date_fn)

    # Daily OHLC over the appropriate calendar; used for PDH/PDL and ATR.
    daily = daily_ohlc_from_bars(bars, asset_class)
    if len(daily) == 0:
        return 0
    daily = daily.sort_values("session_date").reset_index(drop=True)

    # Ordered list of sessions that actually have bars.
    session_dates = sorted(bars["_sdate"].unique())

    rows = []
    for sd in session_dates:
        # Prior session = the last daily row strictly before sd.
        prior = daily[daily["session_date"] < sd]
        if len(prior) == 0:
            pdh = pdl = None
        else:
            last_prior = prior.iloc[-1]
            pdh = float(last_prior["high"])
            pdl = float(last_prior["low"])

        if crypto:
            pmh = pml = None
        else:
            session_bars = bars[bars["_sdate"] == sd]
            pmh, pml = premarket_high_low(session_bars)

        ntz_low, ntz_high, ntz_valid = compute_ntz(pdh, pdl, pmh, pml)

        # ATR uses daily bars up to and including the prior session only.
        atr = atr14_wilder(prior) if len(prior) > 0 else None

        if crypto:
            # is_weekend reflects the (UTC) session day itself.
            is_weekend = sd.weekday() in (5, 6)
        else:
            is_weekend = False

        rows.append(
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "session_date": sd,
                "session_type": session_type,
                "pdh": pdh,
                "pdl": pdl,
                "pmh": pmh,
                "pml": pml,
                "ntz_low": ntz_low,
                "ntz_high": ntz_high,
                "ntz_valid": ntz_valid,
                "atr14": atr,
                "is_weekend": is_weekend,
            }
        )

    return schema.upsert_levels(con, rows)


def build_all(con) -> dict:
    """Build levels for every (symbol, asset_class) present in bars.

    Returns ``{symbol: rows_written}``.
    """
    pairs = con.execute(
        "SELECT DISTINCT symbol, asset_class FROM bars ORDER BY symbol"
    ).fetchall()
    result = {}
    for symbol, asset_class in pairs:
        result[symbol] = build_levels(con, symbol, asset_class)
    return result


if __name__ == "__main__":
    con = schema.connect()
    schema.init_schema(con)
    summary = build_all(con)
    total = sum(summary.values())
    print(
        f"build_levels: {len(summary)} symbol(s), {total} level rows written "
        f"-> {summary}"
    )
