"""Corporate-action back-adjustment utilities (splits, dividends).

In live ingest, Alpaca's ``adjustment=ALL`` is the primary path and already
returns split/dividend-adjusted bars. These utilities are the *tested
fallback* for raw/unadjusted sources and serve as the continuity guarantee:
they let us re-derive adjusted series deterministically and verify that
ingested data is free of un-adjusted discontinuities.

Back-adjustment convention: adjust HISTORY to be consistent with the most
recent (post-event) price scale. The latest bars are left untouched; older
bars are scaled so the series is continuous across each corporate action.

All functions return a NEW DataFrame and never mutate their input. The input
is expected to be sorted ascending by ``ts_utc`` with at least the columns
``open, high, low, close, volume``.
"""

from __future__ import annotations

import pandas as pd


def _to_date(value):
    """Normalize an effective/ex date to a ``datetime.date``."""
    return pd.Timestamp(value).date()


def _bar_dates(df: pd.DataFrame) -> pd.Series:
    """Per-row calendar date derived from ts_utc (fallback to the index)."""
    if "ts_utc" in df.columns:
        src = df["ts_utc"]
    else:
        src = df.index.to_series()
    return pd.to_datetime(src).dt.date


def apply_split_adjustment(df: pd.DataFrame, splits) -> pd.DataFrame:
    """Back-adjust OHLCV for stock splits.

    ``splits`` is a list of ``(effective_date, ratio)`` where
    ``ratio = shares_after / shares_before`` (a 2:1 split is ratio 2.0).

    For every bar whose date is STRICTLY BEFORE an effective date, OHLC is
    divided by that split's ratio and volume multiplied by it. Multiple splits
    compound (a bar before two 2:1 splits is divided by 4.0). The most recent
    bars (on/after every effective date) are unchanged.

    Returns a new DataFrame; the input is not mutated.
    """
    out = df.copy()
    if out is None or len(out) == 0 or not splits:
        return out

    dates = _bar_dates(out)
    # Cumulative split factor per row: product of ratios for splits whose
    # effective date is after the bar.
    factor = pd.Series(1.0, index=out.index)
    for eff_date, ratio in splits:
        eff = _to_date(eff_date)
        pre = dates < eff
        factor = factor * pd.Series(
            [ratio if p else 1.0 for p in pre], index=out.index
        )

    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] / factor
    if "volume" in out.columns:
        out["volume"] = out["volume"] * factor

    return out


def apply_dividend_adjustment(df: pd.DataFrame, dividends) -> pd.DataFrame:
    """Back-adjust OHLC for cash dividends (approximate).

    ``dividends`` is a list of ``(ex_date, cash_amount)``. For each dividend we
    compute a multiplicative factor ``(1 - amount / close_on_ex)`` using the
    close of the FIRST bar on/after the ex-date, and apply it to every bar
    STRICTLY BEFORE the ex-date. Factors from multiple dividends compound.

    Approximation notes: this is the standard "proportional" back-adjustment
    used by most charting tools. It assumes the dividend is fully reflected in
    a same-day price drop and uses the ex-date close as the reference price.
    Volume is not adjusted for dividends. If no bar exists on/after an ex-date,
    that dividend is skipped (nothing to anchor against).

    Returns a new DataFrame; the input is not mutated.
    """
    out = df.copy()
    if out is None or len(out) == 0 or not dividends:
        return out

    dates = _bar_dates(out)
    factor = pd.Series(1.0, index=out.index)

    for ex_date, amount in dividends:
        ex = _to_date(ex_date)
        on_or_after = dates >= ex
        if not on_or_after.any():
            continue  # nothing to anchor the adjustment against
        ref_idx = out.index[on_or_after][0]
        close_on_ex = out.loc[ref_idx, "close"]
        if close_on_ex is None or close_on_ex == 0:
            continue
        adj = 1.0 - (amount / close_on_ex)
        pre = dates < ex
        factor = factor * pd.Series(
            [adj if p else 1.0 for p in pre], index=out.index
        )

    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] * factor

    return out
