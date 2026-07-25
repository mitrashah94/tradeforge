"""Polygon historical bar ingestion for TradeForge (MASTER_PLAN §5, P1 / Phase 0).

Pulls point-in-time intraday bars from the Polygon.io aggregates API into the
DuckDB ``bars`` table — the **consolidated/SIP-quality** sibling of
``data.pipelines.alpaca_ingest`` (which uses the free Alpaca IEX feed). It writes
the *identical* ``bars`` schema (``data.schema.upsert_bars``), so every downstream
consumer — levels, backtest, the gate — is untouched; only the data source changes.

Why this module exists (Phase 0 of the go-live roadmap): no strategy clears the
multiple-testing haircut on Alpaca IEX data, and the strongly-suspected cause is
that IEX (a single venue, ~2-3% of US volume) clips intraday extremes versus the
consolidated SIP tape. Re-pulling from Polygon and re-running the *unchanged* gate
is the highest-ROI step toward a live-worthy edge.

Scope (data layer only — **no trading / strategy logic** lives here):

* Timeframes: ``2m`` and ``5m`` (Polygon custom-multiplier minute aggregates).
* Universe (default): equities ``SPY``, ``QQQ`` and crypto ``BTC/USD``, ``ETH/USD``.
* Lookback: 2 years back by default (overridable via ``days`` / ``start`` / ``end``).

Two correctness invariants this module is responsible for (matching alpaca_ingest):

1. **Point-in-time.** We never store the in-progress (not-yet-closed) bar. ``end``
   defaults to "now" (UTC); after fetching we DROP any bar whose CLOSE time
   (``ts_open + timeframe``) is strictly greater than ``end``. Storing a forming
   bar would leak the future into a backtest (train/serve skew).
2. **Adjustment.** Equity aggregates are requested ``adjusted=True`` (Polygon
   split-adjusts aggregates) and tagged ``adjusted=True``. NOTE: Polygon does NOT
   dividend-adjust aggregates — for short intraday lookbacks this is immaterial to
   PDH/PDL/ATR levels, and the SIP-vs-IEX quality gain is the real win. Crypto has
   no splits/dividends, so it is tagged ``adjusted=False``.

Polygon aggregate timestamps (``t``) are Unix MILLISECONDS at the bar OPEN, UTC.
We convert to tz-naive UTC to match the ``bars`` table contract in ``data.schema``.

The Polygon SDK (``polygon-api-client``) is imported LAZILY inside :func:`run`
(not at module top), so the pure transform :func:`_aggs_to_frame` and this module
import fine for offline tests without the SDK installed. Credentials are read from
``POLYGON_API_KEY`` (see ``.env.example``).

Install the SDK:  ``pip3 install polygon-api-client``
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

import pandas as pd

from data import schema as _schema

__all__ = [
    "TIMEFRAME_MINUTES",
    "DEFAULT_SYMBOLS",
    "DEFAULT_LOOKBACK_DAYS",
    "polygon_ticker",
    "run",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported timeframe labels -> their length in minutes (Polygon multiplier).
TIMEFRAME_MINUTES: dict[str, int] = {"2m": 2, "5m": 5}

#: Default universe: symbol -> asset_class ('equity' | 'crypto').
DEFAULT_SYMBOLS: dict[str, str] = {
    "SPY": "equity",
    "QQQ": "equity",
    "BTC/USD": "crypto",
    "ETH/USD": "crypto",
}

#: Default lookback when neither ``start`` nor ``days`` is supplied (2 years).
DEFAULT_LOOKBACK_DAYS: int = 365 * 2

#: Bar columns the produced DataFrame must contain, in ``data.schema.upsert_bars``
#: order. Kept here so the build matches the documented ``bars`` table contract.
_BAR_COLUMNS: tuple[str, ...] = (
    "symbol",
    "asset_class",
    "timeframe",
    "ts_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "adjusted",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def polygon_ticker(symbol: str, asset_class: str) -> str:
    """Map a TradeForge symbol to a Polygon ticker.

    Equities pass through unchanged (``"QQQ"`` -> ``"QQQ"``). Crypto pairs are
    mapped to Polygon's ``X:`` form with the slash removed
    (``"BTC/USD"`` -> ``"X:BTCUSD"``).
    """
    if asset_class == "crypto":
        return "X:" + symbol.replace("/", "").upper()
    return symbol.upper()


def _now_utc() -> datetime:
    """Current time as a tz-aware UTC datetime (single source of 'now')."""
    return datetime.now(timezone.utc)


def _coerce_utc(value: datetime | str | None, *, default: datetime) -> datetime:
    """Coerce a datetime/ISO-string/None into a tz-aware UTC datetime.

    Naive datetimes (and naive ISO strings) are assumed to already be UTC.
    """
    if value is None:
        dt = default
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        dt = value
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported datetime value: {value!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _require_api_key() -> str:
    """Read the Polygon API key from the environment or fail loudly."""
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError(
            "Polygon API key missing. Set POLYGON_API_KEY in your environment "
            "(see .env.example)."
        )
    return key


def _agg_attr(agg, name: str, *aliases):
    """Read a field from a Polygon Agg object OR a plain dict, trying aliases.

    The Polygon SDK returns ``Agg`` dataclasses (attributes ``open``/``high``/...,
    ``timestamp``, ``transactions``, ``vwap``); the raw REST JSON uses short keys
    (``o``/``h``/``l``/``c``/``v``/``vw``/``n``/``t``). Supporting both keeps the
    transform testable with plain dicts and robust to SDK shape changes.
    """
    if isinstance(agg, Mapping):
        for k in (name, *aliases):
            if k in agg:
                return agg[k]
        return None
    for k in (name, *aliases):
        if hasattr(agg, k):
            return getattr(agg, k)
    return None


def _aggs_to_frame(
    aggs: Iterable,
    *,
    symbol: str,
    asset_class: str,
    timeframe: str,
    adjusted: bool,
    end_utc: datetime,
) -> pd.DataFrame:
    """Build a bars DataFrame from Polygon aggregates for one (symbol, tf).

    Accepts an iterable of Polygon ``Agg`` objects or plain dicts. Handles empty
    responses, normalises the millisecond OPEN timestamp to tz-naive UTC, drops
    the in-progress bar (point-in-time), and returns columns in the
    ``data.schema.upsert_bars`` contract order. Pure / no network.
    """
    rows = []
    for agg in aggs:
        ts_ms = _agg_attr(agg, "timestamp", "t")
        if ts_ms is None:
            continue
        # Polygon `t` is Unix MILLISECONDS at the bar OPEN, UTC.
        ts_open = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
        n = _agg_attr(agg, "transactions", "n")
        rows.append(
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "ts_utc": ts_open.replace(tzinfo=None),  # store tz-naive UTC
                "open": _agg_attr(agg, "open", "o"),
                "high": _agg_attr(agg, "high", "h"),
                "low": _agg_attr(agg, "low", "l"),
                "close": _agg_attr(agg, "close", "c"),
                "volume": _agg_attr(agg, "volume", "v"),
                "trade_count": int(n) if n is not None else None,
                "vwap": _agg_attr(agg, "vwap", "vw"),
                "adjusted": adjusted,
                # transient: bar CLOSE time (tz-naive UTC), used only to drop the
                # forming bar. Kept tz-naive to match ts_utc and end_naive below.
                "_close_utc": (
                    ts_open + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
                ).replace(tzinfo=None),
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))

    df = pd.DataFrame(rows)

    # ---- POINT-IN-TIME: drop the in-progress (not-yet-closed) bar ----------
    # Any bar whose CLOSE time is strictly after `end` (default "now") has not
    # finished forming and must not be persisted.
    end_naive = end_utc.astimezone(timezone.utc).replace(tzinfo=None)
    df = df[df["_close_utc"] <= end_naive]
    df = df.drop(columns=["_close_utc"]).copy()

    if df.empty:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))

    # Deterministic order + exact contract columns.
    df = df.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    return df[list(_BAR_COLUMNS)]


def _fetch_aggs(
    client,
    *,
    ticker: str,
    timeframe: str,
    start_utc: datetime,
    end_utc: datetime,
    adjusted: bool,
    pause: float = 0.0,
    window_days: int | None = None,
) -> list:
    """Fetch Polygon aggregates for one (ticker, timeframe), materialized to a list.

    ``client.list_aggs`` auto-paginates (follows ``next_url``). With a rate-limited
    key, pass ``window_days`` to split the range into sub-windows that each fit in a
    single page (no internal pagination) and ``pause`` to sleep between requests so
    the call rate stays under the plan's per-minute cap. ``window_days=None`` does a
    single ranged call and relies on the client's retry/Retry-After backoff.
    """
    mult = TIMEFRAME_MINUTES[timeframe]

    def _one(from_ms: int, to_ms: int) -> list:
        if pause:
            time.sleep(pause)
        return list(
            client.list_aggs(
                ticker=ticker,
                multiplier=mult,
                timespan="minute",
                from_=from_ms,
                to=to_ms,
                adjusted=adjusted,
                sort="asc",
                limit=50000,
            )
        )

    if not window_days:
        return _one(int(start_utc.timestamp() * 1000), int(end_utc.timestamp() * 1000))

    out: list = []
    win = timedelta(days=window_days)
    cursor = start_utc
    while cursor < end_utc:
        chunk_end = min(cursor + win, end_utc)
        out.extend(_one(int(cursor.timestamp() * 1000), int(chunk_end.timestamp() * 1000)))
        cursor = chunk_end
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    symbols: Mapping[str, str] | None = None,
    timeframes=("2m", "5m"),
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    db_path: str = _schema.DEFAULT_DB_PATH,
    days: int | None = None,
    client=None,
    retries: int = 10,
    pause: float = 0.0,
    window_days: int | None = None,
) -> dict[str, int]:
    """Ingest Polygon historical bars into the DuckDB ``bars`` table.

    Args:
        symbols: mapping of ``symbol -> asset_class`` (``'equity'`` | ``'crypto'``).
            Defaults to :data:`DEFAULT_SYMBOLS` (SPY, QQQ, BTC/USD, ETH/USD).
        timeframes: iterable of timeframe labels; only ``"2m"`` / ``"5m"`` supported.
        start: window start (datetime or ISO string). If omitted, derived from
            ``days`` or the 2-year default. Naive values are treated as UTC.
        end: window end (datetime or ISO string). Defaults to "now" (UTC) and
            anchors the point-in-time drop of the in-progress bar.
        db_path: DuckDB path; defaults to :data:`data.schema.DEFAULT_DB_PATH`.
        days: if given, ``start = now - days`` (overrides the 2-year default).
            Explicit ``start`` still wins over ``days``.
        client: optional pre-built Polygon ``RESTClient`` (injectable for tests).
            When ``None`` a real client is constructed from ``POLYGON_API_KEY``.

    Returns:
        Mapping of ``f"{symbol}:{timeframe}" -> rows_written`` (rows upserted);
        ``-1`` marks a per-target failure (the rest of the universe still ingests).

    Raises:
        RuntimeError: if the Polygon API key is missing and no ``client`` is given.
        ValueError: on an unsupported timeframe or empty window.
    """
    if symbols is None:
        symbols = dict(DEFAULT_SYMBOLS)

    timeframes = tuple(timeframes)
    for tf in timeframes:
        if tf not in TIMEFRAME_MINUTES:
            raise ValueError(
                f"unsupported timeframe {tf!r}; expected one of "
                f"{sorted(TIMEFRAME_MINUTES)}"
            )

    # ---- Resolve the [start, end] window (all tz-aware UTC) -----------------
    now = _now_utc()
    end_utc = _coerce_utc(end, default=now)
    if start is not None:
        start_utc = _coerce_utc(start, default=now)
    elif days is not None:
        start_utc = now - timedelta(days=days)
    else:
        start_utc = now - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    if start_utc >= end_utc:
        raise ValueError(
            f"empty window: start ({start_utc.isoformat()}) is not before "
            f"end ({end_utc.isoformat()})"
        )

    # Lazily construct the Polygon client (keeps the SDK out of the import graph).
    # ``retries`` is passed to the SDK's urllib3 Retry, which honors the 429
    # ``Retry-After`` header — so a rate-limited (free/Starter tier) key
    # self-throttles and eventually succeeds instead of erroring out.
    if client is None:
        from polygon import RESTClient  # lazy: only needed for a real pull

        client = RESTClient(_require_api_key(), retries=retries)

    results: dict[str, int] = {}

    con = _schema.connect(db_path)
    try:
        _schema.init_schema(con)

        for symbol, asset_class in symbols.items():
            asset_class = (asset_class or "").lower()
            if asset_class not in ("equity", "crypto"):
                raise ValueError(
                    f"unknown asset_class {asset_class!r} for symbol {symbol!r}; "
                    "expected 'equity' or 'crypto'"
                )
            ticker = polygon_ticker(symbol, asset_class)
            adjusted = asset_class == "equity"  # split-adjusted equities; crypto N/A

            for tf in timeframes:
                key_label = f"{symbol}:{tf}"
                try:
                    aggs = _fetch_aggs(
                        client,
                        ticker=ticker,
                        timeframe=tf,
                        start_utc=start_utc,
                        end_utc=end_utc,
                        adjusted=adjusted,
                        pause=pause,
                        window_days=window_days,
                    )
                    df = _aggs_to_frame(
                        aggs,
                        symbol=symbol,
                        asset_class=asset_class,
                        timeframe=tf,
                        adjusted=adjusted,
                        end_utc=end_utc,
                    )
                    results[key_label] = int(_schema.upsert_bars(con, df)) if not df.empty else 0
                    print(f"[polygon_ingest] {key_label}: {results[key_label]} rows")
                except Exception as exc:  # noqa: BLE001 - report per-target, keep going
                    print(f"[polygon_ingest] ERROR {key_label}: {exc!r}")
                    results[key_label] = -1
    finally:
        try:
            con.close()
        except Exception:  # pragma: no cover - best-effort close
            pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_symbols(raw: str | None) -> dict[str, str] | None:
    """Parse a comma-separated symbol list into a ``symbol -> asset_class`` map.

    A symbol containing ``/`` (e.g. ``BTC/USD``) is crypto, otherwise equity.
    Returns ``None`` to use :data:`DEFAULT_SYMBOLS`.
    """
    if not raw:
        return None
    out: dict[str, str] = {}
    for token in raw.split(","):
        sym = token.strip()
        if not sym:
            continue
        out[sym] = "crypto" if "/" in sym else "equity"
    return out or None


def _format_summary(results: Mapping[str, int]) -> str:
    """Render a simple aligned summary table of the ingest results."""
    if not results:
        return "(no results)"
    width = max(len(k) for k in results)
    lines = [f"{'TARGET'.ljust(width)}  ROWS", f"{'-' * width}  ----"]
    total = 0
    for label, rows in results.items():
        shown = "ERROR" if rows < 0 else str(rows)
        if rows > 0:
            total += rows
        lines.append(f"{label.ljust(width)}  {shown}")
    lines.append(f"{'-' * width}  ----")
    lines.append(f"{'TOTAL'.ljust(width)}  {total}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """argparse CLI entry point: pull bars and print a summary table."""
    parser = argparse.ArgumentParser(
        prog="polygon_ingest",
        description=(
            "Ingest Polygon (consolidated/SIP) historical 2m/5m bars, "
            "point-in-time, split-adjusted equities, into the TradeForge DuckDB."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="lookback in days (start = now - days); overrides the 2-year default",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="comma list, e.g. 'SPY,QQQ,BTC/USD,ETH/USD' "
        "(symbols with '/' inferred as crypto); default is the standard universe",
    )
    parser.add_argument("--start", type=str, default=None, help="window start (ISO 8601)")
    parser.add_argument("--end", type=str, default=None, help="window end (ISO 8601); default now")
    parser.add_argument(
        "--timeframes",
        type=str,
        default="2m,5m",
        help="comma list of timeframes (default '2m,5m')",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=_schema.DEFAULT_DB_PATH,
        help="DuckDB path (default data.schema.DEFAULT_DB_PATH)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="SDK retries (honors 429 Retry-After; raise for rate-limited keys)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="seconds to pause before each API call (throttle for free/Starter tiers, e.g. 15)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="split the pull into N-day windows (one call each) to stay under per-minute caps",
    )
    args = parser.parse_args(argv)

    timeframes = tuple(t.strip() for t in args.timeframes.split(",") if t.strip())

    results = run(
        symbols=_parse_symbols(args.symbols),
        timeframes=timeframes,
        start=args.start,
        end=args.end,
        db_path=args.db,
        days=args.days,
        retries=args.retries,
        pause=args.sleep,
        window_days=args.window_days,
    )

    print(_format_summary(results))
    return 1 if any(v < 0 for v in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
