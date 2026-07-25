"""Alpaca historical bar ingestion for TradeForge (MASTER_PLAN §5, P1).

Pulls point-in-time, corporate-action-adjusted intraday bars from the Alpaca
historical data API (via ``alpaca-py``) into the DuckDB ``bars`` table.

Scope (data layer only — **no trading / strategy logic** lives here):

* Timeframes: ``2m`` and ``5m``.
* Universe (default): equities ``SPY``, ``QQQ`` and crypto ``BTC/USD``, ``ETH/USD``.
* Lookback: 2 years back by default (overridable via ``days`` / ``start`` / ``end``).

Two correctness invariants this module is responsible for:

1. **Point-in-time.** We never store the in-progress (not-yet-closed) bar.
   ``end`` defaults to "now" (UTC); after fetching we DROP any bar whose CLOSE
   time (``ts_open + timeframe``) is strictly greater than ``end``. The most
   recent bar a live system can legitimately know about is the last one whose
   close time has already passed — storing a forming bar would leak the future
   into a backtest and create train/serve skew.
2. **Corporate-action adjustment (equities only).** We request
   ``Adjustment.ALL`` (split + dividend adjusted) so price levels stay
   continuous across splits/ex-dividend dates, and we tag those rows
   ``adjusted=True``. Crypto has no splits/dividends, so it is requested
   unadjusted and tagged ``adjusted=False``.

Timestamps from Alpaca are tz-aware UTC at the bar OPEN. We strip the tz to
store tz-naive UTC, matching the ``bars`` table contract in ``data.schema``.

This is the *only* module in the data layer that hard-depends on ``alpaca-py``;
sibling modules never import this file, so its broker dependency cannot break
offline tests of ``data.schema`` / ``data.sessions`` / ``data.levels``.

Credentials are read from the environment (``APCA_API_KEY_ID`` /
``APCA_API_SECRET_KEY``); see ``.env.example``.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import Mapping

import pandas as pd

# alpaca-py is a hard dependency of THIS file only (kept at module top per the
# build contract). No other module imports this file, so an absent/uninstalled
# alpaca package never affects offline tests of the rest of the data layer.
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from data import schema as _schema

__all__ = [
    "TIMEFRAME_MINUTES",
    "DEFAULT_SYMBOLS",
    "DEFAULT_LOOKBACK_DAYS",
    "alpaca_timeframe",
    "run",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported timeframe labels -> their length in minutes.
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

def alpaca_timeframe(tf: str) -> TimeFrame:
    """Map a TradeForge timeframe label (``"2m"`` / ``"5m"``) to an Alpaca ``TimeFrame``.

    Raises:
        ValueError: if ``tf`` is not a supported label.
    """
    try:
        minutes = TIMEFRAME_MINUTES[tf]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unsupported timeframe {tf!r}; expected one of "
            f"{sorted(TIMEFRAME_MINUTES)}"
        ) from exc
    return TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute)


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
        # Accept a trailing 'Z' (Python's fromisoformat handles it from 3.11).
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        dt = value
    else:  # pragma: no cover - defensive
        raise TypeError(f"unsupported datetime value: {value!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_feed(feed: DataFeed | str | None) -> DataFeed:
    """Resolve the equities data feed, defaulting to the free IEX feed."""
    if feed is None:
        return DataFeed.IEX
    if isinstance(feed, DataFeed):
        return feed
    # Accept plain strings like "iex" / "sip" from the CLI.
    return DataFeed(str(feed).lower())


def _require_credentials() -> tuple[str, str]:
    """Read Alpaca credentials from the environment or fail loudly."""
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Alpaca credentials missing. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY in your environment (see .env.example)."
        )
    return key, secret


def _barset_to_frame(
    bars,
    *,
    symbol: str,
    asset_class: str,
    timeframe: str,
    adjusted: bool,
    end_utc: datetime,
) -> pd.DataFrame:
    """Build a bars DataFrame from an Alpaca ``BarSet`` for one (symbol, tf).

    Handles empty responses, normalises timestamps to tz-naive UTC, drops the
    in-progress bar, and returns columns in the ``data.schema.upsert_bars``
    contract order.
    """
    # alpaca-py BarSet behaves like a mapping symbol -> list[Bar]. The crypto
    # API may key by the exact requested symbol (e.g. "BTC/USD"); guard for both
    # the requested key and the (rare) stripped variant.
    raw = []
    data = getattr(bars, "data", None)
    if data:
        raw = data.get(symbol) or data.get(symbol.replace("/", "")) or []

    if not raw:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))

    rows = []
    for bar in raw:
        # bar.timestamp is tz-aware UTC at the bar OPEN.
        ts_open = bar.timestamp
        if ts_open.tzinfo is None:
            ts_open = ts_open.replace(tzinfo=timezone.utc)
        else:
            ts_open = ts_open.astimezone(timezone.utc)
        rows.append(
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                # store tz-naive UTC (strip tz after normalising)
                "ts_utc": ts_open.replace(tzinfo=None),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                # trade_count / vwap are absent for some feeds (e.g. IEX) -> None
                "trade_count": getattr(bar, "trade_count", None),
                "vwap": getattr(bar, "vwap", None),
                "adjusted": adjusted,
                # transient: bar CLOSE time, used only to drop the forming bar
                "_close_utc": ts_open + timedelta(minutes=TIMEFRAME_MINUTES[timeframe]),
            }
        )

    df = pd.DataFrame(rows)

    # ---- POINT-IN-TIME: drop the in-progress (not-yet-closed) bar ----------
    # Any bar whose CLOSE time is strictly after `end` (default "now") has not
    # finished forming and must not be persisted. end_utc is tz-aware; the
    # _close_utc column is tz-aware UTC here.
    df = df[df["_close_utc"] <= end_utc].copy()
    df = df.drop(columns=["_close_utc"])

    if df.empty:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))

    # Deterministic order + exact contract columns.
    df = df.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    return df[list(_BAR_COLUMNS)]


def _fetch_equity(
    client: StockHistoricalDataClient,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    feed: DataFeed,
):
    """Fetch split+dividend-adjusted equity bars (alpaca-py auto-paginates)."""
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=alpaca_timeframe(timeframe),
        start=start,
        end=end,
        adjustment=Adjustment.ALL,  # split + dividend adjusted (§5)
        feed=feed,
    )
    return client.get_stock_bars(request)


def _fetch_crypto(
    client: CryptoHistoricalDataClient,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
):
    """Fetch crypto bars (24/7, unadjusted; alpaca-py auto-paginates)."""
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=alpaca_timeframe(timeframe),
        start=start,
        end=end,
        # no adjustment for crypto (no splits/dividends)
    )
    return client.get_crypto_bars(request)


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
    feed: DataFeed | str | None = None,
) -> dict[str, int]:
    """Ingest Alpaca historical bars into the DuckDB ``bars`` table.

    Args:
        symbols: mapping of ``symbol -> asset_class`` (``'equity'`` | ``'crypto'``).
            Defaults to :data:`DEFAULT_SYMBOLS` (SPY, QQQ, BTC/USD, ETH/USD).
        timeframes: iterable of timeframe labels; only ``"2m"`` / ``"5m"`` are
            supported.
        start: window start (datetime or ISO string). If omitted, derived from
            ``days`` or the 2-year default. Naive values are treated as UTC.
        end: window end (datetime or ISO string). Defaults to "now" (UTC) and
            anchors the point-in-time drop of the in-progress bar.
        db_path: DuckDB path; defaults to :data:`data.schema.DEFAULT_DB_PATH`.
        days: if given, ``start = now - days`` (overrides the 2-year default and
            any precedence the default ``start`` would have). Explicit ``start``
            still wins over ``days``.
        feed: equities data feed; defaults to :class:`DataFeed.IEX` (free tier).
            Ignored for crypto.

    Returns:
        Mapping of ``f"{symbol}:{timeframe}" -> rows_written`` (rows upserted).

    Raises:
        RuntimeError: if Alpaca credentials are missing from the environment.
        ValueError: on an unsupported timeframe.
    """
    if symbols is None:
        symbols = dict(DEFAULT_SYMBOLS)

    # Validate timeframes up front for a clear early failure.
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

    # Precedence for the start of the window:
    #   1. explicit `start`        (highest)
    #   2. `days`   -> now - days
    #   3. default  -> now - 2 years
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

    resolved_feed = _resolve_feed(feed)
    key, secret = _require_credentials()

    # Lazily construct only the clients we actually need.
    stock_client: StockHistoricalDataClient | None = None
    crypto_client: CryptoHistoricalDataClient | None = None

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

            for tf in timeframes:
                key_label = f"{symbol}:{tf}"
                try:
                    if asset_class == "equity":
                        if stock_client is None:
                            stock_client = StockHistoricalDataClient(key, secret)
                        bars = _fetch_equity(
                            stock_client,
                            symbol=symbol,
                            timeframe=tf,
                            start=start_utc,
                            end=end_utc,
                            feed=resolved_feed,
                        )
                        adjusted = True  # Adjustment.ALL (split + dividend)
                    else:  # crypto
                        if crypto_client is None:
                            # Public crypto data; keys are still passed for
                            # rate-limit attribution / consistency.
                            crypto_client = CryptoHistoricalDataClient(key, secret)
                        bars = _fetch_crypto(
                            crypto_client,
                            symbol=symbol,
                            timeframe=tf,
                            start=start_utc,
                            end=end_utc,
                        )
                        adjusted = False  # crypto: no splits/dividends

                    df = _barset_to_frame(
                        bars,
                        symbol=symbol,
                        asset_class=asset_class,
                        timeframe=tf,
                        adjusted=adjusted,
                        end_utc=end_utc,
                    )

                    if df.empty:
                        results[key_label] = 0
                        continue

                    results[key_label] = int(_schema.upsert_bars(con, df))
                except Exception as exc:  # noqa: BLE001 - report per-symbol, keep going
                    # One bad symbol/timeframe (e.g. transient API error) must
                    # not abort the whole ingest. Record -1 as a failure marker
                    # and surface it; the rest of the universe still ingests.
                    print(f"[alpaca_ingest] ERROR {key_label}: {exc!r}")
                    results[key_label] = -1
    finally:
        # DuckDB connections must be closed to release the file lock.
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

    Asset class is inferred: a symbol containing ``/`` (e.g. ``BTC/USD``) is
    crypto, otherwise equity. Returns ``None`` to use :data:`DEFAULT_SYMBOLS`.
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
        prog="alpaca_ingest",
        description=(
            "Ingest Alpaca historical 2m/5m bars (point-in-time, "
            "corporate-action-adjusted equities) into the TradeForge DuckDB."
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
        "--feed",
        type=str,
        default="iex",
        help="equities data feed (default 'iex'; e.g. 'sip' with a paid plan)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=_schema.DEFAULT_DB_PATH,
        help="DuckDB path (default data.schema.DEFAULT_DB_PATH)",
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
        feed=args.feed,
    )

    print(_format_summary(results))
    # Non-zero exit if any target failed, so cron/CI can detect partial failure.
    return 1 if any(v < 0 for v in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
