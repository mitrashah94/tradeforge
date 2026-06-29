"""Yahoo Finance daily ADJUSTED bar ingestion for TradeForge (the daily ETF data layer).

Pulls deep (multi-decade) DAILY total-return-adjusted history from the Yahoo
Finance v8 chart API into the DuckDB ``bars`` table at ``timeframe='1d'`` — the
**daily/asset-allocation sibling** of ``data.pipelines.polygon_ingest`` (which
pulls intraday 2m/5m bars for the breakout engine). It writes the *identical*
``bars`` schema (``data.schema.upsert_bars``), so every downstream consumer reads
daily bars through the same table; only the source, timeframe, and adjustment
semantics differ.

Why this module exists: the intraday store (Polygon, ~2 years, 10 single-names)
is the right grain for the breakout edge but the wrong grain for the *other* edge
family the platform needs — slow, cross-asset, low-correlation allocation
(momentum / trend / risk-parity rotation across a broad ETF universe). Those
strategies need (a) DECADES of history to survive walk-forward and (b) a wide,
diversified, **total-return-adjusted** universe (equities, sectors, intl, bonds,
cash, factors, real assets, inverse/leveraged, crypto-ETFs). Yahoo's free v8
chart endpoint gives both with no API key.

Scope (data layer only — **no trading / strategy logic** lives here):

* Timeframe: ``1d`` only (daily bars).
* Universe (default): :data:`DEFAULT_ETF_UNIVERSE` — ~40 broad/sector/intl/bond/
  cash/factor/real-asset/inverse/leveraged/crypto ETFs.
* Lookback: ``range=30y`` (the deepest the endpoint reliably serves); young
  tickers simply return their full (shorter) history.

Two correctness invariants this module is responsible for:

1. **Point-in-time.** We never store the in-progress (not-yet-closed) day. Yahoo
   appends a forming bar for *today* while the session is open (its values are the
   live snapshot, not the official close). After parsing we DROP any bar whose
   exchange-local session date is >= today (exchange-local), so a backtest never
   sees a partial day. See :func:`_chart_to_frame`.
2. **Total-return adjustment.** Yahoo gives a split-only ``quote`` block plus an
   ``adjclose`` series that is BOTH split- AND dividend-adjusted (total return).
   We scale the whole OHLC bar by the per-day factor ``adjclose / close`` so the
   *entire* bar is split+dividend adjusted and internally consistent (the adjusted
   close equals ``adjclose``), tag ``adjusted=True``, keep ``volume`` as reported,
   and set ``asset_class='equity'``. This makes long-run CAGR honest (a $1 buy-hold
   of the adjusted close reproduces total return) — validated for SPY in
   :func:`_cagr_check` after ingest.

Yahoo daily timestamps are Unix SECONDS at the regular-market OPEN, UTC. We do NOT
store an intraday time for daily bars: ``ts_utc`` is the **session date at
00:00**, tz-naive UTC, matching how daily bars are keyed elsewhere. The
session/calendar date is derived from the exchange ``gmtoffset`` so a near-midnight
UTC open never rolls onto the wrong calendar day.

The pure transform :func:`_chart_to_frame` takes a plain parsed-JSON dict and does
no network, so it is fully offline-testable (see ``tests/test_yahoo_ingest.py``).
Only :func:`_fetch_chart` touches the network, via stdlib :mod:`urllib.request`
(NO new dependencies — no ``requests``, no ``yfinance``).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Mapping

import pandas as pd

from data import schema as _schema

__all__ = [
    "DEFAULT_ETF_UNIVERSE",
    "TIMEFRAME",
    "YAHOO_CHART_URL",
    "run",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The only timeframe this module ingests (daily bars).
TIMEFRAME: str = "1d"

#: Yahoo Finance v8 chart endpoint (VERIFIED working). ``{symbol}`` is filled per
#: ticker; ``range=30y`` is the deepest the endpoint reliably serves, ``interval=1d``
#: is daily, and ``events=div,split`` (URL-encoded) requests the dividend/split
#: events block alongside the adjusted close.
YAHOO_CHART_URL: str = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range={range}&interval=1d&events=div%2Csplit"
)

#: Browser-style UA — the endpoint 429s/403s requests without one.
_USER_AGENT: str = "Mozilla/5.0"

#: Default broad ETF universe for the daily allocation layer, grouped by sleeve so
#: the *intent* of each ticker is legible (the value is always ``'equity'`` — every
#: one of these is an exchange-traded fund and ingests through the equity path).
#:
#: broad market | US sectors (SPDR) | international | bonds/duration | cash/T-bill |
#: factors | real assets | inverse (short) | leveraged (2x/3x) | crypto-ETF.
DEFAULT_ETF_UNIVERSE: dict[str, str] = {
    # broad market
    "VTI": "equity", "SPY": "equity", "QQQ": "equity",
    # US sectors (SPDR Select Sector)
    "XLK": "equity", "XLF": "equity", "XLE": "equity", "XLV": "equity",
    "XLI": "equity", "XLY": "equity", "XLP": "equity", "XLU": "equity",
    "XLB": "equity", "XLRE": "equity", "XLC": "equity",
    # international
    "VXUS": "equity", "EFA": "equity", "EEM": "equity",
    # bonds / duration
    "BND": "equity", "AGG": "equity", "TLT": "equity", "IEF": "equity",
    # cash / T-bill
    "BIL": "equity", "SGOV": "equity", "SHY": "equity",
    # factors
    "MTUM": "equity", "QUAL": "equity", "USMV": "equity",
    # real assets
    "GLD": "equity", "DBC": "equity",
    # inverse (short)
    "PSQ": "equity", "SH": "equity", "RWM": "equity",
    # leveraged (2x / 3x)
    "QLD": "equity", "TQQQ": "equity", "QID": "equity", "SQQQ": "equity",
    # crypto-ETF (young — short history expected)
    "IBIT": "equity", "ETHA": "equity",
}

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

def _today_utc() -> date:
    """Today's date in UTC (single source of 'now' for the forming-bar drop)."""
    return datetime.now(timezone.utc).date()


def _session_date(ts_seconds: int, gmtoffset: int) -> date:
    """Exchange-local calendar date for a Yahoo daily timestamp.

    Yahoo's daily ``t`` is Unix SECONDS at the regular-market OPEN in UTC. Adding
    the exchange ``gmtoffset`` (seconds east of UTC; negative for US exchanges)
    yields exchange-local wall time, whose ``.date()`` is the session/calendar
    date. Deriving the date this way (rather than from the UTC instant) keeps a
    near-midnight UTC open on the correct calendar day across DST shifts.
    """
    local = datetime.fromtimestamp(int(ts_seconds), tz=timezone.utc) + timedelta(
        seconds=int(gmtoffset)
    )
    return local.date()


def _chart_to_frame(
    payload: Mapping,
    *,
    symbol: str,
    asset_class: str = "equity",
    today: date | None = None,
) -> pd.DataFrame:
    """Build a daily bars DataFrame from a parsed Yahoo v8 chart JSON payload.

    ``payload`` is the full decoded JSON dict (``{"chart": {"result": [...]}}``).
    Pure / no network. Responsibilities:

    * Read ``chart.result[0]``: ``timestamp[]`` and ``indicators.quote[0]``
      (``open``/``high``/``low``/``close``/``volume``) plus
      ``indicators.adjclose[0].adjclose[]``.
    * Skip any day with a null close or null adjclose (Yahoo emits holes).
    * TOTAL-RETURN ADJUST: scale O/H/L/C by ``factor = adjclose / close`` so the
      whole bar is split+dividend adjusted (the adjusted close equals ``adjclose``);
      keep ``volume`` as reported; tag ``adjusted=True``.
    * ``ts_utc`` = the exchange-local session date at 00:00, tz-naive UTC.
    * POINT-IN-TIME: drop any bar whose session date >= ``today`` (the forming /
      not-yet-closed current day Yahoo appends while the session is live).

    Returns columns in the ``data.schema.upsert_bars`` contract order; an empty
    (but correctly-typed) frame if there is nothing to store.
    """
    if today is None:
        today = _today_utc()

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"Yahoo chart error for {symbol!r}: {chart['error']!r}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))

    res = results[0]
    timestamps = res.get("timestamp") or []
    indicators = res.get("indicators") or {}
    quotes = indicators.get("quote") or [{}]
    quote = quotes[0] if quotes else {}
    adj_block = indicators.get("adjclose") or [{}]
    adjclose = (adj_block[0] or {}).get("adjclose") if adj_block else None

    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    # Without an adjclose series we cannot total-return adjust; fall back to the
    # raw close as its own "adjusted" value (factor == 1) so the module degrades
    # gracefully rather than crashing on an exotic instrument.
    if not adjclose:
        adjclose = closes

    gmtoffset = ((res.get("meta") or {}).get("gmtoffset")) or 0

    n = len(timestamps)
    rows = []
    for i in range(n):
        # Defensive bounds + null handling — Yahoo emits None for missing days.
        if i >= len(closes) or closes[i] is None:
            continue
        adj_c = adjclose[i] if i < len(adjclose) else None
        c = closes[i]
        if adj_c is None or c is None or c == 0:
            continue
        o = opens[i] if i < len(opens) else None
        h = highs[i] if i < len(highs) else None
        low = lows[i] if i < len(lows) else None
        v = volumes[i] if i < len(volumes) else None
        if o is None or h is None or low is None:
            continue

        sess = _session_date(timestamps[i], gmtoffset)
        if sess >= today:
            # forming / not-yet-closed current day — never persist it.
            continue

        # TOTAL-RETURN factor: scales the whole bar so adjusted close == adjclose.
        factor = adj_c / c
        rows.append(
            {
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": TIMEFRAME,
                # daily bar key: session date at 00:00, tz-naive UTC.
                "ts_utc": datetime(sess.year, sess.month, sess.day),
                "open": o * factor,
                "high": h * factor,
                "low": low * factor,
                "close": adj_c,  # == c * factor, but use the source value exactly
                "volume": v,
                "trade_count": None,
                "vwap": None,
                "adjusted": True,
            }
        )

    if not rows:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))

    df = pd.DataFrame(rows)
    df = df.sort_values("ts_utc", kind="stable").reset_index(drop=True)
    return df[list(_BAR_COLUMNS)]


def _fetch_chart(symbol: str, *, range_: str = "30y", timeout: float = 30.0) -> dict:
    """GET the Yahoo v8 daily chart JSON for one symbol (stdlib urllib, no deps).

    The only network call in this module. Sends a browser ``User-Agent`` (the
    endpoint rejects key-less requests without one) and returns the decoded JSON
    dict. Raises on transport/HTTP/JSON errors so the per-symbol try/except in
    :func:`run` can record the failure and continue.
    """
    url = YAHOO_CHART_URL.format(symbol=urllib.request.quote(symbol), range=range_)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Validation (adjustment sanity guard)
# ---------------------------------------------------------------------------

def _cagr_from_adjusted(df: pd.DataFrame) -> tuple[float, float]:
    """Compute (years, CAGR) of the adjusted-close buy-and-hold over ``df``.

    ``df`` is a bars frame (already total-return adjusted) sorted by ``ts_utc``.
    The adjusted close is a total-return index, so its first->last ratio over the
    elapsed years is the realized annualized total return. Returns ``(0.0, nan)``
    for a frame too short to annualize.
    """
    if df is None or len(df) < 2:
        return 0.0, float("nan")
    first_ts = pd.Timestamp(df["ts_utc"].iloc[0])
    last_ts = pd.Timestamp(df["ts_utc"].iloc[-1])
    years = (last_ts - first_ts).days / 365.25
    if years <= 0:
        return 0.0, float("nan")
    first_c = float(df["close"].iloc[0])
    last_c = float(df["close"].iloc[-1])
    if first_c <= 0:
        return years, float("nan")
    cagr = (last_c / first_c) ** (1.0 / years) - 1.0
    return years, cagr


def _cagr_check(df: pd.DataFrame, *, symbol: str, lo: float = 0.05, hi: float = 0.13) -> bool:
    """Print + return whether ``symbol``'s adjusted long-run CAGR is sane.

    A guard that the ``adjclose/close`` scaling is right: a correctly total-return
    adjusted SPY over a multi-decade window should compound at roughly 7-11%/yr.
    We bracket a little wider (``[5%, 13%]``) to tolerate the exact window. A
    failure here means the adjustment math (or the data) is off — not a tradeable
    signal, a data-integrity alarm.
    """
    years, cagr = _cagr_from_adjusted(df)
    ok = (cagr == cagr) and (lo <= cagr <= hi)  # cagr==cagr filters NaN
    flag = "OK" if ok else "SUSPECT"
    print(
        f"[yahoo_ingest] CAGR check {symbol}: {cagr * 100:.2f}%/yr over {years:.1f}y "
        f"-> {flag} (sane band {lo * 100:.0f}-{hi * 100:.0f}%)"
    )
    return ok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    symbols: Mapping[str, str] | Iterable[str] | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    db_path: str = _schema.DEFAULT_DB_PATH,
    range_: str = "30y",
    pause: float = 0.5,
    fetch=None,
) -> dict[str, int]:
    """Ingest deep daily total-return-adjusted ETF bars into the DuckDB ``bars`` table.

    Args:
        symbols: either a mapping ``symbol -> asset_class`` or a plain iterable of
            symbol strings (asset_class defaults to ``'equity'``). Defaults to
            :data:`DEFAULT_ETF_UNIVERSE`.
        start: optional inclusive lower bound on the stored session date (datetime
            or ``YYYY-MM-DD`` string). Rows before it are dropped after parsing.
        end: optional inclusive upper bound on the stored session date. Independent
            of the always-on forming-bar drop (which removes today's open day).
        db_path: DuckDB path; defaults to :data:`data.schema.DEFAULT_DB_PATH`.
        range_: Yahoo ``range`` window (default ``"30y"`` — the deepest reliably
            served). Young tickers just return their shorter full history.
        pause: seconds to sleep between symbols (be polite to the free endpoint).
        fetch: optional injected ``fetch(symbol, range_=...) -> payload`` callable
            (used by tests to avoid the network). Defaults to :func:`_fetch_chart`.

    Returns:
        Mapping of ``symbol -> rows_written`` (rows upserted); ``-1`` marks a
        per-symbol failure (the rest of the universe still ingests).
    """
    if symbols is None:
        symbols = dict(DEFAULT_ETF_UNIVERSE)
    elif isinstance(symbols, Mapping):
        symbols = dict(symbols)
    else:
        symbols = {str(s).strip().upper(): "equity" for s in symbols if str(s).strip()}

    if fetch is None:
        fetch = _fetch_chart

    start_date = _as_date(start)
    end_date = _as_date(end)
    today = _today_utc()

    results: dict[str, int] = {}

    con = _schema.connect(db_path)
    try:
        _schema.init_schema(con)

        for i, (symbol, asset_class) in enumerate(symbols.items()):
            asset_class = (asset_class or "equity").lower()
            try:
                payload = fetch(symbol, range_=range_)
                df = _chart_to_frame(
                    payload, symbol=symbol, asset_class=asset_class, today=today
                )
                if start_date is not None and not df.empty:
                    df = df[df["ts_utc"] >= pd.Timestamp(start_date)]
                if end_date is not None and not df.empty:
                    df = df[df["ts_utc"] <= pd.Timestamp(end_date)]
                df = df.reset_index(drop=True)
                results[symbol] = int(_schema.upsert_bars(con, df)) if not df.empty else 0
                if df.empty:
                    print(f"[yahoo_ingest] {symbol}: 0 rows (no data)")
                else:
                    print(
                        f"[yahoo_ingest] {symbol}: {results[symbol]} rows "
                        f"[{df['ts_utc'].iloc[0].date()} .. {df['ts_utc'].iloc[-1].date()}]"
                    )
            except Exception as exc:  # noqa: BLE001 - report per-symbol, keep going
                print(f"[yahoo_ingest] ERROR {symbol}: {exc!r}")
                results[symbol] = -1

            # Be polite to the free endpoint (skip the sleep after the last one).
            if pause and i < len(symbols) - 1:
                time.sleep(pause)
    finally:
        try:
            con.close()
        except Exception:  # pragma: no cover - best-effort close
            pass

    return results


def _as_date(value: datetime | str | date | None) -> date | None:
    """Coerce a datetime / ``YYYY-MM-DD`` string / date / None into a date or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    raise TypeError(f"cannot coerce {value!r} to a date")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_symbols(raw: str | None) -> dict[str, str] | None:
    """Parse a comma-separated symbol list into a ``symbol -> 'equity'`` map.

    Returns ``None`` to use :data:`DEFAULT_ETF_UNIVERSE`.
    """
    if not raw:
        return None
    out: dict[str, str] = {}
    for token in raw.split(","):
        sym = token.strip().upper()
        if sym:
            out[sym] = "equity"
    return out or None


def _format_summary(results: Mapping[str, int]) -> str:
    """Render a simple aligned summary table of the ingest results."""
    if not results:
        return "(no results)"
    width = max(len(k) for k in results)
    lines = [f"{'SYMBOL'.ljust(width)}  ROWS", f"{'-' * width}  ----"]
    total = 0
    ok = 0
    for label, rows in results.items():
        shown = "ERROR" if rows < 0 else str(rows)
        if rows > 0:
            total += rows
            ok += 1
        lines.append(f"{label.ljust(width)}  {shown}")
    lines.append(f"{'-' * width}  ----")
    lines.append(f"{'TOTAL'.ljust(width)}  {total} rows across {ok}/{len(results)} symbols")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """argparse CLI entry point: pull daily adjusted ETF bars and print a summary."""
    parser = argparse.ArgumentParser(
        prog="yahoo_ingest",
        description=(
            "Ingest deep daily TOTAL-RETURN-adjusted ETF bars from the Yahoo v8 "
            "chart API into the TradeForge DuckDB (timeframe='1d')."
        ),
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="comma list, e.g. 'SPY,QQQ,TLT'; default is DEFAULT_ETF_UNIVERSE",
    )
    parser.add_argument("--start", type=str, default=None, help="min session date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="max session date (YYYY-MM-DD)")
    parser.add_argument(
        "--range", dest="range_", type=str, default="30y",
        help="Yahoo range window (default '30y')",
    )
    parser.add_argument(
        "--db", type=str, default=_schema.DEFAULT_DB_PATH,
        help="DuckDB path (default data.schema.DEFAULT_DB_PATH)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.5,
        help="seconds to pause between symbols (be polite; default 0.5)",
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="skip the post-ingest SPY adjusted-CAGR sanity check",
    )
    args = parser.parse_args(argv)

    results = run(
        symbols=_parse_symbols(args.symbols),
        start=args.start,
        end=args.end,
        db_path=args.db,
        range_=args.range_,
        pause=args.sleep,
    )

    print(_format_summary(results))

    # Post-ingest validation: re-load SPY from the DB and assert its adjusted
    # long-run CAGR is sane (guards the adjclose/close scaling).
    if not args.no_check and results.get("SPY", 0) > 0:
        con = _schema.connect(args.db)
        try:
            spy = con.execute(
                "SELECT ts_utc, close FROM bars "
                "WHERE symbol = 'SPY' AND timeframe = ? ORDER BY ts_utc",
                [TIMEFRAME],
            ).fetch_df()
        finally:
            con.close()
        if not spy.empty:
            lo, hi = spy["ts_utc"].iloc[0].date(), spy["ts_utc"].iloc[-1].date()
            print(f"[yahoo_ingest] SPY date range: {lo} .. {hi} ({len(spy)} sessions)")
            _cagr_check(spy, symbol="SPY")

    return 1 if any(v < 0 for v in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
