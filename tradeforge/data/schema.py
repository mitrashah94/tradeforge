"""DuckDB schema + upsert helpers for the TradeForge market store.

Storage conventions (must match the shared contract exactly):

- ``bars.ts_utc`` is the bar OPEN time, tz-naive, in UTC. The bar's close
  time is ``ts_utc + timeframe`` (see :func:`bar_close_ts`). Anything that
  reasons about "is this bar complete as of time T" must compare against the
  CLOSE time, not the open time.
- Levels are point-in-time: ``build_levels`` (in ``data.levels``) uses only
  completed prior sessions to set PDH/PDL/ATR, never the current session's
  own RTH. This module only provides the persistence primitives.

Tables:
  bars(symbol, asset_class, timeframe, ts_utc, open, high, low, close,
       volume, trade_count, vwap, adjusted)  PK (symbol, timeframe, ts_utc)
  levels(symbol, asset_class, session_date, session_type, pdh, pdl, pmh, pml,
         ntz_low, ntz_high, ntz_valid, atr14, is_weekend)  PK (symbol, session_date)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import duckdb
import pandas as pd

DEFAULT_DB_PATH = "data/duckdb/market.duckdb"

# Canonical column order for the bars table (used for INSERT ... SELECT).
_BARS_COLUMNS = [
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
]

_LEVELS_COLUMNS = [
    "symbol",
    "asset_class",
    "session_date",
    "session_type",
    "pdh",
    "pdl",
    "pmh",
    "pml",
    "ntz_low",
    "ntz_high",
    "ntz_valid",
    "atr14",
    "is_weekend",
]

_TIMEFRAME_MINUTES = {"2m": 2, "5m": 5}


def bar_close_ts(ts_utc: datetime, timeframe: str) -> datetime:
    """Return the bar CLOSE time given its OPEN time and timeframe.

    ``ts_utc`` is the bar open; the close is open + timeframe (2 or 5 minutes).
    """
    try:
        minutes = _TIMEFRAME_MINUTES[timeframe]
    except KeyError as exc:
        raise ValueError(f"unknown timeframe {timeframe!r}") from exc
    return ts_utc + timedelta(minutes=minutes)


def connect(path: str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (creating the parent directory if needed) a DuckDB connection.

    ``path`` may be ``":memory:"`` for an in-memory database (used in tests).
    """
    if path != ":memory:":
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return duckdb.connect(path)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the bars and levels tables if they do not already exist."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bars (
            symbol      VARCHAR,
            asset_class VARCHAR,
            timeframe   VARCHAR,
            ts_utc      TIMESTAMP,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      DOUBLE,
            trade_count BIGINT,
            vwap        DOUBLE,
            adjusted    BOOLEAN,
            PRIMARY KEY (symbol, timeframe, ts_utc)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS levels (
            symbol       VARCHAR,
            asset_class  VARCHAR,
            session_date DATE,
            session_type VARCHAR,
            pdh          DOUBLE,
            pdl          DOUBLE,
            pmh          DOUBLE,
            pml          DOUBLE,
            ntz_low      DOUBLE,
            ntz_high     DOUBLE,
            ntz_valid    BOOLEAN,
            atr14        DOUBLE,
            is_weekend   BOOLEAN,
            PRIMARY KEY (symbol, session_date)
        )
        """
    )


def upsert_bars(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert-or-replace bars by primary key (symbol, timeframe, ts_utc).

    Robust to missing optional columns: ``trade_count`` and ``vwap`` default
    to NULL, ``adjusted`` defaults to True. Returns the number of rows written.
    """
    if df is None or len(df) == 0:
        return 0

    work = df.copy()

    # Fill optional columns if absent.
    if "trade_count" not in work.columns:
        work["trade_count"] = pd.NA
    if "vwap" not in work.columns:
        work["vwap"] = pd.NA
    if "adjusted" not in work.columns:
        work["adjusted"] = True

    missing = [c for c in _BARS_COLUMNS if c not in work.columns]
    if missing:
        raise ValueError(f"bars df missing required columns: {missing}")

    work = work[_BARS_COLUMNS]

    con.register("_bars_src", work)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO bars
                (symbol, asset_class, timeframe, ts_utc, open, high, low,
                 close, volume, trade_count, vwap, adjusted)
            SELECT
                CAST(symbol AS VARCHAR),
                CAST(asset_class AS VARCHAR),
                CAST(timeframe AS VARCHAR),
                CAST(ts_utc AS TIMESTAMP),
                CAST(open AS DOUBLE),
                CAST(high AS DOUBLE),
                CAST(low AS DOUBLE),
                CAST(close AS DOUBLE),
                CAST(volume AS DOUBLE),
                CAST(trade_count AS BIGINT),
                CAST(vwap AS DOUBLE),
                CAST(adjusted AS BOOLEAN)
            FROM _bars_src
            """
        )
    finally:
        con.unregister("_bars_src")
    return len(work)


def upsert_levels(con: duckdb.DuckDBPyConnection, rows) -> int:
    """Insert-or-replace level rows by primary key (symbol, session_date).

    ``rows`` may be a list of dicts or a DataFrame with the levels columns.
    Returns the number of rows written.
    """
    if rows is None:
        return 0

    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        rows = list(rows)
        if not rows:
            return 0
        df = pd.DataFrame(rows)

    if len(df) == 0:
        return 0

    missing = [c for c in _LEVELS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"levels rows missing required columns: {missing}")

    df = df[_LEVELS_COLUMNS]

    con.register("_levels_src", df)
    try:
        con.execute(
            """
            INSERT OR REPLACE INTO levels
                (symbol, asset_class, session_date, session_type, pdh, pdl,
                 pmh, pml, ntz_low, ntz_high, ntz_valid, atr14, is_weekend)
            SELECT
                CAST(symbol AS VARCHAR),
                CAST(asset_class AS VARCHAR),
                CAST(session_date AS DATE),
                CAST(session_type AS VARCHAR),
                CAST(pdh AS DOUBLE),
                CAST(pdl AS DOUBLE),
                CAST(pmh AS DOUBLE),
                CAST(pml AS DOUBLE),
                CAST(ntz_low AS DOUBLE),
                CAST(ntz_high AS DOUBLE),
                CAST(ntz_valid AS BOOLEAN),
                CAST(atr14 AS DOUBLE),
                CAST(is_weekend AS BOOLEAN)
            FROM _levels_src
            """
        )
    finally:
        con.unregister("_levels_src")
    return len(df)
