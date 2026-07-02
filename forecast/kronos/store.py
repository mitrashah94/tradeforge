"""forecast/kronos/store.py — read/write the kronos_forecasts table.

The seam between the SLOW loop (Kronos writes) and the deterministic engine
(reads). Point-in-time by construction: each row is keyed by ``session_date`` and
recomputed every premarket, so the engine only ever reads the forecast that was
knowable as of that session. Pure DuckDB; no torch.

Schema ``kronos_forecasts``::

    symbol, session_date, timeframe, horizon,
    exp_return, vol, downside_cvar, prob_up,
    n_paths, model_id
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

FORECAST_FIELDS = ("exp_return", "vol", "downside_cvar", "prob_up")


def init_forecast_schema(con) -> None:
    """Create ``kronos_forecasts`` if absent."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS kronos_forecasts (
            symbol        VARCHAR,
            session_date  DATE,
            timeframe     VARCHAR,
            horizon       INTEGER,
            exp_return    DOUBLE,
            vol           DOUBLE,
            downside_cvar DOUBLE,
            prob_up       DOUBLE,
            n_paths       INTEGER,
            model_id      VARCHAR,
            PRIMARY KEY (symbol, session_date, timeframe, horizon, model_id)
        )
        """
    )


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    return d


def write_forecast(
    con,
    symbol: str,
    session_date,
    forecast: dict,
    *,
    timeframe: str = "1d",
    horizon: int = 5,
    n_paths: int = 32,
    model_id: str = "kronos-mini",
) -> None:
    """Upsert one forecast row (``forecast`` carries the four distribution fields)."""
    init_forecast_schema(con)
    con.execute(
        "INSERT OR REPLACE INTO kronos_forecasts VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            symbol, _as_date(session_date), timeframe, int(horizon),
            _f(forecast.get("exp_return")), _f(forecast.get("vol")),
            _f(forecast.get("downside_cvar")), _f(forecast.get("prob_up")),
            int(n_paths), model_id,
        ],
    )


def write_forecasts(con, rows: Iterable[dict]) -> int:
    """Bulk-upsert forecast rows. Each row is a dict with ``symbol`` /
    ``session_date`` / the four fields / optional ``timeframe`` / ``horizon`` /
    ``n_paths`` / ``model_id``. Returns the count written."""
    n = 0
    for r in rows:
        write_forecast(
            con, r["symbol"], r["session_date"], r,
            timeframe=r.get("timeframe", "1d"), horizon=r.get("horizon", 5),
            n_paths=r.get("n_paths", 32), model_id=r.get("model_id", "kronos-mini"),
        )
        n += 1
    return n


def read_forecast(
    con, symbol: str, session_date, *,
    timeframe: str = "1d", horizon: int = 5, model_id: Optional[str] = None,
) -> Optional[dict]:
    """Read one symbol's forecast as of ``session_date`` (``None`` if absent)."""
    q = (
        "SELECT exp_return, vol, downside_cvar, prob_up, n_paths, model_id "
        "FROM kronos_forecasts WHERE symbol=? AND session_date=? AND timeframe=? "
        "AND horizon=?"
    )
    params = [symbol, _as_date(session_date), timeframe, int(horizon)]
    if model_id is not None:
        q += " AND model_id=?"
        params.append(model_id)
    try:
        row = con.execute(q, params).fetchone()
    except Exception:  # noqa: BLE001 — table absent
        return None
    if row is None:
        return None
    return {
        "exp_return": row[0], "vol": row[1], "downside_cvar": row[2],
        "prob_up": row[3], "n_paths": row[4], "model_id": row[5],
    }


def read_forecasts_asof(
    con, session_date, *, timeframe: str = "1d", horizon: int = 5,
) -> dict:
    """All symbols' forecasts as of ``session_date`` -> ``{symbol: forecast dict}``.

    The engine's per-day lookup: deterministic, point-in-time, no torch.
    """
    try:
        rows = con.execute(
            "SELECT symbol, exp_return, vol, downside_cvar, prob_up "
            "FROM kronos_forecasts WHERE session_date=? AND timeframe=? AND horizon=?",
            [_as_date(session_date), timeframe, int(horizon)],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    return {
        r[0]: {"exp_return": r[1], "vol": r[2], "downside_cvar": r[3], "prob_up": r[4]}
        for r in rows
    }


def make_forecast_provider(con, *, timeframe: str = "1d", horizon: int = 5):
    """A ``provider(asof, symbol) -> forecast dict | None`` over the table.

    The callable the portfolio engine takes as ``forecast_provider``: a pure,
    point-in-time read of ``kronos_forecasts``. Returns ``None`` for a symbol with
    no forecast as of that session (the engine then simply doesn't veto/blend it).
    """
    def provider(asof, symbol):
        return read_forecast(con, symbol, asof, timeframe=timeframe, horizon=horizon)
    return provider


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
