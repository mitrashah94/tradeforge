"""orchestrator/tools/market_data.py — bar/quote access feeding the fast loop.

Two modes share ONE output shape (the BAR-event payload the fast loop consumes
via ``System.feed_bar`` / ``orchestrator.main``):

  * **ReplayFeed (historical, implemented here)** — reads completed bars from
    ``data/duckdb/market.duckdb`` and yields them grouped by session, tagging the
    session's last bar ``is_eod``. This is the feed Phase-1 REAL paper trading
    uses to drive the deterministic event path over many sessions, exactly as
    ``scripts/paper_dry_run.py`` drives a single session — extracted here so the
    multi-session paper driver and the backtest share one feed.

  * **Live feed (Phase 3, TODO)** — an Alpaca/Polygon websocket or poll loop that
    publishes the SAME ``BAR`` payload onto the bus in real time. Deliberately not
    built yet: live trading is gated behind a validated edge (see the roadmap).

The BAR payload (matches ``orchestrator.fast_loop.engine.LiveBar.from_data`` and
``orchestrator.main.System.feed_bar``):

    {"symbol", "ts_utc" (str), "open", "high", "low", "close", "volume", "is_eod"}

Heavy deps (duckdb/pandas) are imported INSIDE functions per this module's build
contract, so importing the module is cheap and side-effect-free. This module
depends only on the data layer (``data.schema`` / ``data.sessions``) — never on
the backtest engine or the hot-path fast loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator

from data.schema import DEFAULT_DB_PATH


def _infer_asset_class(symbol: str) -> str:
    return "crypto" if "/" in symbol else "equity"


def bar_payload(symbol: str, row, is_eod: bool) -> dict:
    """Build the BAR-event payload from a bars row (a namedtuple/Series-like)."""
    return {
        "symbol": symbol,
        "ts_utc": str(row.ts_utc),
        "open": float(row.open),
        "high": float(row.high),
        "low": float(row.low),
        "close": float(row.close),
        "volume": float(getattr(row, "volume", 0.0) or 0.0),
        "is_eod": bool(is_eod),
    }


@dataclass
class SessionBars:
    """One session's ordered bars as ready-to-feed BAR payloads (last is_eod)."""

    session_date: date
    bars: list[dict]


class ReplayFeed:
    """Historical bar feed: completed bars from market.duckdb, grouped by session.

    Loads RTH-filtered bars for one ``(symbol, timeframe)`` (optionally clipped to
    ``[start, end]`` ET session dates), the symbol's per-session levels, and groups
    the bars by ET session so a driver can, per session: update the armed
    strategy's ``levels`` and feed the session's bars (the last tagged ``is_eod``)
    through ``System.feed_bar``.

    Usage::

        feed = ReplayFeed("QQQ", "5m", start="2024-07-01", end="2026-06-12")
        for s in feed.iter_sessions():
            armed.levels = feed.levels.get(s.session_date, {})
            for payload in s.bars:
                system.mark_price("QQQ", payload["close"])
                system.feed_bar(payload)
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "5m",
        *,
        start=None,
        end=None,
        db_path: str = DEFAULT_DB_PATH,
        con=None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.asset_class = _infer_asset_class(symbol)
        self._sessions: list[SessionBars] = []
        self.levels: dict = {}
        self._load(start, end, db_path, con)

    # ------------------------------------------------------------------ load
    def _load(self, start, end, db_path, con) -> None:
        from data.schema import connect
        from data.sessions import et_session_date, is_rth

        own = con is None
        if own:
            con = connect(db_path)
        try:
            bars = con.execute(
                """
                SELECT ts_utc, open, high, low, close, volume
                FROM bars WHERE symbol = ? AND timeframe = ?
                ORDER BY ts_utc
                """,
                [self.symbol, self.timeframe],
            ).df()
            lv = con.execute(
                """
                SELECT session_date, pdh, pdl, pmh, pml, ntz_low, ntz_high,
                       ntz_valid, atr14
                FROM levels WHERE symbol = ? ORDER BY session_date
                """,
                [self.symbol],
            ).df()
        finally:
            if own:
                con.close()

        if len(bars) == 0:
            return

        if self.asset_class == "equity":
            bars = bars[bars["ts_utc"].apply(is_rth)].reset_index(drop=True)

        s = _as_date(start)
        e = _as_date(end)
        if s is not None or e is not None:
            sess = bars["ts_utc"].apply(et_session_date)
            mask = sess.apply(lambda d: (s is None or d >= s) and (e is None or d <= e))
            bars = bars[mask].reset_index(drop=True)

        # levels map (session_date -> dict), keys normalized to date.
        for row in lv.itertuples(index=False):
            sd = row.session_date.date() if hasattr(row.session_date, "date") else row.session_date
            self.levels[sd] = {
                "pdh": _f(row.pdh), "pdl": _f(row.pdl), "pmh": _f(row.pmh),
                "pml": _f(row.pml), "ntz_low": _f(row.ntz_low),
                "ntz_high": _f(row.ntz_high), "ntz_valid": bool(row.ntz_valid),
                "atr14": _f(row.atr14),
            }

        # group bars by ET session, preserving order, tagging the last is_eod.
        bars["_sd"] = bars["ts_utc"].apply(et_session_date)
        for sd, grp in bars.groupby("_sd", sort=True):
            rows = list(grp.itertuples(index=False))
            payloads = [
                bar_payload(self.symbol, r, is_eod=(i == len(rows) - 1))
                for i, r in enumerate(rows)
            ]
            self._sessions.append(SessionBars(session_date=sd, bars=payloads))

    # ------------------------------------------------------------------ access
    @property
    def session_dates(self) -> list[date]:
        return [s.session_date for s in self._sessions]

    def __len__(self) -> int:
        return len(self._sessions)

    def iter_sessions(self) -> Iterator[SessionBars]:
        """Yield each session's bars in chronological order."""
        yield from self._sessions

    def iter_bars(self) -> Iterator[dict]:
        """Flat chronological stream of BAR payloads (is_eod on each session end)."""
        for s in self._sessions:
            yield from s.bars


def _f(v):
    import pandas as pd
    if v is None:
        return None
    try:
        return None if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return float(v)


def _as_date(d):
    from datetime import datetime
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    if hasattr(d, "date"):
        return d.date()
    raise TypeError(f"cannot coerce {d!r} to a date")
