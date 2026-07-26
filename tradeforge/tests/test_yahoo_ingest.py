"""Offline tests for data.pipelines.yahoo_ingest.

No network: the parser :func:`_chart_to_frame` is exercised with a small fake
Yahoo v8 chart JSON dict, and the full run() path is driven through an injected
fake ``fetch`` into an in-memory DuckDB. Asserts the bars-schema contract, the
total-return ``adjclose/close`` scaling, tz-naive daily ts_utc, and the
forming-bar (current-day) drop — the four invariants the daily layer depends on.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from data.pipelines import yahoo_ingest as yi

# US-equity gmtoffset (EDT, -4h) so session dates land on the right calendar day.
_GMT = -4 * 3600


def _open_ts(y, mo, d) -> int:
    """Unix SECONDS at the 9:30 ET regular-market open for a session date."""
    # 9:30 ET == 13:30 UTC during EDT.
    return int(datetime(y, mo, d, 13, 30, tzinfo=timezone.utc).timestamp())


def _fake_payload(rows, *, symbol="SPY", gmtoffset=_GMT, error=None):
    """Build a Yahoo-shaped chart payload from (date, o, h, l, c, v, adjclose) rows."""
    if error is not None:
        return {"chart": {"result": None, "error": error}}
    ts, o, h, l, c, v, adj = [], [], [], [], [], [], []
    for d, oo, hh, ll, cc, vv, aa in rows:
        ts.append(_open_ts(d.year, d.month, d.day))
        o.append(oo); h.append(hh); l.append(ll); c.append(cc); v.append(vv); adj.append(aa)
    return {
        "chart": {
            "result": [
                {
                    "meta": {"symbol": symbol, "gmtoffset": gmtoffset},
                    "timestamp": ts,
                    "indicators": {
                        "quote": [{"open": o, "high": h, "low": l, "close": c, "volume": v}],
                        "adjclose": [{"adjclose": adj}],
                    },
                }
            ],
            "error": None,
        }
    }


# --------------------------------------------------------------------------- #
# parser: _chart_to_frame                                                      #
# --------------------------------------------------------------------------- #
def test_chart_to_frame_schema_and_columns():
    payload = _fake_payload(
        [(date(2020, 1, 2), 100.0, 101.0, 99.0, 100.0, 1000, 50.0)],
        symbol="SPY",
    )
    df = yi._chart_to_frame(payload, symbol="SPY", today=date(2026, 1, 1))
    # Exact contract columns, in upsert_bars order.
    assert list(df.columns) == list(yi._BAR_COLUMNS)
    row = df.iloc[0]
    assert row["symbol"] == "SPY"
    assert row["asset_class"] == "equity"
    assert row["timeframe"] == "1d"
    assert bool(row["adjusted"]) is True


def test_chart_to_frame_total_return_scaling():
    # adjclose 50 vs close 100 -> factor 0.5 scales the WHOLE bar; close becomes adjclose.
    payload = _fake_payload(
        [(date(2020, 1, 2), 100.0, 120.0, 80.0, 100.0, 1000, 50.0)],
        symbol="SPY",
    )
    df = yi._chart_to_frame(payload, symbol="SPY", today=date(2026, 1, 1))
    row = df.iloc[0]
    assert row["open"] == pytest.approx(50.0)   # 100 * 0.5
    assert row["high"] == pytest.approx(60.0)   # 120 * 0.5
    assert row["low"] == pytest.approx(40.0)    # 80 * 0.5
    assert row["close"] == pytest.approx(50.0)  # == adjclose exactly
    assert row["volume"] == 1000                # volume unscaled
    # Bar stays internally consistent: low <= open,close <= high.
    assert row["low"] <= row["open"] <= row["high"]
    assert row["low"] <= row["close"] <= row["high"]


def test_chart_to_frame_ts_is_tz_naive_daily_date():
    payload = _fake_payload(
        [(date(2020, 3, 16), 10.0, 11.0, 9.0, 10.0, 5, 10.0)],
        symbol="QQQ",
    )
    df = yi._chart_to_frame(payload, symbol="QQQ", today=date(2026, 1, 1))
    ts = df.iloc[0]["ts_utc"]
    # Session date at 00:00, tz-naive UTC.
    assert ts == datetime(2020, 3, 16)
    assert ts.tzinfo is None
    assert (ts.hour, ts.minute, ts.second) == (0, 0, 0)


def test_chart_to_frame_drops_forming_current_day():
    # Two sessions; 'today' is the second one -> the second (forming) day is dropped.
    rows = [
        (date(2026, 6, 25), 100.0, 101.0, 99.0, 100.0, 1000, 100.0),  # closed -> kept
        (date(2026, 6, 26), 100.0, 102.0, 99.0, 101.0, 1200, 101.0),  # today -> dropped
    ]
    payload = _fake_payload(rows, symbol="SPY")
    df = yi._chart_to_frame(payload, symbol="SPY", today=date(2026, 6, 26))
    assert len(df) == 1
    assert df.iloc[0]["ts_utc"] == datetime(2026, 6, 25)


def test_chart_to_frame_skips_null_holes():
    rows = [
        (date(2020, 1, 2), 100.0, 101.0, 99.0, 100.0, 1000, 100.0),
        (date(2020, 1, 3), None, None, None, None, None, None),  # holiday/hole
        (date(2020, 1, 6), 102.0, 103.0, 101.0, 102.5, 900, 102.5),
    ]
    payload = _fake_payload(rows, symbol="SPY")
    df = yi._chart_to_frame(payload, symbol="SPY", today=date(2026, 1, 1))
    assert len(df) == 2
    assert list(df["ts_utc"]) == [datetime(2020, 1, 2), datetime(2020, 1, 6)]


def test_chart_to_frame_sorted_ascending():
    rows = [
        (date(2020, 1, 6), 102.0, 103.0, 101.0, 102.5, 900, 102.5),
        (date(2020, 1, 2), 100.0, 101.0, 99.0, 100.0, 1000, 100.0),
    ]
    payload = _fake_payload(rows, symbol="SPY")
    df = yi._chart_to_frame(payload, symbol="SPY", today=date(2026, 1, 1))
    assert list(df["ts_utc"]) == [datetime(2020, 1, 2), datetime(2020, 1, 6)]


def test_chart_to_frame_empty_and_error():
    empty = {"chart": {"result": [], "error": None}}
    df = yi._chart_to_frame(empty, symbol="SPY", today=date(2026, 1, 1))
    assert list(df.columns) == list(yi._BAR_COLUMNS)
    assert df.empty

    with pytest.raises(ValueError):
        yi._chart_to_frame(
            _fake_payload([], symbol="BADTICK", error={"code": "Not Found"}),
            symbol="BADTICK",
            today=date(2026, 1, 1),
        )


# --------------------------------------------------------------------------- #
# full run() via injected fetch (no network, in-memory DuckDB)                 #
# --------------------------------------------------------------------------- #
def test_run_writes_bars_via_injected_fetch():
    pytest.importorskip("duckdb")
    payloads = {
        "SPY": _fake_payload(
            [
                (date(2020, 1, 2), 100.0, 101.0, 99.0, 100.0, 1000, 50.0),
                (date(2020, 1, 3), 100.0, 102.0, 99.0, 101.0, 1200, 50.5),
            ],
            symbol="SPY",
        ),
        "TLT": _fake_payload(
            [(date(2020, 1, 2), 80.0, 81.0, 79.0, 80.0, 500, 80.0)],
            symbol="TLT",
        ),
    }

    def fake_fetch(symbol, range_="30y"):
        return payloads[symbol]

    results = yi.run(
        symbols=["SPY", "TLT"],
        db_path=":memory:",
        pause=0.0,
        fetch=fake_fetch,
    )
    assert results == {"SPY": 2, "TLT": 1}


def test_run_records_per_symbol_failure_and_continues():
    pytest.importorskip("duckdb")

    def fake_fetch(symbol, range_="30y"):
        if symbol == "BOOM":
            raise RuntimeError("network down")
        return _fake_payload(
            [(date(2020, 1, 2), 10.0, 11.0, 9.0, 10.0, 5, 10.0)], symbol=symbol
        )

    results = yi.run(
        symbols=["BOOM", "SPY"],
        db_path=":memory:",
        pause=0.0,
        fetch=fake_fetch,
    )
    assert results["BOOM"] == -1   # failure recorded
    assert results["SPY"] == 1     # the rest still ingested


def test_run_applies_start_end_window():
    pytest.importorskip("duckdb")
    rows = [
        (date(2019, 12, 31), 100.0, 101.0, 99.0, 100.0, 1, 100.0),
        (date(2020, 1, 2), 100.0, 101.0, 99.0, 100.0, 1, 100.0),
        (date(2020, 1, 3), 100.0, 101.0, 99.0, 100.0, 1, 100.0),
        (date(2020, 6, 1), 100.0, 101.0, 99.0, 100.0, 1, 100.0),
    ]

    def fake_fetch(symbol, range_="30y"):
        return _fake_payload(rows, symbol=symbol)

    results = yi.run(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-01-31",
        db_path=":memory:",
        pause=0.0,
        fetch=fake_fetch,
    )
    assert results["SPY"] == 2  # only the two January sessions survive the window


# --------------------------------------------------------------------------- #
# validation helper                                                            #
# --------------------------------------------------------------------------- #
def test_cagr_from_adjusted_recovers_known_rate():
    import pandas as pd

    # 10% / yr for ~5 years: close 100 -> 100*1.1**5 over exactly 5*365.25 days.
    start = datetime(2015, 1, 1)
    end = start + pd.Timedelta(days=round(5 * 365.25))
    df = pd.DataFrame(
        {
            "ts_utc": [start, end],
            "close": [100.0, 100.0 * (1.10 ** 5)],
        }
    )
    years, cagr = yi._cagr_from_adjusted(df)
    assert years == pytest.approx(5.0, abs=0.01)
    assert cagr == pytest.approx(0.10, abs=1e-3)


def test_parse_symbols():
    assert yi._parse_symbols(None) is None
    assert yi._parse_symbols("spy, qqq , tlt") == {
        "SPY": "equity",
        "QQQ": "equity",
        "TLT": "equity",
    }
