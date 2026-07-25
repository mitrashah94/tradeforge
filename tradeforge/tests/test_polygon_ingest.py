"""Offline tests for data.pipelines.polygon_ingest.

No network and no Polygon SDK required: the transform is exercised with plain
dicts and attribute-style fakes, and the full run() path is driven through an
injected fake client into an in-memory DuckDB. Mirrors the contract that
data.schema.upsert_bars expects so Polygon bars are drop-in with Alpaca bars.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data.pipelines import polygon_ingest as pi

# ts helpers — Polygon `t` is Unix MILLISECONDS at the bar OPEN, UTC.
def _ms(y, mo, d, h, mi) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1000)


def _dict_agg(ts_ms, o, h, l, c, v, vw=None, n=None) -> dict:
    """A raw-REST-style aggregate (short keys)."""
    agg = {"t": ts_ms, "o": o, "h": h, "l": l, "c": c, "v": v}
    if vw is not None:
        agg["vw"] = vw
    if n is not None:
        agg["n"] = n
    return agg


class _AttrAgg:
    """An SDK-style aggregate (Agg dataclass attributes)."""

    def __init__(self, ts_ms, o, h, l, c, v, vw=None, n=None):
        self.timestamp = ts_ms
        self.open, self.high, self.low, self.close, self.volume = o, h, l, c, v
        self.vwap = vw
        self.transactions = n


# --------------------------------------------------------------------------- #
# pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def test_polygon_ticker_mapping():
    assert pi.polygon_ticker("QQQ", "equity") == "QQQ"
    assert pi.polygon_ticker("spy", "equity") == "SPY"
    assert pi.polygon_ticker("BTC/USD", "crypto") == "X:BTCUSD"
    assert pi.polygon_ticker("eth/usd", "crypto") == "X:ETHUSD"


def test_parse_symbols_infers_asset_class():
    assert pi._parse_symbols(None) is None
    assert pi._parse_symbols("SPY,QQQ,BTC/USD") == {
        "SPY": "equity",
        "QQQ": "equity",
        "BTC/USD": "crypto",
    }


# --------------------------------------------------------------------------- #
# transform: _aggs_to_frame                                                    #
# --------------------------------------------------------------------------- #
def test_aggs_to_frame_schema_and_values():
    end = datetime(2024, 6, 13, 21, 0, tzinfo=timezone.utc)  # well after the bars
    aggs = [
        _dict_agg(_ms(2024, 6, 13, 13, 30), 100.0, 101.0, 99.5, 100.5, 1000, vw=100.2, n=42),
        _dict_agg(_ms(2024, 6, 13, 13, 35), 100.5, 102.0, 100.0, 101.5, 1500, vw=101.1, n=51),
    ]
    df = pi._aggs_to_frame(
        aggs, symbol="QQQ", asset_class="equity", timeframe="5m", adjusted=True, end_utc=end
    )
    # Exact contract columns, in order.
    assert list(df.columns) == list(pi._BAR_COLUMNS)
    assert len(df) == 2
    first = df.iloc[0]
    # ts_utc is tz-naive UTC at the bar OPEN.
    assert first["ts_utc"] == datetime(2024, 6, 13, 13, 30)
    assert first["ts_utc"].tzinfo is None
    assert first["open"] == 100.0 and first["close"] == 100.5
    assert first["trade_count"] == 42 and first["vwap"] == 100.2
    assert bool(first["adjusted"]) is True
    assert first["asset_class"] == "equity" and first["timeframe"] == "5m"


def test_aggs_to_frame_accepts_attr_objects():
    end = datetime(2024, 6, 13, 21, 0, tzinfo=timezone.utc)
    aggs = [_AttrAgg(_ms(2024, 6, 13, 13, 30), 10.0, 11.0, 9.0, 10.5, 5, vw=10.1, n=7)]
    df = pi._aggs_to_frame(
        aggs, symbol="SPY", asset_class="equity", timeframe="2m", adjusted=True, end_utc=end
    )
    assert len(df) == 1
    assert df.iloc[0]["high"] == 11.0 and df.iloc[0]["trade_count"] == 7


def test_aggs_to_frame_drops_forming_bar():
    # end falls BETWEEN the open and close of the last 5m bar -> it is forming.
    last_open = datetime(2024, 6, 13, 13, 35, tzinfo=timezone.utc)
    end = last_open  # close would be 13:40 > end -> dropped
    aggs = [
        _dict_agg(_ms(2024, 6, 13, 13, 30), 1, 2, 0.5, 1.5, 100),  # closes 13:35 <= end -> kept
        _dict_agg(_ms(2024, 6, 13, 13, 35), 1, 2, 0.5, 1.5, 100),  # closes 13:40 > end -> dropped
    ]
    df = pi._aggs_to_frame(
        aggs, symbol="QQQ", asset_class="equity", timeframe="5m", adjusted=True, end_utc=end
    )
    assert len(df) == 1
    assert df.iloc[0]["ts_utc"] == datetime(2024, 6, 13, 13, 30)


def test_aggs_to_frame_empty_returns_contract_columns():
    end = datetime(2024, 6, 13, 21, 0, tzinfo=timezone.utc)
    df = pi._aggs_to_frame(
        [], symbol="QQQ", asset_class="equity", timeframe="5m", adjusted=True, end_utc=end
    )
    assert list(df.columns) == list(pi._BAR_COLUMNS)
    assert df.empty


def test_crypto_tagged_unadjusted():
    end = datetime(2024, 6, 13, 21, 0, tzinfo=timezone.utc)
    aggs = [_dict_agg(_ms(2024, 6, 13, 13, 30), 60000, 60500, 59800, 60200, 3)]
    df = pi._aggs_to_frame(
        aggs, symbol="BTC/USD", asset_class="crypto", timeframe="5m", adjusted=False, end_utc=end
    )
    assert bool(df.iloc[0]["adjusted"]) is False
    assert df.iloc[0]["asset_class"] == "crypto"


# --------------------------------------------------------------------------- #
# full run() via injected fake client (no network, in-memory DuckDB)           #
# --------------------------------------------------------------------------- #
class _FakeClient:
    """Returns canned aggregates and records the kwargs it was called with."""

    def __init__(self, by_ticker):
        self._by_ticker = by_ticker
        self.calls = []

    def list_aggs(self, **kwargs):
        self.calls.append(kwargs)
        return self._by_ticker.get(kwargs["ticker"], [])


def test_run_writes_bars_via_injected_client():
    pytest.importorskip("duckdb")
    end = datetime(2024, 6, 13, 21, 0, tzinfo=timezone.utc)
    start = datetime(2024, 6, 13, 13, 0, tzinfo=timezone.utc)
    fake = _FakeClient(
        {
            "QQQ": [
                _dict_agg(_ms(2024, 6, 13, 13, 30), 100, 101, 99, 100.5, 1000, vw=100.2, n=10),
                _dict_agg(_ms(2024, 6, 13, 13, 35), 100.5, 102, 100, 101.5, 1500, vw=101.1, n=12),
            ],
            "X:BTCUSD": [
                _dict_agg(_ms(2024, 6, 13, 13, 30), 60000, 60500, 59800, 60200, 3),
            ],
        }
    )
    results = pi.run(
        symbols={"QQQ": "equity", "BTC/USD": "crypto"},
        timeframes=("5m",),
        start=start,
        end=end,
        db_path=":memory:",
        client=fake,
    )
    assert results == {"QQQ:5m": 2, "BTC/USD:5m": 1}
    # The equity request must be split-adjusted; crypto must not.
    by_ticker = {c["ticker"]: c for c in fake.calls}
    assert by_ticker["QQQ"]["adjusted"] is True
    assert by_ticker["X:BTCUSD"]["adjusted"] is False
    assert by_ticker["QQQ"]["multiplier"] == 5 and by_ticker["QQQ"]["timespan"] == "minute"


def test_run_rejects_bad_timeframe():
    with pytest.raises(ValueError):
        pi.run(symbols={"QQQ": "equity"}, timeframes=("3m",), days=5, client=_FakeClient({}))


def test_run_rejects_empty_window():
    end = datetime(2024, 6, 13, 13, 0, tzinfo=timezone.utc)
    start = datetime(2024, 6, 13, 21, 0, tzinfo=timezone.utc)  # start after end
    with pytest.raises(ValueError):
        pi.run(symbols={"QQQ": "equity"}, timeframes=("5m",), start=start, end=end,
               client=_FakeClient({}))
