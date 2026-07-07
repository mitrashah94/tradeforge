"""tests/test_paper_today.py — the live-session paper-trade bridge.

Deterministic, offline: synthetic RH-shaped JSON (a daily file for the Wilder
ATR14 + PDH/PDL and a 5m session that forces a PDH break→retest) so we can assert
(a) the real BreakoutRetestStrategy fires a trade through the engine and (b) the
$1k fractional vol-target sizing is buying-power-capped on a high-priced
underlying, with the honest risk % reported.
"""

from __future__ import annotations

import json

import pytest

from risk.config import load_limits
from scripts.paper_today import run, size_trade


def _write(path, symbol, bars):
    path.write_text(json.dumps({"data": {"results": [{"symbol": symbol, "bars": bars}]}}))


def _daily(symbol, tmp):
    # 20 flat daily bars (H=100,L=98,C=99) -> PDH=100, PDL=98, ATR~2.
    bars = [{"begins_at": f"2026-06-{d:02d}T00:00:00Z", "open_price": "99",
             "high_price": "100", "low_price": "98", "close_price": "99"}
            for d in range(1, 21)]
    p = tmp / "daily.json"
    _write(p, symbol, bars)
    return p


def _session(symbol, tmp):
    # Trade date 2026-07-06 ET. PDH=100. Break (close>100), then retest, then run.
    def bar(hhmm, o, h, l, c):
        return {"begins_at": f"2026-07-06T{hhmm}:00Z", "open_price": str(o),
                "high_price": str(h), "low_price": str(l), "close_price": str(c),
                "volume": 1000}
    bars = [
        bar("14:00", 100.1, 100.3, 100.0, 100.4),   # break (close 100.4 > PDH 100)
        bar("14:05", 100.4, 100.5, 100.2, 100.5),   # bsb=2
        bar("14:10", 100.5, 100.5, 99.9, 100.3),    # retest: low<=100, close>100 → LONG
        bar("14:15", 100.3, 100.4, 100.2, 100.35),  # entry fills at this open (100.3)
        bar("14:20", 100.4, 101.0, 100.3, 100.9),
        bar("14:25", 100.9, 102.2, 100.8, 102.1),   # high>=target(~101.9) → win
        bar("14:30", 102.1, 102.2, 101.8, 102.0),
    ]
    p = tmp / "intraday.json"
    _write(p, symbol, bars)
    return p


def test_run_fires_trade_and_sizes_on_1k(tmp_path):
    daily = _daily("QQQ", tmp_path)
    intra = _session("QQQ", tmp_path)
    rep = run("QQQ", str(intra), str(daily), None, equity=1000.0)
    assert rep["levels"]["pdh"] == 100.0
    assert rep["levels"]["atr14"] is not None
    assert rep["n_trades"] == 1
    t = rep["trades"][0]
    assert t["side"] == "long"
    assert t["r_multiple"] > 0                    # it resolved a winner
    s = t["sizing"]
    assert s["shares"] > 0
    assert s["shares"] == pytest.approx(min(s["ideal_shares"], s["bp_cap_shares"]))
    assert s["actual_risk_pct"] > 0
    assert t["pnl"] > 0


def test_size_trade_buying_power_capped_on_high_priced_underlying():
    # QQQ ~722, ATR-derived stop ~$4 away: the vol-target share count exceeds
    # what $1k of buying power can hold -> buying_power binds, risk < budget.
    limits = load_limits()
    s = size_trade(1000.0, entry=722.0, stop=718.0, limits=limits)
    lev = limits.level(s["ri"]).leverage_max or 1.0
    assert s["bp_cap_shares"] == pytest.approx(1000.0 * lev / 722.0, abs=1e-3)
    assert s["binding"] == "buying_power"
    assert s["shares"] == pytest.approx(s["bp_cap_shares"])
    # realized risk is a fraction of a percent — the small-account reality.
    assert s["actual_risk_pct"] < limits.level(s["ri"]).per_trade_pct / 100.0


def test_ri8_uses_intraday_leverage_when_leverage_max_null():
    # RI 8 has leverage_max: null ("up to broker intraday max") -> the configured
    # intraday leverage (default 4x) sets buying power, not 1x.
    limits = load_limits()
    assert limits.level(8).leverage_max is None
    s = size_trade(1000.0, entry=751.0, stop=748.5, limits=limits, ri=8,
                   intraday_leverage=4.0)
    assert s["ri"] == 8
    assert s["bp_cap_shares"] == pytest.approx(1000.0 * 4.0 / 751.0, abs=1e-3)
    assert s["risk_budget"] == pytest.approx(20.0)          # 2% of $1k at RI 8
    assert s["binding"] == "buying_power"


def test_run_flat_session_when_no_break(tmp_path):
    daily = _daily("QQQ", tmp_path)
    # session entirely below PDH=100 -> no break -> no trade
    bars = [{"begins_at": f"2026-07-06T14:{m:02d}:00Z", "open_price": "99.0",
             "high_price": "99.5", "low_price": "98.5", "close_price": "99.0",
             "volume": 1000} for m in (0, 5, 10, 15, 20)]
    intra = tmp_path / "flat.json"
    _write(intra, "QQQ", bars)
    rep = run("QQQ", str(intra), str(daily), None, equity=1000.0)
    assert rep["n_trades"] == 0
