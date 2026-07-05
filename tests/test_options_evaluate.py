"""tests/test_options_evaluate.py — the live evaluation runner (offline).

Exercises the pure transforms (BS greeks, realized-vol IV-rank proxy, snapshot
mapping), the rate limiter's spacing (fake clock/sleep — no real waiting), and
the full evaluate() path over the built-in fixture. No key, no SDK, no network.
"""

from __future__ import annotations

from datetime import date

import pytest

from strategies.breakout_retest_options.evaluate import (
    RateLimiter,
    _demo_source,
    atm_iv,
    bs_greeks,
    build_underlying,
    evaluate,
    iv_rank_proxy,
    realized_vol_series,
    snapshot_to_contracts,
)

AS_OF = date(2026, 7, 5)


# ---------------------------------------------------------------- BS greeks
def test_bs_greeks_atm_call_and_put_are_sane():
    dc, tc = bs_greeks(500, 500, 1, 0.20, "call")
    dp, tp = bs_greeks(500, 500, 1, 0.20, "put")
    assert 0.4 < dc < 0.65           # ATM-ish call delta near 0.5
    assert -0.65 < dp < -0.35        # ATM put delta negative
    assert tc < 0 and tp < 0         # long premium bleeds theta
    # deep ITM call -> delta approaches 1; deep OTM call -> approaches 0
    assert bs_greeks(500, 400, 5, 0.20, "call")[0] > 0.9
    assert bs_greeks(500, 600, 5, 0.20, "call")[0] < 0.1


def test_bs_greeks_zero_dte_floored_not_exploded():
    # 0 DTE floored to half a day -> finite greeks, not a blow-up.
    d, t = bs_greeks(500, 500, 0, 0.20, "call")
    assert 0.0 < d < 1.0
    assert t < 0 and t > -1e4


def test_bs_greeks_degenerate_inputs():
    assert bs_greeks(500, 500, 1, 0.0, "call") == (0.0, 0.0)
    assert bs_greeks(0, 500, 1, 0.2, "call") == (0.0, 0.0)


# ---------------------------------------------------- realized vol + IV rank
def test_realized_vol_series_length_and_positive():
    closes = [100.0 * (1.01 ** i if i % 2 else 0.99 ** i) for i in range(60)]
    s = realized_vol_series(closes, window=20)
    assert len(s) == len(closes) - 1 - 20 + 1
    assert all(v > 0 for v in s)


def test_realized_vol_series_too_short_is_empty():
    assert realized_vol_series([100, 101, 102], window=20) == []


def test_iv_rank_proxy_bounds_and_monotone():
    import math
    closes = [100.0 + 5 * math.sin(i / 7.0) for i in range(200)]
    lo = iv_rank_proxy(closes, current_iv=0.01)   # below the envelope -> ~0
    hi = iv_rank_proxy(closes, current_iv=10.0)   # above -> clipped to 1
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)
    assert iv_rank_proxy([1, 2, 3], current_iv=0.2) == 0.5   # insufficient history


# ------------------------------------------------- snapshot -> OptionContract
def test_snapshot_uses_greeks_when_present():
    rows = [{"details": {"strike_price": 500, "expiration_date": "2026-07-06",
                         "contract_type": "call"},
             "last_quote": {"bid": 3.0, "ask": 3.1},
             "greeks": {"delta": 0.53, "theta": -0.12},
             "implied_volatility": 0.19, "open_interest": 900}]
    cs = snapshot_to_contracts("SPY", rows, 500.0, AS_OF)
    assert len(cs) == 1
    assert cs[0].delta == 0.53 and cs[0].theta == -0.12


def test_snapshot_falls_back_to_bs_when_greeks_missing():
    rows = [{"details": {"strike_price": 500, "expiration_date": "2026-07-06",
                         "contract_type": "call"},
             "last_quote": {"bid": 3.0, "ask": 3.1},
             "implied_volatility": 0.19, "open_interest": 900}]
    cs = snapshot_to_contracts("SPY", rows, 500.0, AS_OF)
    assert len(cs) == 1
    assert 0.4 < cs[0].delta < 0.65      # BS-derived
    assert cs[0].theta < 0


def test_snapshot_drops_rows_without_quote_or_iv():
    rows = [
        {"details": {"strike_price": 500, "expiration_date": "2026-07-06",
                     "contract_type": "call"},
         "last_quote": {"bid": 0.0, "ask": 0.0}, "implied_volatility": 0.2},   # no ask
        {"details": {"strike_price": 501, "expiration_date": "2026-07-06",
                     "contract_type": "call"},
         "last_quote": {"bid": 1.0, "ask": 1.1}},                              # no greeks/iv
    ]
    assert snapshot_to_contracts("SPY", rows, 500.0, AS_OF) == []


# --------------------------------------------------------------- rate limiter
def test_rate_limiter_spaces_calls():
    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()
    sleeps: list[float] = []

    def fake_sleep(d):
        sleeps.append(d)
        clock.t += d

    rl = RateLimiter(5, sleep=fake_sleep, clock=clock)  # 12s spacing
    rl.wait()                     # first call: no wait
    clock.t += 2.0                # 2s of real work elapses
    rl.wait()                     # must sleep 12 - 2 = 10s
    assert sleeps == [pytest.approx(10.0)]
    assert rl.calls == 2


# --------------------------------------------------------------- end-to-end
def test_evaluate_offline_produces_decisions_for_both_symbols():
    src = _demo_source()
    out = evaluate(src, ["SPY", "QQQ"], side="long", equity=1000.0,
                   as_of=AS_OF, time_et="10:05")
    assert [u.symbol for u, _ in out] == ["SPY", "QQQ"]
    for u, dec in out:
        assert 0.0 <= u.iv_rank <= 1.0
        assert dec.ok                        # fixture yields an affordable ticket
        assert dec.contracts >= 1
        assert dec.diagnostics["risk_pct_of_equity"] > 0


def test_evaluate_offline_session_gate_skips_outside_window():
    src = _demo_source()
    out = evaluate(src, ["SPY"], side="long", equity=1000.0,
                   as_of=AS_OF, time_et="13:00")
    _, dec = out[0]
    assert not dec.ok
    assert dec.reason == "outside_first_90min_window"


def test_build_underlying_assembles_spot_chain_ivrank():
    src = _demo_source()
    u = build_underlying(src, "SPY", AS_OF)
    assert u.spot == 500.0
    assert len(u.contracts) > 0
    assert u.atm_iv > 0
    assert 0.0 <= u.iv_rank <= 1.0
