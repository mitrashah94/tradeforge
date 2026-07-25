#!/usr/bin/env python3
"""test_option_pricing.py -- plain-assert tests for option_pricing.py.
Run: python3 analysis/test_option_pricing.py   (must print OK). Stdlib only."""
import math
import option_pricing as op


def _close(a, b, tol=1e-3):
    return abs(a - b) <= tol


def test_known_black_scholes_value():
    # textbook: S=K=100, T=1, r=0, sigma=0.20 -> call 7.9656, put 7.9656
    c = op.bs_price(100, 100, 1.0, 0.0, 0.20, "C")
    p = op.bs_price(100, 100, 1.0, 0.0, 0.20, "P")
    assert _close(c, 7.9656), c
    assert _close(p, 7.9656), p


def test_put_call_parity():
    S, K, T, r, s = 60.03, 60.0, 0.02, 0.04, 0.29
    c = op.bs_price(S, K, T, r, s, "C")
    p = op.bs_price(S, K, T, r, s, "P")
    assert _close(c - p, S - K * math.exp(-r * T), 1e-6), (c - p)


def test_price_is_monotonic_in_spot_and_vol():
    base = op.bs_price(60, 60, 0.02, 0.04, 0.29, "C")
    assert op.bs_price(60.5, 60, 0.02, 0.04, 0.29, "C") > base
    assert op.bs_price(60, 60, 0.02, 0.04, 0.40, "C") > base
    assert op.bs_price(60, 60, 0.04, 0.04, 0.29, "C") > base   # more time


def test_implied_vol_round_trip():
    S, K, T, r = 60.03, 60.0, 0.0198, 0.04
    for target in (0.15, 0.29, 0.55):
        px = op.bs_price(S, K, T, r, target, "C")
        iv = op.implied_vol(px, S, K, T, r, "C")
        assert iv is not None and _close(iv, target, 1e-4), (target, iv)


def test_implied_vol_refuses_impossible_marks():
    # below intrinsic -> unfittable, must return None (never guess a vol)
    assert op.implied_vol(0.10, 70.0, 60.0, 0.02, 0.04, "C") is None
    assert op.implied_vol(0.0, 60.0, 60.0, 0.02, 0.04, "C") is None


def test_greeks_sanity():
    g = op.bs_greeks(100, 100, 1.0, 0.0, 0.20, "C")
    assert _close(g["delta"], 0.5398, 1e-3), g
    assert g["gamma"] > 0 and g["vega"] > 0
    assert g["theta"] < 0                      # long option bleeds
    gp = op.bs_greeks(100, 100, 1.0, 0.0, 0.20, "P")
    assert _close(g["delta"] - gp["delta"], 1.0, 1e-6)   # parity on delta


def test_expiry_and_year_fraction():
    e = op.expiry_epoch("2026-07-31")
    sig = 1784901900                            # 2026-07-24 14:05 UTC
    T = op.year_fraction(sig, e)
    assert 0.018 < T < 0.021, T                 # ~7.2 days
    assert op.year_fraction(e + 10, e) > 0      # never zero/negative


def test_calibrate_and_reprice_the_xle_call():
    """Today's XLE Jul31 60C: mark 1.00 at S=60.03 -> IV near the quoted 28.9%,
    delta in the playbook 0.45-0.60 band, and value rises with spot."""
    leg = op.calibrate_leg("XLE Jul31 60C", "C", 60.0, "2026-07-31",
                           1.00, 60.03, 1784901900)
    assert leg is not None
    assert 0.24 < leg.iv < 0.36, leg.iv
    g = leg.greeks(60.03, 1784901900)
    assert 0.45 <= g["delta"] <= 0.60, g
    assert leg.value(60.33, 1784903700) > leg.value(60.03, 1784901900)


def test_structure_value_and_greeks_net_out():
    long = op.calibrate_leg("L", "C", 60.0, "2026-07-31", 1.00, 60.03, 1784901900)
    short = op.PricedLeg("S", "C", 61.0, long.exp_epoch, long.iv, qty=-1)
    legs = [long, short]
    v = op.structure_value(legs, 60.03, 1784901900)
    assert 0 < v < 1.00                          # debit < the long alone
    g = op.structure_greeks(legs, 60.03, 1784901900)
    assert 0 < g["delta"] < long.greeks(60.03, 1784901900)["delta"]
    assert g["theta"] > long.greeks(60.03, 1784901900)["theta"]  # short leg helps


def test_validation_flags_a_bad_model():
    class B:
        def __init__(s, e, c):
            s.epoch, s.close = e, c
    leg = op.calibrate_leg("L", "C", 60.0, "2026-07-31", 1.00, 60.03, 1784901900)
    und = [B(1784901900, 60.03), B(1784903700, 60.38)]
    good = [B(1784901900, 1.00), B(1784903700, 1.20)]
    bad = [B(1784901900, 1.00), B(1784903700, 5.00)]
    assert op.validate_against_bars(leg, und, good, 1784901900)["mae"] < 0.10
    r = op.validate_against_bars(leg, und, bad, 1784901900)
    assert "POOR" in r["verdict"], r
    assert op.validate_against_bars(leg, und, [], 1784901900)["n"] == 0


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("OK  ({} option_pricing tests passed)".format(len(tests)))


if __name__ == "__main__":
    run()
