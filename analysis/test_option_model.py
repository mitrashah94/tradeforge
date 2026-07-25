#!/usr/bin/env python3
"""test_option_model.py -- plain-assert tests for option_model.py.

Locks the 2026-07-24 XLE QUALIFIED reconstruction (the first live signal since
the four-track engine went live) as a regression: the underlying ran +3R while a
0.50-delta call made ~+0.8 account-R, and a 60/61 debit spread captured almost
nothing. Run: python3 analysis/test_option_model.py  (must print OK).
Stdlib only; no network; no fixtures required (bars are inline).
"""
import option_model as om

# --- 2026-07-24 XLE, entry bar closes 60.03 at epoch 1784901900 (09:10 CT) ---
SIG_EPOCH = 1784901900
UND = [  # epoch, o, h, l, c  (only highs matter for the CALL crosses)
    om.Bar(1784901900, 60.01, 60.05, 59.94, 60.03),
    om.Bar(1784902200, 60.04, 60.145, 60.00, 60.14),   # 1R 60.13 first touch
    om.Bar(1784902500, 60.14, 60.195, 60.135, 60.155),
    om.Bar(1784902800, 60.17, 60.24, 60.135, 60.24),   # 2R 60.23 first touch
    om.Bar(1784903100, 60.25, 60.255, 60.13, 60.17),
    om.Bar(1784903400, 60.19, 60.30, 60.18, 60.30),
    om.Bar(1784903700, 60.295, 60.40, 60.28, 60.38),   # 3R 60.33 first touch
]
C60 = [  # XLE Jul31 60C
    om.Bar(1784901900, 0.97, 1.01, 0.97, 1.00),
    om.Bar(1784902200, 1.01, 1.07, 0.99, 1.05),
    om.Bar(1784902500, 1.05, 1.09, 1.05, 1.08),
    om.Bar(1784902800, 1.08, 1.12, 1.06, 1.12),
    om.Bar(1784903100, 1.12, 1.13, 1.08, 1.08),
    om.Bar(1784903400, 1.08, 1.12, 1.08, 1.12),
    om.Bar(1784903700, 1.12, 1.21, 1.12, 1.20),
]
C61 = [  # XLE Jul31 61C (short leg of the spread)
    om.Bar(1784901900, 0.56, 0.58, 0.55, 0.57),
    om.Bar(1784902200, 0.57, 0.57, 0.57, 0.57),
    om.Bar(1784902800, 0.64, 0.64, 0.64, 0.64),
    om.Bar(1784903700, 0.67, 0.72, 0.67, 0.72),
]

C60_CT = om.Contract("XLE Jul31 60C", "C", 60.0, "2026-07-31", 7, 0.50,
                     0.81, 0.85, oi=3099, volume=6941, bars_csv="XLE_260731C60.csv")
C61_CT = om.Contract("XLE Jul31 61C", "C", 61.0, "2026-07-31", 7, 0.34,
                     0.47, 0.51, oi=1200, volume=500, bars_csv="XLE_260731C61.csv")


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_chart_r_and_ladder():
    sig = om.Signal("XLE", "CALL", 60.03, 59.93, SIG_EPOCH)
    assert _approx(sig.chart_r, 0.10), sig.chart_r
    lad = sig.r_ladder(5)
    assert _approx(lad[0], 60.13) and _approx(lad[2], 60.33), lad


def test_first_cross():
    assert om.first_cross_epoch(UND, 60.13, "CALL", SIG_EPOCH) == 1784902200
    assert om.first_cross_epoch(UND, 60.23, "CALL", SIG_EPOCH) == 1784902800
    assert om.first_cross_epoch(UND, 60.33, "CALL", SIG_EPOCH) == 1784903700
    assert om.first_cross_epoch(UND, 99.0, "CALL", SIG_EPOCH) is None


def test_feasibility_and_sizing():
    fe = om.feasibility(0.50, 0.10, 0.04)
    assert _approx(fe["loss_per_contract"], 9.0), fe
    assert fe["feasible"] is True
    # $9 per contract -> the $25 gate already permits 2 contracts
    assert om.risk_normalized_contracts(9.0) == 2
    assert om.risk_normalized_contracts(26.0) == 1   # never below 1


def test_selection_prefers_the_only_clean_contract():
    cands = [
        C60_CT,                                              # dte7 delta.50 spr.04 -> passes
        om.Contract("XLE Aug7 60C", "C", 60.0, "2026-08-07", 14, 0.48, 1.20, 1.30, oi=477),
        om.Contract("XLE Aug14 60C", "C", 60.0, "2026-08-14", 21, 0.49, 1.45, 1.58, oi=269),
        om.Contract("XLE Jul31 59C", "C", 59.0, "2026-07-31", 7, 0.62, 1.33, 1.42, oi=1264),
    ]
    chosen, audit = om.select_contract(cands)
    assert chosen is not None and chosen.label == "XLE Jul31 60C", chosen
    fails = {a["label"]: a["reasons"] for a in audit if not a["passed"]}
    assert "wide_spread" in fails["XLE Aug7 60C"]
    assert "delta_out_of_band" in fails["XLE Jul31 59C"]


def test_frame_reproduces_the_day():
    sig = om.Signal("XLE", "CALL", 60.03, 59.93, SIG_EPOCH)
    f = om.frame_signal(sig, UND, C60, C60_CT, C61, C61_CT, exits=(1, 2, 3))
    assert f["risk_normalized_contracts"] == 2
    by_r = {r["r_multiple"]: r for r in f["exits"]}
    # single long call: monotonic, ~+0.8R at the 3R exit
    assert _approx(by_r[1]["long_1x"]["account_r"], 0.20), by_r[1]["long_1x"]
    assert _approx(by_r[2]["long_1x"]["account_r"], 0.48), by_r[2]["long_1x"]
    assert _approx(by_r[3]["long_1x"]["net"], 20.0), by_r[3]["long_1x"]
    assert _approx(by_r[3]["long_1x"]["account_r"], 0.80), by_r[3]["long_1x"]
    # 2 contracts doubles it, still inside the $25 gate
    assert _approx(by_r[3]["long_risknorm"]["account_r"], 1.60), by_r[3]["long_risknorm"]
    # debit spread captured almost nothing at the 3R cross, and caps at width
    sp = by_r[3]["spread_1x"]
    assert _approx(sp["net"], 5.0), sp
    assert _approx(sp["account_r"], 0.20), sp
    assert _approx(sp["max_value"], 100.0), sp


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("OK  ({} option_model tests passed)".format(len(tests)))


if __name__ == "__main__":
    run()
