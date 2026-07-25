#!/usr/bin/env python3
"""test_structure_lab.py -- plain-assert tests for structure_lab.py.

The important ones are the PRECEDENCE tests: they prove the walker exits on the
rule strategy.md sec 8 says fires first, not on whichever is most flattering.
Run: python3 analysis/test_structure_lab.py   (must print OK). Stdlib only."""
import option_pricing as op
import structure_lab as sl
from option_model import Bar, Signal

SIG_EPOCH = 1784901900          # 2026-07-24 bar labeled 14:05 UTC / 09:05 CT
SIG = Signal("XLE", "CALL", 60.03, 59.93, SIG_EPOCH)   # chart_R = 0.10, 3R = 60.33


def _leg():
    return op.calibrate_leg("XLE Jul31 60C", "C", 60.0, "2026-07-31",
                            1.00, 60.03, SIG_EPOCH)


# real 2026-07-24 path after the signal bar
REAL = [
    Bar(1784902200, 60.04, 60.145, 60.00, 60.14),
    Bar(1784902500, 60.14, 60.195, 60.135, 60.155),
    Bar(1784902800, 60.17, 60.24, 60.135, 60.24),
    Bar(1784903100, 60.25, 60.255, 60.13, 60.17),
    Bar(1784903400, 60.19, 60.30, 60.18, 60.30),
    Bar(1784903700, 60.295, 60.40, 60.28, 60.38),      # crosses 3R 60.33
]


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_real_day_exits_on_the_r_target():
    r = sl.walk_structure([_leg()], REAL, SIG, 1)
    assert r["exit_reason"] == "r_target", r
    assert r["exit_epoch"] == 1784903700, r
    assert _close(r["exit_underlying"], 60.33, 1e-6), r
    assert r["held_minutes"] == 30, r
    assert 0.35 < r["account_r"] < 0.75, r     # ~+0.5R, NOT the chart's +3R
    assert r["max_adverse_r"] >= -0.2, r       # stop was never threatened


def test_priced_at_target_is_stricter_than_bar_close():
    """The 3R bar CLOSED at 60.38, above the 60.33 target. Pricing the exit at
    the target (a resting limit) must pay LESS than pricing it at that close --
    this is the v1 overstatement the walker fixes."""
    leg = _leg()
    at_target = leg.value(60.33, 1784903700)
    at_close = leg.value(60.38, 1784903700)
    assert at_close > at_target, (at_target, at_close)


def test_structural_stop_beats_a_same_bar_target():
    """A bar that both tags 3R and CLOSES through the stop must exit on the
    stop -- adverse branch wins when intrabar order is unknowable."""
    bars = [Bar(1784902200, 60.04, 60.40, 59.80, 59.90)]
    r = sl.walk_structure([_leg()], bars, SIG, 1)
    assert r["exit_reason"] in ("structural_stop", "loss_gate"), r
    assert r["account_r"] < 0, r


def test_loss_gate_fires_before_the_stop_close():
    """Deep adverse excursion inside the bar trips the -$25 gate even though the
    bar closes back above the structural stop (strategy.md sec 8: whichever first)."""
    bars = [Bar(1784902200, 60.03, 60.05, 59.30, 60.00)]
    r = sl.walk_structure([_leg()], bars, SIG, 1)
    assert r["exit_reason"] == "loss_gate", r
    assert r["net"] <= -25.0 + 1e-6, r


def test_stagnation_stop_when_1r_never_comes():
    bars = [Bar(SIG_EPOCH + 300 * i, 60.03, 60.05, 60.00, 60.02)
            for i in range(1, 16)]              # 75 min of nothing, never 1R
    r = sl.walk_structure([_leg()], bars, SIG, 1,
                          sl.ExitRules(stagnation_min=60))
    assert r["exit_reason"] == "stagnation", r
    assert r["held_minutes"] >= 60, r


def test_stagnation_clock_stops_once_1r_is_reached():
    bars = [Bar(1784902200, 60.03, 60.15, 60.00, 60.14)]      # tags 1R 60.13
    bars += [Bar(1784902200 + 300 * i, 60.14, 60.16, 60.10, 60.14)
             for i in range(1, 16)]
    r = sl.walk_structure([_leg()], bars, SIG, 1, sl.ExitRules(stagnation_min=60))
    assert r["exit_reason"] != "stagnation", r


def test_risk_normalized_size_scales_pnl_linearly():
    one = sl.walk_structure([_leg()], REAL, SIG, 1)
    two = sl.walk_structure([_leg()], REAL, SIG, 2)
    assert _close(two["net"], one["net"] * 2, 0.02), (one["net"], two["net"])
    assert _close(two["cost"], one["cost"] * 2, 0.02)


def test_vertical_caps_upside_and_underperforms_on_this_path():
    """The sec 3b claim, now measured: on a small move the short leg rises with
    the long, so the vertical captures materially less than the outright."""
    leg = _leg()
    outright = sl.walk_structure([leg], REAL, SIG, 1)
    vert = sl.walk_structure(sl.synth_vertical(leg, 61.0), REAL, SIG, 1)
    assert vert["account_r"] < outright["account_r"], (vert, outright)
    assert vert["net"] > 0                       # it does win, just far less


def test_compare_runs_every_structure_over_one_path():
    c = sl.compare(SIG, REAL, _leg(), widths=(1.0, 2.0))
    names = [s["name"] for s in c["structures"]]
    assert any("long" in n for n in names) and any("vertical" in n for n in names)
    assert all("exit_reason" in s for s in c["structures"]), names
    # every structure must exit on the SAME rule set, i.e. same path semantics
    assert {s["exit_reason"] for s in c["structures"]} == {"r_target"}, c
    assert c["risk_normalized_contracts"] >= 1


def test_no_bars_after_signal_is_reported_not_faked():
    r = sl.walk_structure([_leg()], [Bar(SIG_EPOCH - 300, 60, 60, 60, 60)], SIG, 1)
    assert "error" in r, r


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("OK  ({} structure_lab tests passed)".format(len(tests)))


if __name__ == "__main__":
    run()
