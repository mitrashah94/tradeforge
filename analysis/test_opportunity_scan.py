#!/usr/bin/env python3
"""
Plain-assert tests for analysis/opportunity_scan.py.

Python 3.9, stdlib only. No brokerage access, no TradingView MCP.

Run:
    python3 analysis/test_opportunity_scan.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.replay_session import run_ticker  # noqa: E402
from analysis.opportunity_scan import (  # noqa: E402
    default_config,
    run_variant,
    resolve_r_outcome,
    excursion,
    r3_before_1r,
    load_session,
    part_a,
    part_b,
    part_c,
    repo_path,
    SESSIONS,
)


def bar(t, o, h, l, c, v=1000.0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def test_excursion_mfe_mae_hand_checkable():
    # Synthetic CALL series: ref_close = 100.
    # Bars after: highs/lows chosen so MFE and MAE are hand-checkable.
    bars = [
        bar(0, 100, 100, 100, 100),   # break bar itself (index 0), ref_close=100
        bar(300, 100, 103, 99, 101),  # high 103 -> +3 fav, low 99 -> -1 adverse
        bar(600, 101, 102, 97, 98),   # high 102 -> +2 fav, low 97 -> -3 adverse (worst)
        bar(900, 98, 106, 98, 105),   # high 106 -> +6 fav (best), low 98 -> -2 adverse
    ]
    mfe, mae = excursion(bars, 0, 3, ref_close=100.0, is_call=True)
    assert mfe == 6.0, "expected MFE=6.0 (bar3 high 106-100), got {}".format(mfe)
    assert mae == 3.0, "expected MAE=3.0 (bar2 low 97, 100-97), got {}".format(mae)

    # PUT direction: favorable = downside, adverse = upside.
    bars_put = [
        bar(0, 50, 50, 50, 50),
        bar(300, 50, 51, 47, 48),   # low 47 -> +3 fav, high 51 -> -1 adverse
        bar(600, 48, 55, 45, 46),   # low 45 -> +5 fav (best), high 55 -> -5 adverse (worst)
    ]
    mfe_p, mae_p = excursion(bars_put, 0, 2, ref_close=50.0, is_call=False)
    assert mfe_p == 5.0, "expected MFE=5.0 for PUT, got {}".format(mfe_p)
    assert mae_p == 5.0, "expected MAE=5.0 for PUT, got {}".format(mae_p)
    print("OK: test_excursion_mfe_mae_hand_checkable")


def test_r3_before_1r_conservative_both_touched():
    # CALL: entry/ref_close=100, risk=1 -> r3=103, stop(=-1R)=99.
    # A single bar whose range touches BOTH 103 and 99 must resolve as LOSS,
    # even though naive "did the high ever reach r3" would say WIN.
    bars = [
        bar(0, 100, 100, 100, 100),         # ref bar (not scanned)
        bar(300, 100, 104, 98, 101),        # touches both r3(103) and stop(99) in one bar
    ]
    result = r3_before_1r(bars, 1, 1, ref_close=100.0, risk=1.0, is_call=True)
    assert result is False, "both-touched-in-one-bar must resolve as loss (-1R), got {}".format(result)

    # Sanity: a bar that ONLY touches r3 (never the stop) resolves True.
    bars_win = [
        bar(0, 100, 100, 100, 100),
        bar(300, 100, 104, 99.5, 103.5),   # low 99.5 never reaches stop=99; high touches r3
    ]
    result_win = r3_before_1r(bars_win, 1, 1, ref_close=100.0, risk=1.0, is_call=True)
    assert result_win is True, "expected clean win, got {}".format(result_win)

    # Sanity: a bar that ONLY touches stop resolves False.
    bars_loss = [
        bar(0, 100, 100, 100, 100),
        bar(300, 100, 102.0, 98.5, 99.0),  # high never reaches r3=103; low touches stop
    ]
    result_loss = r3_before_1r(bars_loss, 1, 1, ref_close=100.0, risk=1.0, is_call=True)
    assert result_loss is False, "expected loss, got {}".format(result_loss)
    print("OK: test_r3_before_1r_conservative_both_touched")


def test_resolve_r_outcome_conservative_both_touched():
    # Mirrors the same both-touched case for the Part B resolver, PUT side.
    # entry=50, risk=1 -> r3 = 50-3=47 (favorable=down), stop=51.
    bars = [
        bar(0, 50, 50, 50, 50),
        bar(300, 50, 51.5, 46.5, 49),  # high touches stop(51), low touches r3(47) -- both in one bar
    ]
    outcome, final_r = resolve_r_outcome(
        bars, 1, 1, entry_price=50.0, risk=1.0, is_call=False,
        stop_price=51.0, r3_price=47.0)
    assert outcome == "LOSS" and final_r == -1.0, (
        "both-touched must resolve LOSS/-1.0, got {}/{}".format(outcome, final_r))
    print("OK: test_resolve_r_outcome_conservative_both_touched")


def test_baseline_variant_matches_run_ticker_zero_qualified():
    """Regression check: run_variant(..., default_config()) must reproduce
    run_ticker's own event stream (in particular: 0 QUALIFIED events) on both
    recorded sessions -- the same result already recorded in
    backtests/session_*/events.json."""
    config = default_config()
    for session_dir in SESSIONS:
        levels_data, session_epochs, tickers, bars_by_ticker = load_session(
            repo_path(session_dir))
        for ticker in tickers:
            bars = bars_by_ticker[ticker]
            static_levels = levels_data["tickers"][ticker]

            base_events, base_levels = run_ticker(
                ticker, bars, static_levels, session_epochs,
                pre_window_reject_rearms=False)
            variant_events, variant_levels = run_variant(
                ticker, bars, static_levels, session_epochs, config)

            base_qualified = [e for e in base_events if e["event"] == "QUALIFIED"]
            variant_qualified = [e for e in variant_events if e["event"] == "QUALIFIED"]
            assert len(base_qualified) == 0, (
                "expected known baseline of 0 QUALIFIED for {} in {}, got {}".format(
                    ticker, session_dir, len(base_qualified)))
            assert len(variant_qualified) == 0, (
                "run_variant(default_config()) must also show 0 QUALIFIED for {} "
                "in {}, got {}".format(ticker, session_dir, len(variant_qualified)))

            # Stronger check: the full event sequence (event/direction/level/time/
            # reason) must match bar-for-bar between the original engine and the
            # parameterized baseline engine.
            def strip(events):
                return [(e["event"], e["direction"], e["level"], e["signal_time_ct"],
                          e.get("reason")) for e in events]

            assert strip(base_events) == strip(variant_events), (
                "run_variant(default_config()) diverged from run_ticker for {} in "
                "{}:\n  base:    {}\n  variant: {}".format(
                    ticker, session_dir, strip(base_events), strip(variant_events)))
            assert base_levels == variant_levels

    print("OK: test_baseline_variant_matches_run_ticker_zero_qualified "
          "(both sessions, all tickers)")


def test_break_bar_rvol_variant_changes_xlf_20260714_orh_outcome():
    """Top hypothesis from CLAUDE.md: XLF 2026-07-14 ORH had break-bar RVOL
    3.28 but confirmation-bar RVOL 0.74 -- 0.74 was the only gate blocking it
    under baseline. Measuring RVOL on the break bar instead of the
    confirmation bar should let this specific setup QUALIFY where the
    baseline does not."""
    levels_data, session_epochs, tickers, bars_by_ticker = load_session(
        repo_path("backtests/session_2026-07-14"))
    ticker = "XLF"
    bars = bars_by_ticker[ticker]
    static_levels = levels_data["tickers"][ticker]

    baseline_events, _ = run_variant(
        ticker, bars, static_levels, session_epochs, default_config())
    baseline_orh_qualified = [e for e in baseline_events
                              if e["event"] == "QUALIFIED" and e["level"] == "ORH"]
    assert len(baseline_orh_qualified) == 0, (
        "baseline should NOT qualify XLF ORH on 2026-07-14 (RVOL gate should "
        "block it), got {}".format(baseline_orh_qualified))

    break_bar_config = default_config()
    break_bar_config["rvol_bar"] = "break"
    variant_events, _ = run_variant(
        ticker, bars, static_levels, session_epochs, break_bar_config)
    variant_orh_qualified = [e for e in variant_events
                              if e["event"] == "QUALIFIED" and e["level"] == "ORH"]
    assert len(variant_orh_qualified) == 1, (
        "expected the break-bar-RVOL variant to qualify exactly one XLF ORH "
        "setup on 2026-07-14 (the known blocked-by-RVOL-alone case), got {}".format(
            variant_orh_qualified))
    print("OK: test_break_bar_rvol_variant_changes_xlf_20260714_orh_outcome")


def test_part_c_finds_xlf_20260714_rvol_only_block():
    """The known single-gate block: XLF 2026-07-14 ORH retest at 09:25 CT,
    blocked by RVOL alone (confirm-bar rvol 0.74 < 1.2), all other gates
    (candle color, VWAP, risk-valid) passing."""
    gate_log = part_c(SESSIONS)
    matches = [g for g in gate_log
               if g["ticker"] == "XLF" and g["date"] == "2026-07-14"
               and g["level"] == "ORH" and g["blocked_by"] == ["rvol"]]
    assert len(matches) >= 1, (
        "expected to find the known XLF 2026-07-14 ORH RVOL-only block in "
        "part_c output, found none. Full gate_log for XLF/2026-07-14: {}".format(
            [g for g in gate_log if g["ticker"] == "XLF" and g["date"] == "2026-07-14"]))
    print("OK: test_part_c_finds_xlf_20260714_rvol_only_block ({} match(es))".format(
        len(matches)))


def test_part_a_runs_and_covers_all_watch_events():
    """Part A should produce exactly one row per WATCH event emitted by the
    baseline engine across both sessions (10 breaks expected: 4 on
    2026-07-13, 6 on 2026-07-14, matching the committed events.json files)."""
    rows = part_a(SESSIONS)
    assert len(rows) == 10, "expected 10 WATCH-derived rows (4 + 6), got {}".format(len(rows))
    for r in rows:
        assert r["resolution_event"] in ("REJECT", "EXPIRED", "QUALIFIED", "DANGLING")
        assert r["mfe_ew"] >= 0.0 and r["mae_ew"] >= 0.0
        assert r["mfe_fb"] >= 0.0 and r["mae_fb"] >= 0.0
        # flat-by window is a superset of the entry-window bound in time, so
        # its excursions can only be >= the entry-window excursions.
        assert r["mfe_fb"] >= r["mfe_ew"] - 1e-9
        assert r["mae_fb"] >= r["mae_ew"] - 1e-9
    print("OK: test_part_a_runs_and_covers_all_watch_events (10 rows)")


def test_part_b_runs_and_baseline_qualifies_zero():
    results, diffs = part_b(SESSIONS)
    assert results["BASELINE"]["qualified_count"] == 0, (
        "BASELINE variant must qualify 0 trades across both sessions, matching "
        "the committed events.json files, got {}".format(results["BASELINE"]["qualified_count"]))
    assert results["BASELINE"]["net_r"] == 0
    for name, res in results.items():
        assert res["qualified_count"] == res["wins"] + res["losses"] + res["unresolved"]
    print("OK: test_part_b_runs_and_baseline_qualifies_zero ({} variants)".format(len(results)))


def main():
    test_excursion_mfe_mae_hand_checkable()
    test_r3_before_1r_conservative_both_touched()
    test_resolve_r_outcome_conservative_both_touched()
    test_baseline_variant_matches_run_ticker_zero_qualified()
    test_break_bar_rvol_variant_changes_xlf_20260714_orh_outcome()
    test_part_c_finds_xlf_20260714_rvol_only_block()
    test_part_a_runs_and_covers_all_watch_events()
    test_part_b_runs_and_baseline_qualifies_zero()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
