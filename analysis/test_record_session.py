#!/usr/bin/env python3
"""Self-contained tests for analysis/record_session.py (plain asserts, no
pytest). Python 3.9 stdlib only; no brokerage access.

Run: python3 /Users/mitrashah/DayTrading/analysis/test_record_session.py
"""

import json
import os
import sys
import tempfile
import shutil

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ANALYSIS_DIR)
sys.path.insert(0, ANALYSIS_DIR)

import record_session as rsn  # noqa: E402

_passed = []


def check(name, cond, detail=""):
    assert cond, "{} FAILED {}".format(name, detail)
    _passed.append(name)
    print("ok - {}".format(name))


# -----------------------------------------------------------------------------
# Test 1: verdict derivation for all four confusion-matrix quadrants.
# -----------------------------------------------------------------------------
def test_verdict_derivation_all_quadrants():
    engine_qualified = {"watch": 1, "qualified": 1, "reject": 0,
                        "expired": 0, "invalidated": 0,
                        "pre_window_rejects": 0}
    engine_none = {"watch": 4, "qualified": 0, "reject": 3,
                   "expired": 1, "invalidated": 0,
                   "pre_window_rejects": 1}
    engine_unknown = dict(rsn.ENGINE_UNKNOWN)

    check("qualified + traded -> CORRECT_TRADE",
          rsn.derive_verdict(engine_qualified, 1) == "CORRECT_TRADE")
    check("qualified + no trade -> MISSED_SIGNAL",
          rsn.derive_verdict(engine_qualified, 0) == "MISSED_SIGNAL")
    check("no qualified + traded -> OFF_PLAN_TRADE",
          rsn.derive_verdict(engine_none, 1) == "OFF_PLAN_TRADE")
    check("no qualified + no trade -> CORRECT_NO_TRADE",
          rsn.derive_verdict(engine_none, 0) == "CORRECT_NO_TRADE")
    # Unknown (--no-events) engine counts must be treated as "no qualified",
    # never as a free pass to claim CORRECT_TRADE/MISSED_SIGNAL.
    check("unknown engine + traded -> OFF_PLAN_TRADE (never asserts a "
          "QUALIFIED that can't be proven)",
          rsn.derive_verdict(engine_unknown, 1) == "OFF_PLAN_TRADE")
    check("unknown engine + no trade -> CORRECT_NO_TRADE",
          rsn.derive_verdict(engine_unknown, 0) == "CORRECT_NO_TRADE")


# -----------------------------------------------------------------------------
# Test 2: duplicate-date refusal (and --force override).
# -----------------------------------------------------------------------------
def test_duplicate_session_date_refused():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)
    try:
        rec1, errs1 = rsn.build_record(
            no_events=True, session_date="2026-07-01",
            trades_taken=0, off_plan_actions=0,
            believed_state="n/a", actual_state="n/a", notes="first")
        check("build_record: no build errors for rec1", errs1 == [], str(errs1))
        assert rsn.validate_record(rec1) == []
        with open(path, "a") as fh:
            fh.write(json.dumps(rec1) + "\n")

        check("existing_session_dates finds the recorded date",
              "2026-07-01" in rsn.existing_session_dates(path))

        # Simulate the CLI duplicate-refusal path directly.
        dup_refused = "2026-07-01" in rsn.existing_session_dates(path)
        check("duplicate session_date detected before append", dup_refused)

        # --force path: main() should still succeed with --force.
        argv = ["--no-events", "--session-date", "2026-07-01",
                "--trades-taken", "0", "--off-plan-actions", "0",
                "--believed-state", "n/a", "--actual-state", "n/a",
                "--notes", "second (forced)",
                "--sessions", path, "--force"]
        rc = rsn.main(argv)
        check("main() with --force succeeds despite duplicate date", rc == 0)

        # Without --force it must be refused (exit code 3).
        argv_no_force = ["--no-events", "--session-date", "2026-07-01",
                         "--trades-taken", "0", "--off-plan-actions", "0",
                         "--believed-state", "n/a", "--actual-state", "n/a",
                         "--notes", "third (should refuse)",
                         "--sessions", path]
        rc2 = rsn.main(argv_no_force)
        check("main() without --force refuses duplicate date (exit 3)",
              rc2 == 3)

        with open(path) as fh:
            lines = [l for l in fh.read().splitlines() if l.strip()]
        check("exactly 2 lines written (original + forced), refusal added none",
              len(lines) == 2, "got {} lines".format(len(lines)))
    finally:
        if os.path.exists(path):
            os.remove(path)


# -----------------------------------------------------------------------------
# Test 3: pre_window_rejects counted correctly from a synthetic events list.
# -----------------------------------------------------------------------------
def test_pre_window_rejects_from_synthetic_events():
    events = [
        {
            "event": "WATCH", "event_id": "X-1", "ticker": "XLF",
            "timeframe": "5", "setup_type": "A_break_retest",
            "direction": "CALL", "level": "PDH", "level_price": 56.03,
            "signal_time_ct": "2026-07-13 08:30", "vwap": 56.07, "rvol": None,
        },
        {
            # Before the 08:45 entry window opens -> pre-window reject.
            "event": "REJECT", "event_id": "X-2", "ticker": "XLF",
            "timeframe": "5", "setup_type": "A_break_retest",
            "direction": "CALL", "level": "PDH", "level_price": 56.03,
            "signal_time_ct": "2026-07-13 08:40", "vwap": 56.05, "rvol": 3.49,
            "reason": "failed_hold_below_level",
        },
        {
            "event": "WATCH", "event_id": "X-3", "ticker": "IWM",
            "timeframe": "5", "setup_type": "B_breakdown_bounce",
            "direction": "PUT", "level": "ORL", "level_price": 294.37,
            "signal_time_ct": "2026-07-13 09:05", "vwap": 295.04, "rvol": 1.58,
        },
        {
            # Inside the entry window -> NOT a pre-window reject.
            "event": "REJECT", "event_id": "X-4", "ticker": "IWM",
            "timeframe": "5", "setup_type": "B_breakdown_bounce",
            "direction": "PUT", "level": "ORL", "level_price": 294.37,
            "signal_time_ct": "2026-07-13 09:10", "vwap": 295.00, "rvol": 1.11,
            "reason": "failed_hold_above_level",
        },
    ]
    counts = rsn.compute_engine_counts(events)
    check("watch count is 2", counts["watch"] == 2, str(counts))
    check("reject count is 2", counts["reject"] == 2, str(counts))
    check("qualified count is 0", counts["qualified"] == 0, str(counts))
    check("expired count is 0", counts["expired"] == 0, str(counts))
    check("invalidated count is 0", counts["invalidated"] == 0, str(counts))
    check("exactly 1 of the 2 REJECTs is pre-window (08:40 < 08:45)",
          counts["pre_window_rejects"] == 1, str(counts))


# -----------------------------------------------------------------------------
# Test 4: engine counts + ticker derivation from the real events.json.
# -----------------------------------------------------------------------------
def test_engine_counts_and_tickers_from_real_events_json():
    session_dir = os.path.join(ROOT, "backtests", "session_2026-07-13")
    check("real session_2026-07-13 events.json exists",
          os.path.exists(os.path.join(session_dir, "events.json")))
    events = rsn.load_events(session_dir)
    engine = rsn.compute_engine_counts(events)
    check("real session: watch == 4", engine["watch"] == 4, str(engine))
    check("real session: qualified == 0", engine["qualified"] == 0, str(engine))
    check("real session: reject == 3", engine["reject"] == 3, str(engine))
    check("real session: expired == 1", engine["expired"] == 1, str(engine))
    check("real session: invalidated == 0", engine["invalidated"] == 0, str(engine))
    check("real session: pre_window_rejects == 1 (XLF PDH REJECT at 08:40)",
          engine["pre_window_rejects"] == 1, str(engine))

    tickers = rsn.derive_tickers(session_dir, events)
    check("real session: tickers derived as XLE, XLF, IWM (levels.json order)",
          tickers == ["XLE", "XLF", "IWM"], str(tickers))

    check("engine_had_qualified is False for this session",
          rsn.engine_had_qualified(engine) is False)


# -----------------------------------------------------------------------------
# Test 5: build_record end-to-end for the real 2026-07-13 no-trade day.
# -----------------------------------------------------------------------------
def test_build_record_correct_no_trade_day():
    session_dir = os.path.join(ROOT, "backtests", "session_2026-07-13")
    rec, errs = rsn.build_record(
        session_dir=session_dir,
        trades_taken=0, off_plan_actions=0,
        believed_state="two setups qualified",
        actual_state="zero qualified",
        notes="hesitation was CORRECT; WATCH mistaken for QUALIFIED")
    check("build_record: no build errors", errs == [], str(errs))
    check("record validates clean", rsn.validate_record(rec) == [],
          str(rsn.validate_record(rec)))
    check("session_date auto-derived from dir name",
          rec["session_date"] == "2026-07-13")
    check("verdict auto-derived as CORRECT_NO_TRADE",
          rec["verdict"] == "CORRECT_NO_TRADE")
    check("net_r defaults to null when not provided", rec["net_r"] is None)


# -----------------------------------------------------------------------------
# Test 6: build_record for a --no-events off-plan day.
# -----------------------------------------------------------------------------
def test_build_record_no_events_off_plan_day():
    rec, errs = rsn.build_record(
        no_events=True, session_date="2026-07-12",
        trades_taken=1, off_plan_actions=5, net_r=-1.04,
        believed_state="unknown", actual_state="unknown",
        notes="chased a failing ORH breakout with no qualified setup; "
              "2 contracts, 4 DTE, also bought stock")
    check("build_record: no build errors", errs == [], str(errs))
    check("record validates clean", rsn.validate_record(rec) == [],
          str(rsn.validate_record(rec)))
    check("engine counts are all null (unknown)",
          all(v is None for v in rec["engine"].values()), str(rec["engine"]))
    check("replay_dir is null", rec["replay_dir"] is None)
    check("verdict auto-derived as OFF_PLAN_TRADE",
          rec["verdict"] == "OFF_PLAN_TRADE")
    check("net_r recorded as -1.04", rec["net_r"] == -1.04)


# -----------------------------------------------------------------------------
# Test 7: schema validation catches bad input.
# -----------------------------------------------------------------------------
def test_validate_record_catches_bad_input():
    bad = {
        "session_date": "13-07-2026",  # wrong format
        "recorded_ct": "not-a-time",
        "tickers": "XLF",  # should be a list
        "replay_dir": None,
        "engine": {"watch": 1},  # missing keys
        "trader": {"trades_taken": -1, "off_plan_actions": 0,
                  "believed_state": "", "actual_state": "x"},
        "verdict": "MAYBE",
        "net_r": "not-a-number",
        "notes": "ok",
    }
    errors = rsn.validate_record(bad)
    check("bad session_date flagged",
          any("session_date" in e for e in errors))
    check("bad recorded_ct flagged", any("recorded_ct" in e for e in errors))
    check("bad tickers flagged", any("tickers" in e for e in errors))
    check("missing engine keys flagged",
          any("engine.qualified" in e for e in errors))
    check("negative trades_taken flagged",
          any("trader.trades_taken" in e for e in errors))
    check("empty believed_state flagged",
          any("trader.believed_state" in e for e in errors))
    check("bad verdict flagged", any("verdict" in e for e in errors))
    check("bad net_r flagged", any("net_r" in e for e in errors))


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def main():
    tests = [
        test_verdict_derivation_all_quadrants,
        test_duplicate_session_date_refused,
        test_pre_window_rejects_from_synthetic_events,
        test_engine_counts_and_tickers_from_real_events_json,
        test_build_record_correct_no_trade_day,
        test_build_record_no_events_off_plan_day,
        test_validate_record_catches_bad_input,
    ]
    for t in tests:
        t()
    print("\n{} checks passed.".format(len(_passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
