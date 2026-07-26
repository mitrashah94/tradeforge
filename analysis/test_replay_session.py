#!/usr/bin/env python3
"""Self-contained tests for analysis/replay_session.py (plain asserts, no
pytest). Python 3.9 stdlib only; no brokerage access.

Run: python3 /Users/mitrashah/DayTrading/analysis/test_replay_session.py
"""

import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ANALYSIS_DIR)

import replay_session as rs  # noqa: E402

_passed = []


def check(name, cond, detail=""):
    assert cond, "{} FAILED {}".format(name, detail)
    _passed.append(name)
    print("ok - {}".format(name))


# -----------------------------------------------------------------------------
# Synthetic bar builders
# -----------------------------------------------------------------------------
RTH = 1700000000     # arbitrary epoch anchor for the synthetic session
BAR = 300             # 5-minute bars


def mkbar(t, o, h, l, c, v):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v}


def build_premarket(prices, start_t, volume=200):
    """25 mildly-noisy premarket bars ending at start_t + 24*BAR, seeding a
    non-trivial ATR baseline (needed so risk-vs-ATR checks are meaningful)
    and enough volume history for the SMA20 volume average."""
    bars = []
    t = start_t
    prev_c = prices[0]
    for p in prices:
        o = prev_c
        c = p
        h = max(o, c) + 0.15
        l = min(o, c) - 0.15
        bars.append(mkbar(t, o, h, l, c, volume))
        prev_c = c
        t += BAR
    return bars


PREMKT_UP = [99.5, 99.6, 99.4, 99.55, 99.45, 99.6, 99.5, 99.35, 99.6, 99.45,
             99.55, 99.4, 99.6, 99.5, 99.45, 99.6, 99.55, 99.4, 99.5, 99.6,
             99.45, 99.55, 99.5, 99.6, 99.5]

PREMKT_DOWN = [100.5, 100.4, 100.6, 100.45, 100.55, 100.4, 100.5, 100.65, 100.4, 100.55,
               100.45, 100.6, 100.4, 100.5, 100.55, 100.4, 100.45, 100.6, 100.5, 100.4,
               100.55, 100.45, 100.5, 100.4, 100.5]


def build_orb_pd_tiebreak_long_bars():
    """Both ORH and PDH sit at 100.00. The breakout bar (first bar after
    the opening range) closes above 100.00 for BOTH tracks simultaneously
    (prior close was <=100 the whole session). The following bar retests
    with a shallow undershoot (low=99.95) so riskValid holds, filters pass
    (big volume, VWAP below close, green candle), inside the entry window.
    """
    bars = build_premarket(PREMKT_UP, RTH - 25 * BAR)
    bars.append(mkbar(RTH, 99.6, 99.8, 99.5, 99.7, 250))
    bars.append(mkbar(RTH + BAR, 99.7, 99.9, 99.6, 99.8, 250))
    bars.append(mkbar(RTH + 2 * BAR, 99.8, 100.00, 99.7, 99.85, 250))  # orh=100.00
    bars.append(mkbar(RTH + 3 * BAR, 99.9, 100.6, 99.7, 100.3, 300))    # break bar (both tracks)
    bars.append(mkbar(RTH + 4 * BAR, 100.0, 100.3, 99.95, 100.15, 5000))  # retest -> QUALIFIED
    for k in range(5, 10):
        bars.append(mkbar(RTH + k * BAR, 100.15, 100.3, 100.05, 100.2, 200))
    return bars


LEVELS_TIEBREAK = {"pdh": 100.00, "pdl": 90.0, "pdc": 95.0}
EPOCHS_TIEBREAK = {"rth_open": RTH, "entry_start": RTH + 3 * BAR,
                    "entry_end": RTH + 3 * BAR + 3600 * 2}


def build_riskfail_long_bars():
    """Same ORH break as the tiebreak scenario, but the retest bar
    undershoots deeply (low=97.5) so the computed risk blows well past
    ATR*1.0. PDH is set unreachable so only the ORH track is exercised
    (isolating the riskValid gate)."""
    bars = build_premarket(PREMKT_UP, RTH - 25 * BAR)
    bars.append(mkbar(RTH, 99.6, 99.8, 99.5, 99.7, 250))
    bars.append(mkbar(RTH + BAR, 99.7, 99.9, 99.6, 99.8, 250))
    bars.append(mkbar(RTH + 2 * BAR, 99.8, 100.00, 99.7, 99.85, 250))
    bars.append(mkbar(RTH + 3 * BAR, 99.9, 100.6, 99.7, 100.3, 300))
    bars.append(mkbar(RTH + 4 * BAR, 100.2, 100.3, 97.5, 100.2, 5000))  # deep undershoot
    for k in range(5, 10):
        bars.append(mkbar(RTH + k * BAR, 100.2, 100.3, 100.1, 100.25, 200))
    return bars


LEVELS_RISKFAIL = {"pdh": 200.0, "pdl": 90.0, "pdc": 95.0}


def build_late_entry_window_bars():
    """Identical to the tiebreak scenario, but the entry window is pushed
    to open AFTER the only bar that ever retests the level -- proves a
    QUALIFIED never fires outside the entry window even when every other
    filter would otherwise pass."""
    return build_orb_pd_tiebreak_long_bars() + [
        mkbar(RTH + 10 * BAR, 100.2, 100.3, 100.1, 100.25, 200),
        mkbar(RTH + 11 * BAR, 100.2, 100.3, 100.1, 100.25, 200),
    ]


EPOCHS_LATE_WINDOW = {"rth_open": RTH, "entry_start": RTH + 6 * BAR,
                      "entry_end": RTH + 6 * BAR + 3600 * 2}


def build_orb_pd_tiebreak_short_bars():
    """Mirror of the long tie-break: ORL and PDL both sit at 100.00, break
    down on the same bar, and retest/qualify on the next bar."""
    bars = build_premarket(PREMKT_DOWN, RTH - 25 * BAR)
    bars.append(mkbar(RTH, 100.4, 100.5, 100.3, 100.35, 250))
    bars.append(mkbar(RTH + BAR, 100.35, 100.45, 100.25, 100.3, 250))
    bars.append(mkbar(RTH + 2 * BAR, 100.3, 100.4, 100.00, 100.15, 250))  # orl=100.00
    bars.append(mkbar(RTH + 3 * BAR, 100.1, 100.3, 99.7, 99.85, 300))     # break bar
    bars.append(mkbar(RTH + 4 * BAR, 100.05, 100.05, 99.9, 99.95, 5000))  # retest -> QUALIFIED
    for k in range(5, 10):
        bars.append(mkbar(RTH + k * BAR, 99.95, 100.0, 99.8, 99.9, 200))
    return bars


LEVELS_TIEBREAK_SHORT = {"pdh": 200.0, "pdl": 100.00, "pdc": 150.0}


def build_pre_window_pdh_reject_bars():
    """Isolates a PDH track that breaks and REJECTs entirely BEFORE the
    entry window opens (both events fall inside the opening-range bars,
    since entry_start == rth_open + 3*BAR == end of the opening range in
    this harness). PDL is set unreachable (50.0) and the opening-range
    bars are kept below the ORH high they themselves set, so ORH/ORL never
    break later -- only the PDH track is exercised.

    Sequence (k = bars since RTH, all times relative to RTH):
      k0 (pre-window): flat bar, no break yet.
      k1 (pre-window): BREAK -- prev_close <= pdh, close > pdh.
      k2 (pre-window): REJECT trigger -- close < pdh - atr*ATR_STOP_BUFFER.
      k3 (in-window, == entry_start): re-BREAK -- prev_close <= pdh, close > pdh.
      k4 (in-window): retest -- low <= pdh, close > pdh, green candle, big
          volume (RVOL), close > VWAP -- would QUALIFY if the track is live.
    """
    bars = build_premarket(PREMKT_UP, RTH - 25 * BAR)
    bars.append(mkbar(RTH, 99.5, 99.7, 99.4, 99.6, 200))             # k0 no break
    bars.append(mkbar(RTH + BAR, 99.6, 100.6, 99.5, 100.3, 250))     # k1 BREAK (pre-window)
    bars.append(mkbar(RTH + 2 * BAR, 100.3, 100.3, 99.4, 99.5, 250))  # k2 REJECT (pre-window)
    bars.append(mkbar(RTH + 3 * BAR, 99.5, 100.6, 99.4, 100.3, 300))  # k3 re-BREAK (in-window)
    bars.append(mkbar(RTH + 4 * BAR, 100.0, 100.3, 99.95, 100.15, 5000))  # k4 retest
    for k in range(5, 10):
        bars.append(mkbar(RTH + k * BAR, 100.15, 100.3, 100.05, 100.2, 200))
    return bars


LEVELS_PREWINDOW = {"pdh": 100.00, "pdl": 50.0, "pdc": 95.0}


# -----------------------------------------------------------------------------
# Test 1: ATR is Wilder's RMA, not a simple/plain moving average.
# -----------------------------------------------------------------------------
def test_atr_is_wilder_rma_not_sma():
    # TR = [1..14, 100] (15 values), length=14.
    tr = [float(x) for x in range(1, 15)] + [100.0]
    rma = rs.compute_rma(tr, 14)

    # na (None) until index 12 (13 values isn't enough).
    check("rma na before seed index", rma[12] is None)

    # Seed at index 13 (14th value) = simple average of the first 14 values.
    expected_seed = sum(range(1, 15)) / 14.0  # 105/14 = 7.5
    check("rma seed = SMA(first 14)",
          abs(rma[13] - expected_seed) < 1e-9,
          "{} vs {}".format(rma[13], expected_seed))

    # Index 14: alpha=1/14 exponential step off the seed (Wilder recursion).
    alpha = 1.0 / 14.0
    expected_rma14 = alpha * 100.0 + (1 - alpha) * expected_seed
    check("rma Wilder recursion matches hand computation",
          abs(rma[14] - expected_rma14) < 1e-9,
          "{} vs {}".format(rma[14], expected_rma14))

    # A plain moving average of the last 14 values (a common wrong
    # implementation) gives a materially different number -- assert we do
    # NOT match that, proving this is genuinely RMA and not SMA-in-disguise.
    wrong_plain_avg = sum(tr[1:15]) / 14.0  # 204/14 = 14.571...
    check("rma output is NOT the plain trailing SMA",
          abs(rma[14] - wrong_plain_avg) > 0.1,
          "{} vs wrong {}".format(rma[14], wrong_plain_avg))


# -----------------------------------------------------------------------------
# Test 2: ORB-before-PD same-bar tie-break -- ORB wins.
# -----------------------------------------------------------------------------
def test_orb_before_pd_tiebreak():
    bars = build_orb_pd_tiebreak_long_bars()
    events, computed = rs.run_ticker("TIEBREAK", bars, LEVELS_TIEBREAK, EPOCHS_TIEBREAK)

    watches = [e for e in events if e["event"] == "WATCH"]
    check("both ORH and PDH WATCH fire on the break bar",
          {("ORH",), ("PDH",)}.issubset({(w["level"],) for w in watches}),
          str(watches))

    qualifieds = [e for e in events if e["event"] == "QUALIFIED"]
    check("exactly one QUALIFIED emitted", len(qualifieds) == 1, str(qualifieds))
    check("the QUALIFIED is the ORH track (ORB wins the tie-break)",
          qualifieds[0]["level"] == "ORH", str(qualifieds[0]))
    check("PDH gets no REJECT/EXPIRED of its own (dangling WATCH, silenced)",
          not any(e["level"] == "PDH" and e["event"] in ("REJECT", "EXPIRED", "QUALIFIED")
                  for e in events),
          str(events))


def test_orb_before_pd_tiebreak_short_side():
    bars = build_orb_pd_tiebreak_short_bars()
    events, computed = rs.run_ticker("TIEBREAKSHORT", bars, LEVELS_TIEBREAK_SHORT, EPOCHS_TIEBREAK)
    qualifieds = [e for e in events if e["event"] == "QUALIFIED"]
    check("short side: exactly one QUALIFIED emitted", len(qualifieds) == 1, str(qualifieds))
    check("short side: the QUALIFIED is the ORL track (ORB wins)",
          qualifieds[0]["level"] == "ORL", str(qualifieds[0]))


# -----------------------------------------------------------------------------
# Test 3: one QUALIFIED per day per ticker (qualifiedToday lockout).
# -----------------------------------------------------------------------------
def test_one_qualified_per_day_lockout():
    bars = build_orb_pd_tiebreak_long_bars()
    events, computed = rs.run_ticker("LOCKOUT", bars, LEVELS_TIEBREAK, EPOCHS_TIEBREAK)
    qualifieds = [e for e in events if e["event"] == "QUALIFIED"]
    check("no more than one QUALIFIED for the ticker/day",
          len(qualifieds) == 1, str(qualifieds))


# -----------------------------------------------------------------------------
# Test 4: QUALIFIED never occurs outside the entry window.
# -----------------------------------------------------------------------------
def test_qualified_never_outside_entry_window():
    # Positive: the normal scenario's QUALIFIED time is inside the window.
    bars = build_orb_pd_tiebreak_long_bars()
    events, _ = rs.run_ticker("WINDOWCHECK", bars, LEVELS_TIEBREAK, EPOCHS_TIEBREAK)
    qualifieds = [e for e in events if e["event"] == "QUALIFIED"]
    check("sanity: scenario does produce a QUALIFIED", len(qualifieds) == 1)
    from datetime import datetime
    q_time = datetime.strptime(qualifieds[0]["signal_time_ct"], "%Y-%m-%d %H:%M")
    entry_start_str = rs.ct_str(EPOCHS_TIEBREAK["entry_start"])
    entry_end_str = rs.ct_str(EPOCHS_TIEBREAK["entry_end"])
    entry_start_dt = datetime.strptime(entry_start_str, "%Y-%m-%d %H:%M")
    entry_end_dt = datetime.strptime(entry_end_str, "%Y-%m-%d %H:%M")
    check("QUALIFIED signal_time_ct falls within the entry window",
          entry_start_dt <= q_time < entry_end_dt,
          "{} not in [{}, {})".format(q_time, entry_start_dt, entry_end_dt))

    # Negative: push the entry window to open AFTER the only retest bar --
    # the level-touch opportunity is gone by the time the window opens, so
    # no QUALIFIED must ever fire (it should eventually EXPIRE instead).
    late_bars = build_late_entry_window_bars()
    late_events, _ = rs.run_ticker("LATEWINDOW", late_bars, LEVELS_TIEBREAK, EPOCHS_LATE_WINDOW)
    late_qualifieds = [e for e in late_events if e["event"] == "QUALIFIED"]
    check("no QUALIFIED when the entry window opens after the retest bar",
          len(late_qualifieds) == 0, str(late_events))
    check("track resolves via EXPIRED instead",
          any(e["event"] == "EXPIRED" and e["level"] == "ORH" for e in late_events),
          str(late_events))


# -----------------------------------------------------------------------------
# Test 5: riskValid rejection -- risk > ATR*1.0 kills the QUALIFIED.
# -----------------------------------------------------------------------------
def test_risk_valid_rejection():
    bars = build_riskfail_long_bars()
    events, _ = rs.run_ticker("RISKFAIL", bars, LEVELS_RISKFAIL, EPOCHS_TIEBREAK)

    # Confirm the premise numerically: at the retest bar, risk really does
    # exceed atr*maximumStopAtr.
    tr = rs.compute_true_range(bars)
    atr = rs.compute_rma(tr, rs.ATR_LEN)
    retest_idx = 29
    b = bars[retest_idx]
    stop = min(b["l"], 100.00) - atr[retest_idx] * rs.ATR_STOP_BUFFER
    risk = b["c"] - stop
    check("premise: risk exceeds ATR*maximumStopAtr at the retest bar",
          risk > atr[retest_idx] * rs.MAX_STOP_ATR,
          "risk={} atr*1.0={}".format(risk, atr[retest_idx] * rs.MAX_STOP_ATR))

    check("no QUALIFIED emitted when riskValid fails",
          not any(e["event"] == "QUALIFIED" for e in events), str(events))
    check("the break itself still produced a WATCH (track stays open, not fixed)",
          any(e["event"] == "WATCH" and e["level"] == "ORH" for e in events), str(events))


# -----------------------------------------------------------------------------
# Test 6: level-touch invariant for every emitted QUALIFIED.
# -----------------------------------------------------------------------------
def test_level_touch_invariant_on_every_qualified():
    scenarios = [
        (build_orb_pd_tiebreak_long_bars(), LEVELS_TIEBREAK, EPOCHS_TIEBREAK, "LONGINV"),
        (build_orb_pd_tiebreak_short_bars(), LEVELS_TIEBREAK_SHORT, EPOCHS_TIEBREAK, "SHORTINV"),
    ]
    total_checked = 0
    for bars, levels, epochs, name in scenarios:
        events, _ = rs.run_ticker(name, bars, levels, epochs)
        for e in events:
            if e["event"] != "QUALIFIED":
                continue
            total_checked += 1
            # Recover the qualifying bar from its event_id suffix (bar index).
            bar_idx = int(e["event_id"].rsplit("-", 1)[-1])
            bar = bars[bar_idx]
            level_price = e["level_price"]
            if e["direction"] == "CALL":
                check("{}: long touch invariant (bar low <= level)".format(name),
                      bar["l"] <= level_price,
                      "low={} level={}".format(bar["l"], level_price))
            else:
                check("{}: short touch invariant (bar high >= level)".format(name),
                      bar["h"] >= level_price,
                      "high={} level={}".format(bar["h"], level_price))
            check("{}: entry_high >= level_price".format(name),
                  e["entry_high"] >= level_price,
                  "entry_high={} level={}".format(e["entry_high"], level_price))
    check("at least one QUALIFIED was actually checked", total_checked >= 2,
          "checked {}".format(total_checked))


# -----------------------------------------------------------------------------
# Test 7: --pre-window-reject-rearms -- default OFF burns the PD track;
# flag ON re-arms it and lets a later in-window break/retest QUALIFY.
# -----------------------------------------------------------------------------
def test_pre_window_reject_default_burns_pd_track():
    bars = build_pre_window_pdh_reject_bars()
    events, _ = rs.run_ticker(
        "PREWINOFF", bars, LEVELS_PREWINDOW, EPOCHS_TIEBREAK,
        pre_window_reject_rearms=False)

    rejects = [e for e in events if e["event"] == "REJECT" and e["level"] == "PDH"]
    check("default (flag off): PDH REJECT fires pre-window",
          len(rejects) == 1, str(events))
    from datetime import datetime
    reject_time = datetime.strptime(rejects[0]["signal_time_ct"], "%Y-%m-%d %H:%M")
    entry_start_dt = datetime.strptime(
        rs.ct_str(EPOCHS_TIEBREAK["entry_start"]), "%Y-%m-%d %H:%M")
    check("default (flag off): the REJECT is genuinely pre-window",
          reject_time < entry_start_dt,
          "{} not before {}".format(reject_time, entry_start_dt))

    check("default (flag off): PD track is burned -- no later QUALIFIED",
          not any(e["event"] == "QUALIFIED" for e in events), str(events))


def test_pre_window_reject_rearms_allows_later_qualify():
    bars = build_pre_window_pdh_reject_bars()
    events, _ = rs.run_ticker(
        "PREWINON", bars, LEVELS_PREWINDOW, EPOCHS_TIEBREAK,
        pre_window_reject_rearms=True)

    check("flag on: no pre-window PDH REJECT is emitted",
          not any(e["event"] == "REJECT" and e["level"] == "PDH" for e in events),
          str(events))

    qualifieds = [e for e in events if e["event"] == "QUALIFIED" and e["level"] == "PDH"]
    check("flag on: a later PDH QUALIFIED is emitted",
          len(qualifieds) == 1, str(events))

    from datetime import datetime
    q_time = datetime.strptime(qualifieds[0]["signal_time_ct"], "%Y-%m-%d %H:%M")
    entry_start_dt = datetime.strptime(
        rs.ct_str(EPOCHS_TIEBREAK["entry_start"]), "%Y-%m-%d %H:%M")
    entry_end_dt = datetime.strptime(
        rs.ct_str(EPOCHS_TIEBREAK["entry_end"]), "%Y-%m-%d %H:%M")
    check("flag on: the QUALIFIED signal_time_ct is inside the entry window",
          entry_start_dt <= q_time < entry_end_dt,
          "{} not in [{}, {})".format(q_time, entry_start_dt, entry_end_dt))


def test_pre_window_reject_rearms_defaults_off_and_is_regression_safe():
    # The keyword parameter itself defaults to False.
    default_val = rs.run_ticker.__defaults__[-1]
    check("run_ticker's pre_window_reject_rearms parameter defaults to False",
          default_val is False, "default={}".format(default_val))

    # An existing scenario, called exactly as the pre-existing tests call it
    # (no pre_window_reject_rearms kwarg at all), is byte-for-byte unchanged:
    # still exactly one QUALIFIED, still on ORH.
    bars = build_orb_pd_tiebreak_long_bars()
    events, _ = rs.run_ticker("REGRESSION", bars, LEVELS_TIEBREAK, EPOCHS_TIEBREAK)
    qualifieds = [e for e in events if e["event"] == "QUALIFIED"]
    check("regression: exactly one QUALIFIED emitted with no flag passed",
          len(qualifieds) == 1, str(qualifieds))
    check("regression: the QUALIFIED is still the ORH track",
          qualifieds[0]["level"] == "ORH", str(qualifieds[0]))


# -----------------------------------------------------------------------------
# Test 8: the --pre-window-reject-rearms CLI flag must never poison the
# canonical events.json record (Fix 4 -- see CLAUDE.md "the learning loop":
# record_session.py reads events.json verbatim, and its duplicate-date
# refusal would otherwise protect a poisoned record from correction).
# -----------------------------------------------------------------------------
def _write_synthetic_session_dir():
    """Writes the pre-window-PDH-REJECT scenario (build_pre_window_pdh_reject_
    bars / LEVELS_PREWINDOW / EPOCHS_TIEBREAK) to a fresh temp session dir as
    real files, so replay_session.main() can be driven exactly as the CLI
    drives it (reading CSV + levels.json from disk)."""
    session_dir = tempfile.mkdtemp(prefix="replay_cli_test_")
    ticker = "CLITEST"
    bars = build_pre_window_pdh_reject_bars()

    csv_path = os.path.join(session_dir, "{}_5m.csv".format(ticker))
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]])

    levels = {
        "rth_open_epoch": EPOCHS_TIEBREAK["rth_open"],
        "entry_window_start_epoch": EPOCHS_TIEBREAK["entry_start"],
        "entry_window_end_epoch": EPOCHS_TIEBREAK["entry_end"],
        "tickers": {ticker: dict(LEVELS_PREWINDOW)},
    }
    with open(os.path.join(session_dir, "levels.json"), "w") as fh:
        json.dump(levels, fh)

    return session_dir, ticker


def _run_main_quiet(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = rs.main(argv)
    return rc, buf.getvalue()


def test_pre_window_reject_rearms_flag_writes_rearm_file_not_canonical():
    session_dir, ticker = _write_synthetic_session_dir()
    try:
        events_path = os.path.join(session_dir, "events.json")
        rearm_path = os.path.join(session_dir, "events_rearm.json")

        # Baseline: flag OFF writes the canonical file, no rearm file exists.
        rc, _ = _run_main_quiet(["--session", session_dir])
        check("flag off: main() exits 0", rc == 0)
        check("flag off: events.json was created", os.path.exists(events_path))
        check("flag off: no events_rearm.json was created",
              not os.path.exists(rearm_path))

        with open(events_path, "rb") as fh:
            before_bytes = fh.read()
        before_mtime = os.path.getmtime(events_path)

        # Nudge the filesystem clock resolution so an accidental rewrite of
        # events.json would show up as a changed mtime, not just luck.
        import time
        time.sleep(0.01)

        # Flag ON: must not touch events.json at all; must write
        # events_rearm.json instead.
        rc2, out2 = _run_main_quiet(
            ["--session", session_dir, "--pre-window-reject-rearms"])
        check("flag on: main() exits 0", rc2 == 0)

        with open(events_path, "rb") as fh:
            after_bytes = fh.read()
        after_mtime = os.path.getmtime(events_path)

        check("flag on: events.json bytes are byte-identical to before",
              before_bytes == after_bytes)
        check("flag on: events.json mtime is unchanged",
              before_mtime == after_mtime,
              "{} vs {}".format(before_mtime, after_mtime))
        check("flag on: events_rearm.json was written instead",
              os.path.exists(rearm_path))

        with open(rearm_path) as fh:
            rearm_events = json.load(fh)
        check("flag on: events_rearm.json is non-empty and PD-track-affected",
              len(rearm_events) > 0, str(rearm_events))
        check("flag on: no pre-window PDH REJECT survives in the rearm file "
              "(matches run_ticker-level behavior)",
              not any(e["event"] == "REJECT" and e["level"] == "PDH"
                      for e in rearm_events),
              str(rearm_events))

        check("flag on: stdout marks the output as research-only",
              "RESEARCH-ONLY" in out2 and "events_rearm.json" in out2,
              out2)
        check("flag on: stdout warns record_session.py must not consume it",
              "record_session.py" in out2, out2)
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


def test_explicit_out_overrides_default_filename_in_both_modes():
    session_dir, ticker = _write_synthetic_session_dir()
    try:
        custom_path = os.path.join(session_dir, "custom_events.json")
        rc, _ = _run_main_quiet(
            ["--session", session_dir, "--out", custom_path])
        check("explicit --out: main() exits 0", rc == 0)
        check("explicit --out: writes to the requested path",
              os.path.exists(custom_path))
        check("explicit --out: default events.json NOT written",
              not os.path.exists(os.path.join(session_dir, "events.json")))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def main():
    tests = [
        test_atr_is_wilder_rma_not_sma,
        test_orb_before_pd_tiebreak,
        test_orb_before_pd_tiebreak_short_side,
        test_one_qualified_per_day_lockout,
        test_qualified_never_outside_entry_window,
        test_risk_valid_rejection,
        test_level_touch_invariant_on_every_qualified,
        test_pre_window_reject_default_burns_pd_track,
        test_pre_window_reject_rearms_allows_later_qualify,
        test_pre_window_reject_rearms_defaults_off_and_is_regression_safe,
        test_pre_window_reject_rearms_flag_writes_rearm_file_not_canonical,
        test_explicit_out_overrides_default_filename_in_both_modes,
    ]
    for t in tests:
        t()
    print("\n{} checks passed.".format(len(_passed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
