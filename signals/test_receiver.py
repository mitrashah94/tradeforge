#!/usr/bin/env python3
"""Self-contained tests for signals/receiver.py (plain asserts, no pytest).

Run: python3 /Users/mitrashah/DayTrading/signals/test_receiver.py
"""

import copy
import json
import os
import re
import sys
import tempfile

SIGNALS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIGNALS_DIR)

import receiver  # noqa: E402

FIXTURES = os.path.join(SIGNALS_DIR, "fixtures")

BANNED = ("order", "buy_to_open", "sell_to_open", "submit", "execute trade")

_passed = []


def check(name, cond, detail=""):
    assert cond, "{} FAILED {}".format(name, detail)
    _passed.append(name)
    print("ok - {}".format(name))


def load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as fh:
        return json.load(fh)


def card_for(fixture_name):
    payload = load_fixture(fixture_name)
    # No-dedupe processing via the importable pipeline function.
    return receiver.process_payload(payload, dedupe=False)


def assert_no_banned_words(name, card):
    low = card.lower()
    for word in BANNED:
        check("{}: card free of {!r}".format(name, word), word not in low,
              "found {!r} in card".format(word))


def test_qualified_xlf_call():
    card = card_for("qualified_xlf_call.json")
    for needle in ("XLF", "CALL", "51.40", "51.48", "51.22",
                   "51.58", "51.76", "51.94", "52.12", "52.30",
                   "09:20", receiver.FOOTER):
        check("xlf_call card contains {!r}".format(needle), needle in card)
    check("xlf_call R levels verified", "R levels verified" in card)
    check("xlf_call is not a validation error",
          "VALIDATION ERROR" not in card)
    assert_no_banned_words("xlf_call", card)
    return card


def test_qualified_xle_put():
    card = card_for("qualified_xle_put.json")
    for needle in ("XLE", "PUT", "54.75",
                   "54.25", "54.00", "53.75", "53.50", "53.25",
                   receiver.FOOTER):
        check("xle_put card contains {!r}".format(needle), needle in card)
    check("xle_put R levels verified", "R levels verified" in card)
    check("xle_put is not a validation error",
          "VALIDATION ERROR" not in card)
    assert_no_banned_words("xle_put", card)


def test_terminal_events():
    cases = (
        ("rejected_wide_stop.json", "REJECT", "risk_above_max_atr"),
        ("expired_signal.json", "EXPIRED", "retest_window_elapsed"),
        ("invalidated_signal.json", "INVALIDATED",
         "closed_through_stop_after_qualified"),
    )
    for fname, event, reason in cases:
        card = card_for(fname)
        check("{} card contains event {}".format(fname, event),
              event in card)
        check("{} card contains exact reason {!r}".format(fname, reason),
              reason in card)
        check("{} card has footer".format(fname), receiver.FOOTER in card)
        check("{} not a validation error".format(fname),
              "VALIDATION ERROR" not in card)
        assert_no_banned_words(fname, card)


def test_watch_and_demo_cards_clean():
    for payload in receiver.demo_payloads():
        card = receiver.process_payload(payload, dedupe=False)
        check("demo {} has footer".format(payload["event"]),
              receiver.FOOTER in card)
        check("demo {} not validation error".format(payload["event"]),
              "VALIDATION ERROR" not in card)
        assert_no_banned_words("demo " + payload["event"], card)


def test_dedupe():
    payload = load_fixture("qualified_xlf_call.json")
    fd, store = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(store)  # start with no store file
    try:
        first = receiver.process_payload(payload, dedupe=True,
                                         store_path=store)
        second = receiver.process_payload(payload, dedupe=True,
                                          store_path=store)
        check("dedupe: first pass renders a card",
              "QUALIFIED SETUP" in first)
        check("dedupe: second pass is duplicate notice",
              second.startswith("DUPLICATE event_id")
              and payload["event_id"] in second)
        check("dedupe: duplicate notice is one line", "\n" not in second)
        bypass = receiver.process_payload(payload, dedupe=False,
                                          store_path=store)
        check("dedupe: --no-dedupe bypass still renders card",
              "QUALIFIED SETUP" in bypass)
    finally:
        if os.path.exists(store):
            os.remove(store)


def test_validation_error_wrong_side_stop():
    payload = load_fixture("qualified_xlf_call.json")
    corrupted = copy.deepcopy(payload)
    corrupted["stop"] = 51.55  # above entry_low: wrong side for a CALL
    card = receiver.process_payload(corrupted, dedupe=False)
    check("corrupted payload yields VALIDATION ERROR card",
          "VALIDATION ERROR" in card)
    check("violation names the stop-side rule",
          "stop (51.55) must be below entry_low (51.40) for a CALL" in card)
    check("validation error card has footer", receiver.FOOTER in card)


def test_validation_lists_every_violation():
    payload = load_fixture("qualified_xle_put.json")
    corrupted = copy.deepcopy(payload)
    corrupted["stop"] = 54.20          # wrong side for PUT
    corrupted["level"] = "PDH"         # PUT must be PDL
    corrupted["score"] = 9             # out of range
    del corrupted["expiration_price"]  # missing key
    card = receiver.process_payload(corrupted, dedupe=False)
    check("multi-violation: is validation error", "VALIDATION ERROR" in card)
    check("multi-violation: stop side listed",
          "must be above entry_high" in card)
    check("multi-violation: direction/level listed",
          "direction PUT requires level in PDL/ORL, got PDH" in card)
    check("multi-violation: score listed", "score must be an integer" in card)
    check("multi-violation: missing key listed",
          "missing required key: expiration_price" in card)


def test_risk_math_discrepancy_reported():
    payload = load_fixture("qualified_xlf_call.json")
    tweaked = copy.deepcopy(payload)
    tweaked["r3"] = 52.10  # should be 51.94; still monotone so passes schema
    card = receiver.process_payload(tweaked, dedupe=False)
    check("r-level discrepancy is not a validation error",
          "VALIDATION ERROR" not in card)
    check("r-level discrepancy reported",
          "DISCREPANCIES FOUND" in card and "r3" in card)


def test_plan_gate_flags():
    xlf = card_for("qualified_xlf_call.json")
    check("xlf gates: rvol PASS", "PASS  RVOL >= 1.2" in xlf)
    check("xlf gates: room PASS", "PASS  Room >= 3R" in xlf)
    check("xlf gates: score PASS", "PASS  Score >= 3" in xlf)
    check("xlf gates: time PASS", "PASS  Signal time 08:45-10:30 CT" in xlf)

    xle = card_for("qualified_xle_put.json")
    check("xle gates: room FAIL flagged (2.40 < 3)",
          "FAIL  Room >= 3R  (room_r = 2.40)" in xle)
    check("xle gates: failure did not suppress card",
          "QUALIFIED SETUP: XLE PUT" in xle)

    # Late signal time -> time gate FAIL, card still renders.
    late = load_fixture("qualified_xlf_call.json")
    late = copy.deepcopy(late)
    late["signal_time_ct"] = "2026-07-13 11:05"
    card = receiver.process_payload(late, dedupe=False)
    check("late signal: time gate FAIL",
          "FAIL  Signal time 08:45-10:30 CT" in card)
    check("late signal: card still rendered", "QUALIFIED SETUP" in card)


def test_qualified_orh_call():
    card = card_for("qualified_orh_call.json")
    for needle in ("IWM", "CALL", "ORH", "225.10", "225.25", "225.33",
                   "225.07", "225.43", "225.61", "225.79", "225.97",
                   "226.15", "226.40", receiver.FOOTER):
        check("orh_call card contains {!r}".format(needle), needle in card)
    check("orh_call R levels verified", "R levels verified" in card)
    check("orh_call is not a validation error",
          "VALIDATION ERROR" not in card)
    assert_no_banned_words("orh_call", card)


def test_qualified_orl_put():
    card = card_for("qualified_orl_put.json")
    for needle in ("QQQ", "PUT", "ORL", "555.80", "555.60", "555.82",
                   "555.85", "555.35", "555.10", "554.85", "554.60",
                   "554.35", receiver.FOOTER):
        check("orl_put card contains {!r}".format(needle), needle in card)
    check("orl_put R levels verified", "R levels verified" in card)
    check("orl_put is not a validation error",
          "VALIDATION ERROR" not in card)
    assert_no_banned_words("orl_put", card)


def test_invalidated_orl_put():
    card = card_for("invalidated_orl_put.json")
    check("invalidated_orl_put contains event INVALIDATED",
          "INVALIDATED" in card)
    check("invalidated_orl_put contains level ORL", "ORL" in card)
    check("invalidated_orl_put contains known reason",
          "closed_through_stop_after_qualified" in card)
    check("invalidated_orl_put not a validation error",
          "VALIDATION ERROR" not in card)
    check("invalidated_orl_put has footer", receiver.FOOTER in card)
    assert_no_banned_words("invalidated_orl_put", card)


def test_orh_call_obstacle_is_pdh():
    # The ORH-call fixture's next_obstacle is a PDH sitting overhead --
    # exercises the "obstacle is a different level type" case.
    card = card_for("qualified_orh_call.json")
    check("orh_call obstacle shows PDH price 226.40",
          "226.40" in card)
    room_r = 6.39
    check("orh_call obstacle room_r matches fixture",
          "{:.2f}".format(room_r) in card)
    check("orh_call obstacle room_r passes gate",
          "PASS  Room >= 3R  (room_r = 6.39)" in card)
    check("orh_call R levels verify alongside obstacle check",
          "R levels verified" in card)


def test_qualified_fixtures_are_engine_reachable():
    """Every QUALIFIED fixture must describe a state the Pine engine can
    actually produce -- otherwise a fixture can 'pass' while encoding a setup
    that could never fire live.

    The Pine retest condition requires the confirmation bar to TOUCH the level:
        CALL: low  <= level  and close > level   -> entry_high >= entry_low
              and the bar reclaimed the level, so entry_high >= level_price.
        PUT:  high >= level  and close < level   -> entry_high >= level_price.
    In both directions the confirmation bar's HIGH must reach the level, and
    entry_high is that bar's high (Pine: entry_high = high). Also cross-check
    expiration_price against the documented do-not-chase formula:
        CALL: entry_high + 0.5R      PUT: entry_low - 0.5R
    """
    import glob
    paths = sorted(glob.glob(os.path.join(FIXTURES, "qualified_*.json")))
    check("found QUALIFIED fixtures to check", len(paths) > 0)
    for path in paths:
        name = os.path.basename(path)
        p = load_fixture(name)
        level = p["level_price"]
        entry_low = p["entry_low"]
        entry_high = p["entry_high"]
        stop = p["stop"]
        risk = abs(entry_low - stop)

        # The confirmation bar must have touched the broken level.
        check("{}: confirmation bar touches the level "
              "(entry_high {} >= level {})".format(name, entry_high, level),
              entry_high >= level - receiver.TICK)

        # Close must be on the correct side of the level.
        if p["direction"] == "CALL":
            check("{}: CALL closes above the level".format(name),
                  entry_low > level - receiver.TICK)
            expected_exp = entry_high + 0.5 * risk
        else:
            check("{}: PUT closes below the level".format(name),
                  entry_low < level + receiver.TICK)
            expected_exp = entry_low - 0.5 * risk

        check("{}: expiration_price {} matches formula {:.3f} "
              "(within a tick)".format(name, p["expiration_price"],
                                       expected_exp),
              abs(p["expiration_price"] - expected_exp) <= receiver.TICK + 1e-9)


def test_direction_level_pairing_call_orh_and_put_orl_validate_clean():
    for fname, direction, level in (
        ("qualified_orh_call.json", "CALL", "ORH"),
        ("qualified_orl_put.json", "PUT", "ORL"),
    ):
        payload = load_fixture(fname)
        check("{}: fixture direction is {}".format(fname, direction),
              payload["direction"] == direction)
        check("{}: fixture level is {}".format(fname, level),
              payload["level"] == level)
        violations = receiver.validate_payload(payload)
        check("{}: {} + {} validates clean".format(fname, direction, level),
              violations == [], detail=repr(violations))
        card = receiver.process_payload(payload, dedupe=False)
        check("{}: renders QUALIFIED SETUP card".format(fname),
              "QUALIFIED SETUP" in card)


def test_direction_level_pairing_negative_call_orl_and_put_orh():
    # CALL+ORL is a schema violation.
    call_orl = copy.deepcopy(load_fixture("qualified_orh_call.json"))
    call_orl["level"] = "ORL"
    card = receiver.process_payload(call_orl, dedupe=False)
    check("CALL+ORL yields VALIDATION ERROR", "VALIDATION ERROR" in card)
    check("CALL+ORL names allowed set",
          "direction CALL requires level in PDH/ORH, got ORL" in card)
    check("CALL+ORL card has footer", receiver.FOOTER in card)
    assert_no_banned_words("CALL+ORL", card)

    # PUT+ORH is a schema violation.
    put_orh = copy.deepcopy(load_fixture("qualified_orl_put.json"))
    put_orh["level"] = "ORH"
    card = receiver.process_payload(put_orh, dedupe=False)
    check("PUT+ORH yields VALIDATION ERROR", "VALIDATION ERROR" in card)
    check("PUT+ORH names allowed set",
          "direction PUT requires level in PDL/ORL, got ORH" in card)
    check("PUT+ORH card has footer", receiver.FOOTER in card)
    assert_no_banned_words("PUT+ORH", card)


def test_pdh_and_orh_watch_event_ids_distinct_and_dedupe_independently():
    pdh_watch = {
        "event": "WATCH",
        "event_id": "SPY-20260713-CALL-PDH-WATCH-9601",
        "ticker": "SPY", "timeframe": "5",
        "setup_type": "A_break_retest", "direction": "CALL",
        "level": "PDH", "level_price": 628.40,
        "signal_time_ct": "2026-07-13 08:52",
        "vwap": 627.90, "rvol": 1.31,
    }
    orh_watch = copy.deepcopy(pdh_watch)
    orh_watch["event_id"] = "SPY-20260713-CALL-ORH-WATCH-9602"
    orh_watch["level"] = "ORH"
    orh_watch["level_price"] = 628.55

    check("PDH-WATCH and ORH-WATCH event_ids are distinct",
          pdh_watch["event_id"] != orh_watch["event_id"])

    fd, store = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(store)
    try:
        pdh_card = receiver.process_payload(pdh_watch, dedupe=True,
                                            store_path=store)
        orh_card = receiver.process_payload(orh_watch, dedupe=True,
                                            store_path=store)
        check("PDH-WATCH renders (first time seen)",
              "WATCH" in pdh_card and "VALIDATION ERROR" not in pdh_card)
        check("ORH-WATCH renders (first time seen, distinct event_id)",
              "WATCH" in orh_card and "VALIDATION ERROR" not in orh_card)
        assert_no_banned_words("PDH-WATCH", pdh_card)
        assert_no_banned_words("ORH-WATCH", orh_card)

        pdh_again = receiver.process_payload(pdh_watch, dedupe=True,
                                             store_path=store)
        orh_again = receiver.process_payload(orh_watch, dedupe=True,
                                             store_path=store)
        check("re-feeding PDH-WATCH is caught as DUPLICATE",
              pdh_again.startswith("DUPLICATE event_id")
              and pdh_watch["event_id"] in pdh_again)
        check("re-feeding ORH-WATCH is caught as DUPLICATE",
              orh_again.startswith("DUPLICATE event_id")
              and orh_watch["event_id"] in orh_again)
    finally:
        if os.path.exists(store):
            os.remove(store)


def test_watch_card_leads_with_human_factors_warning():
    payload = {
        "event": "WATCH",
        "event_id": "SPY-20260713-CALL-PDH-WATCH-9701",
        "ticker": "SPY", "timeframe": "5",
        "setup_type": "A_break_retest", "direction": "CALL",
        "level": "PDH", "level_price": 628.40,
        "signal_time_ct": "2026-07-13 08:52",
        "vwap": 627.90, "rvol": 1.31,
    }
    card = receiver.process_payload(payload, dedupe=False)
    check("WATCH card leads with the human-factors warning verbatim",
          "WATCH - NOT A SETUP. NOT AN ENTRY. Do nothing." in card)
    lines = card.splitlines()
    warning_idx = next(i for i, l in enumerate(lines)
                       if "NOT A SETUP" in l)
    ticker_idx = next(i for i, l in enumerate(lines)
                      if l.startswith("SPY CALL"))
    check("the warning line appears before the ticker/direction line",
          warning_idx < ticker_idx)
    check("WATCH card still has the mandatory footer", receiver.FOOTER in card)
    assert_no_banned_words("watch_human_factors", card)


# -----------------------------------------------------------------------------
# Feasibility block tests (Fix 1/2/3 in the receiver-builder brief): the
# shipped bug rendered "max delta ~1.39" -- a value outside the trader's own
# 0.45-0.60 allowed band (strategy.md section 7) -- with ZERO test coverage.
# These tests close that gap.
# -----------------------------------------------------------------------------
_QUALIFIED_TEMPLATE = {
    "event": "QUALIFIED",
    "ticker": "TST", "timeframe": "5",
    "setup_type": "A_break_retest", "direction": "CALL",
    "level": "PDH", "level_price": 99.95,
    "signal_time_ct": "2026-07-13 09:05",
    "vwap": 99.98, "rvol": 1.40,
    "next_obstacle": None, "room_r": None, "score": 4,
}


def make_call_payload(risk, event_id, entry_low=100.00):
    """Builds a schema-valid CALL QUALIFIED payload with entry_high ==
    entry_low (zero-width confirmation bar) so the worst-fill risk is
    exactly 1.5x the best-fill risk (expiration_price = entry_high + 0.5R),
    which keeps a chosen risk cleanly inside one feasibility branch for
    best AND worst fill in the isolation tests below."""
    p = copy.deepcopy(_QUALIFIED_TEMPLATE)
    stop = entry_low - risk
    p.update({
        "event_id": event_id,
        "entry_low": entry_low, "entry_high": entry_low, "stop": stop,
        "r1": entry_low + 1 * risk, "r2": entry_low + 2 * risk,
        "r3": entry_low + 3 * risk, "r4": entry_low + 4 * risk,
        "r5": entry_low + 5 * risk,
        "expiration_time_ct": "2026-07-13 09:20",
        "expiration_price": entry_low + 0.5 * risk,
    })
    return p


def _feasibility_block(card):
    """Extract just the feasibility sub-section of a rendered card (from the
    'Feasibility  :' line through the next rule line)."""
    lines = card.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(
        "Feasibility"))
    end = next(i for i in range(start + 1, len(lines))
              if lines[i].strip().startswith("---") or not lines[i].strip())
    return "\n".join(lines[start:end])


def test_feasibility_cap_not_binding_branch():
    # risk 0.15 -> best 0.15, worst 1.5x = 0.225; both comfortably under the
    # 0.3333 threshold where (0.60*R + 0.05)*100 <= 25.
    payload = make_call_payload(0.15, "TST-20260713-CALL-PDH-QUALIFIED-1001")
    violations = receiver.validate_payload(payload)
    check("cap-not-binding payload validates clean", violations == [],
          repr(violations))
    card = receiver.process_payload(payload, dedupe=False)
    block = _feasibility_block(card)
    check("cap-not-binding: verdict is FEASIBLE (pre-check)",
          "FEASIBLE (pre-check)" in block, block)
    check("cap-not-binding: exact required wording present",
          "$25 cap not binding: the entire 0.45-0.60 delta band passes at "
          "spread <= 0.05." in block, block)
    check("cap-not-binding: no delta number rendered anywhere in the block",
          "delta ~" not in block, block)
    check("cap-not-binding: not flagged INFEASIBLE", "INFEASIBLE" not in block,
          block)


def test_feasibility_cap_binds_inside_band_branch():
    # risk 0.35 -> best 0.35, worst 1.5x = 0.525; both inside (0.3334, 0.5556)
    # where the cap binds but delta 0.45 at spread 0.00 still clears $25.
    payload = make_call_payload(0.35, "TST-20260713-CALL-PDH-QUALIFIED-1002")
    card = receiver.process_payload(payload, dedupe=False)
    block = _feasibility_block(card)
    check("cap-binds: verdict is FEASIBLE (pre-check)",
          "FEASIBLE (pre-check)" in block, block)
    check("cap-binds: prints a clamped delta ceiling",
          re.search(r"max usable delta ~0\.\d\d", block) is not None, block)
    check("cap-binds: ceiling caveat names the 0.60 band top",
          "below the 0.60 band top" in block, block)
    check("cap-binds: not flagged INFEASIBLE", "INFEASIBLE" not in block,
          block)


def test_feasibility_infeasible_branch():
    # risk 0.60 -> best 0.60, worst 1.5x = 0.90; both exceed 0.5556, so even
    # delta 0.45 at spread 0.00 already fails the $25 gate.
    payload = make_call_payload(0.60, "TST-20260713-CALL-PDH-QUALIFIED-1003")
    card = receiver.process_payload(payload, dedupe=False)
    block = _feasibility_block(card)
    check("infeasible: verdict is INFEASIBLE", "INFEASIBLE" in block, block)
    check("infeasible: names the failing delta/spread combination",
          "even delta 0.45 at spread 0.00 exceeds $25" in block, block)
    check("infeasible: no delta number rendered anywhere in the block",
          "delta ~" not in block, block)
    check("infeasible: card still renders fully (never suppressed)",
          "QUALIFIED SETUP" in card)


def test_feasibility_crossing_branches_best_vs_worst_and_governing_note():
    # The fixture that shipped the original bug: entry_low 51.40, stop 51.22
    # (risk 0.18, best fill = OPEN) vs expiration_price 51.57 (worst-fill
    # risk 0.35, CAPPED) -- the two fills land in DIFFERENT branches, so the
    # card must surface that and let the worse (worst-fill) case govern.
    card = card_for("qualified_xlf_call.json")
    block = _feasibility_block(card)
    check("crossing branches: worst-fill note is present",
          "best-fill and worst-fill verdicts differ" in block, block)
    check("crossing branches: governing verdict is the worse one (CAPPED, "
          "not the OPEN best-fill result)",
          "max usable delta ~0.57" in block, block)

    best_r = float(re.search(r"Best  fill \(chart R ([\d.]+)", block).group(1))
    worst_r = float(re.search(r"Worst fill \(chart R ([\d.]+)", block).group(1))
    check("worst-fill chart-R is >= best-fill chart-R",
          worst_r >= best_r, "{} vs {}".format(worst_r, best_r))


def test_feasibility_line_itself_carries_precheck_caveat():
    for fixture in ("qualified_xlf_call.json", "qualified_xle_put.json",
                    "qualified_orh_call.json", "qualified_orl_put.json"):
        card = card_for(fixture)
        lines = card.splitlines()
        feas_line = next(l for l in lines if l.strip().startswith(
            "Feasibility"))
        check("{}: the Feasibility line itself carries the pre-check "
              "caveat".format(fixture),
              "assumed spread <= 0.05" in feas_line
              and "recheck" in feas_line.lower(), feas_line)


def test_feasibility_strike_line_references_current_price():
    card = card_for("qualified_xlf_call.json")
    strike_line = next(l for l in card.splitlines()
                       if l.strip().startswith("Strike"))
    check("strike line points at current price, not the stale signal close",
          "CURRENT PRICE" in strike_line, strike_line)
    check("strike line still notes the signal close for reference",
          "51.40" in strike_line, strike_line)


def test_feasibility_delta_invariant_across_all_fixtures_and_synthetics():
    """The test that would have caught the shipped bug: no rendered card,
    for ANY QUALIFIED fixture or a sweep of synthetic payloads spanning every
    feasibility branch, may ever print a delta value outside [0.45, 0.60]."""
    import glob
    cards = []
    for path in sorted(glob.glob(os.path.join(FIXTURES, "qualified_*.json"))):
        cards.append((os.path.basename(path),
                      receiver.process_payload(load_fixture(
                          os.path.basename(path)), dedupe=False)))

    # Sweep chart-R across a wide range, including values that would have
    # produced the original out-of-band "~1.39" bug (very small R).
    for i, risk in enumerate([0.01, 0.05, 0.10, 0.15, 0.18, 0.25, 0.30,
                              0.35, 0.40, 0.50, 0.55, 0.60, 0.75, 1.00,
                              2.00]):
        payload = make_call_payload(
            risk, "TST-20260713-CALL-PDH-QUALIFIED-{}".format(2000 + i))
        card = receiver.process_payload(payload, dedupe=False)
        cards.append(("synthetic risk={}".format(risk), card))

    checked_any_delta = False
    for name, card in cards:
        for m in re.finditer(r"delta ~([\d.]+)", card):
            checked_any_delta = True
            val = float(m.group(1))
            check("{}: rendered delta {} is inside [0.45, 0.60]".format(
                name, val),
                receiver.FEASIBILITY_DELTA_MIN - 1e-9 <= val
                <= receiver.FEASIBILITY_DELTA_MAX + 1e-9,
                "{} out of band in {}".format(val, name))
    check("at least one 'delta ~' value was actually exercised by the sweep",
          checked_any_delta)


def test_no_brokerage_references_in_source():
    with open(os.path.join(SIGNALS_DIR, "receiver.py")) as fh:
        src = fh.read().lower()
    for token in ("import requests", "import urllib", "import http.client",
                  "robin_stocks", "ibapi", "alpaca", "tradier"):
        check("source free of {!r}".format(token), token not in src)


def main():
    card = test_qualified_xlf_call()
    test_qualified_xle_put()
    test_terminal_events()
    test_watch_and_demo_cards_clean()
    test_dedupe()
    test_validation_error_wrong_side_stop()
    test_validation_lists_every_violation()
    test_risk_math_discrepancy_reported()
    test_plan_gate_flags()
    test_qualified_orh_call()
    test_qualified_orl_put()
    test_invalidated_orl_put()
    test_qualified_fixtures_are_engine_reachable()
    test_orh_call_obstacle_is_pdh()
    test_direction_level_pairing_call_orh_and_put_orl_validate_clean()
    test_direction_level_pairing_negative_call_orl_and_put_orh()
    test_pdh_and_orh_watch_event_ids_distinct_and_dedupe_independently()
    test_watch_card_leads_with_human_factors_warning()
    test_feasibility_cap_not_binding_branch()
    test_feasibility_cap_binds_inside_band_branch()
    test_feasibility_infeasible_branch()
    test_feasibility_crossing_branches_best_vs_worst_and_governing_note()
    test_feasibility_line_itself_carries_precheck_caveat()
    test_feasibility_strike_line_references_current_price()
    test_feasibility_delta_invariant_across_all_fixtures_and_synthetics()
    test_no_brokerage_references_in_source()
    print()
    print("ALL {} ASSERTIONS PASSED".format(len(_passed)))
    print()
    print("Sample QUALIFIED card (qualified_xlf_call.json):")
    print(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
