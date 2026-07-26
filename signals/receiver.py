#!/usr/bin/env python3
"""
Day-trading signal receiver: TradingView alert JSON -> human-review decision cards.

This module is DISPLAY / VALIDATION logic only. It never calls, imports, or
references any brokerage API. Every rendered card ends with the fixed footer:

    NO BROKERAGE ACTION - human decision required

Pipeline: parse -> schema-validate -> dedupe (by event_id) -> render card.

Payload schema (emitted by the companion Pine indicator)
--------------------------------------------------------
Common envelope (all events):
    event            "WATCH" | "QUALIFIED" | "REJECT" | "INVALIDATED" | "EXPIRED"
    event_id         unique string, e.g. "XLF-20260713-CALL-PDH-QUALIFIED-9412"
    ticker           string
    timeframe        string, e.g. "5"
    setup_type       "A_break_retest" | "B_breakdown_bounce"
    direction        "CALL" | "PUT"
    level            "PDH" | "PDL" | "ORH" | "ORL"
                     (CALL -> PDH or ORH, PUT -> PDL or ORL)
    level_price      number
    signal_time_ct   "yyyy-MM-dd HH:mm"
    vwap             number | null
    rvol             number | null

Four level tracks run per day: PDH-long, ORH-long, PDL-short, ORL-short.
All four are produced by the identical break -> retest -> confirm engine
(same retest window, VWAP/RVOL/EMA filters, ATR stop, risk-validity checks)
-- ORB events (ORH/ORL) are just another track through that same engine,
gated additionally on the opening range being complete. `event_id` embeds
the level token, so an ORH event and a PDH event on the same ticker/day are
distinct and independently deduped. `next_obstacle` on an ORB QUALIFIED may
legitimately be the opposing prior-day level (e.g. PDH as the obstacle for
an ORH-long trade) -- nothing in the risk math below changes for that case.

QUALIFIED adds:
    entry_low, entry_high, stop, r1..r5   numbers
    next_obstacle    number | null
    room_r           number | null
    score            int 0-5
    expiration_time_ct    string
    expiration_price      number

REJECT / INVALIDATED / EXPIRED add:
    reason           string
        REJECT:      "failed_hold_below_level", "failed_hold_above_level",
                     "risk_above_max_atr" (wide stop)
        EXPIRED:     "retest_window_elapsed"
        INVALIDATED: "closed_through_stop_after_qualified"

Risk anchoring convention (used by validator, fixtures, and Pine indicator)
---------------------------------------------------------------------------
Risk per share is ALWAYS anchored at entry_low:

    risk = abs(entry_low - stop)

and target levels are measured from entry_low in the trade direction:

    CALL:  rN = entry_low + N * risk      (stop below entry_low, rN ascending)
    PUT:   rN = entry_low - N * risk      (stop above entry_high, rN descending)

For a PUT, entry_low is the lower edge of the entry zone (the confirmation
close), so risk = stop - entry_low spans the whole zone plus the buffer to
the stop. Each rN must match the formula within one tick (0.01).

Plan gates (flags for the human; failures never suppress a card):
    rvol >= 1.2
    room_r >= 3  (next_obstacle null counts as PASS: "open space")
    score >= 3
    signal time between 08:45 and 10:30 CT
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNALS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SIGNALS_DIR, "fixtures")
SEEN_STORE_PATH = os.path.join(SIGNALS_DIR, ".seen_events.json")

TICK = 0.01

EVENTS = ("WATCH", "QUALIFIED", "REJECT", "INVALIDATED", "EXPIRED")
SETUP_TYPES = ("A_break_retest", "B_breakdown_bounce")
DIRECTIONS = ("CALL", "PUT")
LEVELS = ("PDH", "PDL", "ORH", "ORL")

# Direction/level pairing enforced by the validator (see AGENTS.md):
# CALL -> PDH or ORH; PUT -> PDL or ORL.
DIRECTION_LEVELS = {
    "CALL": ("PDH", "ORH"),
    "PUT": ("PDL", "ORL"),
}

COMMON_KEYS = (
    "event", "event_id", "ticker", "timeframe", "setup_type", "direction",
    "level", "level_price", "signal_time_ct", "vwap", "rvol",
)
QUALIFIED_KEYS = (
    "entry_low", "entry_high", "stop", "r1", "r2", "r3", "r4", "r5",
    "next_obstacle", "room_r", "score", "expiration_time_ct",
    "expiration_price",
)
REASON_EVENTS = ("REJECT", "INVALIDATED", "EXPIRED")

KNOWN_REASONS = {
    "REJECT": ("failed_hold_below_level", "failed_hold_above_level",
               "risk_above_max_atr"),
    "EXPIRED": ("retest_window_elapsed",),
    "INVALIDATED": ("closed_through_stop_after_qualified",),
}

GATE_RVOL_MIN = 1.2
GATE_ROOM_R_MIN = 3.0
GATE_SCORE_MIN = 3
GATE_TIME_START = (8, 45)
GATE_TIME_END = (10, 30)

# Feasibility math (strategy.md section 7, authoritative):
#   estimated option loss at chart stop = (delta * chart_R + full_spread) * 100
#   must be <= $25. Rearranged for the maximum delta that still clears the
# cap at a given spread: delta_max = (0.25 - spread) / chart_R.
FEASIBILITY_CAP_PER_SHARE = 0.25   # $25 / 100 shares-equivalent
FEASIBILITY_WIDE_SPREAD = 0.05     # strategy.md section 4 "wide spread" boundary
FEASIBILITY_DELTA_MIN = 0.45
FEASIBILITY_DELTA_MAX = 0.60
DTE_MIN_DAYS = 7
DTE_MAX_DAYS = 21

WIDTH = 68
FOOTER = "NO BROKERAGE ACTION - human decision required"

OPTION_CHECKLIST = (
    "Premium <= settled cash on hand",
    "Delta between 0.45 and 0.60",
    "Est. option loss at underlying stop <= $25",
    "Tight bid/ask spread",
    "7-21 DTE",
    "No earnings report before expiry",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_num_or_null(v):
    return v is None or _is_num(v)


def _fmt(v):
    if v is None:
        return "n/a"
    if _is_num(v):
        return "{:.2f}".format(v)
    return str(v)


def _parse_ct(value):
    """Parse 'yyyy-MM-dd HH:mm'; return datetime or None."""
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

def validate_payload(payload):
    """Return a list of violation strings; empty list means valid."""
    v = []
    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]

    for key in COMMON_KEYS:
        if key not in payload:
            v.append("missing required key: {}".format(key))
    if v:
        # Without the envelope we cannot go much further, but still check
        # what we can below using .get().
        pass

    event = payload.get("event")
    if "event" in payload and event not in EVENTS:
        v.append("event must be one of {}, got {!r}".format(
            "/".join(EVENTS), event))

    for key in ("event_id", "ticker", "timeframe", "signal_time_ct"):
        if key in payload and not isinstance(payload[key], str):
            v.append("{} must be a string".format(key))

    if "setup_type" in payload and payload["setup_type"] not in SETUP_TYPES:
        v.append("setup_type must be one of {}, got {!r}".format(
            "/".join(SETUP_TYPES), payload["setup_type"]))

    direction = payload.get("direction")
    level = payload.get("level")
    if "direction" in payload and direction not in DIRECTIONS:
        v.append("direction must be CALL or PUT, got {!r}".format(direction))
    if "level" in payload and level not in LEVELS:
        v.append("level must be one of {}, got {!r}".format(
            "/".join(LEVELS), level))

    if "level_price" in payload and not _is_num(payload["level_price"]):
        v.append("level_price must be a number")
    for key in ("vwap", "rvol"):
        if key in payload and not _is_num_or_null(payload[key]):
            v.append("{} must be a number or null".format(key))

    if isinstance(payload.get("signal_time_ct"), str) \
            and _parse_ct(payload["signal_time_ct"]) is None:
        v.append("signal_time_ct must match 'yyyy-MM-dd HH:mm'")

    # Direction / level consistency: CALL -> PDH or ORH, PUT -> PDL or ORL.
    if direction in DIRECTIONS and level in LEVELS:
        allowed = DIRECTION_LEVELS[direction]
        if level not in allowed:
            v.append("direction {} requires level in {}, got {}".format(
                direction, "/".join(allowed), level))

    # Event-specific keys
    if event == "QUALIFIED":
        v.extend(_validate_qualified(payload))
    elif event in REASON_EVENTS:
        reason = payload.get("reason")
        if "reason" not in payload:
            v.append("missing required key: reason")
        elif not isinstance(reason, str) or not reason:
            v.append("reason must be a non-empty string")

    return v


def _validate_qualified(payload):
    v = []
    for key in QUALIFIED_KEYS:
        if key not in payload:
            v.append("missing required key: {}".format(key))

    num_keys = ("entry_low", "entry_high", "stop",
                "r1", "r2", "r3", "r4", "r5", "expiration_price")
    for key in num_keys:
        if key in payload and not _is_num(payload[key]):
            v.append("{} must be a number".format(key))
    for key in ("next_obstacle", "room_r"):
        if key in payload and not _is_num_or_null(payload[key]):
            v.append("{} must be a number or null".format(key))

    score = payload.get("score")
    if "score" in payload:
        if not isinstance(score, int) or isinstance(score, bool) \
                or not (0 <= score <= 5):
            v.append("score must be an integer 0-5, got {!r}".format(score))

    if "expiration_time_ct" in payload \
            and not isinstance(payload["expiration_time_ct"], str):
        v.append("expiration_time_ct must be a string")

    # Structural checks only when the numbers are actually numbers.
    entry_low = payload.get("entry_low")
    entry_high = payload.get("entry_high")
    stop = payload.get("stop")
    direction = payload.get("direction")
    rs = [payload.get("r{}".format(i)) for i in range(1, 6)]

    if _is_num(entry_low) and _is_num(entry_high) and entry_low > entry_high:
        v.append("entry_low ({}) must be <= entry_high ({})".format(
            _fmt(entry_low), _fmt(entry_high)))

    if direction in DIRECTIONS and _is_num(stop) \
            and _is_num(entry_low) and _is_num(entry_high):
        if direction == "CALL" and stop >= entry_low:
            v.append("stop ({}) must be below entry_low ({}) for a CALL"
                     .format(_fmt(stop), _fmt(entry_low)))
        if direction == "PUT" and stop <= entry_high:
            v.append("stop ({}) must be above entry_high ({}) for a PUT"
                     .format(_fmt(stop), _fmt(entry_high)))

    if direction in DIRECTIONS and all(_is_num(r) for r in rs):
        for i in range(4):
            a, b = rs[i], rs[i + 1]
            if direction == "CALL" and b <= a:
                v.append("r{} ({}) must be above r{} ({}) for a CALL".format(
                    i + 2, _fmt(b), i + 1, _fmt(a)))
            if direction == "PUT" and b >= a:
                v.append("r{} ({}) must be below r{} ({}) for a PUT".format(
                    i + 2, _fmt(b), i + 1, _fmt(a)))

    return v


# ---------------------------------------------------------------------------
# 3. Risk math cross-check (QUALIFIED only)
# ---------------------------------------------------------------------------

def risk_check(payload):
    """Cross-check r1..r5 against entry_low +/- N*risk.

    Returns (risk_per_share, list_of_report_lines). Convention documented in
    the module docstring: risk = abs(entry_low - stop), anchored at entry_low.
    """
    entry_low = payload["entry_low"]
    stop = payload["stop"]
    direction = payload["direction"]
    risk = abs(entry_low - stop)
    sign = 1.0 if direction == "CALL" else -1.0

    discrepancies = []
    for n in range(1, 6):
        expected = entry_low + sign * n * risk
        actual = payload["r{}".format(n)]
        if abs(actual - expected) > TICK + 1e-9:
            discrepancies.append(
                "r{}: payload {} vs computed {} (off by {:.2f})".format(
                    n, _fmt(actual), _fmt(expected), abs(actual - expected)))

    if discrepancies:
        lines = ["R level check: DISCREPANCIES FOUND"]
        lines.extend("  " + d for d in discrepancies)
    else:
        lines = ["R levels verified (each within one tick of "
                 "entry_low {} N x {:.2f})".format(
                     "+" if sign > 0 else "-", risk)]
    return risk, lines


# ---------------------------------------------------------------------------
# 4. Plan gates
# ---------------------------------------------------------------------------

def plan_gates(payload):
    """Return list of 'PASS/FAIL  <gate>' lines for a QUALIFIED payload."""
    lines = []

    rvol = payload.get("rvol")
    if _is_num(rvol):
        ok = rvol >= GATE_RVOL_MIN
        lines.append(("PASS" if ok else "FAIL",
                      "RVOL >= {:.1f}  (rvol = {:.2f})".format(GATE_RVOL_MIN,
                                                               rvol)))
    else:
        lines.append(("FAIL", "RVOL >= {:.1f}  (rvol missing)".format(
            GATE_RVOL_MIN)))

    room_r = payload.get("room_r")
    obstacle = payload.get("next_obstacle")
    if obstacle is None:
        lines.append(("PASS", "Room >= {:.0f}R  (no obstacle - open space)"
                      .format(GATE_ROOM_R_MIN)))
    elif _is_num(room_r):
        ok = room_r >= GATE_ROOM_R_MIN
        lines.append(("PASS" if ok else "FAIL",
                      "Room >= {:.0f}R  (room_r = {:.2f})".format(
                          GATE_ROOM_R_MIN, room_r)))
    else:
        lines.append(("FAIL", "Room >= {:.0f}R  (room_r missing)".format(
            GATE_ROOM_R_MIN)))

    score = payload.get("score")
    ok = isinstance(score, int) and score >= GATE_SCORE_MIN
    lines.append(("PASS" if ok else "FAIL",
                  "Score >= {}  (score = {})".format(GATE_SCORE_MIN, score)))

    dt = _parse_ct(payload.get("signal_time_ct"))
    if dt is not None:
        hm = (dt.hour, dt.minute)
        ok = GATE_TIME_START <= hm <= GATE_TIME_END
        lines.append(("PASS" if ok else "FAIL",
                      "Signal time 08:45-10:30 CT  (time = {:02d}:{:02d})"
                      .format(dt.hour, dt.minute)))
    else:
        lines.append(("FAIL", "Signal time 08:45-10:30 CT  (unparseable)"))

    return ["{}  {}".format(flag, text) for flag, text in lines]


# ---------------------------------------------------------------------------
# 5. Card rendering
# ---------------------------------------------------------------------------

def _rule(ch):
    return ch * WIDTH


def _boxed(title_lines, body_lines):
    out = [_rule("=")]
    out.extend(title_lines)
    out.append(_rule("="))
    out.extend(body_lines)
    out.append(_rule("-"))
    out.append(FOOTER)
    out.append(_rule("="))
    return "\n".join(out)


def render_validation_error_card(payload, violations):
    ident = "unknown"
    if isinstance(payload, dict) and isinstance(payload.get("event_id"), str):
        ident = payload["event_id"]
    body = ["The payload failed schema validation "
            "({} violation{}):".format(len(violations),
                                       "" if len(violations) == 1 else "s"),
            ""]
    body.extend("  * {}".format(x) for x in violations)
    return _boxed(["VALIDATION ERROR", "event_id: {}".format(ident)], body)


def _feasibility_for_risk(risk):
    """Evaluate the strategy.md section 7 $25 loss gate for one chart-R value.

    The trader's own allowed delta band is 0.45-0.60 (strategy.md section 7).
    This NEVER returns a delta outside that band -- if the unclamped math
    would put the max-feasible delta above 0.60 or below 0.45, that fact is
    reported in words (band fully open / fully closed) instead of as a
    number, because a number that looks computed but sits outside the
    trader's own rules invites hunting a strike the plan does not allow.

    Returns a dict: {"verdict": "OPEN"|"CAPPED"|"INFEASIBLE"|"NA",
                     "detail": str, "delta": float or None}.
    The evaluation is done at FEASIBILITY_WIDE_SPREAD (0.05), the strategy's
    own "wide spread" boundary -- i.e. it is the conservative pre-check, not
    a guarantee for whatever spread is actually on the chain.
    """
    if risk is None or risk <= 0:
        return {"verdict": "NA", "detail": "n/a (non-positive chart-R)",
                "delta": None}

    spread = FEASIBILITY_WIDE_SPREAD
    cap_cents = FEASIBILITY_CAP_PER_SHARE * 100.0

    # Worst combination still inside the allowed band: delta at the band top,
    # spread at the wide-spread boundary. If that still clears $25, the
    # whole band is open -- no delta number needs to be shown at all.
    cost_band_top = (FEASIBILITY_DELTA_MAX * risk + spread) * 100.0
    if cost_band_top <= cap_cents + 1e-9:
        return {
            "verdict": "OPEN",
            "detail": "$25 cap not binding: the entire 0.45-0.60 delta "
                      "band passes at spread <= 0.05.",
            "delta": None,
        }

    # Best combination inside the band: delta at the band floor, spread 0.
    # If even that fails, no strike in the allowed band can pass -- skip.
    cost_band_bottom_best_spread = (FEASIBILITY_DELTA_MIN * risk) * 100.0
    if cost_band_bottom_best_spread > cap_cents + 1e-9:
        return {
            "verdict": "INFEASIBLE",
            "detail": "even delta {:.2f} at spread 0.00 exceeds $25 - "
                      "skip.".format(FEASIBILITY_DELTA_MIN),
            "delta": None,
        }

    # Cap binds somewhere inside the band: report the ceiling, clamped so it
    # can never print outside [DELTA_MIN, DELTA_MAX].
    raw_ceiling = (FEASIBILITY_CAP_PER_SHARE - spread) / risk
    delta_ceiling = max(FEASIBILITY_DELTA_MIN,
                        min(FEASIBILITY_DELTA_MAX, raw_ceiling))
    return {
        "verdict": "CAPPED",
        "detail": "max usable delta ~{:.2f} (below the {:.2f} band top)."
                  .format(delta_ceiling, FEASIBILITY_DELTA_MAX),
        "delta": delta_ceiling,
    }


def _feasibility_render(result):
    if result["verdict"] == "INFEASIBLE":
        return "INFEASIBLE - {}".format(result["detail"])
    return "FEASIBLE - {}".format(result["detail"])


def _feasibility_header(result):
    if result["verdict"] == "INFEASIBLE":
        return "INFEASIBLE - {}".format(result["detail"])
    return "FEASIBLE (pre-check) - {}".format(result["detail"])


def _worst_fill_risk(payload):
    """Chart-R at the worst allowed fill: the expiration_price (do-not-chase
    limit), not entry_low. strategy.md section 4 permits entry above the
    confirmation candle's high, so a fill anywhere up to expiration_price is
    in-plan and understates loss-at-stop if only entry_low is checked."""
    stop = payload.get("stop")
    expiration_price = payload.get("expiration_price")
    direction = payload.get("direction")
    if not (_is_num(stop) and _is_num(expiration_price)):
        return None
    if direction == "CALL":
        return expiration_price - stop
    if direction == "PUT":
        return stop - expiration_price
    return None


_FEASIBILITY_SEVERITY = {"OPEN": 0, "CAPPED": 1, "INFEASIBLE": 2}


def _contract_selection_lines(payload, risk):
    """Computed contract-selection + feasibility lines for a QUALIFIED card.

    Everything here is derived deterministically from the payload and the
    strategy.md section 7 formula. Strike/limit lines remain hand-check
    reminders (the live option chain is not available offline) -- see
    OPTION_CHECKLIST for the boxes that still require a manual look.
    """
    p = payload
    lines = []

    if p["direction"] == "CALL":
        lines.append("Direction    : CALL (bullish break/hold)")
    else:
        lines.append("Direction    : PUT (bearish breakdown)")

    lines.append("Chart-R      : {:.2f}".format(risk))

    dt = _parse_ct(p.get("signal_time_ct"))
    if dt is not None:
        earliest = dt + timedelta(days=DTE_MIN_DAYS)
        latest = dt + timedelta(days=DTE_MAX_DAYS)
        lines.append(
            "Target expiry: first listed between {} and {} "
            "({}-{} DTE)".format(earliest.strftime("%Y-%m-%d"),
                                 latest.strftime("%Y-%m-%d"),
                                 DTE_MIN_DAYS, DTE_MAX_DAYS))
    else:
        lines.append("Target expiry: {}-{} DTE (signal date unparseable)"
                     .format(DTE_MIN_DAYS, DTE_MAX_DAYS))

    lines.append(
        "Strike       : nearest-ATM to CURRENT PRICE (signal close was {}, "
        "price may have moved since) - delta {:.2f}-{:.2f} "
        "(prefer 0.50-0.55, slightly ITM)".format(
            _fmt(p["entry_low"]), FEASIBILITY_DELTA_MIN,
            FEASIBILITY_DELTA_MAX))

    lines.append("Limit        : at Mark (mid of bid/ask); nudge "
                 "+0.01-0.02 toward ask if unfilled")

    lines.append("Stop         : {} on the underlying, or -$25 on the "
                 "option (whichever first)".format(_fmt(p["stop"])))

    best = _feasibility_for_risk(risk)
    if best["verdict"] == "NA":
        lines.append("Feasibility  : n/a (non-positive chart-R)")
        return lines

    worst_risk = _worst_fill_risk(p)
    worst = None
    if worst_risk is not None and worst_risk > 0:
        worst = _feasibility_for_risk(worst_risk)

    governing = best
    if worst is not None and (_FEASIBILITY_SEVERITY[worst["verdict"]]
                              >= _FEASIBILITY_SEVERITY[best["verdict"]]):
        governing = worst

    lines.append(
        "Feasibility  : {} Pre-check at assumed spread <= 0.05; recheck "
        "with the ACTUAL delta and spread from the chain."
        .format(_feasibility_header(governing)))
    lines.append(
        "  Best  fill (chart R {:.2f}, entry near {}): {}".format(
            risk, _fmt(p["entry_low"]), _feasibility_render(best)))
    if worst is not None:
        lines.append(
            "  Worst fill (chart R {:.2f}, at do-not-chase limit {}): {}"
            .format(worst_risk, _fmt(p["expiration_price"]),
                    _feasibility_render(worst)))
        if worst["verdict"] != best["verdict"]:
            lines.append(
                "  NOTE: best-fill and worst-fill verdicts differ - the "
                "worse case (worst fill) governs.")
    else:
        lines.append("  Worst fill: n/a (expiration_price/stop not usable)")

    return lines


def render_qualified_card(payload):
    p = payload
    risk, risk_lines = risk_check(p)
    gates = plan_gates(p)

    if p["next_obstacle"] is None:
        room_line = "Room to next obstacle : open space (no obstacle mapped)"
    else:
        room_line = ("Room to next obstacle : {}R to {} "
                     .format(_fmt(p["room_r"]), _fmt(p["next_obstacle"])))
        room_line = room_line.rstrip()

    body = [
        "Setup type   : {}".format(p["setup_type"]),
        "Level        : {} at {}".format(p["level"], _fmt(p["level_price"])),
        "Timeframe    : {} min".format(p["timeframe"]),
        _rule("-"),
        "Entry zone   : {} - {}".format(_fmt(p["entry_low"]),
                                        _fmt(p["entry_high"])),
        "Stop         : {}".format(_fmt(p["stop"])),
        "Risk / share : {:.2f}".format(risk),
        "  1R target  : {}".format(_fmt(p["r1"])),
        "  2R target  : {}".format(_fmt(p["r2"])),
        "  3R target  : {}".format(_fmt(p["r3"])),
        "  4R target  : {}".format(_fmt(p["r4"])),
        "  5R target  : {}".format(_fmt(p["r5"])),
        room_line,
        "Score        : {} / 5".format(p["score"]),
        "VWAP         : {}    RVOL: {}".format(_fmt(p["vwap"]),
                                               _fmt(p["rvol"])),
        "Signal time  : {} CT".format(p["signal_time_ct"]),
        "Expires      : {} CT at {} "
        "(do not chase beyond)".format(p["expiration_time_ct"],
                                       _fmt(p["expiration_price"])),
        _rule("-"),
        "Risk math cross-check:",
    ]
    body.extend("  " + line for line in risk_lines)
    body.append(_rule("-"))
    body.append("Plan gates (flags only - human decides):")
    body.extend("  " + g for g in gates)
    body.append(_rule("-"))
    body.append("Contract selection & feasibility "
                "(computed; strike/limit remain hand-check reminders):")
    body.extend("  " + line
                for line in _contract_selection_lines(p, risk))
    body.append(_rule("-"))
    body.append("Option-contract checklist "
                "(check by hand in Robinhood - reminders only, NOT computed):")
    body.extend("  [ ] {}".format(item) for item in OPTION_CHECKLIST)

    title = [
        "QUALIFIED SETUP: {} {}".format(p["ticker"], p["direction"]),
        "event_id: {}".format(p["event_id"]),
    ]
    return _boxed(title, body)


def render_watch_card(payload):
    p = payload
    body = [
        "{} {} broken at {} - watching for retest.".format(
            p["level"], _fmt(p["level_price"]),
            p["signal_time_ct"] + " CT"),
        "Setup type : {}   Timeframe: {} min".format(p["setup_type"],
                                                     p["timeframe"]),
        "VWAP: {}    RVOL: {}".format(_fmt(p["vwap"]), _fmt(p["rvol"])),
    ]
    title = [
        "WATCH - NOT A SETUP. NOT AN ENTRY. Do nothing.",
        "{} {} - level broken, watching for retest".format(
            p["ticker"], p["direction"]),
        "event_id: {}".format(p["event_id"]),
    ]
    return _boxed(title, body)


def render_terminal_card(payload):
    """REJECT / INVALIDATED / EXPIRED."""
    p = payload
    reason = p["reason"]
    known = reason in KNOWN_REASONS.get(p["event"], ())
    body = [
        "REASON: {}".format(reason),
    ]
    if not known:
        body.append("(note: reason not in the known list for {})".format(
            p["event"]))
    body.extend([
        _rule("-"),
        "Level      : {} at {}".format(p["level"], _fmt(p["level_price"])),
        "Setup type : {}   Timeframe: {} min".format(p["setup_type"],
                                                     p["timeframe"]),
        "Time       : {} CT".format(p["signal_time_ct"]),
        "VWAP: {}    RVOL: {}".format(_fmt(p["vwap"]), _fmt(p["rvol"])),
    ])
    title = [
        "{}: {} {}".format(p["event"], p["ticker"], p["direction"]),
        "event_id: {}".format(p["event_id"]),
    ]
    return _boxed(title, body)


def render_card(payload):
    """Validate and render. Always returns card text (never raises on bad
    payload content)."""
    violations = validate_payload(payload)
    if violations:
        return render_validation_error_card(payload, violations)
    event = payload["event"]
    if event == "QUALIFIED":
        return render_qualified_card(payload)
    if event == "WATCH":
        return render_watch_card(payload)
    return render_terminal_card(payload)


# ---------------------------------------------------------------------------
# 2. Dedup store
# ---------------------------------------------------------------------------

def load_seen(store_path=SEEN_STORE_PATH):
    try:
        with open(store_path, "r") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return set(x for x in data if isinstance(x, str))
    except (OSError, ValueError):
        pass
    return set()


def save_seen(seen, store_path=SEEN_STORE_PATH):
    with open(store_path, "w") as fh:
        json.dump(sorted(seen), fh, indent=1)


def reset_seen(store_path=SEEN_STORE_PATH):
    if os.path.exists(store_path):
        os.remove(store_path)


def process_payload(payload, dedupe=True, store_path=SEEN_STORE_PATH):
    """Full pipeline for one payload: validate -> dedupe -> card.

    Returns the text to display (a card, or a one-line duplicate notice).
    """
    event_id = payload.get("event_id") if isinstance(payload, dict) else None
    if dedupe and isinstance(event_id, str) and event_id:
        seen = load_seen(store_path)
        if event_id in seen:
            return "DUPLICATE event_id {} - already processed, ignored.".format(
                event_id)
        seen.add(event_id)
        save_seen(seen, store_path)
    return render_card(payload)


# ---------------------------------------------------------------------------
# Demo payloads (built-in, no files needed)
# ---------------------------------------------------------------------------

def demo_payloads():
    return [
        {
            "event": "WATCH",
            "event_id": "SPY-20260713-CALL-PDH-WATCH-9001",
            "ticker": "SPY", "timeframe": "5",
            "setup_type": "A_break_retest", "direction": "CALL",
            "level": "PDH", "level_price": 628.40,
            "signal_time_ct": "2026-07-13 08:52",
            "vwap": 627.90, "rvol": 1.31,
        },
        {
            "event": "QUALIFIED",
            "event_id": "SPY-20260713-CALL-PDH-QUALIFIED-9002",
            "ticker": "SPY", "timeframe": "5",
            "setup_type": "A_break_retest", "direction": "CALL",
            "level": "PDH", "level_price": 628.40,
            "signal_time_ct": "2026-07-13 09:07",
            "vwap": 628.05, "rvol": 1.38,
            "entry_low": 628.60, "entry_high": 628.80, "stop": 628.20,
            "r1": 629.00, "r2": 629.40, "r3": 629.80,
            "r4": 630.20, "r5": 630.60,
            "next_obstacle": None, "room_r": None, "score": 4,
            "expiration_time_ct": "2026-07-13 09:22",
            "expiration_price": 628.95,
        },
        {
            "event": "INVALIDATED",
            "event_id": "SPY-20260713-CALL-PDH-INVALIDATED-9003",
            "ticker": "SPY", "timeframe": "5",
            "setup_type": "A_break_retest", "direction": "CALL",
            "level": "PDH", "level_price": 628.40,
            "signal_time_ct": "2026-07-13 09:31",
            "vwap": 628.10, "rvol": 1.22,
            "reason": "closed_through_stop_after_qualified",
        },
        {
            "event": "EXPIRED",
            "event_id": "QQQ-20260713-PUT-PDL-EXPIRED-9004",
            "ticker": "QQQ", "timeframe": "5",
            "setup_type": "B_breakdown_bounce", "direction": "PUT",
            "level": "PDL", "level_price": 556.10,
            "signal_time_ct": "2026-07-13 09:40",
            "vwap": 556.60, "rvol": 1.05,
            "reason": "retest_window_elapsed",
        },
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="receiver.py",
        description="Render TradingView alert JSON payloads as human-review "
                    "decision cards. Display/validation only - no brokerage "
                    "connectivity of any kind.")
    parser.add_argument("files", nargs="*",
                        help="JSON payload file(s) to process")
    parser.add_argument("--all", action="store_true",
                        help="process every fixture in signals/fixtures/ "
                             "in name sort")
    parser.add_argument("--demo", action="store_true",
                        help="run built-in sample payloads through the "
                             "complete pipeline")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="bypass the event_id dedupe store")
    parser.add_argument("--reset-seen", action="store_true",
                        help="clear the dedupe store and continue")
    args = parser.parse_args(argv)

    if args.reset_seen:
        reset_seen()
        print("Dedupe store cleared: {}".format(SEEN_STORE_PATH))

    dedupe = not args.no_dedupe

    paths = list(args.files)
    if args.all:
        try:
            names = sorted(n for n in os.listdir(FIXTURES_DIR)
                           if n.endswith(".json"))
        except OSError as exc:
            print("ERROR: cannot list fixtures dir: {}".format(exc),
                  file=sys.stderr)
            return 2
        paths.extend(os.path.join(FIXTURES_DIR, n) for n in names)

    if not paths and not args.demo:
        if args.reset_seen:
            return 0
        parser.print_usage()
        print("No input given. Pass payload files, --all, or --demo.",
              file=sys.stderr)
        return 2

    exit_code = 0
    for path in paths:
        try:
            with open(path, "r") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            print("ERROR: cannot read/parse {}: {}".format(path, exc),
                  file=sys.stderr)
            exit_code = 2
            continue
        print("### {}".format(os.path.basename(path)))
        print(process_payload(payload, dedupe=dedupe))
        print()

    if args.demo:
        for payload in demo_payloads():
            print("### demo: {} {}".format(payload["event"],
                                           payload["event_id"]))
            print(process_payload(payload, dedupe=dedupe))
            print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
