#!/usr/bin/env python3
"""record_session.py -- validate and append one session-decision-accuracy
record: did the engine actually QUALIFIED anything that day, and did the
trader actually act -- with the trader's own before/after belief noted.

Closes the blind spot where the learning loop only recorded TRADES and
BACKTEST RUNS: a no-trade day (the best-behaved outcome in the system) used
to leave zero rows anywhere.

Usage:
    # Normal case: a replay directory with events.json already exists.
    python3 analysis/record_session.py \\
        --session backtests/session_2026-07-13 \\
        --trades-taken 0 --off-plan-actions 0 \\
        --believed-state "two setups qualified" \\
        --actual-state "zero qualified" \\
        --notes "hesitation was CORRECT; WATCH mistaken for QUALIFIED"

    # No replay exists for the day (e.g. an earlier off-plan session):
    python3 analysis/record_session.py \\
        --no-events --session-date 2026-07-12 \\
        --trades-taken 1 --off-plan-actions 5 --net-r -1.04 \\
        --believed-state "unknown" --actual-state "unknown" \\
        --notes "chased a failing ORH breakout with no qualified setup"

`verdict` is NEVER taken as a CLI argument -- it is always derived from the
2x2 confusion matrix (engine had a QUALIFIED?) x (trader took a trade?):

    QUALIFIED>0, traded   -> CORRECT_TRADE
    QUALIFIED>0, no trade -> MISSED_SIGNAL
    QUALIFIED=0, traded   -> OFF_PLAN_TRADE
    QUALIFIED=0, no trade -> CORRECT_NO_TRADE

Appends one JSON object as a single line to backtests/sessions.jsonl
(path relative to the DayTrading root, resolved from this script's
location; override with --sessions for testing).

Exit codes:
    0  record appended
    2  invalid input / schema validation failure (stderr names the field)
    3  duplicate session_date (record refused; use --force to override)

ADVISORY-ONLY TOOLING. No brokerage access. Records describe engine replay
output and the trader's own actions; nothing here places, stages, or
recommends orders.
"""

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SESSIONS = os.path.join(ROOT, "backtests", "sessions.jsonl")

EVENT_TYPES = ("WATCH", "QUALIFIED", "REJECT", "EXPIRED", "INVALIDATED")
ENGINE_KEYS = ("watch", "qualified", "reject", "expired", "invalidated",
               "pre_window_rejects")
VERDICTS = ("CORRECT_TRADE", "CORRECT_NO_TRADE", "MISSED_SIGNAL",
            "OFF_PLAN_TRADE")

# Entry window opens 08:45 CT (see strategy.md / AGENTS.md); a REJECT whose
# signal_time_ct clock time is strictly before this is a pre-window reject.
ENTRY_WINDOW_START = (8, 45)

ENGINE_UNKNOWN = {k: None for k in ENGINE_KEYS}


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _parse_ct(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- events.json

def load_events(session_dir):
    """Load <session_dir>/events.json. Raises OSError/ValueError on failure
    (caller decides how to report it)."""
    path = os.path.join(session_dir, "events.json")
    with open(path, "r") as fh:
        return json.load(fh)


def compute_engine_counts(events):
    """Return the engine dict (see ENGINE_KEYS) derived from a list of
    canonical-schema event dicts (see AGENTS.md)."""
    counts = {k: 0 for k in ENGINE_KEYS}
    for e in events:
        if not isinstance(e, dict):
            continue
        etype = e.get("event")
        key = etype.lower() if isinstance(etype, str) else None
        if key in counts:
            counts[key] += 1
        if etype == "REJECT":
            dt = _parse_ct(e.get("signal_time_ct"))
            if dt is not None and (dt.hour, dt.minute) < ENTRY_WINDOW_START:
                counts["pre_window_rejects"] += 1
    return counts


def derive_tickers(session_dir, events):
    """Prefer levels.json's ticker order (matches the replay's own
    per-ticker processing order); fall back to first-appearance order in
    events.json if levels.json is unavailable or malformed."""
    levels_path = os.path.join(session_dir, "levels.json")
    if os.path.exists(levels_path):
        try:
            with open(levels_path, "r") as fh:
                data = json.load(fh)
            tickers = data.get("tickers")
            if isinstance(tickers, dict) and tickers:
                return list(tickers.keys())
        except (OSError, ValueError):
            pass
    seen = []
    for e in events:
        if not isinstance(e, dict):
            continue
        t = e.get("ticker")
        if isinstance(t, str) and t not in seen:
            seen.append(t)
    return seen


# ------------------------------------------------------------------ verdict

def engine_had_qualified(engine):
    """True iff the engine dict shows at least one QUALIFIED. None/unknown
    counts (--no-events mode) is treated as "no" -- absence of an events.json
    can never be used to assert a QUALIFIED occurred."""
    return bool(engine.get("qualified") or 0)


def derive_verdict(engine, trades_taken):
    qualified = engine_had_qualified(engine)
    traded = bool(trades_taken and trades_taken > 0)
    if qualified and traded:
        return "CORRECT_TRADE"
    if qualified and not traded:
        return "MISSED_SIGNAL"
    if (not qualified) and traded:
        return "OFF_PLAN_TRADE"
    return "CORRECT_NO_TRADE"


# --------------------------------------------------------------- validation

def validate_record(rec):
    """Return a list of 'field: problem' strings; empty list means valid."""
    errors = []
    if not isinstance(rec, dict):
        return ["record: must be a JSON object"]

    session_date = rec.get("session_date")
    if not isinstance(session_date, str):
        errors.append("session_date: required string 'YYYY-MM-DD'")
    else:
        try:
            datetime.strptime(session_date, "%Y-%m-%d")
        except ValueError:
            errors.append("session_date: must match 'YYYY-MM-DD'")

    recorded_ct = rec.get("recorded_ct")
    if not isinstance(recorded_ct, str):
        errors.append("recorded_ct: required string 'YYYY-MM-DD HH:MM'")
    else:
        try:
            datetime.strptime(recorded_ct, "%Y-%m-%d %H:%M")
        except ValueError:
            errors.append("recorded_ct: must match 'YYYY-MM-DD HH:MM'")

    tickers = rec.get("tickers")
    if not isinstance(tickers, list) or not all(
            isinstance(t, str) and t for t in tickers):
        errors.append("tickers: required list of non-empty strings "
                       "(may be empty list)")

    if "replay_dir" in rec and rec["replay_dir"] is not None \
            and not isinstance(rec["replay_dir"], str):
        errors.append("replay_dir: must be a string or null")

    engine = rec.get("engine")
    if not isinstance(engine, dict):
        errors.append("engine: required object")
    else:
        for key in ENGINE_KEYS:
            if key not in engine:
                errors.append("engine.%s: key required (value may be null)"
                              % key)
                continue
            v = engine[key]
            if v is None:
                continue
            if not _is_int(v) or v < 0:
                errors.append("engine.%s: must be a non-negative integer "
                              "or null" % key)

    trader = rec.get("trader")
    if not isinstance(trader, dict):
        errors.append("trader: required object")
    else:
        for key in ("trades_taken", "off_plan_actions"):
            v = trader.get(key)
            if key not in trader:
                errors.append("trader.%s: key required" % key)
            elif not _is_int(v) or v < 0:
                errors.append("trader.%s: must be a non-negative integer"
                              % key)
        for key in ("believed_state", "actual_state"):
            v = trader.get(key)
            if key not in trader:
                errors.append("trader.%s: key required" % key)
            elif not isinstance(v, str) or not v.strip():
                errors.append("trader.%s: must be a non-empty string" % key)

    verdict = rec.get("verdict")
    if verdict not in VERDICTS:
        errors.append("verdict: must be one of %s, got %r"
                      % (VERDICTS, verdict))

    net_r = rec.get("net_r")
    if "net_r" not in rec:
        errors.append("net_r: key required (value may be null)")
    elif net_r is not None and not _is_number(net_r):
        errors.append("net_r: must be a number or null")

    if "notes" not in rec or not isinstance(rec.get("notes"), str):
        errors.append("notes: required string (may be empty)")
    elif "\n" in rec["notes"]:
        errors.append("notes: must not contain newlines")

    return errors


def existing_session_dates(path):
    dates = set()
    if not os.path.exists(path):
        return dates
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue  # malformed lines cannot claim a session_date
            if isinstance(obj, dict) and isinstance(obj.get("session_date"), str):
                dates.add(obj["session_date"])
    return dates


# ---------------------------------------------------------------- building

def session_date_from_dir(session_dir):
    """'backtests/session_2026-07-13' -> '2026-07-13', else None."""
    base = os.path.basename(os.path.normpath(session_dir))
    prefix = "session_"
    if base.startswith(prefix):
        candidate = base[len(prefix):]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            return None
    return None


def build_record(session_dir=None, session_date=None, no_events=False,
                 tickers_override=None, trades_taken=0, off_plan_actions=0,
                 believed_state="", actual_state="", notes="", net_r=None,
                 recorded_ct=None):
    """Assemble a session record (without validating/writing it). Returns
    (record_dict, list_of_build_errors). Build errors are things that
    prevent even constructing a candidate record (e.g. missing events.json);
    they are distinct from schema validation errors."""
    build_errors = []

    if no_events:
        engine = dict(ENGINE_UNKNOWN)
        events = []
        replay_dir = None
        derived_date = None
        if tickers_override is not None:
            tickers = list(tickers_override)
        else:
            tickers = []
    else:
        if session_dir is None:
            build_errors.append("--session is required unless --no-events "
                                "is given")
            return None, build_errors
        try:
            events = load_events(session_dir)
        except (OSError, ValueError) as exc:
            build_errors.append("could not read events.json in %s: %s"
                                % (session_dir, exc))
            return None, build_errors
        if not isinstance(events, list):
            build_errors.append("events.json in %s is not a JSON list"
                                % session_dir)
            return None, build_errors
        engine = compute_engine_counts(events)
        replay_dir = os.path.relpath(os.path.abspath(session_dir), ROOT)
        derived_date = session_date_from_dir(session_dir)
        if tickers_override is not None:
            tickers = list(tickers_override)
        else:
            tickers = derive_tickers(session_dir, events)

    final_session_date = session_date or derived_date
    if not final_session_date:
        build_errors.append(
            "session_date could not be determined: pass --session-date "
            "explicitly (the --session directory name did not match "
            "'session_<YYYY-MM-DD>')")
        return None, build_errors

    verdict = derive_verdict(engine, trades_taken)

    rec = {
        "session_date": final_session_date,
        "recorded_ct": recorded_ct or datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tickers": tickers,
        "replay_dir": replay_dir,
        "engine": engine,
        "trader": {
            "trades_taken": trades_taken,
            "off_plan_actions": off_plan_actions,
            "believed_state": believed_state,
            "actual_state": actual_state,
        },
        "verdict": verdict,
        "net_r": net_r,
        "notes": notes,
    }
    return rec, build_errors


# ---------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and append one session-decision-accuracy "
                    "record (advisory tooling only).")
    default_session = os.path.join(
        "backtests", "session_%s" % datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--session", default=default_session,
                        help="path to the replay session directory "
                             "containing events.json (default: %(default)s)")
    parser.add_argument("--session-date", default=None,
                        help="override session_date; required with "
                             "--no-events unless --session's dirname is "
                             "'session_<YYYY-MM-DD>'")
    parser.add_argument("--no-events", action="store_true",
                        help="record a session with no events.json "
                             "available; engine counts are recorded as "
                             "unknown (null)")
    parser.add_argument("--tickers", default=None,
                        help="comma-separated ticker override (required "
                             "with --no-events if you want tickers listed; "
                             "otherwise auto-derived from levels.json / "
                             "events.json)")
    parser.add_argument("--trades-taken", type=int, required=True)
    parser.add_argument("--off-plan-actions", type=int, required=True)
    parser.add_argument("--believed-state", required=True)
    parser.add_argument("--actual-state", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--net-r", type=float, default=None)
    parser.add_argument("--recorded-ct", default=None,
                        help="override recorded_ct 'YYYY-MM-DD HH:MM' "
                             "(default: now)")
    parser.add_argument("--sessions", default=DEFAULT_SESSIONS,
                        help="path to sessions.jsonl (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="allow appending a duplicate session_date")
    args = parser.parse_args(argv)

    tickers_override = None
    if args.tickers is not None:
        tickers_override = [t.strip() for t in args.tickers.split(",")
                            if t.strip()]

    rec, build_errors = build_record(
        session_dir=None if args.no_events else args.session,
        session_date=args.session_date,
        no_events=args.no_events,
        tickers_override=tickers_override,
        trades_taken=args.trades_taken,
        off_plan_actions=args.off_plan_actions,
        believed_state=args.believed_state,
        actual_state=args.actual_state,
        notes=args.notes,
        net_r=args.net_r,
        recorded_ct=args.recorded_ct,
    )
    if build_errors:
        sys.stderr.write("error: could not build record:\n")
        for err in build_errors:
            sys.stderr.write("  - %s\n" % err)
        return 2

    errors = validate_record(rec)
    if errors:
        sys.stderr.write("Schema validation failed:\n")
        for err in errors:
            sys.stderr.write("  - %s\n" % err)
        return 2

    if not args.force and rec["session_date"] in existing_session_dates(args.sessions):
        sys.stderr.write(
            "error: duplicate session_date '%s' already recorded in %s -- "
            "refused (use --force to override)\n"
            % (rec["session_date"], args.sessions))
        return 3

    sessions_dir = os.path.dirname(os.path.abspath(args.sessions))
    if sessions_dir and not os.path.isdir(sessions_dir):
        os.makedirs(sessions_dir)
    with open(args.sessions, "a") as fh:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print("recorded session_date '%s' -> %s (verdict: %s)"
         % (rec["session_date"], args.sessions, rec["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
