#!/usr/bin/env python3
"""record_backtest.py -- validate and append one backtest run record.

Usage:
    python3 analysis/record_backtest.py --json '{...whole record...}'

Appends one JSON object as a single line to backtests/results.jsonl
(path relative to the DayTrading root, resolved from this script's
location; override with --results for testing).

Exit codes:
    0  record appended
    2  invalid JSON or schema validation failure (stderr names the field)
    3  duplicate run_id (record refused)

ADVISORY-ONLY TOOLING. No brokerage access. Records describe simulated
backtests; nothing here places, stages, or recommends orders.
"""

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(ROOT, "backtests", "results.jsonl")

LEVEL_SOURCES = ("PDH/PDL", "ORH/ORL")
ENTRY_MODES = ("Retest only", "Break only", "Break or retest")
EXIT_MODES = ("Fixed 2R", "Fixed 3R", "Fixed 5R", "Trail after 2R")
METRIC_KEYS = (
    "trades", "wins", "win_rate_pct", "net_r", "avg_r",
    "profit_factor_r", "max_dd_r", "best_r", "worst_r",
)
INT_METRICS = ("trades", "wins")


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def validate_record(rec):
    """Return a list of 'field: problem' strings; empty list means valid."""
    errors = []
    if not isinstance(rec, dict):
        return ["record: must be a JSON object"]

    def req_str(field):
        v = rec.get(field)
        if not isinstance(v, str) or not v.strip():
            errors.append("%s: required non-empty string" % field)
            return None
        if "\n" in v:
            errors.append("%s: must not contain newlines" % field)
            return None
        return v

    req_str("run_id")
    req_str("symbol")
    req_str("strategy")

    recorded_ct = rec.get("recorded_ct")
    if not isinstance(recorded_ct, str):
        errors.append("recorded_ct: required string 'YYYY-MM-DD HH:MM'")
    else:
        try:
            datetime.strptime(recorded_ct, "%Y-%m-%d %H:%M")
        except ValueError:
            errors.append("recorded_ct: must match 'YYYY-MM-DD HH:MM'")

    level_source = rec.get("level_source")
    if level_source not in LEVEL_SOURCES:
        errors.append("level_source: must be one of %s" % (LEVEL_SOURCES,))

    config = rec.get("config")
    if not isinstance(config, dict):
        errors.append("config: required object")
    else:
        if config.get("entry_mode") not in ENTRY_MODES:
            errors.append("config.entry_mode: must be one of %s" % (ENTRY_MODES,))
        if config.get("exit_mode") not in EXIT_MODES:
            errors.append("config.exit_mode: must be one of %s" % (EXIT_MODES,))
        rvol = config.get("rvol_min")
        if not _is_number(rvol) or rvol <= 0:
            errors.append("config.rvol_min: must be a positive number")
        for key in ("vwap_filter", "volume_filter"):
            if not isinstance(config.get(key), bool):
                errors.append("config.%s: must be true or false" % key)

    period = rec.get("period")
    if not isinstance(period, dict):
        errors.append("period: required object")
    else:
        dates = {}
        for key in ("from", "to"):
            v = period.get(key)
            if not isinstance(v, str):
                errors.append("period.%s: required string 'YYYY-MM-DD'" % key)
                continue
            try:
                dates[key] = datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                errors.append("period.%s: must match 'YYYY-MM-DD'" % key)
        if "from" in dates and "to" in dates and dates["from"] > dates["to"]:
            errors.append("period.from: must not be after period.to")
        if not isinstance(period.get("bars"), str) or not period.get("bars"):
            errors.append("period.bars: required non-empty string (e.g. '5m')")

    metrics = rec.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics: required object")
    else:
        for key in METRIC_KEYS:
            if key not in metrics:
                errors.append("metrics.%s: key required (value may be null)" % key)
                continue
            v = metrics[key]
            if v is None:
                continue  # any metric may be null (tester showed n/a)
            if key in INT_METRICS:
                if not _is_int(v) or v < 0:
                    errors.append("metrics.%s: must be a non-negative integer or null" % key)
            elif not _is_number(v):
                errors.append("metrics.%s: must be a number or null" % key)
        trades = metrics.get("trades")
        wins = metrics.get("wins")
        if _is_int(trades) and _is_int(wins) and wins > trades:
            errors.append("metrics.wins: must not exceed metrics.trades")

    return errors


def existing_run_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue  # malformed lines cannot claim a run_id
            if isinstance(obj, dict) and isinstance(obj.get("run_id"), str):
                ids.add(obj["run_id"])
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and append one backtest run record (advisory tooling only).")
    parser.add_argument("--json", required=True,
                        help="whole record as a single JSON object string")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="path to results.jsonl (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        rec = json.loads(args.json)
    except ValueError as exc:
        sys.stderr.write("error: --json is not valid JSON: %s\n" % exc)
        return 2
    if not isinstance(rec, dict):
        sys.stderr.write("error: record: must be a JSON object\n")
        return 2

    errors = validate_record(rec)
    if errors:
        sys.stderr.write("Schema validation failed:\n")
        for err in errors:
            sys.stderr.write("  - %s\n" % err)
        return 2

    if rec["run_id"] in existing_run_ids(args.results):
        sys.stderr.write(
            "error: duplicate run_id '%s' already recorded in %s -- refused\n"
            % (rec["run_id"], args.results))
        return 3

    results_dir = os.path.dirname(os.path.abspath(args.results))
    if results_dir and not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    with open(args.results, "a") as fh:
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print("recorded run_id '%s' -> %s" % (rec["run_id"], args.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
