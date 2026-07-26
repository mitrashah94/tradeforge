#!/usr/bin/env python3
"""Plain-assert tests for record_backtest.py and learning_loop.py.

Run:  python3 analysis/test_learning_loop.py
All fixtures are synthetic and live in a temp directory; the real
journal and real results file are never touched.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import learning_loop  # noqa: E402
import record_backtest  # noqa: E402

RECORD_CLI = os.path.join(HERE, "record_backtest.py")

JOURNAL_HEADERS = [
    "Date", "Time (CT)", "Ticker", "Direction", "Setup", "Market bias",
    "SPY/QQQ aligned", "Entry (und.)", "Stop (und.)", "Chart R ($)",
    "1R", "2R", "3R", "4R", "5R", "Strike", "Expiration", "DTE", "Delta",
    "IV", "Bid", "Ask", "Spread", "Fill", "Contracts", "Max loss cap ($)",
    "Est. loss @ stop ($)", "Risk check", "Invalidation statement",
    "Screenshot", "Exit price (opt.)", "Net P&L ($)", "Realized R",
    "Highest chart R", "MFE ($)", "MAE ($)", "Exit reason",
    "Followed plan", "Chased", "Widened stop", "Lesson", "Week start",
]
assert JOURNAL_HEADERS[29] == "Screenshot"       # col 30
assert JOURNAL_HEADERS[32] == "Realized R"       # col 33
assert JOURNAL_HEADERS[37] == "Followed plan"    # col 38
assert JOURNAL_HEADERS[38] == "Chased"           # col 39
assert JOURNAL_HEADERS[39] == "Widened stop"     # col 40

PASS = []


def ok(name):
    PASS.append(name)
    print("PASS  %s" % name)


def valid_record(run_id="20260501-XLE-pdh-retest-3R", symbol="XLE",
                 level="PDH/PDL", entry="Retest only", exit_mode="Fixed 3R",
                 trades=4, wins=2, net_r=1.8,
                 period=("2026-04-20", "2026-07-10")):
    win_rate = (100.0 * wins / trades) if trades else None
    return {
        "run_id": run_id,
        "recorded_ct": "2026-07-12 19:45",
        "symbol": symbol,
        "strategy": "Asymmetric Backtest v1",
        "level_source": level,
        "config": {"entry_mode": entry, "exit_mode": exit_mode,
                   "rvol_min": 1.2, "vwap_filter": True,
                   "volume_filter": True},
        "period": {"from": period[0], "to": period[1], "bars": "5m"},
        "metrics": {"trades": trades, "wins": wins,
                    "win_rate_pct": win_rate, "net_r": net_r,
                    "avg_r": (net_r / trades) if trades else None,
                    "profit_factor_r": 1.9 if net_r > 0 else 0.5,
                    "max_dd_r": 1.2, "best_r": 3.0, "worst_r": -1.1},
    }


def run_cli(results_path, record):
    return subprocess.run(
        [sys.executable, RECORD_CLI, "--results", results_path,
         "--json", json.dumps(record)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)


# --------------------------------------------------------------- fixtures

def synthetic_results(path):
    """>= 6 runs: mixed symbols/level_sources, one zero-trade config
    (IWM), one net-negative config with >= 10 trades (XLF)."""
    runs = [
        valid_record("r1-xle-pdh", "XLE", "PDH/PDL", trades=4, wins=2,
                     net_r=1.8),
        valid_record("r2-xle-orb", "XLE", "ORH/ORL", trades=12, wins=7,
                     net_r=6.5),
        valid_record("r3-xlf-pdh-neg", "XLF", "PDH/PDL", trades=12, wins=3,
                     net_r=-4.2),
        valid_record("r4-spy-pdh", "SPY", "PDH/PDL", trades=3, wins=2,
                     net_r=2.1),
        valid_record("r5-iwm-zero", "IWM", "PDH/PDL", trades=0, wins=0,
                     net_r=0.0),
        valid_record("r6-iwm-zero-orb", "IWM", "ORH/ORL", trades=0, wins=0,
                     net_r=0.0),
        valid_record("r7-qqq-orb", "QQQ", "ORH/ORL", trades=5, wins=3,
                     net_r=3.3),
    ]
    # a metrics-null run must also survive
    nulls = valid_record("r8-xle-nulls", "XLE", "PDH/PDL",
                         entry="Break only", trades=2, wins=1, net_r=0.4)
    nulls["metrics"]["profit_factor_r"] = None
    nulls["metrics"]["max_dd_r"] = None
    runs.append(nulls)
    with open(path, "w") as fh:
        for r in runs:
            fh.write(json.dumps(r) + "\n")
    return runs


def synthetic_journal(path):
    """Trade Log with real headers, one worked-example row (must be
    excluded), and 5 real rows: Followed plan Y,Y,Y,N,(blank) ->
    adherence 75.0% over 4 scored; 1 chase; 1 widened stop."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Trade Log"
    ws.append(JOURNAL_HEADERS)

    def row(ticker, setup, screenshot, realized_r, highest_r, mfe, mae,
            exit_reason, followed, chased, widened, lesson):
        cells = [None] * len(JOURNAL_HEADERS)
        cells[2] = ticker          # Ticker (col 3)
        cells[4] = setup           # Setup (col 5)
        cells[29] = screenshot     # Screenshot (col 30)
        cells[32] = realized_r     # Realized R (col 33)
        cells[33] = highest_r      # Highest chart R (col 34)
        cells[34] = mfe            # MFE ($) (col 35)
        cells[35] = mae            # MAE ($) (col 36)
        cells[36] = exit_reason    # Exit reason (col 37)
        cells[37] = followed       # Followed plan (col 38)
        cells[38] = chased         # Chased (col 39)
        cells[39] = widened        # Widened stop (col 40)
        cells[40] = lesson         # Lesson (col 41)
        ws.append(cells)

    # worked example (row 2 of the real journal) -- must be excluded
    row("XLF", "A", "screenshots/2026-07-13-xlf.png", 3.0, 3.0, 60, -8,
        "3R target hit", "Y", "N", "N",
        "Waited for the retest close instead of the break")
    # real rows
    row("XLE", "A", "screenshots/a1.png", 2.0, 3.0, 55, -10,
        "3R target", "Y", "N", "N", "clean")
    row("XLE", "A", "screenshots/a2.png", -1.0, 0.5, 10, -25,
        "stopped", "Y", "N", "N", "valid loss")
    row("XLF", "B", "screenshots/b1.png", 1.5, 2.0, 40, -12,
        "trail exit", "Y", "N", "N", "good sim replay drill")
    row("IWM", "A", "screenshots/a3.png", -1.4, 0.2, 5, -35,
        "stopped", "N", "Y", "Y", "chased past expiration price")
    row("QQQ", "B", "screenshots/b2.png", None, None, None, None,
        "open", None, "N", "N", "")
    wb.save(path)


# ------------------------------------------------------------------ tests

def test_record_valid_append(tmp):
    results = os.path.join(tmp, "results.jsonl")
    proc = run_cli(results, valid_record())
    assert proc.returncode == 0, proc.stderr
    with open(results) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["run_id"] == "20260501-XLE-pdh-retest-3R"
    assert parsed["metrics"]["trades"] == 4
    ok("record_backtest appends a valid record as one JSONL line")


def test_record_duplicate_run_id(tmp):
    results = os.path.join(tmp, "results.jsonl")
    assert run_cli(results, valid_record()).returncode == 0
    proc = run_cli(results, valid_record())  # identical run_id
    assert proc.returncode == 3, (proc.returncode, proc.stderr)
    assert "duplicate run_id" in proc.stderr
    with open(results) as fh:
        assert len(fh.read().splitlines()) == 1  # nothing appended
    ok("record_backtest refuses duplicate run_id with exit 3")


def test_record_invalid_schema(tmp):
    results = os.path.join(tmp, "results.jsonl")
    bad = valid_record()
    bad["config"]["entry_mode"] = "YOLO market buy"
    proc = run_cli(results, bad)
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert "config.entry_mode" in proc.stderr  # names the field
    assert not os.path.exists(results) or open(results).read() == ""

    bad2 = valid_record()
    del bad2["metrics"]["net_r"]
    proc2 = run_cli(results, bad2)
    assert proc2.returncode == 2
    assert "metrics.net_r" in proc2.stderr

    bad3 = valid_record()
    bad3["level_source"] = "moon phase"
    proc3 = run_cli(results, bad3)
    assert proc3.returncode == 2 and "level_source" in proc3.stderr

    proc4 = subprocess.run(
        [sys.executable, RECORD_CLI, "--results", results,
         "--json", "{not json"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True)
    assert proc4.returncode == 2 and "not valid JSON" in proc4.stderr
    ok("record_backtest exits 2 on invalid schema and names the field")


def test_record_zero_trades_and_nulls_valid(tmp):
    results = os.path.join(tmp, "results.jsonl")
    zero = valid_record(run_id="zero-run", trades=0, wins=0, net_r=0.0)
    zero["metrics"]["win_rate_pct"] = None
    zero["metrics"]["avg_r"] = None
    zero["metrics"]["profit_factor_r"] = None
    zero["metrics"]["best_r"] = None
    zero["metrics"]["worst_r"] = None
    proc = run_cli(results, zero)
    assert proc.returncode == 0, proc.stderr
    ok("record_backtest accepts trades == 0 and null metrics")


def test_learning_loop_synthetic_no_journal(tmp):
    results = os.path.join(tmp, "results.jsonl")
    synthetic_results(results)
    out = os.path.join(tmp, "reports", "report.md")
    report = learning_loop.generate(
        results, os.path.join(tmp, "no_such_journal.xlsx"), out)
    assert os.path.exists(out)

    # league table sorted by net R desc
    net_col = []
    for line in report.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if (line.startswith("| r") and line.count("|") >= 11
                and cells[1] != "run_id"):
            net_col.append(float(cells[7]))
    assert len(net_col) == 8, net_col
    assert net_col == sorted(net_col, reverse=True), net_col

    # tiny-sample flags on runs with < 10 trades
    for line in report.splitlines():
        if line.startswith("| r1-xle-pdh") or line.startswith("| r5-iwm"):
            assert "tiny sample" in line, line
        if line.startswith("| r2-xle-orb") or line.startswith("| r3-xlf"):
            assert "tiny sample" not in line, line

    # rule (a): net-negative config with >= 10 trades called out
    assert "Recommend AGAINST" in report
    assert "XLF / PDH/PDL / Retest only / Fixed 3R" in report

    # rule (c): zero-trade symbol called out with a named filter
    assert "IWM produced zero trades in every tested config" in report
    assert "rvol_min" in report

    # rule (e) must NOT fire (several configs exceed 1 trade/month)
    assert "cannot be validated at this frequency" not in report

    # missing journal handled in section 4, report still complete
    assert "Journal unavailable" in report
    assert "## 6. Next experiments queue" in report

    # inventory numbers
    assert "Backtest runs recorded: **8**" in report
    assert "Total simulated trades: 38" in report

    # footer is the exact required line, at the end
    assert learning_loop.FOOTER == (
        "Recommendations are hypotheses to test in simulation - "
        "not live-trading instructions.")
    assert report.rstrip().endswith(learning_loop.FOOTER)
    ok("learning_loop synthetic run: sorting, flags, rules (a)+(c), footer")


def test_learning_loop_confidence_caps(tmp):
    results = os.path.join(tmp, "results.jsonl")
    synthetic_results(results)
    out = os.path.join(tmp, "reports", "report2.md")
    report = learning_loop.generate(
        results, os.path.join(tmp, "none.xlsx"), out)
    # rule (a) sample is 12 trades (< 20) -> LOW; nothing above MEDIUM ever
    for line in report.splitlines():
        if "Recommend AGAINST" in line:
            assert "[LOW]" in line, line
    assert "[HIGH]" not in report
    ok("confidence stays LOW under 20 trades and never exceeds MEDIUM")


def test_learning_loop_with_journal(tmp):
    results = os.path.join(tmp, "results.jsonl")
    synthetic_results(results)
    journal = os.path.join(tmp, "journal.xlsx")
    synthetic_journal(journal)
    out = os.path.join(tmp, "reports", "report3.md")
    report = learning_loop.generate(results, journal, out)

    # example row excluded: 5 real rows, not 6
    loaded = learning_loop.load_journal(journal)
    assert loaded["found"]
    assert loaded["examples_excluded"] == 1
    assert len(loaded["rows"]) == 5
    assert all(r["ticker"] != "XLF" or r["setup"] == "B"
               for r in loaded["rows"])  # the example XLF/A row is gone
    assert "Trades journaled: 5" in report

    # adherence math: 3 Y of 4 scored = 75.0%
    stats = learning_loop.journal_stats(loaded["rows"])
    assert stats["adherence_n"] == 4
    assert abs(stats["adherence_pct"] - 75.0) < 1e-9
    assert "Adherence rate: 75.0% (over 4 scored trades)" in report
    assert stats["chases"] == 1 and stats["widened"] == 1
    assert "Chase count: 1" in report
    assert "Widened-stop count: 1" in report

    # realized R by setup: A -> (2.0 - 1.0 - 1.4)/3, B -> 1.5
    avg_a, tot_a, n_a = stats["r_by_setup"]["A"]
    assert n_a == 3 and abs(tot_a - (-0.4)) < 1e-9
    avg_b, tot_b, n_b = stats["r_by_setup"]["B"]
    assert n_b == 1 and abs(avg_b - 1.5) < 1e-9

    # correlation callout phrased carefully
    assert "where plan was not followed, avg R was" in report

    # rule (d): adherence < 80% -> FIRST recommendation is process
    rec_lines = [ln for ln in report.splitlines()
                 if ln.startswith(("1. ", "2. ", "3. "))
                 and "**[" in ln]
    assert rec_lines and "PROCESS FIRST" in rec_lines[0], rec_lines[:1]

    # sim detection: the 'sim replay drill' lesson row counts as sim
    assert "live: 4, sim: 1" in report
    assert report.rstrip().endswith(learning_loop.FOOTER)
    ok("learning_loop journal: adherence math, example exclusion, rule (d)")


def test_learning_loop_empty_results_bootstrap(tmp):
    results = os.path.join(tmp, "empty.jsonl")
    open(results, "w").close()
    out = os.path.join(tmp, "reports", "report4.md")
    report = learning_loop.generate(
        results, os.path.join(tmp, "none.xlsx"), out)
    assert "Backtest runs recorded: **0**" in report
    assert "Total simulated trades: 0" in report
    # bootstrap queue: PDH/PDL vs ORB on all five watchlist symbols
    for sym in ("SPY", "QQQ", "XLF", "XLE", "IWM"):
        assert ("%s -- variable: level_source (PDH/PDL vs ORH/ORL)" % sym
                ) in report, sym
    assert report.rstrip().endswith(learning_loop.FOOTER)
    ok("empty results file renders with zeros and bootstrap queue")


def test_learning_loop_malformed_lines(tmp):
    results = os.path.join(tmp, "mixed.jsonl")
    with open(results, "w") as fh:
        fh.write(json.dumps(valid_record("good-run")) + "\n")
        fh.write("this is not json at all\n")
        fh.write('{"run_id": "half-a-record"}\n')
        fh.write(json.dumps(valid_record("good-run-2", "QQQ",
                                         "ORH/ORL")) + "\n")
    out = os.path.join(tmp, "reports", "report5.md")
    report = learning_loop.generate(
        results, os.path.join(tmp, "none.xlsx"), out)
    assert "Backtest runs recorded: **2**" in report
    assert "## Appendix: skipped/malformed input" in report
    assert "line 2: not valid JSON" in report
    assert "line 3: missing" in report
    assert report.rstrip().endswith(learning_loop.FOOTER)
    ok("malformed JSONL lines are skipped and listed in the appendix")


def test_validate_record_direct():
    assert record_backtest.validate_record(valid_record()) == []
    errs = record_backtest.validate_record({"run_id": ""})
    assert any(e.startswith("run_id") for e in errs)
    bad = valid_record()
    bad["metrics"]["wins"] = 99  # wins > trades
    assert any("metrics.wins" in e
               for e in record_backtest.validate_record(bad))
    bad2 = valid_record()
    bad2["recorded_ct"] = "yesterday"
    assert any("recorded_ct" in e
               for e in record_backtest.validate_record(bad2))
    ok("validate_record unit checks")


def main():
    tests = [
        test_record_valid_append,
        test_record_duplicate_run_id,
        test_record_invalid_schema,
        test_record_zero_trades_and_nulls_valid,
        test_learning_loop_synthetic_no_journal,
        test_learning_loop_confidence_caps,
        test_learning_loop_with_journal,
        test_learning_loop_empty_results_bootstrap,
        test_learning_loop_malformed_lines,
    ]
    test_validate_record_direct()
    for test in tests:
        tmp = tempfile.mkdtemp(prefix="learning_loop_test_")
        try:
            test(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS (%d tests)" % (len(tests) + 1))


if __name__ == "__main__":
    main()
