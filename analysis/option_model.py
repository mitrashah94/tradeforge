#!/usr/bin/env python3
"""option_model.py -- offline option-structure modeling for the Asymmetric campaign.

WHY THIS EXISTS
---------------
The backtester and replay harness model the UNDERLYING only. strategy.md sec 6
warns that chart-R and account-R diverge through delta/theta/IV/spread, and
memory sec 3b parks spreads until "option-level modeling is built into the
backtester". This module is that precondition: given a QUALIFIED signal and the
option bars that would have been traded, it prices the ACTUAL structure and
reports the outcome in account-R (1R = $25, strategy.md sec 1/6).

HARD RULES (AGENTS.md)
----------------------
- Python 3.9 STDLIB ONLY. No third-party deps. No network calls in this module.
- NO brokerage access. Nothing here places, stages, or recommends an order.
- Option bars + contract metadata are fetched OUT OF BAND by the orchestrator
  (read-only quotes/chains/historicals) and saved into the session dir as
  CSV/JSON, exactly like the equity <TICKER>_5m.csv bars. This module only reads
  those files and does arithmetic.

MODELED STRUCTURES
------------------
- single long call/put (the current campaign structure)
- vertical debit spread (the structure sec 3b is evaluating)

FILL CONVENTION (deterministic, documented so tests are stable)
---------------------------------------------------------------
- entry fill  = close of the confirmation bar (enter at signal-bar close)
- exit  fill  = close of the bar in which the underlying FIRST crosses the R
                target (a resting limit at the R level; bar close is the proxy)
Both are option marks read from the saved option bars. This is a model, not a
promise of fills; real fills depend on the live bid/ask at that instant.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import argparse
import csv
import json
import math
import os

# ---- campaign constants (authoritative values live in strategy.md) ----
RISK_UNIT_USD = 25.0      # 1R, strategy.md sec 1 (Settings!B4)
CONTRACT_MULT = 100
DTE_MIN, DTE_MAX = 7, 21  # strategy.md sec 7
DELTA_LO, DELTA_HI = 0.45, 0.60
DELTA_TARGET = 0.52       # "slightly ITM is preferred"
MAX_SPREAD = 0.05         # strategy.md sec 5 "wide spread"
MAX_FEASIBLE_LOSS = 25.0  # feasibility gate, strategy.md sec 7


@dataclass
class Bar:
    epoch: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Contract:
    label: str          # e.g. "XLE Jul31 60C"
    right: str          # "C" or "P"
    strike: float
    expiry: str         # YYYY-MM-DD
    dte: int
    delta: float        # abs delta at (approx) entry time
    bid: float
    ask: float
    oi: int = 0
    volume: int = 0
    bars_csv: Optional[str] = None  # path to this contract's saved 5m bars

    @property
    def mark(self) -> float:
        return round((self.bid + self.ask) / 2.0, 4)

    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 4)


# --------------------------------------------------------------------------
# IO helpers (read saved fixtures only)
# --------------------------------------------------------------------------
def read_bars(path: str) -> List[Bar]:
    """Read a <...>_5m.csv with header: time,open,high,low,close[,volume]."""
    out: List[Bar] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(Bar(
                epoch=int(float(row["time"])),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
            ))
    out.sort(key=lambda b: b.epoch)
    return out


def bar_at(bars: List[Bar], epoch: int) -> Optional[Bar]:
    """Last bar whose epoch <= the given epoch (the bar covering that time)."""
    chosen = None
    for b in bars:
        if b.epoch <= epoch:
            chosen = b
        else:
            break
    return chosen


def first_cross_epoch(bars: List[Bar], level: float, direction: str,
                      after_epoch: int) -> Optional[int]:
    """Epoch of the first bar (strictly at/after after_epoch) whose range
    touches `level` in the trade direction. CALL -> high>=level, PUT -> low<=level."""
    up = direction.upper() == "CALL"
    for b in bars:
        if b.epoch < after_epoch:
            continue
        if (up and b.high >= level) or (not up and b.low <= level):
            return b.epoch
    return None


# --------------------------------------------------------------------------
# selection + gates
# --------------------------------------------------------------------------
def select_contract(cands: List[Contract]
                    ) -> Tuple[Optional[Contract], List[Dict]]:
    """Pick the playbook contract: DTE in [7,21], |delta| in [0.45,0.60],
    spread <= $0.05; among survivors prefer delta nearest DELTA_TARGET, then
    highest open interest. Returns (chosen, audit) where audit lists every
    candidate with pass/fail reasons (never hides why one was dropped)."""
    audit: List[Dict] = []
    survivors: List[Contract] = []
    for c in cands:
        reasons = []
        if not (DTE_MIN <= c.dte <= DTE_MAX):
            reasons.append("dte_out_of_window")
        if not (DELTA_LO <= abs(c.delta) <= DELTA_HI):
            reasons.append("delta_out_of_band")
        if c.spread > MAX_SPREAD:
            reasons.append("wide_spread")
        audit.append({"label": c.label, "delta": c.delta, "spread": c.spread,
                      "dte": c.dte, "oi": c.oi, "passed": not reasons,
                      "reasons": reasons})
        if not reasons:
            survivors.append(c)
    if not survivors:
        return None, audit
    survivors.sort(key=lambda c: (abs(abs(c.delta) - DELTA_TARGET), -c.oi))
    return survivors[0], audit


def feasibility(delta: float, chart_r: float, spread: float) -> Dict:
    """(|delta|*chartR + spread) * 100 <= $25  (strategy.md sec 7)."""
    loss = round((abs(delta) * chart_r + spread) * CONTRACT_MULT, 2)
    return {"loss_per_contract": loss, "feasible": loss <= MAX_FEASIBLE_LOSS,
            "gate": MAX_FEASIBLE_LOSS}


def risk_normalized_contracts(loss_per_contract: float) -> int:
    """Contracts that fit the $25 loss gate. floor($25 / per-contract risk),
    minimum 1. This is the lever behind 'buy more contracts' -- it uses the
    risk budget already in strategy.md, no new structure required."""
    if loss_per_contract <= 0:
        return 1
    return max(1, int(math.floor(RISK_UNIT_USD / loss_per_contract)))


# --------------------------------------------------------------------------
# P&L
# --------------------------------------------------------------------------
def _acct_r(net_usd: float) -> float:
    return round(net_usd / RISK_UNIT_USD, 2)


def long_option_pnl(entry_mark: float, exit_mark: float, contracts: int) -> Dict:
    cost = round(entry_mark * CONTRACT_MULT * contracts, 2)
    proceeds = round(exit_mark * CONTRACT_MULT * contracts, 2)
    net = round(proceeds - cost, 2)
    return {"structure": "long_option", "contracts": contracts,
            "entry_mark": entry_mark, "exit_mark": exit_mark,
            "cost": cost, "proceeds": proceeds, "net": net,
            "account_r": _acct_r(net)}


def debit_spread_pnl(long_entry: float, short_entry: float,
                     long_exit: float, short_exit: float,
                     width: float, contracts: int) -> Dict:
    debit = round((long_entry - short_entry), 4)
    exit_val = round((long_exit - short_exit), 4)
    net = round((exit_val - debit) * CONTRACT_MULT * contracts, 2)
    return {"structure": "debit_spread", "contracts": contracts,
            "debit": round(debit * CONTRACT_MULT * contracts, 2),
            "exit_value": round(exit_val * CONTRACT_MULT * contracts, 2),
            "max_value": round(width * CONTRACT_MULT * contracts, 2),
            "net": net, "account_r": _acct_r(net),
            "note": "debit spread caps upside at width; pays near expiry only"}


# --------------------------------------------------------------------------
# integrator
# --------------------------------------------------------------------------
@dataclass
class Signal:
    ticker: str
    direction: str          # CALL / PUT
    entry: float            # confirmation close (entry fill on the underlying)
    stop: float
    signal_epoch: int       # epoch of the confirmation bar
    r_levels: List[float] = field(default_factory=list)  # r1..rN; derived if empty

    @property
    def chart_r(self) -> float:
        return round(abs(self.entry - self.stop), 4)

    def r_ladder(self, n: int = 5) -> List[float]:
        if self.r_levels:
            return self.r_levels
        sgn = 1.0 if self.direction.upper() == "CALL" else -1.0
        return [round(self.entry + sgn * k * self.chart_r, 4) for k in range(1, n + 1)]


def frame_signal(sig: Signal, underlying: List[Bar],
                 long_bars: List[Bar], long_ct: Contract,
                 short_bars: Optional[List[Bar]] = None,
                 short_ct: Optional[Contract] = None,
                 exits: Tuple[int, ...] = (1, 2, 3, 4, 5)) -> Dict:
    """Full account-R frame for one QUALIFIED signal across R exits, for a single
    long option and (optionally) a same-expiry vertical debit spread."""
    feas = feasibility(long_ct.delta, sig.chart_r, long_ct.spread)
    size_rn = risk_normalized_contracts(feas["loss_per_contract"])

    long_entry = bar_at(long_bars, sig.signal_epoch)
    short_entry = bar_at(short_bars, sig.signal_epoch) if short_bars else None
    width = round(abs(short_ct.strike - long_ct.strike), 4) if short_ct else None

    ladder = sig.r_ladder(max(exits))
    rows: List[Dict] = []
    for k in exits:
        level = ladder[k - 1]
        cross = first_cross_epoch(underlying, level, sig.direction, sig.signal_epoch)
        row: Dict = {"r_multiple": k, "level": level,
                     "reached": cross is not None,
                     "cross_epoch": cross}
        if cross is not None and long_entry is not None:
            lx = bar_at(long_bars, cross)
            if lx is not None:
                row["long_1x"] = long_option_pnl(long_entry.close, lx.close, 1)
                row["long_risknorm"] = long_option_pnl(long_entry.close, lx.close, size_rn)
                if short_bars and short_entry is not None and width is not None:
                    sx = bar_at(short_bars, cross)
                    if sx is not None:
                        row["spread_1x"] = debit_spread_pnl(
                            long_entry.close, short_entry.close,
                            lx.close, sx.close, width, 1)
        rows.append(row)

    return {
        "ticker": sig.ticker, "direction": sig.direction,
        "entry": sig.entry, "stop": sig.stop, "chart_r": sig.chart_r,
        "long_contract": long_ct.label,
        "short_contract": short_ct.label if short_ct else None,
        "feasibility": feas,
        "risk_normalized_contracts": size_rn,
        "exits": rows,
    }


# --------------------------------------------------------------------------
# CLI: consume a saved session dir
# --------------------------------------------------------------------------
def _load_contracts(path: str) -> List[Contract]:
    with open(path) as fh:
        raw = json.load(fh)
    return [Contract(**c) for c in raw]


def run_session(session_dir: str) -> Optional[Dict]:
    """Wire session files -> frame. Expects:
        <dir>/<TICKER>_5m.csv           underlying bars
        <dir>/options/contracts.json    candidate Contract dicts (bars_csv set)
        <dir>/signal.json               a Signal dict (from events.json QUALIFIED)
    Missing pieces are reported, not fabricated."""
    opt_dir = os.path.join(session_dir, "options")
    cpath = os.path.join(opt_dir, "contracts.json")
    spath = os.path.join(session_dir, "signal.json")
    if not (os.path.exists(cpath) and os.path.exists(spath)):
        print("[option_model] need options/contracts.json and signal.json; "
              "run the replay to emit signal.json and save the option bars first.")
        return None
    with open(spath) as fh:
        sig = Signal(**json.load(fh))
    cands = _load_contracts(cpath)
    chosen, audit = select_contract(cands)
    if chosen is None:
        print("[option_model] no candidate passed the playbook filters:")
        for a in audit:
            print("  -", a["label"], a["reasons"])
        return {"selection_audit": audit, "chosen": None}
    underlying = read_bars(os.path.join(session_dir, sig.ticker + "_5m.csv"))
    long_bars = read_bars(os.path.join(opt_dir, chosen.bars_csv))
    # pick a one-strike-wider short leg of the same expiry for the spread, if present
    short_ct = None
    short_bars = None
    for c in cands:
        if (c.expiry == chosen.expiry and c.right == chosen.right
                and abs(c.strike - chosen.strike) == 1.0
                and ((chosen.right == "C" and c.strike > chosen.strike)
                     or (chosen.right == "P" and c.strike < chosen.strike))):
            short_ct = c
            short_bars = read_bars(os.path.join(opt_dir, c.bars_csv))
            break
    frame = frame_signal(sig, underlying, long_bars, chosen, short_bars, short_ct)
    frame["selection_audit"] = audit
    out = os.path.join(session_dir, "options_model.json")
    with open(out, "w") as fh:
        json.dump(frame, fh, indent=2)
    _print_frame(frame)
    print("[option_model] wrote", out)
    return frame


def _print_frame(f: Dict) -> None:
    print("=" * 66)
    print("{ticker} {direction}  entry {entry}  stop {stop}  chart_R {chart_r}"
          .format(**f))
    print("long: {}   short: {}".format(f["long_contract"], f["short_contract"]))
    fe = f["feasibility"]
    print("feasibility: ${:.0f}/contract  {}  -> risk-normalized size = {}x"
          .format(fe["loss_per_contract"],
                  "FEASIBLE" if fe["feasible"] else "INFEASIBLE",
                  f["risk_normalized_contracts"]))
    print("-" * 66)
    print("{:>3}  {:>7}  {:>6}  {:>9}  {:>9}  {:>9}".format(
        "R", "level", "hit", "long 1x", "long RN", "spread 1x"))
    for r in f["exits"]:
        def rr(key):
            return "{:+.2f}R".format(r[key]["account_r"]) if key in r else "  -  "
        print("{:>3}  {:>7}  {:>6}  {:>9}  {:>9}  {:>9}".format(
            r["r_multiple"], r["level"], "yes" if r["reached"] else "no",
            rr("long_1x"), rr("long_risknorm"), rr("spread_1x")))
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Offline option-structure model.")
    ap.add_argument("session_dir", nargs="?", help="backtests/session_YYYY-MM-DD")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        import test_option_model  # noqa: F401
        test_option_model.run()
    elif args.session_dir:
        run_session(args.session_dir)
    else:
        ap.print_help()
