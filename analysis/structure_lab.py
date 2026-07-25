#!/usr/bin/env python3
"""structure_lab.py -- compare option STRUCTURES under the campaign's real exit rules.

WHY THIS EXISTS
---------------
`option_model.py` (v1) answered "what would each R exit have paid?" but it only
ever checked whether an R target was REACHED. That is a correctness gap: it can
report a winner on a path that strategy.md would have exited first, because it
ignored sec 8 exit precedence, the stagnation stop, and the flat-by-14:55 rule.

This module walks the path bar by bar and applies the WRITTEN rules in order, so
a modeled outcome is one the plan would actually have produced. It then prices
several structures over that same path -- long call/put vs vertical debit spreads
of varying width -- which is the comparison memory sec 3b requires before spreads
can be reconsidered.

EXIT PRECEDENCE (strategy.md sec 8, sec 10, sec 3)
--------------------------------------------------
Per bar, ADVERSE conditions are evaluated BEFORE favorable ones (intrabar order
is unknowable from 5m OHLC, so the model takes the pessimistic branch):
  1. option/structure loss gate -- value down 1R ($25) on the trade
  2. structural invalidation  -- a CONFIRMED 5m CLOSE through the stop
  3. R take-profit            -- underlying touches the R target (priced AT the
                                 target, not the bar close: a resting limit)
  4. stagnation stop          -- no 1R within N minutes
  5. flat by 14:55 CT         -- a day trade never becomes an overnight hold

HARD RULES (AGENTS.md): Python 3.9 stdlib only. No network. No brokerage access.
Nothing here places, stages, or recommends an order.

MODEL LIMITS: inherits every caveat in option_pricing.py (constant per-leg IV,
calendar-time decay, European pricing, mid-price world). Structures whose legs
were never observed are MODELED -- run option_pricing.validate_against_bars()
and check the verdict before trusting a width comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import os

import option_pricing as op
from option_model import (Bar, Signal, read_bars, RISK_UNIT_USD,
                          CONTRACT_MULT, risk_normalized_contracts, feasibility)

try:
    from zoneinfo import ZoneInfo
    _CT = ZoneInfo("America/Chicago")
except Exception:                                    # pragma: no cover
    _CT = None


@dataclass
class ExitRules:
    r_target: float = 3.0            # sec 10: 3R is the learning-phase default
    max_loss_usd: float = RISK_UNIT_USD   # sec 8: -$25 on the trade
    stagnation_min: int = 60         # sec 8: no 1R in ~45-60 min -> exit
    flat_by_ct: str = "14:55"        # sec 3
    use_stagnation: bool = True
    use_flat_by: bool = True


def _ct_hhmm(epoch: int) -> str:
    if _CT is None:                                  # pragma: no cover
        return "00:00"
    return datetime.fromtimestamp(epoch, _CT).strftime("%H:%M")


def _is_call(direction: str) -> bool:
    return direction.upper() == "CALL"


# --------------------------------------------------------------------------
# structure construction
# --------------------------------------------------------------------------
def build_long(label: str, right: str, strike: float, expiry: str,
               mark: float, S: float, epoch: int) -> Optional[List[op.PricedLeg]]:
    leg = op.calibrate_leg(label, right, strike, expiry, mark, S, epoch, qty=+1)
    return [leg] if leg else None


def build_vertical(long_label: str, right: str, long_strike: float,
                   short_strike: float, expiry: str,
                   long_mark: float, short_mark: float,
                   S: float, epoch: int) -> Optional[List[op.PricedLeg]]:
    """Debit vertical: long the nearer strike, short the further OTM one."""
    a = op.calibrate_leg(long_label, right, long_strike, expiry, long_mark,
                         S, epoch, qty=+1)
    b = op.calibrate_leg("short%.0f" % short_strike, right, short_strike,
                         expiry, short_mark, S, epoch, qty=-1)
    if a is None or b is None:
        return None
    return [a, b]


def synth_vertical(base: op.PricedLeg, short_strike: float,
                   iv_for_short: Optional[float] = None) -> List[op.PricedLeg]:
    """Build a vertical from an already-calibrated long leg by MODELING the short
    leg on the same (or supplied) vol. Used to test widths we never observed --
    flagged as modeled by the caller."""
    short = op.PricedLeg("short%.0f" % short_strike, base.right, short_strike,
                         base.exp_epoch, iv_for_short or base.iv, qty=-1,
                         rate=base.rate)
    return [base, short]


# --------------------------------------------------------------------------
# the path walk
# --------------------------------------------------------------------------
def walk_structure(legs: List[op.PricedLeg], bars: List[Bar], sig: Signal,
                   contracts: int = 1, rules: ExitRules = ExitRules()) -> Dict:
    """Walk the underlying path applying sec 8 precedence. Returns the single
    outcome the written plan would have produced."""
    call = _is_call(sig.direction)
    chart_r = sig.chart_r
    ladder = sig.r_ladder(max(5, int(rules.r_target)))
    target = sig.entry + (1 if call else -1) * rules.r_target * chart_r
    one_r = ladder[0]

    entry_val = op.structure_value(legs, sig.entry, sig.signal_epoch)
    if entry_val <= 0:
        return {"error": "non-positive entry value", "entry_value": entry_val}
    cost = round(entry_val * CONTRACT_MULT * contracts, 2)

    def pnl_at(S: float, epoch: int) -> float:
        v = op.structure_value(legs, S, epoch)
        return round((v - entry_val) * CONTRACT_MULT * contracts, 2)

    reached_1r_epoch = None
    mae_usd = 0.0
    path = []

    for b in bars:
        if b.epoch <= sig.signal_epoch:
            continue
        adverse_S = b.low if call else b.high
        favorable_S = b.high if call else b.low
        adv_pnl = pnl_at(adverse_S, b.epoch)
        mae_usd = min(mae_usd, adv_pnl)
        path.append({"epoch": b.epoch, "ct": _ct_hhmm(b.epoch),
                     "close": b.close, "pnl_at_close": pnl_at(b.close, b.epoch)})

        # --- 1. structure loss gate (adverse first, pessimistic) ---
        if adv_pnl <= -rules.max_loss_usd:
            return _result("loss_gate", b, adverse_S, adv_pnl, cost, contracts,
                           entry_val, legs, mae_usd, sig, path)

        # --- 2. structural invalidation: CONFIRMED close through the stop ---
        through = (b.close <= sig.stop) if call else (b.close >= sig.stop)
        if through:
            p = pnl_at(b.close, b.epoch)
            return _result("structural_stop", b, b.close, p, cost, contracts,
                           entry_val, legs, mae_usd, sig, path)

        # --- 3. R take-profit, priced AT the target (resting limit) ---
        hit = (b.high >= target) if call else (b.low <= target)
        if hit:
            p = pnl_at(target, b.epoch)
            return _result("r_target", b, target, p, cost, contracts,
                           entry_val, legs, mae_usd, sig, path)

        # track 1R for the stagnation clock
        if reached_1r_epoch is None:
            if (favorable_S >= one_r) if call else (favorable_S <= one_r):
                reached_1r_epoch = b.epoch

        # --- 4. stagnation stop ---
        if (rules.use_stagnation and reached_1r_epoch is None
                and (b.epoch - sig.signal_epoch) >= rules.stagnation_min * 60):
            p = pnl_at(b.close, b.epoch)
            return _result("stagnation", b, b.close, p, cost, contracts,
                           entry_val, legs, mae_usd, sig, path)

        # --- 5. flat by 14:55 CT ---
        if rules.use_flat_by and _ct_hhmm(b.epoch) >= rules.flat_by_ct:
            p = pnl_at(b.close, b.epoch)
            return _result("flat_by_close", b, b.close, p, cost, contracts,
                           entry_val, legs, mae_usd, sig, path)

    if not path:
        return {"error": "no bars after signal"}
    last = bars[-1]
    p = pnl_at(last.close, last.epoch)
    return _result("data_end", last, last.close, p, cost, contracts,
                   entry_val, legs, mae_usd, sig, path)


def _result(reason, bar, exit_S, net, cost, contracts, entry_val, legs,
            mae_usd, sig, path) -> Dict:
    return {
        "exit_reason": reason,
        "exit_epoch": bar.epoch, "exit_ct": _ct_hhmm(bar.epoch),
        "exit_underlying": round(exit_S, 4),
        "entry_value_per_contract": round(entry_val, 4),
        "cost": cost, "contracts": contracts,
        "net": net, "account_r": round(net / RISK_UNIT_USD, 2),
        "max_adverse_usd": round(mae_usd, 2),
        "max_adverse_r": round(mae_usd / RISK_UNIT_USD, 2),
        "held_minutes": int((bar.epoch - sig.signal_epoch) / 60),
        "entry_greeks": op.structure_greeks(legs, sig.entry, sig.signal_epoch),
        "bars_walked": len(path),
    }


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def compare(sig: Signal, bars: List[Bar], base_leg: op.PricedLeg,
            widths: Tuple[float, ...] = (1.0, 2.0, 3.0),
            rules: ExitRules = ExitRules(),
            observed_short: Optional[op.PricedLeg] = None,
            spread_usd: float = 0.04) -> Dict:
    """Run long-1x, long-risk-normalized, and debit verticals of several widths
    over the identical path and exit rules."""
    call = _is_call(sig.direction)
    long_only = [base_leg]
    feas = feasibility(base_leg.greeks(sig.entry, sig.signal_epoch)["delta"],
                       sig.chart_r, spread_usd)
    size_rn = risk_normalized_contracts(feas["loss_per_contract"])

    out: Dict = {"ticker": sig.ticker, "direction": sig.direction,
                 "entry": sig.entry, "stop": sig.stop, "chart_r": sig.chart_r,
                 "r_target": rules.r_target,
                 "feasibility": feas, "risk_normalized_contracts": size_rn,
                 "iv_long_leg": round(base_leg.iv, 4), "structures": []}

    out["structures"].append(dict(
        name="long %s x1" % base_leg.label, modeled=False,
        **walk_structure(long_only, bars, sig, 1, rules)))
    if size_rn > 1:
        out["structures"].append(dict(
            name="long %s x%d (risk-normalized)" % (base_leg.label, size_rn),
            modeled=False,
            **walk_structure(long_only, bars, sig, size_rn, rules)))

    for w in widths:
        ss = base_leg.strike + (w if call else -w)
        if observed_short is not None and abs(observed_short.strike - ss) < 1e-9:
            legs, modeled = [base_leg, observed_short], False
        else:
            legs, modeled = synth_vertical(base_leg, ss), True
        res = walk_structure(legs, bars, sig, 1, rules)
        out["structures"].append(dict(
            name="debit vertical %.0f/%.0f (w=%.0f) x1"
                 % (base_leg.strike, ss, w),
            modeled=modeled, max_structure_value=round(w * CONTRACT_MULT, 2),
            **res))
    return out


def print_comparison(c: Dict) -> None:
    print("=" * 78)
    print("{ticker} {direction}  entry {entry}  stop {stop}  chart_R {chart_r}"
          "  target {r_target}R".format(**c))
    fe = c["feasibility"]
    print("long-leg IV {:.1%}   feasibility ${:.0f}/contract -> risk-normalized {}x"
          .format(c["iv_long_leg"], fe["loss_per_contract"],
                  c["risk_normalized_contracts"]))
    print("-" * 78)
    print("{:<38}{:>8}{:>9}{:>8}{:>7}".format(
        "structure", "acct-R", "exit", "MAE-R", "min"))
    for s in c["structures"]:
        if "error" in s:
            print("{:<38}{:>8}".format(s.get("name", "?"), "ERR"))
            continue
        tag = "*" if s.get("modeled") else " "
        print("{:<38}{:>+8.2f}{:>9}{:>8.2f}{:>7}".format(
            s["name"][:37] + tag, s["account_r"], s["exit_reason"][:9],
            s["max_adverse_r"], s["held_minutes"]))
    print("-" * 78)
    print("* = short leg MODELED (never observed); validate before trusting")
    print("=" * 78)


def run_session(session_dir: str, widths=(1.0, 2.0, 3.0),
                r_target: float = 3.0) -> Optional[Dict]:
    """Read a saved session (same layout option_model.py uses), calibrate the
    long leg from its observed entry mark, VALIDATE the model against the
    observed option bars, then compare structures over the real path."""
    opt_dir = os.path.join(session_dir, "options")
    spath = os.path.join(session_dir, "signal.json")
    cpath = os.path.join(opt_dir, "contracts.json")
    if not (os.path.exists(spath) and os.path.exists(cpath)):
        print("[structure_lab] need signal.json + options/contracts.json")
        return None
    with open(spath) as fh:
        sig = Signal(**json.load(fh))
    with open(cpath) as fh:
        contracts = json.load(fh)

    from option_model import select_contract, Contract
    chosen, _audit = select_contract([Contract(**c) for c in contracts])
    if chosen is None or not chosen.bars_csv:
        print("[structure_lab] no playbook contract with saved bars")
        return None

    und = read_bars(os.path.join(session_dir, sig.ticker + "_5m.csv"))
    obars = read_bars(os.path.join(opt_dir, chosen.bars_csv))
    entry_bar = [b for b in obars if b.epoch <= sig.signal_epoch]
    if not entry_bar:
        print("[structure_lab] no option bar at the signal epoch")
        return None
    entry_mark = entry_bar[-1].close

    leg = op.calibrate_leg(chosen.label, chosen.right, chosen.strike,
                           chosen.expiry, entry_mark, sig.entry, sig.signal_epoch)
    if leg is None:
        print("[structure_lab] could not calibrate IV from the entry mark")
        return None

    val = op.validate_against_bars(leg, und, obars, sig.signal_epoch)
    print("model check vs observed bars: n={} MAE=${} max=${} IV={:.1%} -> {}"
          .format(val["n"], val["mae"], val["max_abs_error"],
                  val["iv_used"], val["verdict"]))

    # use the real short leg when we actually recorded it
    observed_short = None
    for c in contracts:
        if (c.get("bars_csv") and c["expiry"] == chosen.expiry
                and c["right"] == chosen.right and c["strike"] != chosen.strike):
            sb = read_bars(os.path.join(opt_dir, c["bars_csv"]))
            se = [b for b in sb if b.epoch <= sig.signal_epoch]
            if se:
                observed_short = op.calibrate_leg(
                    c["label"], c["right"], c["strike"], c["expiry"],
                    se[-1].close, sig.entry, sig.signal_epoch, qty=-1)
            break

    post = [b for b in und if b.epoch > sig.signal_epoch]
    comp = compare(sig, post, leg, widths, ExitRules(r_target=r_target),
                   observed_short, spread_usd=round(chosen.ask - chosen.bid, 4))
    comp["model_validation"] = {k: v for k, v in val.items() if k != "rows"}
    print_comparison(comp)
    out = os.path.join(session_dir, "structure_comparison.json")
    with open(out, "w") as fh:
        json.dump(comp, fh, indent=2)
    print("[structure_lab] wrote", out)
    return comp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compare option structures.")
    ap.add_argument("session_dir", nargs="?")
    ap.add_argument("--r-target", type=float, default=3.0)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        import test_structure_lab
        test_structure_lab.run()
    elif a.session_dir:
        run_session(a.session_dir, r_target=a.r_target)
    else:
        ap.print_help()
