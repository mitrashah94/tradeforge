#!/usr/bin/env python3
"""eval_engine_regression.py -- deterministic-engine regression + premarket
selection accuracy, tracked over time.

Two suites, both non-LLM (Braintrust supports deterministic tasks):

  engine-regression   pins the modelled outputs that today's analysis produced.
                      If a change to option_pricing / structure_lab / the
                      scorecard silently moves a number, this drops. Goldens are
                      the values verified on 2026-07-24 -- see daytrading_memory.md.

  premarket-selection did the locked shortlist contain the ticker that actually
                      QUALIFIED? HONEST CAVEAT: n is tiny. This metric means
                      nothing until many sessions accumulate, and a high score
                      on n=1 is noise. It is here to START accumulating.

Run:  bt eval evals/eval_engine_regression.py  |  python3 evals/run_offline.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "analysis"))

from common import run_eval, print_report                      # noqa: E402
from scorers import matches_golden, within_tolerance, shortlist_contained_winner  # noqa: E402


# --------------------------------------------------------------------------
# engine regression
# --------------------------------------------------------------------------
def _xle_20260724_case():
    """The 2026-07-24 XLE QUALIFIED, recomputed live from the committed modules."""
    import option_pricing as op
    import structure_lab as sl
    from option_model import Bar, Signal

    sig = Signal("XLE", "CALL", 60.03, 59.93, 1784901900)
    bars = [Bar(1784902200, 60.04, 60.145, 60.00, 60.14),
            Bar(1784902500, 60.14, 60.195, 60.135, 60.155),
            Bar(1784902800, 60.17, 60.24, 60.135, 60.24),
            Bar(1784903100, 60.25, 60.255, 60.13, 60.17),
            Bar(1784903400, 60.19, 60.30, 60.18, 60.30),
            Bar(1784903700, 60.295, 60.40, 60.28, 60.38)]
    leg = op.calibrate_leg("XLE Jul31 60C", "C", 60.0, "2026-07-31",
                           1.00, 60.03, 1784901900)
    outright = sl.walk_structure([leg], bars, sig, 1)
    vertical = sl.walk_structure(sl.synth_vertical(leg, 61.0), bars, sig, 1)
    return {"chart_r": sig.chart_r,
            "exit_reason": outright["exit_reason"],
            "held_minutes": outright["held_minutes"],
            "outright_r": outright["account_r"],
            "vertical_r": vertical["account_r"],
            "vertical_underperforms": vertical["account_r"] < outright["account_r"]}


ENGINE_CASES = [
    {"name": "xle_2026-07-24_structure_walk",
     "input": {"session": "2026-07-24", "ticker": "XLE"},
     "expected": {"chart_r": 0.1, "exit_reason": "r_target", "held_minutes": 30,
                  "vertical_underperforms": True}},
]


def engine_task(input):
    if input.get("session") == "2026-07-24":
        return _xle_20260724_case()
    return {}


# --------------------------------------------------------------------------
# premarket selection accuracy
# --------------------------------------------------------------------------
SELECTION_CASES = [
    {"name": "2026-07-14 (zero QUALIFIED - correct no-trade day)",
     "input": {"session": "2026-07-14"},
     "expected": {"qualified_ticker": None}},
    {"name": "2026-07-24 (XLE QUALIFIED, missed)",
     "input": {"session": "2026-07-24"},
     "expected": {"qualified_ticker": "XLE"}},
]

# Locked cards. 07-14 was produced by premarket_scorecard from that day's saved
# bars; 07-24 is RECONSTRUCTED (no card existed - the scorecard did not yet
# exist), so treat it as illustrative, not as evidence the tool picked it.
SELECTION_CARDS = {
    "2026-07-14": {"primary": "XLE", "backup": "IWM",
                   "context_only": ["XLF"], "reconstructed": False},
    "2026-07-24": {"primary": "XLE", "backup": "IWM",
                   "context_only": [], "reconstructed": True},
}


def selection_task(input):
    return SELECTION_CARDS.get(input.get("session"), {})


def main(force_offline: bool = False) -> bool:
    ok = True
    r1 = run_eval("engine-regression", data=ENGINE_CASES, task=engine_task,
                  scores=[matches_golden],
                  metadata={"suite": "engine_regression",
                            "goldens_verified": "2026-07-24"},
                  force_offline=force_offline)
    ok &= print_report(r1)

    r2 = run_eval("premarket-selection", data=SELECTION_CASES,
                  task=selection_task, scores=[shortlist_contained_winner],
                  metadata={"suite": "premarket_selection",
                            "caveat": "n is tiny; 2026-07-24 card is reconstructed"},
                  force_offline=force_offline)
    ok &= print_report(r2)
    print("NOTE: premarket-selection is NOT yet evidence. n=2, and the 2026-07-24")
    print("      card was reconstructed after the fact. Accumulate sessions first.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
