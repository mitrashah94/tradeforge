"""scripts/prop_strategy_search.py — hunt the strategy that reaches $100k in payouts.

Uses the PROP-EVAL SIMULATOR AS THE FITNESS FUNCTION: every deployable config in
the repo (the blended book, each sleeve standalone at full allocation, and their
named variants — including the leveraged-rotation levers) is backtested on a FIT
window, run through the leverage sweep (P(pass) / funded survival / E[payout] /
EV per attempt), and then through the $1k CAMPAIGN (payouts reinvested, fleet
capped) to score THE question: P(gross payouts >= $100k within 378 trading days
≈ 1.5 years). The top configs are re-scored on a HOLDOUT window.

MULTIPLE-TESTING HONESTY: this script tries ~13 configs on the same history and
picks winners — classic selection bias. The winner's numbers are OPTIMISTIC by
construction; the holdout columns are the first defense, and nothing here
promotes anything (a winner still owes the full validation gate + a paper track
record before a dollar follows it).

CLI: PYTHONPATH=. .venv/bin/python scripts/prop_strategy_search.py
     (fit 2018-01-01..2023-12-31, holdout 2024-01-01..2026-06-26)
"""

from __future__ import annotations

import sys

from backtest.daily.bracket_engine import BracketConfig
from backtest.daily.portfolio_backtester import run_portfolio
from backtest.daily.validation import returns_metrics
from data.schema import DEFAULT_DB_PATH, connect
from portfolio.model import SleeveSpec
from prop.campaign import run_campaign
from prop.rules import load_firm
from prop.simulate import expected_value

FIT = ("2018-01-01", "2023-12-31")
HOLDOUT = ("2024-01-01", "2026-06-26")
LEVERAGES = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
FIRM = "equities_25k"
TOP_N_HOLDOUT = 5

BR_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "JPM",
               "XLK", "QQQ", "SPY"]


# --------------------------------------------------------------------------- #
# config builders
# --------------------------------------------------------------------------- #
def _rot(variant):
    from strategies.momentum_rotation.strategy import MomentumRotationStrategy, load_params
    return MomentumRotationStrategy(load_params(variant))


def _mr(variant):
    from strategies.swing_meanrev.strategy import SwingMeanRevStrategy, load_params
    return SwingMeanRevStrategy(load_params(variant))


def _br(variant):
    from strategies.swing_breakout.strategy import SwingBreakoutStrategy, load_params
    p = load_params(variant)
    strat = SwingBreakoutStrategy(p)
    bracket = BracketConfig(
        atr_window=int(p["atr_window"]), stop_atr_mult=float(p["stop_atr_mult"]),
        tp1_R=float(p["tp1_R"]), tp1_fraction=float(p["tp1_fraction"]),
        trail_atr_mult=float(p["trail_atr_mult"]), use_trail=bool(p["use_trail"]),
        hard_target_R=(None if p["hard_target_R"] is None else float(p["hard_target_R"])),
    )
    return strat, bracket


def _spec_rot(variant, alloc):
    return SleeveSpec(name="rotation", strategy=_rot(variant), kind="weight",
                      grade="B", family="equity", benchmark="SPY", allocation=alloc)


def _spec_mr(variant, alloc):
    return SleeveSpec(name="swing_meanrev", strategy=_mr(variant), kind="weight",
                      grade="B", family="meanrev", benchmark="SPY", allocation=alloc)


def _spec_br(variant, alloc, grade="A"):
    strat, bracket = _br(variant)
    return SleeveSpec(name="swing_breakout", strategy=strat, kind="score",
                      grade=grade, family="trend", benchmark="SPY",
                      allocation=alloc, bracket=bracket)


def _spec_lt(variant, alloc):
    from strategies.lev_trend.strategy import LevTrendStrategy, load_params
    return SleeveSpec(name="lev_trend", strategy=LevTrendStrategy(load_params(variant)),
                      kind="weight", grade="B", family="lev_trend", benchmark="SPY",
                      allocation=alloc)


def configs() -> dict:
    """{name: (sleeves, extra_universe)} — every deployable config in the search space.

    Families: the original three sleeves + their variants; the LEVERAGED
    TREND-GATE family (the missing high-MAR shape); and blends of the spiky
    survivors (the maximization thesis applied to the EVAL objective — stack
    decorrelated spiky edges to raise the blend's return-per-drawdown).
    """
    return {
        # --- original blends ---
        "blend_default": ([_spec_rot("DEFAULT", 0.4), _spec_mr("DEFAULT", 0.3), _spec_br("DEFAULT", 0.3)], []),
        "blend_rot_lev": ([_spec_rot("LEVERAGED_ON", 0.4), _spec_mr("DEFAULT", 0.3), _spec_br("DEFAULT", 0.3)], []),
        "brk_mr_5050": ([_spec_mr("DEFAULT", 0.5), _spec_br("DEFAULT", 0.5)], []),
        # --- singles at full allocation ---
        "rotation": ([_spec_rot("DEFAULT", 1.0)], []),
        "rotation_lev": ([_spec_rot("LEVERAGED_ON", 1.0)], []),
        "rotation_both": ([_spec_rot("BOTH_LEVERS_ON", 1.0)], []),
        "meanrev": ([_spec_mr("DEFAULT", 1.0)], []),
        "meanrev_strict": ([_spec_mr("STRICT", 1.0)], []),
        "meanrev_fastexit": ([_spec_mr("FAST_EXIT", 1.0)], []),
        "breakout": ([_spec_br("DEFAULT", 1.0)], []),
        "breakout_tight": ([_spec_br("TIGHT_STOP", 1.0)], []),
        "breakout_wide": ([_spec_br("WIDE_STOP", 1.0)], []),
        "breakout_fastdonch": ([_spec_br("FAST_DONCHIAN", 1.0)], []),
        # --- NEW: the leveraged trend-gate family ---
        "levtrend_qld": ([_spec_lt("DEFAULT", 1.0)], []),
        "levtrend_tqqq": ([_spec_lt("TQQQ", 1.0)], []),
        "levtrend_tqqq_tlt": ([_spec_lt("TQQQ_TLT", 1.0)], []),
        "levtrend_qqq_gate": ([_spec_lt("QQQ_GATE", 1.0)], []),
        "levtrend_dualgate": ([_spec_lt("DUAL_GATE", 1.0)], []),
        # --- NEW: breakout over a widened universe (gold + bitcoin-ETF trends) ---
        "breakout_wide_gold": ([_spec_br("WIDE_STOP", 1.0)], ["GLD", "IBIT"]),
        # --- NEW: blends of the spiky survivors ---
        "mrS_brkW_5050": ([_spec_mr("STRICT", 0.5), _spec_br("WIDE_STOP", 0.5)], []),
        "mrS_lt_5050": ([_spec_mr("STRICT", 0.5), _spec_lt("TQQQ", 0.5)], []),
        "mrS_brkW_lt_thirds": ([_spec_mr("STRICT", 0.34), _spec_br("WIDE_STOP", 0.33),
                                _spec_lt("TQQQ", 0.33)], []),
    }


def _universe(sleeves, extra=()) -> list[str]:
    syms = set(BR_UNIVERSE) | set(extra)
    for sp in sleeves:
        strat = sp.strategy
        if hasattr(strat, "extra_symbols"):
            syms.update(strat.extra_symbols())
        if hasattr(strat, "universe"):
            syms.update(strat.universe)
    return sorted(syms)


# --------------------------------------------------------------------------- #
# scoring: backtest -> eval sweep -> campaign
# --------------------------------------------------------------------------- #
def score_config(name, sleeves, window, con, rules, *, extra=(), n_paths=1000) -> dict:
    res = run_portfolio(sleeves, _universe(sleeves, extra), start=window[0], end=window[1],
                        initial_equity=1000.0, con=con)
    rets = res.twr_returns
    rm = returns_metrics(rets)
    mar = (rm["ann_return"] / rm["max_drawdown"]
           if rm["max_drawdown"] and rm["max_drawdown"] > 0 else float("nan"))
    best = None
    for lev in LEVERAGES:
        ev = expected_value(rets, rules, leverage=lev, horizon=60,
                            n_paths=n_paths, block=10, seed=0)
        if best is None or ev["ev_one_attempt"] > best["ev_one_attempt"]:
            best = ev
    camp = run_campaign(rets, rules, leverage=best["leverage"], initial_cash=1000.0,
                        horizon_days=378, max_concurrent=10,
                        checkpoints=(252, 378), n_paths=400, block=10, seed=5)
    return {
        "name": name,
        "ann_return": rm["ann_return"],
        "max_dd": rm["max_drawdown"],
        "mar": mar,
        "sharpe": rm["sharpe"],
        "best_lev": best["leverage"],
        "p_pass": best["p_pass"],
        "survival": best["funded_survival_rate"],
        "payout_acct": best["mean_annual_payout_per_account"],
        "ev_attempt": best["ev_one_attempt"],
        "p100k_1yr": camp["p_target_252d"],
        "p100k_18mo": camp["p_target_378d"],
        "median_18mo": camp["median_payout_378d"],
    }


def _print_table(rows, title):
    print(f"\n===== {title} =====")
    hdr = (f"  {'config':<20s} {'ann':>7s} {'maxDD':>7s} {'MAR':>5s} {'lev':>4s} "
           f"{'P(pass)':>8s} {'surv':>6s} {'$/acct':>8s} {'EV':>7s} "
           f"{'P100k@1y':>9s} {'P100k@18m':>10s} {'med@18m':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['name']:<20s} {r['ann_return']:>7.1%} {r['max_dd']:>7.1%} "
              f"{r['mar']:>5.2f} {r['best_lev']:>4.0f} {r['p_pass']:>8.1%} "
              f"{r['survival']:>6.1%} ${r['payout_acct']:>6,.0f} "
              f"${r['ev_attempt']:>5,.0f} {r['p100k_1yr']:>9.1%} "
              f"{r['p100k_18mo']:>10.1%} ${r['median_18mo']:>7,.0f}")


def main(argv) -> int:
    rules = load_firm(FIRM)
    con = connect(DEFAULT_DB_PATH)
    try:
        fit_rows = []
        for name, (sleeves, extra) in configs().items():
            try:
                row = score_config(name, sleeves, FIT, con, rules, extra=extra)
            except Exception as e:  # noqa: BLE001 — one bad config must not kill the sweep
                print(f"  !! {name} failed: {e}")
                continue
            fit_rows.append(row)
            print(f"  scored {name}: MAR {row['mar']:.2f}, "
                  f"P100k@18m {row['p100k_18mo']:.1%}", flush=True)
        fit_rows.sort(key=lambda r: (r["p100k_18mo"], r["ev_attempt"]), reverse=True)
        _print_table(fit_rows, f"FIT {FIT[0]}..{FIT[1]}  (firm {FIRM}, $1k campaign, cap 10)")

        top = [r["name"] for r in fit_rows[:TOP_N_HOLDOUT]]
        hold_rows = []
        cfg = configs()
        for name in top:
            sleeves, extra = cfg[name]
            hold_rows.append(score_config(name, sleeves, HOLDOUT, con, rules, extra=extra))
        _print_table(hold_rows, f"HOLDOUT {HOLDOUT[0]}..{HOLDOUT[1]}  (top {TOP_N_HOLDOUT} only)")

        n_cfg = len(configs())
        print(f"\nHONESTY: {n_cfg} configs were tried on the fit window (cumulative "
              "across search rounds — the haircut grows with every round). The "
              "winner's fit numbers are optimistic by selection. Trust the HOLDOUT "
              "column, and even that is one sample; nothing here is promoted without "
              "the full validation gate + a forward paper record.")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
