"""scripts/prop_1k_report.py — the honest read on a $1,000 account.

Three views, one strategy (the blended daily book — the validated edge):
  1. The $1k ACCOUNT under prop-style rules (``personal_1k``): P(pass) is
     scale-invariant (identical to the $25k profile), but every payout dollar is
     1/25th — showing why account SIZE, not pass rate, is what a prop firm sells.
  2. The $1k PERSONAL MOONSHOT (``compound_to_target``): P($1k -> $100k in 252 /
     378 trading days) across leverels, with ruin allowed all the way to zero
     (the most charitable assumption; real margin rules stop far sooner).
  3. The $1k CAMPAIGN: $1,000 of eval fees on the REAL $25k firm, payouts
     reinvested into more evals — P(gross payouts >= $100k) at 12 and 18 months.

CLI: PYTHONPATH=. .venv/bin/python scripts/prop_1k_report.py [start] [end]
"""

from __future__ import annotations

import sys

from data.schema import DEFAULT_DB_PATH, connect
from prop.campaign import run_campaign
from prop.rules import load_firm
from prop.simulate import compound_to_target, expected_value
from scripts.prop_eval_report import build_sleeves

from backtest.daily.portfolio_backtester import run_portfolio


def main(argv: list[str]) -> int:
    start = argv[1] if len(argv) > 1 else "2018-01-01"
    end = argv[2] if len(argv) > 2 else "2025-12-31"

    con = connect(DEFAULT_DB_PATH)
    try:
        sleeves, universe = build_sleeves()
        res = run_portfolio(sleeves, universe, start=start, end=end,
                            initial_equity=1000.0, con=con)
    finally:
        con.close()
    rets = res.twr_returns

    print("TradeForge — THE $1,000 QUESTION (blended daily book, "
          f"{start} -> {end}, twr_cagr {res.twr_cagr():.2%})")

    # ---- 1. the $1k account under prop rules (scale invariance) ----
    p1k = load_firm("personal_1k")
    p25k = load_firm("equities_25k")
    print("\n[1] $1k ACCOUNT under prop-style rules (vs the real $25k firm)")
    print("    (no real firm sells $1k accounts; P(pass) is scale-invariant, dollars are 1/25th)")
    hdr = f"    {'acct':>10s} {'lev':>4s} {'P(pass)':>8s} {'E[payout]/acct-yr':>18s}"
    print(hdr)
    for rules in (p1k, p25k):
        for lev in (4.0, 8.0):
            ev = expected_value(rets, rules, leverage=lev, horizon=60,
                                n_paths=1500, block=10, seed=0)
            print(f"    {rules.name:>10s} {lev:>4.0f} {ev['p_pass']:>8.1%} "
                  f"${ev['mean_annual_payout_per_account']:>16,.0f}")

    # ---- 2. the personal moonshot: compound $1k -> $100k ----
    print("\n[2] $1k PERSONAL MOONSHOT — P(reach $100k), ruin allowed to zero")
    hdr = (f"    {'lev':>4s} {'P($100k/1yr)':>13s} {'P($100k/1.5yr)':>15s} "
           f"{'P(ruin)':>8s} {'median end':>11s} {'p99 end':>10s}")
    print(hdr)
    for lev in (1.0, 2.0, 4.0, 8.0, 16.0, 28.0):
        r252 = compound_to_target(rets, initial=1000, target=100_000, leverage=lev,
                                  days=252, n_paths=3000, block=10, seed=2)
        r378 = compound_to_target(rets, initial=1000, target=100_000, leverage=lev,
                                  days=378, n_paths=3000, block=10, seed=2)
        print(f"    {lev:>4.0f} {r252['p_target']:>13.2%} {r378['p_target']:>15.2%} "
              f"{r378['p_ruin']:>8.1%} ${r378['median_terminal']:>9,.0f} "
              f"${r378['p99_terminal']:>8,.0f}")

    # ---- 3. the campaign: $1k of eval fees on the real firm ----
    print("\n[3] $1k CAMPAIGN — eval fees on the REAL $25k firm, payouts reinvested")
    hdr = (f"    {'lev':>4s} {'cap':>4s} {'P($100k/1yr)':>13s} {'P($100k/1.5yr)':>15s} "
           f"{'median 1.5yr':>13s} {'mean 1.5yr':>11s} {'evals passed':>13s}")
    print(hdr)
    for lev in (6.0, 8.0):
        for cap in (10, 20):
            c = run_campaign(rets, p25k, leverage=lev, initial_cash=1000.0,
                             horizon_days=378, max_concurrent=cap,
                             checkpoints=(252, 378), n_paths=600, block=10, seed=5)
            print(f"    {lev:>4.0f} {cap:>4d} {c['p_target_252d']:>13.1%} "
                  f"{c['p_target_378d']:>15.1%} ${c['median_payout_378d']:>11,.0f} "
                  f"${c['mean_gross_payout']:>9,.0f} {c['mean_evals_passed']:>13.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
