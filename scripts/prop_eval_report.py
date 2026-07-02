"""scripts/prop_eval_report.py — run a validated strategy through the prop-eval math.

Builds the blended DAILY book (rotation + swing_meanrev + swing_breakout — the one
book that clears the validation haircut, PF 1.76), takes its flow-free TWR daily
returns, and reports — for a chosen firm profile — the LEVERAGE SWEEP of P(pass) /
P(bust) / expected annual payout / end-to-end EV net of the eval fee, plus how many
funded accounts the mean payout implies to reach a $100k payout year.

This is the honest EV read on the prop path: it tells you whether this edge, sized
to a given aggression, actually passes an evaluation and produces payouts — the
comparison against gambling $1k for 100x.

CLI:
  PYTHONPATH=. .venv/bin/python scripts/prop_eval_report.py [firm] [start] [end]
  e.g. PYTHONPATH=. .venv/bin/python scripts/prop_eval_report.py equities_25k 2018-01-01 2025-12-31
"""

from __future__ import annotations

import sys

from backtest.daily.bracket_engine import BracketConfig
from backtest.daily.portfolio_backtester import run_portfolio
from data.schema import DEFAULT_DB_PATH, connect
from portfolio.model import SleeveSpec
from prop.rules import load_firm
from prop.simulate import sweep_leverage


def build_sleeves():
    from strategies.momentum_rotation.strategy import (
        MomentumRotationStrategy, load_params as rl,
    )
    from strategies.swing_meanrev.strategy import (
        SwingMeanRevStrategy, load_params as ml,
    )
    from strategies.swing_breakout.strategy import (
        SwingBreakoutStrategy, load_params as bl,
    )
    rot = MomentumRotationStrategy(rl("DEFAULT"))
    mr = SwingMeanRevStrategy(ml("DEFAULT"))
    br = SwingBreakoutStrategy(bl("DEFAULT"))
    br_universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "JPM", "XLK", "QQQ", "SPY"]
    universe = sorted(set(rot.extra_symbols()) | set(mr.universe) | set(br_universe))
    sleeves = [
        SleeveSpec(name="rotation", strategy=rot, kind="weight", grade="B",
                   family="equity", benchmark="SPY", allocation=0.4),
        SleeveSpec(name="swing_meanrev", strategy=mr, kind="weight", grade="B",
                   family="meanrev", benchmark="SPY", allocation=0.3),
        SleeveSpec(name="swing_breakout", strategy=br, kind="score", grade="A",
                   family="trend", benchmark="SPY", allocation=0.3,
                   bracket=BracketConfig(atr_window=14)),
    ]
    return sleeves, universe


def main(argv: list[str]) -> int:
    firm_name = argv[1] if len(argv) > 1 else "equities_25k"
    start = argv[2] if len(argv) > 2 else "2018-01-01"
    end = argv[3] if len(argv) > 3 else "2025-12-31"
    rules = load_firm(firm_name)

    con = connect(DEFAULT_DB_PATH)
    try:
        sleeves, universe = build_sleeves()
        res = run_portfolio(sleeves, universe, start=start, end=end,
                            initial_equity=1000.0, con=con)
    finally:
        con.close()

    rets = res.twr_returns
    ann_vol = float(rets.std() * (252 ** 0.5)) if len(rets) > 1 else float("nan")
    print(f"TradeForge — PROP-EVAL EV REPORT  ({firm_name})")
    print(f"strategy: blended daily book (TWR)  {start} -> {end}")
    print(f"  n_days={len(rets)}  twr_cagr={res.twr_cagr():.2%}  ann_vol={ann_vol:.2%}")
    print(f"firm: acct ${rules.account_size:,.0f}  target +{rules.profit_target_pct:.0%} "
          f"(${rules.profit_target:,.0f})  maxDD {rules.max_drawdown_pct:.0%} "
          f"(${rules.max_drawdown:,.0f}, {'trailing' if rules.trailing else 'static'})  "
          f"split {rules.profit_split:.0%}  fee ${rules.eval_fee:,.0f}")
    print()
    hdr = (f"  {'lev':>4s} {'P(pass)':>8s} {'P(bust)':>8s} {'days':>6s} "
           f"{'E[payout]/acct':>15s} {'survive':>8s} {'EV/attempt':>12s} {'#acct→$100k':>12s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = sweep_leverage(rets, rules, leverages=(1, 2, 3, 4, 6, 8, 12),
                          horizon=60, n_paths=3000, block=10, seed=0)
    best = None
    for r in rows:
        n_acct = r["accounts_for_target_payout"]
        print(f"  {r['leverage']:>4.0f} {r['p_pass']:>8.1%} {r['p_fail']:>8.1%} "
              f"{r['median_days_to_pass']:>6.0f} "
              f"${r['mean_annual_payout_per_account']:>13,.0f} "
              f"{r['funded_survival_rate']:>8.1%} "
              f"${r['ev_one_attempt']:>10,.0f} "
              f"{('n/a' if n_acct is None else str(n_acct)):>12s}")
        if best is None or r["ev_one_attempt"] > best["ev_one_attempt"]:
            best = r
    print()
    if best is not None:
        print(f"  BEST EV: leverage {best['leverage']:.0f}x -> P(pass) {best['p_pass']:.1%}, "
              f"E[payout]/acct ${best['mean_annual_payout_per_account']:,.0f}, "
              f"EV/attempt ${best['ev_one_attempt']:,.0f}, "
              f"~{best['accounts_for_target_payout']} funded accounts for a $100k year.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
