"""Backtest the 6 semis x 4 breakout_retest configs on our pipeline.

Reproduces a pasted external table on TradeForge's own engine + data
(Alpaca IEX 5m, ~2y). The four configs map to breakout_retest params:
  Break+2R     entry_type=break,  target_mode=fixed_2r
  Break+Trail  entry_type=break,  target_mode=trailing (partial+runner*)
  Retest+2R    entry_type=retest, target_mode=fixed_2r   (== V0)
  Retest+Trail entry_type=retest, target_mode=trailing (partial+runner*)

* Our trailing exit is the V3-style partial(50%@1R)+breakeven+prior-bar trail;
  the external table's "Trail" is a pure prior-bar trail (no partial). So the
  2R rows are apples-to-apples; the Trail rows are directional only.

$ figures use percent_of_equity=1.0 on $100k initial (the Pine basis), so they
are sizing-dependent; Win%/PF/Trades/ExpR/MaxDD are the robust comparisons.
"""
from __future__ import annotations

import numpy as np

from backtest.runner import run_strategy
from data.schema import DEFAULT_DB_PATH, connect
from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

TICKERS = ["NVDA", "AVGO", "TSM", "ASML", "MU", "AMD"]
CONFIGS = {
    "Break+2R":     {"entry_type": "break",  "target_mode": "fixed_2r", "partial_runner": False},
    "Break+Trail":  {"entry_type": "break",  "target_mode": "trailing", "partial_runner": True},
    "Retest+2R":    {"entry_type": "retest", "target_mode": "fixed_2r", "partial_runner": False},
    "Retest+Trail": {"entry_type": "retest", "target_mode": "trailing", "partial_runner": True},
}


def _avg_win_loss_dollar(result):
    t = result.trades
    if t is None or len(t) == 0:
        return float("nan"), float("nan")
    pnl = t["pnl"].astype(float).to_numpy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(abs(losses.mean())) if len(losses) else 0.0
    return aw, al


def run_one(symbol, cfg_overrides, cost_profile, con):
    params = load_params("V0")           # defaults
    params.update(cfg_overrides)
    strat = BreakoutRetestStrategy(params=params)
    res = run_strategy(strat, symbol, "5m", cost_profile=cost_profile,
                       initial_equity=100_000.0, percent_of_equity=1.0, con=con)
    s = res.summary()
    aw, al = _avg_win_loss_dollar(res)
    return {
        "n": s["n_trades"], "win": s["win_rate"], "pf": s["profit_factor"],
        "aw": aw, "al": al, "dd": s["max_drawdown_pct"],
        "exp_d": s["expectancy_dollar"], "exp_r": s["expectancy_R"],
    }


def _pf(x):
    if x != x:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.2f}"


def _win(x):
    return "n/a" if x != x else f"{x*100:.1f}%"


def main():
    con = connect(DEFAULT_DB_PATH)
    rows = []  # (ticker, cfg, tv_metrics, real_metrics)
    for tk in TICKERS:
        for cfg, ov in CONFIGS.items():
            tv = run_one(tk, ov, "tv_style", con)
            rl = run_one(tk, ov, "realistic", con)
            rows.append((tk, cfg, tv, rl))

    # ---- table 1: tv_style (mirrors the external columns) ----
    print("=" * 104)
    print("TradeForge backtest — 6 semis x 4 configs — Alpaca IEX 5m, ~2y — COST PROFILE: tv_style")
    print("=" * 104)
    hdr = f"{'Ticker':<6}{'Config':<13}{'Trades':>7}{'Win%':>7}{'PF':>6}{'AvgW$':>8}{'AvgL$':>8}{'MaxDD':>7}{'Exp$':>7}{'ExpR':>8}"
    print(hdr)
    print("-" * len(hdr))
    for tk, cfg, tv, _ in rows:
        warn = " " if tv["n"] >= 100 else "*"
        print(f"{tk:<6}{cfg:<13}{tv['n']:>5}{warn} {_win(tv['win']):>6}{_pf(tv['pf']):>6}"
              f"{tv['aw']:>8.0f}{tv['al']:>8.0f}{tv['dd']:>6.1f}%{tv['exp_d']:>+7.0f}{tv['exp_r']:>+8.2f}R")
    print("  (* = under 100 trades -> underpowered, wide CIs)")

    # ---- table 2: realistic (the honest net-of-costs read) ----
    print()
    print("=" * 70)
    print("Net of REALISTIC small-account costs (PF / ExpR) — the honest read")
    print("=" * 70)
    hdr2 = f"{'Ticker':<6}{'Config':<13}{'Trades':>7}{'Win%':>7}{'PF':>6}{'ExpR':>8}"
    print(hdr2)
    print("-" * len(hdr2))
    for tk, cfg, _, rl in rows:
        print(f"{tk:<6}{cfg:<13}{rl['n']:>7}{_win(rl['win']):>7}{_pf(rl['pf']):>6}{rl['exp_r']:>+8.2f}R")

    # ---- what survives realistic costs (PF>1 and ExpR>0) ----
    print()
    print("Configs profitable NET of realistic costs (PF>1.0 and ExpR>0):")
    survivors = [(tk, cfg, rl) for tk, cfg, _, rl in rows if rl["pf"] == rl["pf"] and rl["pf"] > 1.0 and rl["exp_r"] > 0]
    if not survivors:
        print("  (none)")
    for tk, cfg, rl in sorted(survivors, key=lambda x: -x[2]["exp_r"]):
        print(f"  {tk:<6} {cfg:<13} PF {rl['pf']:.2f}  ExpR {rl['exp_r']:+.2f}R  n={rl['n']}")
    con.close()


if __name__ == "__main__":
    main()
