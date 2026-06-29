#!/usr/bin/env python3
"""scripts/research_breadth.py — RESEARCH PROBE: does BREADTH power up the edge?

Reads everything, writes NOTHING live (research firewall). It does not touch
registry.yaml, limits.yaml, or the shared hypothesis_log.jsonl — it only prints.

Motivation: the single-symbol regime-gate probe (scripts/research_regime_gate.py)
showed the trend-gated breakout (v0_atr_stop, entered only when the PRIOR session
was a TREND regime) flips expectancy positive but is UNDERPOWERED (~52 trades / 2y
on QQQ) — too few trend days on one symbol to clear the multiple-testing haircut.
The natural lever (MASTER_PLAN §1.A "edge portfolio / breadth"): run the SAME
gated edge across MANY liquid names and POOL the trades. More trend days across
the universe -> more trades -> tighter CI. Does the pooled gated edge clear?

DISCIPLINE: same lookahead-free gate as the single-symbol probe — each symbol is
gated on ITS OWN prior-session regime (realizable premarket). Costs are the
realistic profile. OOS is the locked vault window (entries inside it). We report
the gated pool AND the ungated pool (every trade) so the gating effect is visible,
plus a per-symbol breakdown. A pass is a HYPOTHESIS to confirm, never an auto-
promote — and breadth across K symbols is itself multiple testing (the haircut
threshold rises accordingly).

Run:  PYTHONPATH=. .venv/bin/python scripts/research_breadth.py
      PYTHONPATH=. .venv/bin/python scripts/research_breadth.py --symbols SPY,QQQ,AAPL,...
"""

from __future__ import annotations

import argparse

import numpy as np

from backtest.runner import run_strategy
from backtest.stats.confidence import is_underpowered, pf_ci
from backtest.stats.multiple_testing import min_pf_threshold
from backtest.stats.oos import read_vault
from backtest.stats.regime import TREND, VOL_SHOCK, regime_tags
from data.schema import DEFAULT_DB_PATH, connect
from data.sessions import et_session_date

FULL_START, FULL_END = "2024-06-13", "2026-06-12"
COST = "realistic"
VARIANT = "v0_atr_stop"
DEFAULT_UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD"]
# Existing honest trial count (rigorous_stats=14) + the 6 single-symbol gate
# configs already probed; this breadth pass adds more (counted below).
PRIOR_TRIALS = 14 + 6


def lagged(tags: dict) -> dict:
    dates = sorted(tags)
    return {d: tags[dates[i - 1]] for i, d in enumerate(dates) if i > 0}


def gated_trades(trades, gate_map: dict, allowed: set):
    es = trades["entry_ts"].apply(et_session_date)
    reg = es.map(gate_map)
    return trades[reg.isin(allowed)]


def metrics(pnl: np.ndarray, r: np.ndarray) -> dict:
    n = int(pnl.size)
    if n == 0:
        return {"n": 0, "pf": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "expR": float("nan"), "win": float("nan")}
    if n >= 2:
        pf, lo, hi, _ = pf_ci(pnl, n_boot=2000, seed=0)
    else:
        gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        pf = (gp / gl) if gl > 0 else float("inf")
        lo = hi = float("nan")
    return {"n": n, "pf": float(pf), "lo": float(lo), "hi": float(hi),
            "expR": float(r.mean()), "win": float((pnl > 0).mean())}


def _fpf(x: float) -> str:
    return "n/a" if x != x else ("inf" if x == float("inf") else f"{x:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TradeForge breadth research probe")
    p.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    p.add_argument("--allowed", default="trend", help="favorable regimes (comma): trend[,vol_shock]")
    p.add_argument("--db", default=DEFAULT_DB_PATH)
    args = p.parse_args(argv)

    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    allow_map = {"trend": TREND, "vol_shock": VOL_SHOCK}
    allowed = {allow_map[a.strip()] for a in args.allowed.split(",") if a.strip() in allow_map}

    from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

    con = connect(args.db)
    try:
        vault = read_vault()
        oos_s, oos_e = vault["oos_start"], vault["oos_end"]
        import datetime as _dt
        oos_s_d, oos_e_d = _dt.date.fromisoformat(oos_s), _dt.date.fromisoformat(oos_e)

        # Which symbols actually have bars?
        have = {r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM bars WHERE timeframe='5m'").fetchall()}
        universe = [s for s in universe if s in have]
        missing = [s for s in (x.strip().upper() for x in args.symbols.split(",")) if s and s not in have]

        print("TradeForge — BREADTH research probe (v0_atr_stop, trend-gated, realistic costs)")
        print(f"gate = each symbol's PRIOR-session regime in {{{','.join(sorted(allowed))}}} "
              "(lookahead-free)")
        if missing:
            print(f"  (skipping symbols with no bars: {', '.join(missing)})")

        per_symbol = []
        gated_pnl, gated_r = [], []
        ungated_pnl, ungated_r = [], []
        gated_oos_pnl = []

        for sym in universe:
            strat = BreakoutRetestStrategy(params=load_params(VARIANT))
            res = run_strategy(strat, sym, "5m", start=FULL_START, end=FULL_END,
                               cost_profile=COST, con=con)
            tr = res.trades
            if tr is None or len(tr) == 0:
                per_symbol.append((sym, 0, float("nan"), 0, float("nan")))
                continue
            tags = lagged(regime_tags(symbol=sym, con=con))
            g = gated_trades(tr, tags, allowed)

            ungated_pnl.append(tr["pnl"].astype(float).to_numpy())
            ungated_r.append(tr["r_multiple"].astype(float).to_numpy())
            gp = g["pnl"].astype(float).to_numpy()
            gr = g["r_multiple"].astype(float).to_numpy()
            gated_pnl.append(gp)
            gated_r.append(gr)
            # OOS slice (entry session inside the locked vault)
            es = g["entry_ts"].apply(et_session_date)
            goos = g[(es >= oos_s_d) & (es <= oos_e_d)]
            gated_oos_pnl.append(goos["pnl"].astype(float).to_numpy())

            m = metrics(gp, gr) if gp.size else {"n": 0, "pf": float("nan"), "expR": float("nan")}
            per_symbol.append((sym, len(tr), res.summary()["profit_factor"], m["n"], m["pf"]))

        def cat(arrs):
            return np.concatenate(arrs) if arrs else np.array([], dtype="float64")

        gP, gR = cat(gated_pnl), cat(gated_r)
        uP, uR = cat(ungated_pnl), cat(ungated_r)
        gOOS = cat(gated_oos_pnl)

        # honest multiple-testing count: prior trials + one per symbol probed + the
        # ungated-pool and gated-pool aggregates.
        n_trials = PRIOR_TRIALS + len(universe) + 2
        thr = min_pf_threshold(n_trials)

        print(f"\n  universe: {', '.join(universe)}  ({len(universe)} symbols)")
        print(f"  honest n_trials = {PRIOR_TRIALS} prior + {len(universe)} symbols + 2 pools = "
              f"{n_trials}  ->  PF haircut >= {thr:.3f}")

        print("\n===== per-symbol (v0_atr_stop) =====")
        print(f"  {'symbol':<7s} {'all_n':>6s} {'all_PF':>7s} {'gated_n':>8s} {'gated_PF':>9s}")
        for sym, an, apf, gn, gpf in per_symbol:
            print(f"  {sym:<7s} {an:>6d} {_fpf(apf):>7s} {gn:>8d} {_fpf(gpf):>9s}")

        print("\n===== POOLED across the universe =====")
        um = metrics(uP, uR)
        gm = metrics(gP, gR)
        om = metrics(gOOS, gOOS)  # OOS PF only needs pnl; r unused for pf
        print(f"  {'pool':<16s} {'n':>5s} {'PF':>6s} {'PF 95% CI':>16s} {'expR':>6s} {'win':>5s}")
        print(f"  {'ungated (all)':<16s} {um['n']:>5d} {_fpf(um['pf']):>6s} "
              f"[{_fpf(um['lo'])},{_fpf(um['hi'])}]".rjust(16) +
              f" {um['expR']:>+6.2f} {um['win']*100:>4.0f}%")
        print(f"  {'GATED (trend)':<16s} {gm['n']:>5d} {_fpf(gm['pf']):>6s} "
              f"[{_fpf(gm['lo'])},{_fpf(gm['hi'])}]".rjust(16) +
              f" {gm['expR']:>+6.2f} {gm['win']*100:>4.0f}%")
        print(f"  GATED OOS (vault {oos_s}..{oos_e}): n={om['n']}  PF={_fpf(om['pf'])}")

        print("\n===== verdict =====")
        powered = gm["n"] >= 100 and not is_underpowered(gm["n"])
        clears = (gm["pf"] == gm["pf"]) and gm["pf"] >= thr and powered
        print(f"  pooled GATED: n={gm['n']} ({'POWERED' if powered else 'UNDERPOWERED'}), "
              f"PF={_fpf(gm['pf'])} vs haircut {thr:.3f}  ->  "
              f"{'CLEARS (confirm OOS + walk-forward next)' if clears else 'does NOT clear'}")
        print("  Breadth multiplies trades (power) but each added symbol also raises the haircut;")
        print("  a genuine edge must beat the RISING bar, not just accumulate trades.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
