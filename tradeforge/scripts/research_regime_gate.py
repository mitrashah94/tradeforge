#!/usr/bin/env python3
"""scripts/research_regime_gate.py — RESEARCH PROBE: does regime-gating salvage an edge?

Reads everything, writes NOTHING live (research firewall, CLAUDE.md §"Self-
improvement"). It does not touch registry.yaml, limits.yaml, or the shared
hypothesis_log.jsonl — it only prints.

Motivation (2026-06-28): on consolidated Polygon data NO strategy clears the
multiple-testing haircut, BUT the by-regime breakdown shows a real conditional
signal — breakout_retest/v0_atr_stop PF ~1.95 in TREND, momentum_thrust PF ~3.6
in VOL_SHOCK, level_meanrev best in CHOP. This probe asks: if we only trade each
edge in its favorable regime, does a gated variant (or a regime-ROUTED blend)
clear the bar — with an HONEST trade count, CI, and OOS, and with NO lookahead?

LOOKAHEAD DISCIPLINE (the crux): the regime tagger (backtest.stats.regime)
classifies a session from THAT session's own daily OHLC (today's true range,
today's close vs SMA). Using today's tag to gate today's entries would be
lookahead — not realizable live. So the tradeable gate uses the PRIOR session's
regime (lagged-by-one), exactly as the live regime_reader sets policy premarket
from completed data. We ALSO print the same-day (lookahead) number as an
untradeable "attribution ceiling" so the realizable-vs-ideal gap is visible.

Run:  PYTHONPATH=. .venv/bin/python scripts/research_regime_gate.py
"""

from __future__ import annotations

import numpy as np

from backtest.runner import run_strategy
from backtest.stats.confidence import is_underpowered, pf_ci
from backtest.stats.multiple_testing import min_pf_threshold
from backtest.stats.oos import read_vault
from backtest.stats.regime import CHOP, TREND, VOL_SHOCK, regime_tags
from data.schema import DEFAULT_DB_PATH, connect
from data.sessions import et_session_date

FULL_START, FULL_END = "2024-06-13", "2026-06-12"
COST = "realistic"

# Existing honest trial count in the hypothesis log (rigorous_stats: 14 variants).
# Every gated config we test below is a NEW trial and must be added to keep the
# multiple-testing haircut honest.
BASE_TRIALS = 14


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def lagged_regime_map(tags: dict) -> dict:
    """Map each session_date -> the PRIOR session's regime (lookahead-free gate)."""
    dates = sorted(tags)
    return {d: tags[dates[i - 1]] for i, d in enumerate(dates) if i > 0}


def _entry_regimes(trades, regime_map: dict):
    """Series of each trade's gating regime, by its ENTRY session date."""
    es = trades["entry_ts"].apply(et_session_date)
    return es.map(regime_map)


def filter_by_regime(trades, regime_map: dict, allowed: set):
    reg = _entry_regimes(trades, regime_map)
    return trades[reg.isin(allowed)]


def metrics(trades) -> dict:
    """PF (+bootstrap 95% CI), expectancy_R, win rate, n for a trade subset."""
    n = int(len(trades))
    if n == 0:
        return {"n": 0, "pf": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "expR": float("nan"), "win": float("nan")}
    pnl = trades["pnl"].astype(float).to_numpy()
    r = trades["r_multiple"].astype(float).to_numpy()
    if n >= 2:
        pf, lo, hi, _ = pf_ci(pnl, n_boot=2000, seed=0)
    else:
        gp = pnl[pnl > 0].sum()
        gl = -pnl[pnl < 0].sum()
        pf = (gp / gl) if gl > 0 else float("inf")
        lo = hi = float("nan")
    return {"n": n, "pf": float(pf), "lo": float(lo), "hi": float(hi),
            "expR": float(r.mean()), "win": float((pnl > 0).mean())}


def _fpf(x: float) -> str:
    if x != x:
        return "n/a"
    return "inf" if x == float("inf") else f"{x:.3f}"


def _in_window(trades, start: str, end: str):
    import datetime as _dt
    s = _dt.date.fromisoformat(start)
    e = _dt.date.fromisoformat(end)
    es = trades["entry_ts"].apply(et_session_date)
    return trades[(es >= s) & (es <= e)]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    con = connect(DEFAULT_DB_PATH)
    try:
        from strategies.breakout_retest.strategy import (
            BreakoutRetestStrategy, load_params as br_load)
        from strategies.level_meanrev.strategy import (
            LevelMeanRevStrategy, load_params as lmr_load)
        from strategies.momentum_thrust.strategy import (
            MomentumThrustStrategy, load_params as mt_load)

        tags = regime_tags(symbol="QQQ", con=con)
        lagged = lagged_regime_map(tags)
        vault = read_vault()
        oos_start, oos_end = vault["oos_start"], vault["oos_end"]

        # ---- full-period realistic runs for the three base edges ----
        runs = {
            "v0_atr_stop": run_strategy(
                BreakoutRetestStrategy(params=br_load("v0_atr_stop")),
                "QQQ", "5m", start=FULL_START, end=FULL_END, cost_profile=COST, con=con),
            "momentum_thrust": run_strategy(
                MomentumThrustStrategy(params=mt_load("DEFAULT")),
                "QQQ", "5m", start=FULL_START, end=FULL_END, cost_profile=COST, con=con),
            "level_meanrev": run_strategy(
                LevelMeanRevStrategy(params=lmr_load("DEFAULT")),
                "QQQ", "5m", start=FULL_START, end=FULL_END, cost_profile=COST, con=con),
        }
        trades = {k: v.trades for k, v in runs.items()}

        # ---- the gated configs under test (each is a NEW trial) ----
        # (label, base_edge, allowed_regimes)
        configs = [
            ("v0_atr_stop @ trend",            "v0_atr_stop",    {TREND}),
            ("v0_atr_stop @ trend+volshock",   "v0_atr_stop",    {TREND, VOL_SHOCK}),
            ("momentum_thrust @ volshock",     "momentum_thrust", {VOL_SHOCK}),
            ("momentum_thrust @ trend+vsk",    "momentum_thrust", {TREND, VOL_SHOCK}),
            ("level_meanrev @ chop",           "level_meanrev",  {CHOP}),
        ]
        n_trials = BASE_TRIALS + len(configs) + 1  # +1 for the routed blend
        thr = min_pf_threshold(n_trials)

        print("TradeForge — RESEARCH PROBE: regime-gated edges (realistic costs, QQQ 5m)")
        print(f"full period {FULL_START} -> {FULL_END}; gate = PRIOR-session regime "
              "(lookahead-free)")
        print(f"honest n_trials = {BASE_TRIALS} prior + {len(configs)+1} new = {n_trials}"
              f"  ->  PF haircut >= {thr:.3f}")

        print("\n===== gated variants (TRADEABLE: prior-session regime gate) =====")
        hdr = (f"  {'config':<32s} {'n':>4s} {'PF':>6s} {'PF 95% CI':>16s} "
               f"{'expR':>6s} {'win':>5s} {'OOS_PF':>7s} {'OOSn':>5s} {'clears?':>8s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        def report_row(label, tr_subset):
            m = metrics(tr_subset)
            oos = metrics(_in_window(tr_subset, oos_start, oos_end))
            ci = f"[{_fpf(m['lo'])},{_fpf(m['hi'])}]"
            up = "!" if (m["n"] and is_underpowered(m["n"])) else " "
            clears = "YES" if (m["pf"] == m["pf"] and m["pf"] >= thr
                              and not is_underpowered(m["n"])) else "no"
            win = f"{m['win']*100:.0f}%" if m["win"] == m["win"] else "n/a"
            print(f"  {label:<32s} {m['n']:>4d}{up} {_fpf(m['pf']):>5s} {ci:>16s} "
                  f"{m['expR']:>+6.2f} {win:>5s} {_fpf(oos['pf']):>7s} {oos['n']:>5d} "
                  f"{clears:>8s}")
            return m

        for label, edge, allowed in configs:
            report_row(label, filter_by_regime(trades[edge], lagged, allowed))

        # ---- regime-ROUTED blend: each edge only in its best regime ----
        routed = pd_concat([
            filter_by_regime(trades["v0_atr_stop"], lagged, {TREND}),
            filter_by_regime(trades["momentum_thrust"], lagged, {VOL_SHOCK}),
            filter_by_regime(trades["level_meanrev"], lagged, {CHOP}),
        ])
        print("  " + "-" * (len(hdr) - 2))
        report_row("ROUTED blend (trend/vsk/chop)", routed)

        # ---- attribution ceiling (SAME-DAY tag = LOOKAHEAD, NOT tradeable) ----
        print("\n===== attribution ceiling (SAME-DAY regime = LOOKAHEAD, NOT tradeable) =====")
        print("  (shows the gap between the ideal same-day signal and the realizable gate)")
        for label, edge, allowed in [
            ("v0_atr_stop @ trend [ideal]", "v0_atr_stop", {TREND}),
            ("momentum_thrust @ vsk [ideal]", "momentum_thrust", {VOL_SHOCK}),
        ]:
            m = metrics(filter_by_regime(trades[edge], tags, allowed))
            print(f"  {label:<32s} n={m['n']:>4d}  PF={_fpf(m['pf'])}  "
                  f"expR={m['expR']:+.2f}  (NOT a tradeable result)")

        print("\n===== verdict =====")
        print(f"  A gated config 'clears' only if PF >= {thr:.3f} (haircut at n_trials="
              f"{n_trials}) AND it is not underpowered (>= ~100 trades).")
        print("  Underpowered (!) configs cannot clear regardless of point PF — too few")
        print("  trend/vol-shock days in 2y to be statistically credible. Treat any pass")
        print("  as a HYPOTHESIS to confirm with more data / OOS, never an auto-promote.")
        return 0
    finally:
        con.close()


def pd_concat(frames):
    import pandas as pd
    frames = [f for f in frames if f is not None and len(f) > 0]
    if not frames:
        import pandas as pd
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raise SystemExit(main())
