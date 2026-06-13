"""backtest/rigorous_stats.py — the integration-stage rigorous validation pass.

Runs the full MASTER_PLAN §5 validation discipline over the headline
breakout_retest variants (V0, V3, v0_atr_stop) plus the two complements
(level_meanrev/DEFAULT, momentum_thrust/DEFAULT), and emits the numbers the
REPORT.md files and the registry consume:

  * PF with a bootstrap CI + trade count (flag underpowered <100 trades);
  * IS vs OOS performance on the LOCKED vault range (split_is_oos); OOS is the
    locked 2026-01-18 -> 2026-06-12 slice — evaluated, never tuned on;
  * a rolling walk-forward aggregate (pooled OOS PF/expectancy);
  * a by-regime breakdown (trend / chop / vol_shock) of the daily returns;
  * a HypothesisLog entry for EVERY variant evaluated (survivors AND failures —
    the multiple-testing discipline), then the min_pf_threshold(n_trials)
    haircut applied with n_trials = the honest total tried.

ALL runs share ONE DuckDB connection. Costs are the **realistic** profile (the
honest small-account picture) for the headline numbers. Timestamps are the fixed
ISO string so the log is deterministic.

Run:  PYTHONPATH=. .venv/bin/python backtest/rigorous_stats.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.runner import daily_returns_pct, run_strategy
from backtest.stats.confidence import is_underpowered, pf_ci, expectancy_ci
from backtest.stats.metrics import (
    sharpe,
    sharpe_per_period,
    trade_pnls,
    trade_r_multiples,
)
from backtest.stats.multiple_testing import (
    HypothesisLog,
    deflated_sharpe,
    min_pf_threshold,
)
from backtest.stats.oos import read_vault, split_is_oos
from backtest.stats.regime import by_regime, regime_tags
from backtest.stats.walk_forward import aggregate_folds, walk_forward
from data.schema import DEFAULT_DB_PATH, connect

# Deterministic timestamp for the hypothesis log (no datetime.now in the path).
RUN_TS = "2026-06-13"
COST_PROFILE = "realistic"
FULL_START, FULL_END = "2024-06-13", "2026-06-12"


# --------------------------------------------------------------------------- #
# Variant catalog (each = its own hypothesis -> logged, survivor or not)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    name: str               # log name, e.g. "breakout_retest/V0"
    factory: object
    params_loader: object   # callable(variant_key) -> params dict
    variant_key: str
    symbol: str = "QQQ"
    timeframe: str = "5m"
    headline: bool = False  # True -> gets a tear sheet + a REPORT mention

    def build(self):
        return self.factory(self.params_loader(self.variant_key))


def all_variants() -> list[Variant]:
    """Every variant evaluated this pass — the honest multiple-testing count.

    The 3 headline breakout variants + the 2 complement DEFAULTs are marked
    ``headline``; the remaining breakout/complement variants are still RUN and
    LOGGED (they were tried, so they count against the haircut) but are not
    featured. This is the "log every hypothesis, not just survivors" rule.
    """
    from strategies.breakout_retest.strategy import (
        BreakoutRetestStrategy,
        load_params as br_load,
    )
    from strategies.level_meanrev.strategy import (
        LevelMeanRevStrategy,
        load_params as lmr_load,
    )
    from strategies.momentum_thrust.strategy import (
        MomentumThrustStrategy,
        load_params as mt_load,
    )

    br = lambda p: BreakoutRetestStrategy(params=p)
    lmr = lambda p: LevelMeanRevStrategy(params=p)
    mt = lambda p: MomentumThrustStrategy(params=p)

    return [
        # breakout_retest ablation ladder (V0/V3/v0_atr_stop headline)
        Variant("breakout_retest/V0", br, br_load, "V0", headline=True),
        Variant("breakout_retest/V1", br, br_load, "V1"),
        Variant("breakout_retest/V2", br, br_load, "V2"),
        Variant("breakout_retest/V3", br, br_load, "V3", headline=True),
        Variant("breakout_retest/V4", br, br_load, "V4"),
        Variant("breakout_retest/v0_atr_stop", br, br_load, "v0_atr_stop",
                headline=True),
        # level_meanrev (DEFAULT headline)
        Variant("level_meanrev/DEFAULT", lmr, lmr_load, "DEFAULT", headline=True),
        Variant("level_meanrev/V0", lmr, lmr_load, "V0"),
        Variant("level_meanrev/V1", lmr, lmr_load, "V1"),
        Variant("level_meanrev/V2", lmr, lmr_load, "V2"),
        # momentum_thrust (DEFAULT headline)
        Variant("momentum_thrust/DEFAULT", mt, mt_load, "DEFAULT", headline=True),
        Variant("momentum_thrust/V0", mt, mt_load, "V0"),
        Variant("momentum_thrust/V1", mt, mt_load, "V1"),
        Variant("momentum_thrust/V2", mt, mt_load, "V2"),
    ]


# --------------------------------------------------------------------------- #
# Per-variant evaluation
# --------------------------------------------------------------------------- #
@dataclass
class VariantStats:
    name: str
    headline: bool
    # full-period (realistic) headline
    n_trades: int
    pf: float
    pf_lo: float
    pf_hi: float
    underpowered: bool
    expectancy_r: float
    exp_lo: float
    exp_hi: float
    sharpe: float
    dsr: float
    # IS vs OOS
    is_n: int
    is_pf: float
    is_expr: float
    oos_n: int
    oos_pf: float
    oos_expr: float
    # walk-forward
    wf_folds: int
    wf_n: int
    wf_pf: float
    wf_expr: float
    # by-regime (PF per regime, on full period)
    regime: dict = field(default_factory=dict)


def evaluate_variant(
    v: Variant, con, tags: dict, oos_start: str, oos_end: str,
    is_start: str, is_end: str,
) -> VariantStats:
    """Compute the full rigorous bundle for one variant (shared connection)."""
    # ---- full-period run (realistic) ----
    full = run_strategy(v.build(), v.symbol, v.timeframe,
                        start=FULL_START, end=FULL_END,
                        cost_profile=COST_PROFILE, con=con)
    s = full.summary()
    pnls = trade_pnls(full)
    rmults = trade_r_multiples(full)
    pf, pf_lo, pf_hi, n = pf_ci(pnls, n_boot=2000, seed=0)
    _expm, exp_lo, exp_hi, _en = expectancy_ci(rmults, kind="r", n_boot=2000, seed=0)
    dr = daily_returns_pct(full)
    shp = sharpe(dr) if len(dr) >= 2 else float("nan")

    # Deflated Sharpe (per-observation Sharpe, n_trials filled at report time;
    # here use a placeholder n_trials=1 and recompute with the real count below).
    spp = sharpe_per_period(dr) if len(dr) >= 2 else float("nan")

    # ---- IS vs OOS (locked vault) ----
    is_res = run_strategy(v.build(), v.symbol, v.timeframe,
                          start=is_start, end=is_end,
                          cost_profile=COST_PROFILE, con=con)
    is_s = is_res.summary()
    oos_res = run_strategy(v.build(), v.symbol, v.timeframe,
                           start=oos_start, end=oos_end,
                           cost_profile=COST_PROFILE, con=con)
    oos_s = oos_res.summary()

    # ---- walk-forward (rolling 6m IS / 1m OOS) ----
    def run_fn(s_, e_):
        return run_strategy(v.build(), v.symbol, v.timeframe,
                            start=s_, end=e_, cost_profile=COST_PROFILE, con=con)

    folds = walk_forward(run_fn, FULL_START, FULL_END,
                         is_months=6, oos_months=1, keep_results=True)
    wf = aggregate_folds(folds)

    # ---- by-regime ----
    reg = by_regime(dr, tags)
    regime_pf = {k: reg[k]["profit_factor"] for k in reg}
    regime_full = reg

    return VariantStats(
        name=v.name, headline=v.headline,
        n_trades=n, pf=pf, pf_lo=pf_lo, pf_hi=pf_hi,
        underpowered=is_underpowered(n),
        expectancy_r=s["expectancy_R"], exp_lo=exp_lo, exp_hi=exp_hi,
        sharpe=shp, dsr=spp,  # store per-period sharpe; DSR computed at report
        is_n=int(is_s["n_trades"]), is_pf=is_s["profit_factor"],
        is_expr=is_s["expectancy_R"],
        oos_n=int(oos_s["n_trades"]), oos_pf=oos_s["profit_factor"],
        oos_expr=oos_s["expectancy_R"],
        wf_folds=int(wf["n_folds"]), wf_n=int(wf["n_trades"]),
        wf_pf=wf["profit_factor"], wf_expr=wf["expectancy_r"],
        regime=regime_full,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _fmt_pf(x: float) -> str:
    if x != x:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.3f}"


def run(con=None, log_path=None) -> dict:
    """Evaluate every variant, log each hypothesis, apply the haircut.

    Returns a dict with the per-variant ``VariantStats`` list, the trial count,
    the PF threshold, and the survivors — consumed by the REPORT writers.
    """
    own = con is None
    if own:
        con = connect(DEFAULT_DB_PATH)

    log = HypothesisLog() if log_path is None else HypothesisLog(path=log_path)
    # Fresh log for this pass (deterministic, reproducible).
    log.clear()

    try:
        vault = read_vault()
        oos_start, oos_end = vault["oos_start"], vault["oos_end"]
        (is_r, _oos_r) = split_is_oos(FULL_START, FULL_END, oos_fraction=0.2)
        is_start, is_end = is_r

        tags = regime_tags(symbol="QQQ", con=con)

        variants = all_variants()
        stats: list[VariantStats] = []
        for v in variants:
            vs = evaluate_variant(v, con, tags, oos_start, oos_end,
                                  is_start, is_end)
            stats.append(vs)

        n_trials = len(variants)
        pf_threshold = min_pf_threshold(n_trials)

        # ---- log every hypothesis (survivor or not) ----
        for vs in stats:
            passed = (vs.pf == vs.pf) and (vs.pf >= pf_threshold)
            dsr = deflated_sharpe(vs.dsr, n_trials=n_trials,
                                  n_obs=max(vs.n_trades, 2))
            note = (f"OOS pf={_fmt_pf(vs.oos_pf)} (n={vs.oos_n}); "
                    f"wf pf={_fmt_pf(vs.wf_pf)} (n={vs.wf_n}); "
                    f"DSR={dsr:.3f}; "
                    f"{'UNDERPOWERED' if vs.underpowered else 'powered'}; "
                    f"threshold={pf_threshold:.3f}")
            log.append(
                name=vs.name,
                params={"symbol": "QQQ", "tf": "5m", "cost": COST_PROFILE},
                n_trades=vs.n_trades,
                pf=vs.pf,
                expectancy_r=vs.expectancy_r,
                sharpe=vs.sharpe,
                timestamp=RUN_TS,
                passed=passed,
                note=note,
            )

        survivors = [vs.name for vs in stats
                     if (vs.pf == vs.pf) and (vs.pf >= pf_threshold)]

        return {
            "stats": stats,
            "n_trials": n_trials,
            "pf_threshold": pf_threshold,
            "survivors": survivors,
            "is_range": (is_start, is_end),
            "oos_range": (oos_start, oos_end),
            "tags": tags,
        }
    finally:
        if own:
            con.close()


def print_report(out: dict) -> None:
    stats = out["stats"]
    print("TradeForge — rigorous stats pass (realistic costs, QQQ 5m)")
    print(f"IS  range: {out['is_range'][0]} -> {out['is_range'][1]}")
    print(f"OOS range (LOCKED vault): {out['oos_range'][0]} -> {out['oos_range'][1]}")
    print(f"variants tried (n_trials): {out['n_trials']}  "
          f"-> min_pf_threshold = {out['pf_threshold']:.3f}")

    print("\n===== full-period PF (with bootstrap 95% CI) + IS/OOS + walk-forward =====")
    hdr = (f"  {'variant':<30s} {'n':>4s} {'PF':>6s} {'PF 95% CI':>16s} "
           f"{'expR':>6s} {'IS_PF':>6s} {'OOS_PF':>7s} {'OOSn':>5s} {'WF_PF':>6s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for vs in stats:
        ci = f"[{_fmt_pf(vs.pf_lo)},{_fmt_pf(vs.pf_hi)}]"
        flag = "*" if vs.headline else " "
        up = "!" if vs.underpowered else " "
        print(f" {flag}{vs.name:<30s} {vs.n_trades:>4d}{up} {_fmt_pf(vs.pf):>5s} "
              f"{ci:>16s} {vs.expectancy_r:>+6.2f} {_fmt_pf(vs.is_pf):>6s} "
              f"{_fmt_pf(vs.oos_pf):>7s} {vs.oos_n:>5d} {_fmt_pf(vs.wf_pf):>6s}")
    print("  (* = headline variant, ! = underpowered <100 trades)")

    print("\n===== by-regime PF (full period, daily returns) =====")
    print(f"  {'variant':<30s} {'trend':>10s} {'chop':>10s} {'vol_shock':>10s}")
    for vs in stats:
        if not vs.headline:
            continue
        r = vs.regime
        print(f"  {vs.name:<30s} "
              f"{_fmt_pf(r['trend']['profit_factor']):>10s} "
              f"{_fmt_pf(r['chop']['profit_factor']):>10s} "
              f"{_fmt_pf(r['vol_shock']['profit_factor']):>10s}")

    print("\n===== multiple-testing haircut =====")
    print(f"  threshold (n_trials={out['n_trials']}): PF >= {out['pf_threshold']:.3f}")
    print(f"  survivors clearing the haircut: "
          f"{out['survivors'] if out['survivors'] else 'NONE'}")


def main() -> int:
    con = connect(DEFAULT_DB_PATH)
    try:
        out = run(con=con)
        print_report(out)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
