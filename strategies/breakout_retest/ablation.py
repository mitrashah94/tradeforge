"""strategies/breakout_retest/ablation.py — the V0->V4 ablation runner.

Runs the breakout_retest ablation ladder (MASTER_PLAN.md §5) on QQQ 5m over the
full ~2y history, under BOTH cost profiles, and prints a raw comparison table so
the deltas between components are visible:

    V0          PDH/PDL break->retest + Fixed 2R + role-reversal 1-tick stop
    V1 = V0 +   NTZ no-trade filter (block entries whose level is in the NTZ)
    V2 = V1 +   PMH/PML levels (a second level set -> more setups/day)
    V3 = V2 +   partial + runner (scale 50% at +1R, breakeven, trail the rest)
    V4          full playbook (V3 + break_buffer_atr clean-break filter)
    v0_atr_stop V0 but stop = level -/+ k*ATR14 (probes the tight-stop fragility)

This produces ONLY the raw comparison (n_trades, win_rate, PF, expectancy_R,
max_dd per cost profile). It uses ONLY the engine + result summary — the
rigorous CI / OOS / walk-forward / multiple-testing analysis lives in the
integration stage (backtest.stats), which this file deliberately does NOT touch.

Run:  PYTHONPATH=. .venv/bin/python strategies/breakout_retest/ablation.py
"""

from __future__ import annotations

from backtest.engine.cost import CostModel
from backtest.engine.engine import BacktestEngine, bars_from_df
from backtest.engine.result import BacktestResult
from backtest.run_gate import load_qqq_5m, _session_of
from data.schema import DEFAULT_DB_PATH, connect
from data.sessions import et_session_date
from strategies.breakout_retest.strategy import BreakoutRetestStrategy, load_params

VARIANTS = ["V0", "V1", "V2", "V3", "V4", "v0_atr_stop"]
COST_PROFILES = ["tv_style", "realistic"]


def run_variant(
    bars_df, levels_by_session, variant: str, cost_profile: str
) -> BacktestResult:
    """Run one ablation ``variant`` under one ``cost_profile`` and return result.

    Fill realism is matched to the cost profile exactly as the V0 gate does:
    ``tv_style`` uses TradingView-parity optimistic fills (stops fill at the stop
    price); ``realistic`` models adverse stop gaps (a stop whose bar opens beyond
    it fills at the open). This keeps every variant comparable to the V0 gate.
    """
    params = load_params(variant)
    strat = BreakoutRetestStrategy(params=params)
    cost = CostModel.from_profile(cost_profile)
    engine = BacktestEngine(
        strat,
        cost,
        symbol="QQQ",
        asset_class="equity",
        tick=0.01,
        initial_equity=100_000.0,
        percent_of_equity=1.0,
        model_stop_gaps=(cost_profile != "tv_style"),
    )
    bars = bars_from_df(bars_df)
    return engine.run(bars, levels_by_session, _session_of)


def _fmt_pf(pf: float) -> str:
    if pf != pf:  # nan
        return "n/a"
    if pf == float("inf"):
        return "inf"
    return f"{pf:.3f}"


def _row(variant: str, s: dict) -> str:
    wr = s["win_rate"]
    wr_s = "n/a" if wr != wr else f"{wr * 100:5.1f}%"
    exp = s["expectancy_R"]
    return (
        f"  {variant:<12s} {s['n_trades']:>8d} {wr_s:>9s} "
        f"{_fmt_pf(s['profit_factor']):>8s} {exp:>+13.3f} "
        f"{s['max_drawdown_pct']:>9.2f}%"
    )


def _header() -> str:
    return (
        f"  {'variant':<12s} {'n_trades':>8s} {'win_rate':>9s} "
        f"{'PF':>8s} {'expectancy_R':>13s} {'max_dd':>10s}"
    )


def main() -> int:
    con = connect(DEFAULT_DB_PATH)
    bars_df, levels_by_session = load_qqq_5m(con)
    n_sessions = bars_df["ts_utc"].apply(et_session_date).nunique()

    print("TradeForge — breakout_retest V0->V4 ablation (QQQ 5m)")
    print(f"loaded {len(bars_df)} RTH 5m bars across {n_sessions} sessions")
    print("ladder: V0 base | V1 +NTZ | V2 +PMH/PML | V3 +partial/runner | "
          "V4 full | v0_atr_stop (k*ATR stop)")

    # variant -> {cost_profile -> summary}
    results: dict[str, dict[str, dict]] = {}
    for variant in VARIANTS:
        results[variant] = {}
        for cp in COST_PROFILES:
            res = run_variant(bars_df, levels_by_session, variant, cp)
            results[variant][cp] = res.summary()

    for cp in COST_PROFILES:
        print(f"\n===== cost profile: {cp} =====")
        print(_header())
        for variant in VARIANTS:
            print(_row(variant, results[variant][cp]))

    # ---- raw read of which components moved expectancy (tv_style, ascending) --
    print("\n===== component deltas (raw read — expectancy_R, tv_style) =====")
    ladder = [("V0", "base"),
              ("V1", "+ NTZ filter"),
              ("V2", "+ PMH/PML"),
              ("V3", "+ partial/runner"),
              ("V4", "+ break_buffer (full)")]
    prev = None
    for variant, label in ladder:
        exp = results[variant]["tv_style"]["expectancy_R"]
        if prev is None:
            print(f"  {variant:<4s} {label:<24s} expR={exp:+.3f}")
        else:
            d = exp - prev
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
            print(f"  {variant:<4s} {label:<24s} expR={exp:+.3f}  "
                  f"(Δ vs prior {d:+.3f} {arrow})")
        prev = exp
    exp_v0 = results["V0"]["tv_style"]["expectancy_R"]
    exp_atr = results["v0_atr_stop"]["tv_style"]["expectancy_R"]
    print(f"  v0_atr_stop  vs V0 (k*ATR vs 1-tick stop) expR={exp_atr:+.3f}  "
          f"(Δ {exp_atr - exp_v0:+.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
