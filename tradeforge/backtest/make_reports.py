"""backtest/make_reports.py — generate all tear-sheet + portfolio PNGs.

Writes, under ``backtest/reports/``:
  * one tear sheet per headline breakout variant (V0, V3, v0_atr_stop),
  * one per complement (level_meanrev/DEFAULT, momentum_thrust/DEFAULT),
  * one for the best blend (min-variance),
  * the correlation heatmap,
  * the blend-vs-singles g chart.

Full-period (2024-06-13 -> 2026-06-12) under the **realistic** cost profile, on
one shared DuckDB connection.

Run:  PYTHONPATH=. .venv/bin/python backtest/make_reports.py
"""

from __future__ import annotations

from pathlib import Path

from backtest.portfolio import analyze, blend_returns, default_edge_set
from backtest.runner import run_strategy
from data.schema import DEFAULT_DB_PATH, connect
from reporting.charts import blend_vs_singles, correlation_heatmap, tear_sheet

REPORTS = Path(__file__).resolve().parent / "reports"
FULL_START, FULL_END = "2024-06-13", "2026-06-12"
COST = "realistic"


def _headline_specs():
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

    return [
        ("breakout_retest_V0",
         lambda: BreakoutRetestStrategy(params=br_load("V0")), "QQQ", "5m",
         "breakout_retest / V0 (baseline, realistic costs)"),
        ("breakout_retest_V3",
         lambda: BreakoutRetestStrategy(params=br_load("V3")), "QQQ", "5m",
         "breakout_retest / V3 (partial + runner, realistic costs)"),
        ("breakout_retest_v0_atr_stop",
         lambda: BreakoutRetestStrategy(params=br_load("v0_atr_stop")), "QQQ", "5m",
         "breakout_retest / v0_atr_stop (ATR stop survivor, realistic costs)"),
        ("level_meanrev_DEFAULT",
         lambda: LevelMeanRevStrategy(params=lmr_load("DEFAULT")), "QQQ", "5m",
         "level_meanrev / DEFAULT (chop fade, realistic costs)"),
        ("momentum_thrust_DEFAULT",
         lambda: MomentumThrustStrategy(params=mt_load("DEFAULT")), "QQQ", "5m",
         "momentum_thrust / DEFAULT (trend follow, realistic costs)"),
    ]


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    con = connect(DEFAULT_DB_PATH)
    written: list[str] = []
    try:
        # ---- per-variant / per-complement tear sheets ----
        for slug, build, symbol, tf, title in _headline_specs():
            res = run_strategy(build(), symbol, tf, start=FULL_START,
                               end=FULL_END, cost_profile=COST, con=con)
            p = tear_sheet(res, title, REPORTS / f"tearsheet_{slug}.png")
            written.append(p)
            print(f"wrote {p}")

        # ---- portfolio analysis (full period) ----
        an = analyze(edges=default_edge_set(), start=FULL_START, end=FULL_END,
                     cost_profile=COST, con=con)

        # best blend tear sheet (min-variance)
        mv_blend = blend_returns(an.returns, an.minvar_weights)
        p = tear_sheet(mv_blend,
                       "BLEND / min-variance (decorrelated edge stack, realistic costs)",
                       REPORTS / "tearsheet_blend_minvar.png")
        written.append(p)
        print(f"wrote {p}")

        # correlation heatmap
        p = correlation_heatmap(an.corr, REPORTS / "correlation_heatmap.png")
        written.append(p)
        print(f"wrote {p}")

        # blend-vs-singles g chart
        p = blend_vs_singles(an.g_table, REPORTS / "blend_vs_singles_g.png")
        written.append(p)
        print(f"wrote {p}")

        print(f"\n{len(written)} PNGs written under {REPORTS}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
