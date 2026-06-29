#!/usr/bin/env python3
"""scripts/backtest_daily_sleeves.py — RESEARCH: run the daily sleeves vs SPY.

Drives the Phase-2 daily sleeves (momentum_rotation, swing_meanrev) through
backtest.daily.run_daily on the deep daily ETF data, net of realistic cost +
short-term tax, and prints CAGR / maxDD / Sharpe / after-tax vs SPY buy-and-hold.
Reads everything, writes nothing live (research firewall).

Run:  PYTHONPATH=. .venv/bin/python scripts/backtest_daily_sleeves.py
"""
from __future__ import annotations

from backtest.daily import run_daily, run_param_grid


class BuyHold:
    """Trivial benchmark: hold one symbol at 100%."""
    def __init__(self, sym: str = "SPY"):
        self.sym = sym
    def target_weights(self, asof_date, history) -> dict:
        return {self.sym: 1.0}


def _get(s: dict, *keys):
    for k in keys:
        if k in s and s[k] is not None:
            return s[k]
    return None


def _cagr(s: dict):
    return _get(s, "CAGR", "cagr")


def _fmt(s: dict) -> str:
    def pct(*keys):
        v = _get(s, *keys)
        return "n/a" if v is None or v != v else f"{v*100:+.2f}%"
    def num(*keys):
        v = _get(s, *keys)
        return "n/a" if v is None or v != v else f"{v:.2f}"
    return (f"CAGR {pct('CAGR','cagr')}  maxDD {pct('max_drawdown')}  Sharpe {num('sharpe')}  "
            f"vol {pct('ann_vol')}  afterTaxCAGR {pct('after_tax_CAGR','after_tax_cagr')}  "
            f"turn/yr {num('turnover_per_year')}")


def _row(label: str, result, bench_cagr=None) -> None:
    s = result.summary() if hasattr(result, "summary") else result
    extra = ""
    c = _cagr(s)
    if bench_cagr is not None and c is not None and c == c:
        extra = f"   vsSPY {(c-bench_cagr)*100:+.2f}%/yr"
    print(f"  {label:<34s} {_fmt(s)}{extra}")


def main() -> int:
    from strategies.momentum_rotation.strategy import MomentumRotationStrategy, load_params as mr_load
    from strategies.swing_meanrev.strategy import SwingMeanRevStrategy

    SEP = "=" * 100
    print(SEP)
    print("DAILY SLEEVES — realistic cost (2bps) + short-term tax, monthly rebalance, vs SPY buy&hold")
    print(SEP)

    # ---- window A: deep (incl. 2008), levers OFF ----
    A0, A1 = "2006-01-01", "2026-06-26"
    spyA = run_daily(BuyHold("SPY"), ["SPY"], start=A0, end=A1)
    spyA_cagr = spyA.summary().get("cagr")
    mr = MomentumRotationStrategy()
    uniA = sorted(set(mr.extra_symbols()) | {"SPY"})
    print(f"\n[A] {A0} .. {A1}  (deep, incl. 2008/2020/2022; levers OFF)   universe={len(uniA)} ETFs")
    _row("SPY buy&hold (benchmark)", spyA)
    _row("momentum_rotation (default)", run_daily(mr, uniA, start=A0, end=A1), spyA_cagr)

    # swing mean-reversion (use its own universe if it exposes one, else a sane set)
    sw = SwingMeanRevStrategy()
    sw_uni = sorted(set(getattr(sw, "extra_symbols", lambda: [])() or
                        ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "BIL"]))
    try:
        _row("swing_meanrev (default)", run_daily(sw, sw_uni, start=A0, end=A1), spyA_cagr)
    except Exception as e:
        print(f"  swing_meanrev: ERROR {type(e).__name__}: {str(e)[:120]}")

    # ---- window B: 2011+ (all GEM/lever ETFs alive), levers OFF vs ON ----
    B0, B1 = "2011-01-01", "2026-06-26"
    spyB = run_daily(BuyHold("SPY"), ["SPY"], start=B0, end=B1)
    spyB_cagr = spyB.summary().get("cagr")
    mrB = MomentumRotationStrategy()
    p_on = mr_load("DEFAULT"); p_on["offensive_risk_off"] = True; p_on["leveraged_long"] = True
    mrB_on = MomentumRotationStrategy(params=p_on)
    uniB = sorted(set(mrB.extra_symbols()) | set(mrB_on.extra_symbols()) | {"SPY"})
    print(f"\n[B] {B0} .. {B1}  (modern; growth levers OFF vs ON)   universe={len(uniB)} ETFs")
    _row("SPY buy&hold (benchmark)", spyB)
    _row("momentum_rotation (levers OFF)", run_daily(mrB, uniB, start=B0, end=B1), spyB_cagr)
    _row("momentum_rotation (levers ON)", run_daily(mrB_on, uniB, start=B0, end=B1), spyB_cagr)

    # ---- robustness: lookback grid (GEM-fragility guard) ----
    print(f"\n[C] robustness — momentum_rotation 12m-lookback grid (CAGR/Sharpe/maxDD dispersion), {A0}..{A1}")
    try:
        grid = run_param_grid(
            lambda v: MomentumRotationStrategy(params={**mr_load("DEFAULT"), "lookback_12m": v}),
            "lookback_12m", [189, 210, 231, 252, 273],
            universe=uniA, start=A0, end=A1,
        )
        print(f"  {grid}")
    except Exception as e:
        print(f"  run_param_grid: ERROR {type(e).__name__}: {str(e)[:160]}")

    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
