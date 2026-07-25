# Research probe: long-call overlay on SPY / QQQ — "tune to ≥3:1 RtoR"

**Status:** RESEARCH ONLY — NOT PROMOTED, not paper-eligible. Fails multiple promotion gates (see verdict).
**Date:** 2026-07-07
**Author:** strategy-researcher (operator-requested)

## Request
Backtest a call-options strategy on SPY & QQQ; configure parameters until reward-to-risk (RtoR = avg win / avg loss) ≥ 3:1.

## Result in one line
**Target trivially met** (hundreds of configs > 3:1; best full-sample RtoR ≈ 8.8, OOS ≈ 7.9) — **but the metric is an artifact and the strategy fails validation.** A 3:1 RtoR is a *mechanical property of buying OTM calls* (capped downside, unbounded upside), not evidence of edge.

## Method & honest limitations
- **No historical options data** (Polygon free tier = EOD equity only). Option P&L is **modeled via Black-Scholes**, IV = realized-vol(20d) × 1.15 + 0.02 (regime-responsive: you pay up for premium in high vol). Costs = 3% half-spread on entry & exit. **These are modeled fills, not real ones.**
- **Only ~2 years of data available** (2024-07-08 → 2026-07-02, 499 bars/ticker). Free tier caps history at 2y.
- Signal (convex momentum): long only, close > 200d SMA **and** close > N-day high (breakout); exit on profit-target / stop / time.
- Split: in-sample = first 60% of each ticker, out-of-sample = last 40%.

## Recommended "least-bad" config (if forced to name one)
`breakout=40d, moneyness=1.10 (OTM), DTE=60, profit_target=+500%, stop=-75%, max_hold=20d`

| Window | n | RtoR | PF | Win% | Exp/trade | Best | Worst |
|---|---|---|---|---|---|---|---|
| In-sample | 11 | 11.7 | 1.17 | 9% | +10% | +727% | −83% |
| Out-of-sample | 14 | 7.9 | 4.40 | 36% | +152% | +1303% | −91% |
| Full | 25 | 8.8 | 2.79 | 24% | +89% | +1303% | −91% |

Benchmark over the same window: **buy&hold SPY +36%, QQQ +52%.**

## Why this is NOT a validated edge (the verdict)
1. **RtoR is gameable by construction.** Long OTM calls lose ≤100% and can win 500%+, so avg-win/avg-loss is high *by design*. Win rate is only **9–36%** — the high RtoR exists *because you rarely win*. RtoR alone says nothing about making money.
2. **Sample far too small.** n = 25 total (11 IS / 14 OOS) vs the gate's **≥100 trades**. The +89% avg is carried by **1–2 monster winners** (+727%, +1303%).
3. **No regime diversity.** The 2-year window was one of the strongest bull runs on record (QQQ +52%). A breakout-call strategy is *long convexity into a trending market* — of course it prints. **Zero bear-market OOS.** First real drawdown regime likely flips expectancy negative.
4. **Multiple-testing haircut is fatal.** ~**3,240 configs** searched. By chance alone ~162 clear p<0.05; Bonferroni alpha for one real finding is **1.5e-5**. A PF≈1.2 on n≈12 is nowhere near surviving that.
5. **Modeled, not real, fills.** Real IV crush around events, weekend theta, and wider spreads on OTM strikes would erode the modeled result.
6. **Doesn't beat buy&hold** on a risk-adjusted basis once you account for the −91% single-trade drawdowns and the concentration in a couple of winners.

## What would make this answerable (not done here)
- Real historical options chains (paid data) for true fills/IV.
- ≥10 years spanning multiple regimes (2018 Q4, 2020, 2022 bear).
- Pre-registered single config (no grid), walk-forward with anchored folds, deflated Sharpe.
- Capital gate: at ~$1k, a single SPY/QQQ contract can't be sized to ≤1% risk anyway (see limits.yaml RI-5 `defined_risk_small`); this is moot below ~$5–10k.

## Recommendation
Do **not** promote. If the thesis is interesting, carry as PAPER-only and revisit only with real options data + multi-regime history. Treat any "3:1 RtoR" claim on OTM calls as null until win-rate/PF/expectancy clear the gate on ≥100 trades out-of-sample.
