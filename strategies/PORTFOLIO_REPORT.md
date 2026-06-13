# PORTFOLIO_REPORT — Edge-Portfolio & Maximization Synthesis (P2 Stage 2)

**Symbol/timeframe:** QQQ 5m · **Cost profile:** `realistic` · Generated from `backtest/portfolio.py`.
The maximization thesis (MASTER_PLAN §0/§1.B): geometric growth `g ~= mean - 1/2*var`; stacking **uncorrelated** edges cuts blend variance -> raises g (and, operationally, lets you size larger at the same drawdown -> compound faster).

This synthesis tests it honestly on the three-strategy edge set:
`breakout_retest/v0_atr_stop`, `level_meanrev/DEFAULT`, `momentum_thrust/DEFAULT`.

---

## 1. Live correlation matrix (in-sample 392 sessions; full-period in parens)

|                | breakout (v0_atr_stop) | level_meanrev | momentum_thrust |
|---|---:|---:|---:|
| **breakout**       | +1.000 | -0.140 (-0.135) | +0.005 (-0.016) |
| **level_meanrev**  | -0.140 | +1.000 | -0.066 (-0.036) |
| **momentum_thrust**| +0.005 | -0.066 | +1.000 |

**The decorrelation is real and by construction.** Every off-diagonal is near zero or negative. level_meanrev fades the same levels breakout rides (-0.14); momentum_thrust is an independent trend-expansion stream (~0). Heatmap: `backtest/reports/correlation_heatmap.png`.

---

## 2. Blend weights

- **Equal-weight:** 0.333 / 0.333 / 0.333.
- **Min-variance (long-only, sum=1):** breakout **0.614** / level_meanrev **0.306** / momentum_thrust **0.080** (in-sample); full-period **0.576 / 0.320 / 0.103**. momentum_thrust's high single-sleeve variance pushes its weight down; the low-variance, decorrelated breakout anchor gets the most.

---

## 3. g-table (in-sample, g = mean - 1/2*var)

| strategy / blend | n | mean (bps) | var (1e-4) | Sharpe | g (bps) |
|---|---:|---:|---:|---:|---:|
| breakout_retest/v0_atr_stop | 392 | 1.786 | 0.0869 | 0.96 | 1.743 |
| level_meanrev/DEFAULT | 392 | 0.800 | 0.2000 | 0.28 | 0.700 |
| momentum_thrust/DEFAULT | 392 | 2.559 | 0.6809 | 0.49 | 2.218 |
| **BLEND_equal** | 392 | 1.715 | 0.0983 | 0.87 | 1.666 |
| **BLEND_minvar** | 392 | 1.546 | **0.0479** | **1.12** | 1.522 |

Chart: `backtest/reports/blend_vs_singles_g.png`. Blend tear sheet: `backtest/reports/tearsheet_blend_minvar.png`.

---

## 4. The maximization verdict — honest, two-part

### (a) Raw g: the blend does NOT beat the best single.
Best single g = **2.218 bps** (momentum_thrust); best blend g = **1.666 bps** (equal-weight). Uplift **-0.55 bps**. The blend loses on the literal proxy because at these tiny daily magnitudes the `1/2*var` penalty is negligible (best single's `1/2*var` is only ~0.34 bps), so a high-mean / high-variance single wins the unscaled number. **We do not dress this up: on the literal g, the single dominates.**

### (b) The variance cut and the scale-invariant mechanism: the thesis HOLDS.
- Blend **variance < best single's** (0.048e-4 minvar vs 0.087e-4 breakout vs 0.681e-4 momentum). **TRUE.**
- The min-variance blend has the **highest Sharpe of any sleeve (1.12 vs 0.96 / 0.28 / 0.49).**
- **Vol-targeted g** — scaling every sleeve to a common daily risk budget (sd ~29.5 bps), which is what the risk engine actually does (fixed $-risk per trade) — the min-variance blend wins: **g = +2.04 bps vs best single +1.74 bps (+0.30 bps uplift).**

**Why the two readings disagree:** raw g rewards a high-mean stream regardless of how much risk it runs; the risk engine never lets a sleeve run at uncontrolled size, so the operationally honest comparison is at a fixed risk budget — where higher Sharpe == higher growth. The decorrelated blend delivers the highest Sharpe, so under the constraint that actually binds (a fixed risk of ruin / drawdown band), **the blend compounds fastest.** That is the maximization thesis in the form that matters.

### Bottom line
The structural maximization benefit is **real** (lower variance, higher Sharpe, higher vol-targeted g from genuine decorrelation), even though every individual edge is marginal-to-weak net of realistic costs and **none clears the multiple-testing haircut** (PF >= 1.696 at n_trials=14). Nothing is promoted to LIVE. The portfolio's job here is proven-out as a variance machine; it now needs stronger underlying edges (a consolidated/SIP data re-pull, or new edges off the research conveyor) to become live-worthy.

---

## 5. Rigorous-stats appendix (realistic costs, full period)

IS = 2024-06-13 -> 2026-01-17; OOS = **locked vault** 2026-01-18 -> 2026-06-12 (never tuned on).

| variant | n | PF [95% CI] | expR | IS PF | OOS PF (n) | WF PF |
|---|---:|---|---:|---:|---:|---:|
| breakout_retest/v0_atr_stop | 188 | 1.318 [0.95, 1.84] | +0.05 | 1.269 | 1.436 (50) | 1.261 |
| level_meanrev/DEFAULT | 707 | 1.021 [0.86, 1.21] | +0.03 | 1.036 | 0.977 (155) | 1.125 |
| momentum_thrust/DEFAULT | 498 | 1.155 [0.83, 1.59] | +0.01 | 1.134 | 1.247 (101) | 1.310 |

All 14 evaluated variants (survivors and failures) are in `backtest/stats/hypothesis_log.jsonl`. Multiple-testing: `min_pf_threshold(14) = 1.696`; **survivors clearing the haircut: NONE.**
