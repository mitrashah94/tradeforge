# breakout_retest — Validation Report (P2 Stage 2 integration)

**Symbol/timeframe:** QQQ 5m · **Period:** 2024-06-13 -> 2026-06-12 (~2y)
**Cost profile:** `realistic` (zero-commission equity + 1c QQQ half-spread + 1-tick adverse slippage) — the honest small-account picture, not the optimistic `tv_style` gate.
**Data caveat:** bars are Alpaca's **IEX** feed (thin), which clips high/low extremes vs a consolidated/SIP feed and is the binding constraint on the V0 reproduction (see `backtest/run_gate.py`). All numbers below are net of realistic costs.

Generated from `backtest/rigorous_stats.py` and `backtest/portfolio.py`. This report also carries the **top-level portfolio / maximization synthesis** (last section); the full version is `strategies/PORTFOLIO_REPORT.md`.

---

## 1. Headline variants

| variant | n | PF | PF 95% CI | expectancy_R | IS PF | OOS PF (n) | walk-fwd PF | underpowered |
|---|---:|---:|---|---:|---:|---:|---:|:--:|
| **V0** (baseline, 1-tick stop) | 188 | 0.961 | [0.617, 1.449] | **-0.76** | 0.675 | 1.929 (50) | 1.196 | no |
| **V3** (partial + runner) | 284 | 0.998 | [0.722, 1.387] | **-0.32** | 1.041 | 0.890 (60) | 0.973 | no |
| **v0_atr_stop** (ATR stop) | 188 | **1.318** | [0.951, 1.844] | **+0.05** | 1.269 | **1.436 (50)** | **1.261** | no |

Tear sheets: `backtest/reports/tearsheet_breakout_retest_V0.png`, `..._V3.png`, `..._v0_atr_stop.png`.

### What actually earned its place
- **v0_atr_stop is the survivor.** Replacing V0's role-reversal **1-tick** stop with a **0.25*ATR14** stop is the single most important fix. V0's tight stop is whipsawed by slippage (avg loss ~ 2.5R), dragging expectancy to -0.76R and PF to 0.96. The ATR stop lifts **expectancy_R from -0.76 -> +0.05**, **PF 0.961 -> 1.318**, and is the only breakout config positive net of realistic costs. It holds out-of-sample (OOS PF **1.436** on the locked vault) and on walk-forward (**1.261**).
- **V3's partial/runner skew is real.** The ablation ladder (`strategies/breakout_retest/ablation.py`) shows partial-at-1R + breakeven + trailing-runner lifts expectancy from **-0.76R (V0) -> -0.32R (V3)** — the positive-skew structure from MASTER_PLAN §1.B. But V3's PF is still **< 1** net of costs, so the skew improvement alone does not make V0's IEX signal set tradable.

---

## 2. By-regime (full period, daily returns; PF per regime)

| variant | trend | chop | vol_shock |
|---|---:|---:|---:|
| V0 | 1.648 | 0.869 | 0.152 |
| V3 | 1.346 | 0.958 | 0.614 |
| **v0_atr_stop** | **2.047** | **1.088** | **1.157** |

As designed (a continuation edge), breakout_retest earns in **trend** and bleeds in **chop**. The crucial difference: **v0_atr_stop is the only variant that stays > 1 in every regime**, including vol_shock — the wider stop survives the gappy days that destroy the tight-stop V0 (V0 vol_shock PF 0.15).

---

## 3. Multiple-testing haircut

14 variants were evaluated this pass (breakout V0-V4 + v0_atr_stop, plus the two complements' DEFAULT/V0/V1/V2). Every one is logged to `backtest/stats/hypothesis_log.jsonl` (survivors **and** failures — the §5/§6 discipline). With `n_trials = 14`, `min_pf_threshold = 1.696`.

- **No breakout variant clears 1.696.** v0_atr_stop's PF 1.318 is the best; its CI upper bound (1.844) reaches the bar, but the point estimate does not.
- **Honest verdict:** breakout_retest does **not** clear the haircut and is **not** live-worthy standalone on this data. Its value is (a) the v0_atr_stop expectancy fix and (b) being the lowest-variance, decorrelated anchor of the blend (§5).

---

## 4. Correlation to the complements (full period)

| | breakout (v0_atr_stop) | level_meanrev | momentum_thrust |
|---|---:|---:|---:|
| **breakout (v0_atr_stop)** | 1.000 | **-0.135** | **-0.016** |

Negatively correlated to level_meanrev (it fades the same levels breakout rides) and near-zero to momentum_thrust — the decorrelation-by-construction the edge portfolio needs. Heatmap: `backtest/reports/correlation_heatmap.png`.

---

## 5. Verdict

**Keep at PAPER.** v0_atr_stop is the config to carry forward (the registry headline): best realistic-cost breakout config (PF 1.318, expectancy +0.05R, OOS 1.436, all-regime PF > 1), but it does **not** clear the multiple-testing haircut, so PAPER not LIVE. Next step: re-pull QQQ 5m from a consolidated/SIP feed (Polygon) and re-run the gate — the IEX feed is the diagnosed ceiling.

---

## 6. Portfolio / maximization synthesis (top-level)

Full analysis: `strategies/PORTFOLIO_REPORT.md`. Headline:

- **Correlations are genuinely low** (-0.135, -0.016, -0.036) — complements decorrelated by design.
- **Min-variance blend** weights: breakout **0.576** / level_meanrev **0.320** / momentum_thrust **0.103**.
- On the **raw** proxy g = mean - 1/2*var, the high-mean/high-variance single (momentum_thrust) wins, because at these tiny daily magnitudes the 1/2*var penalty is negligible. **Honest: the blend does NOT beat the best single on raw g.**
- But the blend **does** cut variance below the best single's, has the **highest Sharpe** (1.12 vs best single 0.96), and **wins on vol-targeted g** (+0.30 bps) once every sleeve is levered to a common risk budget — the operational form of the maximization mechanism (size to a fixed risk -> compound faster). The structural benefit is real; the absolute edges are too weak to be live-worthy yet.

Charts: `backtest/reports/blend_vs_singles_g.png`, `.../tearsheet_blend_minvar.png`, `.../correlation_heatmap.png`.
