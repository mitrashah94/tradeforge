# momentum_thrust — Validation Report (P2 Stage 2 integration)

**Symbol/timeframe:** QQQ 5m · **Period:** 2024-06-13 -> 2026-06-12 · **Cost profile:** `realistic`
Generated from `backtest/rigorous_stats.py`. The trend/expansion-regime FOLLOW complement (with-trend thrust + chandelier trailing exit).

---

## 1. Headline variant (DEFAULT)

| variant | n | PF | PF 95% CI | expectancy_R | IS PF | OOS PF (n) | walk-fwd PF | underpowered |
|---|---:|---:|---|---:|---:|---:|---:|:--:|
| **DEFAULT** (2-bar thrust, range+vol, 1.5 ATR trail) | 498 | **1.155** | [0.826, 1.586] | +0.01 | 1.134 | **1.247 (101)** | **1.310** | no |
| V0 (range-only thrust) | 504 | 1.169 | [0.878, 1.585] | +0.01 | 1.209 | 1.025 (106) | 1.178 | no |
| V1 (3-bar thrust, tight trail) | 421 | 1.071 | [0.758, 1.491] | -0.00 | 1.068 | 1.086 (85) | 1.165 | no |
| V2 (trail after 1R + time stop) | 723 | 1.096 | [0.837, 1.422] | +0.01 | 1.105 | 1.060 (143) | 1.148 | no |

Tear sheet: `backtest/reports/tearsheet_momentum_thrust_DEFAULT.png`.

---

## 2. By-regime (full period; PF per regime)

| regime | trend | chop | vol_shock |
|---|---:|---:|---:|
| DEFAULT | 0.867 | 0.787 | **7.345** |

The trailing-runner structure harvests the **vol_shock / big-expansion days (PF 7.3)** the fade misses entirely — the positive-skew, fat-right-tail behavior from MASTER_PLAN §1.B. It is roughly breakeven in routine trend/chop (the high-variance cost of letting winners run), so its edge is concentrated in the explosive tail.

---

## 3. Multiple-testing haircut

Logged to `backtest/stats/hypothesis_log.jsonl`. With `n_trials = 14`, `min_pf_threshold = 1.696`. **DEFAULT (PF 1.155) does not clear the haircut**, though it has the best aggregate PF of the complements and the best OOS (1.247) and walk-forward (1.310) — the most encouraging stability of the three strategies.

---

## 4. Correlation to the others (full period)

| | breakout | level_meanrev | momentum_thrust |
|---|---:|---:|---:|
| **momentum_thrust** | **-0.016** | **-0.036** | 1.000 |

Near-zero correlation to both other sleeves — a clean, independent return stream.

---

## 5. Verdict

**Keep RESEARCH.** Best aggregate PF (1.155) and the steadiest OOS/walk-forward of the complements, with a genuine vol_shock specialty (PF 7.3) and near-zero correlation to the rest. It does **not** clear the multiple-testing haircut, so it stays RESEARCH. Its high single-sleeve variance (the trailing runner) keeps its min-variance weight small (**0.103**), but it contributes the explosive-day exposure the other two sleeves lack. Promote on a data re-pull that confirms the edge, or wire conviction/vol-targeted sizing to tame its variance.
