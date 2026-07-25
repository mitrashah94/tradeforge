# level_meanrev — Validation Report (P2 Stage 2 integration)

**Symbol/timeframe:** QQQ 5m · **Period:** 2024-06-13 -> 2026-06-12 · **Cost profile:** `realistic`
Generated from `backtest/rigorous_stats.py`. The chop/range-regime FADE complement to breakout_retest.

---

## 1. Headline variant (DEFAULT)

| variant | n | PF | PF 95% CI | expectancy_R | IS PF | OOS PF (n) | walk-fwd PF | underpowered |
|---|---:|---:|---|---:|---:|---:|---:|:--:|
| **DEFAULT** (PDH/PDL/PMH/PML fade, 1R) | 707 | 1.021 | [0.862, 1.209] | +0.03 | 1.036 | 0.977 (155) | 1.125 | no |
| V0 (PDH/PDL only, 1.2R) | 412 | 1.048 | [0.830, 1.299] | +0.01 | 1.151 | 0.755 (91) | 1.141 | no |
| V1 (== DEFAULT) | 707 | 1.021 | [0.862, 1.209] | +0.03 | 1.036 | 0.977 (155) | 1.125 | no |
| V2 (session-mid target) | 470 | 0.693 | [0.570, 0.852] | -0.15 | 0.670 | 0.779 (112) | 0.737 | no |

Tear sheet: `backtest/reports/tearsheet_level_meanrev_DEFAULT.png`.

---

## 2. By-regime (full period; PF per regime)

| regime | trend | chop | vol_shock |
|---|---:|---:|---:|
| DEFAULT | 0.901 | **1.259** | 0.197 |

This is the whole point: level_meanrev earns its keep **in chop (PF 1.26)** — its design regime — and bleeds in trend (0.90) and vol_shock (0.20), exactly the mirror image of the breakout/momentum sleeves. It is a regime specialist, not an all-weather edge.

---

## 3. Multiple-testing haircut

Logged to `backtest/stats/hypothesis_log.jsonl`. With `n_trials = 14`, `min_pf_threshold = 1.696`. **DEFAULT (PF 1.021) does not clear the haircut** — not close. As a standalone aggregate edge it is barely above breakeven.

---

## 4. Correlation to the others (full period)

| | breakout | level_meanrev | momentum_thrust |
|---|---:|---:|---:|
| **level_meanrev** | **-0.135** | 1.000 | **-0.036** |

**Negatively correlated to both** other sleeves — the strongest decorrelation in the set. This is the value it adds: it is right (in chop) when the continuation edges are flat or wrong.

---

## 5. Verdict

**Keep RESEARCH.** Aggregate edge is marginal (PF ~1.02, expectancy +0.03R) and fails the haircut, and OOS slips to 0.977. But it is doing its job as a **decorrelated chop specialist** (chop PF 1.26, -0.135 correlation to breakout) and pulls the min-variance blend's variance down — it gets allocation **0.320** for that reason, not for a standalone PF. Promote only if a re-pull on better data lifts the aggregate edge, or if a regime-gated version (only arm in chop) clears its own gate.
