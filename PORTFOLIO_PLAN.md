# TradeForge — Edge Portfolio Plan (v1, 2026-06-29)

> Companion to [MASTER_PLAN.md](MASTER_PLAN.md) (architecture) and [CLAUDE.md](CLAUDE.md) (operating
> contract). This document answers: **which edges are actually applicable to a $500→$100k Robinhood
> DCA account, and how do they compose into a portfolio that compounds while surviving costs?**
>
> It exists because the original intraday `breakout_retest` family was tested three independent ways
> (data quality, regime-gating, breadth) and has **no validated, generalizable, cost-surviving edge**.
> The conclusion that survived: at this size the binding constraint is **cost drag + variance**, so the
> winning design is a **stack of 3–5 low-correlation, low-turnover, cost-surviving sleeves** — not one
> high-frequency edge. Each claim below was evidence-checked (sources noted inline).

## 0. Thesis — a growth engine with a catastrophe floor

**The goal is to GROW** — a growth engine with a catastrophe floor, not survival for its own sake.
Geometric growth ≈ **mean − variance/2**, so cutting variance isn't timidity — it's what lets us **size
up aggressively at the same drawdown**. We grow by **leaning hard into validated edges in favorable
regimes** and **trading directionally both ways** (profit from downtrends, not just dodge them), while
the always-on aborts make blow-up structurally impossible.

Under ~$10k the **weekly deposit still dominates** every trading edge, so don't bleed it to costs — but
the framing is **growth-engine-with-a-floor, not ballast-first**. We stack low-correlation edges (the
variance cut is precisely what *permits* bigger size), trade the risk-OFF leg **offensively** via inverse
ETFs, lean the exposure scalar >1 in favorable regimes, and ramp risk as paper proves out — admitting
every sleeve to capital only on the **measured** numbers, never assumed ones. The catastrophe floor
(−20%/−35% aborts, reconciliation halt, dead-man's switch) stays non-negotiable; that floor is what makes
aggression *safe*, not optional.

## 0.2 First validation results (2026-06-29) — the taxable-account crux

The daily data + backtester + the momentum/swing sleeves are **built and validated** (38 ETFs 1996→2026;
`scripts/backtest_daily_sleeves.py`). Real numbers, net of cost + short-term tax, monthly rebalance:

| 2006–2026 (incl. 2008) | CAGR | **after-tax CAGR** | maxDD | Sharpe | turn/yr |
|---|---|---|---|---|---|
| **SPY buy & hold** | 10.96% | **10.96%** | 55.2% | 0.64 | 0.05 |
| momentum_rotation (default) | 8.04% | **5.95%** | **22.2%** | 0.70 | 5.25 |
| momentum (GEM-only, low-turn) | 8.27% | **6.60%** | 22.7% | 0.73 | 3.61 |
| swing_meanrev | 3.96% | 1.92% | 32.7% | 0.46 | 5.65 |

**What's confirmed:** the trend filter **robustly halves drawdown** (22% vs 55%) at higher Sharpe and lower
vol, and is **robust across lookbacks** (CAGR cv ~6% — not GEM-fragile). The variance-cut thesis is real.

**The hard finding:** in a **taxable** account, after-tax CAGR **badly trails buy-and-hold** (5.95% vs
10.96%) because turnover realizes short-term gains while buy-and-hold defers all tax. Lower turnover
(GEM-only, drop sector RS) narrows it (6.60%) but ~4.4%/yr behind remains. The **growth levers detract**
in the modern bull (Sharpe 0.80→0.71) — keep OFF / strict-gate only. Sector-RS rotation adds turnover with
~no CAGR benefit — **drop it**.

**Refinement to the plan (folds into §2/§3):** for a taxable DCA accumulator the realistic **CORE is
low-turnover buy-and-hold (VTI/SPY, tax-deferred)** with the 200d trend filter used as **rare drawdown
insurance** (de-risk only in deep, confirmed downtrends — minimal flips), NOT a frequent rotation.
Momentum-rotation and swing become **small decorrelation satellites**, admitted on after-tax + measured
correlation, never standalone return. Tax-advantaged space (a Roth IRA sleeve) is where the
higher-turnover rotation/VRP edges actually keep their edge — worth weighing.

## 0.3 Active bracketed swing — the growth engine (the answer to "don't just hold")

Holding gives up the active capture of uptrends. So we built the **actively-managed daily book** you
asked for: each day decide hold/buy/sell across a liquid universe, hold **multiple fractional positions**,
and **every position carries a stop-loss + a profit-taking exit** (partial-TP → breakeven → trailing
runner, or a hard take-profit). Engine: `backtest/daily/bracket_engine.py`; strategy:
`strategies/swing_breakout/` (Donchian N-day-high breakout, trend-gated, momentum-ranked); driver:
`scripts/backtest_swing.py`. Verified by running it directly (2006–2026, 46 names, cost 2bps + 30% ST-tax):

| variant | CAGR | after-tax | maxDD | Sharpe | win% | avgR | tr/yr |
|---|---|---|---|---|---|---|---|
| SPY buy & hold | 10.96% | 10.96%* | 55.2% | 0.64 | — | — | 0.05 |
| **HARD_TARGET** (capped sell, low-turn) | **15.08%** | **10.35%** | 33.8% | 1.03 | 32.9 | +0.63 | 29 |
| FAST_DONCHIAN (20d) | 14.28% | 7.21% | 30.6% | **1.14** | 50.8 | +0.35 | 49 |
| DEFAULT (partial+trail) | 12.92% | 6.23% | 23.5% | 1.06 | 51.7 | +0.35 | 49 |

\*buy & hold defers (doesn't avoid) tax — an over-generous bar.

**You were right: active beats holding.** Pre-tax the breakout beats SPY by **+4.1%/yr** (15.1% vs 11.0%)
with **~half the drawdown** and **~1.7× the Sharpe**. The one headwind is **short-term tax on turnover** —
after-tax in a taxable account it ~ties an (untaxed) buy & hold (HARD_TARGET −0.61%/yr), and lower-turnover
variants keep the most.

**The aggressive-growth unlocks (this is how we press):**
1. **Size up the low-vol curve.** Half the drawdown + far higher Sharpe means at the same drawdown budget
   you can run it **bigger** (RI 6→8, or a trend-gated leveraged tilt) → beat buy & hold's *return* at
   equal risk. Variance reduction isn't defense here — it's the license to be aggressive.
2. **Roth IRA sleeve.** The tax drag is the whole after-tax gap. Run the high-turnover active engine in
   **tax-advantaged space** and the full **+15% pre-tax** edge compounds untaxed — the cleanest growth machine.
3. **Lower-turnover bracket** (HARD_TARGET, 29 trades/yr) for the taxable sleeve — banks the gain, fewer
   taxable events, keeps +10.35% after-tax with half SPY's drawdown.

**Status: RESEARCH.** The 20-yr in-sample result across 46 names is strong but must still clear OOS +
walk-forward + the multiple-testing haircut before paper→live. This is now the **lead growth sleeve**;
momentum-rotation / factor-core become the lower-octane ballast, swing-meanrev the decorrelating fade.

## 1. The edge menu (what's real, what fits)

| Family | Verdict | Evidence grade | Role | Turns on |
|---|---|---|---|---|
| **Momentum / Dual-Momentum ETF rotation** (GEM-lite + sector RS + trend gate) | **CORE** | Trend leg ROBUST; relative-strength leg DECAYED | Foundational growth+crash-filter anchor | **$500** |
| **Factor / Beta core + trend overlay** (VTI/SPY + QUAL/USMV/MTUM, 200d/12m gate) | **CORE** | Beta ROBUST (risk premium); tilts decayed | Ballast / DCA sink / don't-die base | **$500** |
| **Multi-asset trend-following / crisis alpha** (ETF TSMOM ensemble or DBMF) | SATELLITE | ROBUST but ETF-lossy | Variance-reducer, tail-cutter (crisis years) | $5k |
| **Swing mean-reversion** (RSI(2)/%-below-MA, multi-day, 200d-gated) + overnight-as-a-rule | SATELLITE | RSI(2) ROBUST; seasonality folklore | Buys weakness → decorrelates the trend cores | $5k |
| **Crypto trend-following** (BTC/ETH slow TSMOM; via IBIT/ETHA first) | SATELLITE | ROBUST (survived 2022) | Uncorrelated-*when-it-matters* diversifier | $5k (ETF route) |
| **Options VRP / income** (defined-risk put-credit spreads → wheel/overwrite) | DEFER→core-income | VRP ROBUST in existence, decayed in magnitude | Capital-efficient income, low-corr to trend | $25k |
| **Event / earnings options** (IV-crush, defined-risk) | DEFER | Thin / negative-skew | Tiny diversifier only | $25k |
| **Pairs / stat-arb** | DEFER (mostly reject) | DECAYED + crowded; needs short/borrow | Not viable retail-small; only long-only ETF-ratio rotation is near-term | $25k+ |

**Key evidence anchors:** trend-following positive in every decade since 1880, net Sharpe ~0.76, near-zero
stock/bond correlation (AQR, *A Century of Evidence*); the 200-day filter historically ~halves max
drawdown; cross-sectional momentum decayed ~10%/yr→~2%/yr and **crashes** in post-decline rebounds
(Daniel–Moskowitz, NBER w20439); GEM is parameter-fragile (9- vs 10-month lookback diverged ~2000bps —
ThinkNewfound); RSI(2) oversold-bounce remains effective on index ETFs through 2025; crypto TSMOM Sharpe
~1.3–1.9 surviving the −65% 2022 bear; VRP is an *insurance* premium (CBOE PUT/BXM ≈ S&P return at lower
vol) so it persists but is thin and negatively skewed.

## 2. Portfolio sleeves (the stack)

1. **Momentum / Rotation Core** — most of the directional beta; sector RS for the return tilt, an
   absolute-momentum/200d gate for the crash filter. Job: market-like return at ~30–50% lower maxDD.
2. **Factor / Beta Ballast** — cheapest equity-risk-premium capture and the literal DCA deposit sink;
   high correlation to the book *on purpose* (it's ballast, not a diversifier — deflate it as real
   decorrelators come online).
3. **Trend / Crisis-Alpha satellite** — near-zero/negative correlation to equities *in sustained
   drawdowns*; held for the 1-in-5 crisis year, expected to lag SPY in calm bulls.
4. **Swing Mean-Reversion satellite** — buys panic dips (the cores buy strength) → strong negative
   correlation to them; the 200d gate + small weight are mandatory (correlation flips positive in
   vol-shocks).
5. **Crypto-Trend satellite** — diversification lives in the **trend-OFF filter** (raw BTC *amplifies*
   equity drawdowns, corr has hit 0.87); default to IBIT/ETHA in the equity MCP to dodge 35–85bps crypto
   spreads until $25k justifies the native channel.
6. **Options VRP Income** (≥$25k) — harvests implied>realized vol via **defined-risk** spreads sized so a
   simultaneous-max-loss crash day stays inside the −20%/−35% abort lines.

**Instrument note (research items, validate before any live use):** the **Momentum/Rotation Core** and
**Trend/Crisis-Alpha** sleeves' **risk-OFF destination = { cash/bonds (defensive) OR −1x inverse ETF
(offensive) }**, chosen by signal strength — on a cash account you can't short, so a −1x inverse
(PSQ/SH/RWM) is the only way to *profit* from a downtrend instead of sitting in cash. The **risk-ON** leg
may use a leveraged-long tilt (QLD/TQQQ) **under the trend gate**. Both legs are research items that must
clear the normal backtest→paper→haircut gate (the inverse leg specifically must beat going-to-bonds *net
of* the inverse ETF's daily-rebalance decay, cost, and short-term tax).

## 2.1 Aggression levers (grow, not just survive)

These are how the floor-protected book actually *presses*. Each is a research item gated like everything
else — they earn capital only on the measured numbers.

- **Directional risk-OFF (inverse ETFs).** When the downtrend signal is strong/confirmed, the trend &
  rotation risk-OFF leg rotates into a **−1x inverse ETF (PSQ/SH/RWM)** instead of cash/bonds → make money
  when the market falls. *Validate it beats going-to-bonds net of inverse-ETF daily-rebalance decay + cost
  + short-term tax; deploy only on strong, confirmed signals.*
- **Trend-gated leverage.** In confirmed strong uptrends (price > 200d **and** momentum strong), allow a
  **leveraged-long** tilt (QLD/TQQQ); in confirmed downtrends, a **short, fast-exit leveraged-inverse**
  (QID/SQQQ). ⚠️ Leveraged ETFs are built for ~1-day holds and **decay in chop** — only ever **trend-gated
  with fast exits, never buy-and-hold**, sized for −50%+ drawdowns. *Validate before any live use.*
- **Lean-in sizing.** Let the regime **exposure scalar exceed 1.0** (toward the band cap) in favorable
  regimes; conviction tiering A+→RI 8 so the best regimes get the most capital.
- **RI posture — start hot, ramp faster.** Open at **RI 6** (not the RI-5 floor) and step toward **8
  faster** as paper proves out — still **manual, still gated**, never auto-tuned from P&L, never widened
  into a drawdown. (Requires the `risk_index.default: 6` hand-edit in `limits.yaml`; the RI-5 floor stays
  as the catastrophe reference.)
- **Inverse ETFs as *signals*: skipped** — inverse-ETF strength is ~redundant with the price trend we
  already compute (SQQQ ripping ⇔ QQQ falling, same information). Use them as **instruments, not a
  separate signal.**

> **Honest flag:** the inverse/leveraged legs raise **turnover, tax drag, and tail risk**, and
> leveraged-ETF **decay is real** — so they go through the same gate as everything else and earn capital
> only on the measured numbers. With that discipline they're exactly the right "grow" levers for a cash
> account; without it they're gambling.

## 3. Capital-tier ladder

| Tier | Objective | Active sleeves (≈ risk weight) |
|---|---|---|
| **$500** | Don't bleed the deposit; capture beta + a crash filter | Combined **Core** (broad ETF + 200d/12m trend overlay) ~100%. Optional tiny BTC-trend toehold via IBIT. Single-position GEM-lite/SPY-trend (no diversification yet). |
| **$5k** | Build the first real low-correlation stack | Rotation Core 45–55% · Factor Ballast 25–35% · Trend 10–15% · Swing MR 5–10% · (crypto toehold small). T+1 cash account fine at monthly/weekly cadence. |
| **$25k** | Edge starts to matter; PDT worry gone | Rotation 30–40% · Beta 15–25% · Trend ~15% · Swing 10–15% · **Options VRP 5–15%** · Crypto 5–10% (native channel optional). |
| **$100k** | Trading edge drives growth, not deposits | Rotation 25–35% · Beta 10–20% · Trend 15–20% · Swing 10–15% · **Options VRP 15–20%** (wheel+overwrite+condors) · Crypto 10–20%. |

(Weights are targets; the live engine normalizes via the existing min-variance blend on the **measured**
correlation matrix, not these nominal numbers.)

## 4. DCA, reinvestment & protection mechanics

- **DCA rule:** route 100% of each weekly $50–$500 into the **most-underweight** sleeve (cash-funded
  rebalancing) — at $500–$5k that's almost always the Factor/Beta core (VTI/SPLG fractional). This
  rebalances **without selling**, avoiding realized short-term gains in the taxable account. Only sell to
  rebalance when a sleeve drifts >~10pp off target and new cash can't close it within a month.
- **Reinvest:** automatic — per-trade $ risk = (per_trade_pct/100) × **current** equity, recomputed every
  trade ([risk/limits.yaml](risk/limits.yaml)), so profits + deposits both auto-scale the next position.
- **Tax:** keep a tax-reserve ledger line (short-term/ordinary rate); judge everything on **after-tax**
  equity. Prefer 12-month absolute momentum over the 200-SMA to cut taxable flips.
- **Milestones & ratchet (growth-first):** ladder = **$1k · $2.5k · $5k · $10k · $25k · $50k · $100k**.
  **No vault sweep below $10k** — sub-$10k milestones are *growth checkpoints* (mark the win, gate an RI
  step-up review) with **0% swept**, so every early dollar compounds (sweeping 25% into an untouchable
  vault at $500→$5k would shrink the base exactly when compounding matters most). From **$10k+**, sweep
  **25% of gains** above the last baseline into the **vault sleeve** (BIL/SGOV, untouchable by the active
  engine); press with 75%. The 25% protection kicks in only once the base is big enough that giveback
  hurts more than compounding helps.
- **Risk dial — start hot, ramp faster:** open at **RI 6** (not the RI-5 floor) and step toward 8
  **faster** as paper proves out — still **manual, still gated**, never auto-tuned from P&L, never widened
  into a drawdown. Conviction tiering B→5/A→6–7/A+→8; **lean the exposure scalar >1** toward the band cap
  in favorable regimes. The RI-5 floor remains as the catastrophe reference.
- **Always-on aborts:** −20%/month → review; −35%/peak → halt+manual restart; reconciliation halt;
  dead-man's switch. **Cost-viability gate:** take a trade only if expected edge ≥ 2× modeled round-trip
  cost — else raise selectivity, never widen risk.

## 5. Validation order (highest ROI / easiest-to-validate-honestly first)

1. **Absolute-momentum trend filter + GEM-lite** — most decay-resistant, cheapest data (daily
   dividend-adjusted ETF bars only), and honesty is enforceable via a **lookback grid** (robustness, not
   peak backtest). Unlocks the cost-surviving RH-native core fastest.
2. **RSI(2)/%-below-MA swing** — same daily-ETF data; validating it second **proves the decorrelation
   thesis** (expected strong negative correlation to the trend core).
3. **Sector relative-strength rotation** — more researcher degrees of freedom → heavier multiple-testing
   haircut; promote only after a lookback-grid survives.
4. **Defer** options VRP and crypto-native until their data pipelines exist (IV history needs months of
   self-snapshotting; crypto needs 5yr+ covering a bear). **Do not** prioritize stat-arb/single-name
   earnings — data-mining minefields with thin, cost-fragile edges.
- Across all: lock a true OOS vault, count every variant as a trial, apply the haircut (PF ≥ ~1.7 at
  n_trials≈14), and force a **pre-2010 vs post-2010 regime split** to measure decay honestly.

## 6. What to build first (the concrete near-term engineering)

The whole core keys off **daily dividend-adjusted ETF data**, which the platform does **not** have today
(DuckDB holds only 5m bars for ~12 names). In order:

1. **Daily adjusted ETF ingest** — extend [data/pipelines/polygon_ingest.py](data/pipelines/polygon_ingest.py)
   to a `1d` timeframe and ingest ~15 ETFs (VTI/SPY/SPLG, 11 XL* sectors, QQQ, VEU/VXUS, AGG/BND,
   BIL/SGOV/SHY, QUAL/USMV/MTUM), **15–30yr** history (supplement Polygon's lookback cap with Stooq/Tiingo),
   with the `adjusted` column **actually populated** (split+dividend). *Validate the adjustment against a
   known total-return index before trusting any backtest.*
2. **Daily NAV backtest path** — the engine is intraday/R-multiple oriented; add a daily
   equity-curve backtester measuring CAGR, maxDD, Sharpe, **and after-tax return**, reusing the existing
   `daily_returns`/correlation utilities + [backtest/portfolio.py](backtest/portfolio.py) min-variance blend.
3. **Two strategy modules** under `strategies/`: `momentum_rotation` (blended 3/6/12m ranking + 200d/12m
   gate + top-N + vol-scaling) and `swing_meanrev` (RSI(2)/%-below-MA + 200d gate), each with `params.yaml`
   + a `registry.yaml` RESEARCH entry wired into the live correlation matrix.
4. **Regime-reader extension** — [orchestrator/agents/regime_reader.py](orchestrator/agents/regime_reader.py)
   already emits a trend tag + exposure scalar; extend it to emit the monthly arm/disarm signal per sleeve
   + the vol-scaling scalar (LLM stays out of the hot path; the deterministic loop stages monthly orders
   for the human-confirmed equity MCP).
5. **Tax-reserve / after-tax-equity wiring** so compounding uses after-tax dollars.

This single build validates the **first two sleeves** (Rotation Core + Swing MR) and needs **no
options/IV or fundamentals pipeline** — deliberately the cheapest path to a live, cost-surviving,
decorrelated core.

## 7. Top risks (and mitigations)

- **Fake diversification** — Beta, Rotation, and the long leg of Swing MR are all long-equity; they
  correlate→1 in a vol-shock. *Mitigate:* admit capital only on the **measured** correlation matrix,
  deflate Beta as true decorrelators (trend, VRP, crypto-trend-OFF) come online, stress-test the
  all-correlate case.
- **Bad dividend adjustment fabricates momentum** — the entire core is price-signal-based. *Validate the
  `adjusted` column first.*
- **Short-sample / regime-blind validation** — 2yr of data misses 2008/2020/2022. *Acquire 15–30yr daily
  history; force pre/post-2010 split; lock OOS.*
- **Tax drag** in the taxable wrapper from every flip/rotation/roll. *Favor 12m abs-momentum, rebalance
  with new cash, judge after-tax.*
- **Parameter fragility / overfitting** (GEM ±2000bps on lookback; options grids). *Gate on robustness
  across a grid, pre-register rules, apply the haircut.*
- **Options short-vol tail** correlates across positions in a crash; **crypto decorrelation is
  conditional** on the trend-OFF filter. *Size for simultaneous max loss; validate the filter before
  sizing crypto; default to the ETF route early.*
- **Leverage / inverse decay & tax drag** — leveraged & inverse ETFs lose value to daily-rebalance
  **volatility decay** over the multi-week holds a trend filter produces (the right direction can still
  bleed in chop), and the extra turnover realizes short-term gains. *Strictly trend-gate with fast exits,
  never buy-and-hold; require the inverse leg to beat bonds AND the leveraged leg to beat 1× net of
  decay+tax in the backtest; size for −50%+ drawdowns; admit only on the measured after-tax numbers.*

## 8. Bottom line

Build the **daily-ETF data + daily backtester + momentum-rotation & swing-MR modules** (Section 6),
validate the **trend filter / GEM-lite first, RSI(2) second** (Section 5), and stand up a **2-sleeve
cost-surviving core** that the DCA deposits feed — then admit trend/crypto/options satellites on proven
low correlation as capital crosses $5k/$25k. The edge here is **the stack and the discipline**, not any
single strategy.
