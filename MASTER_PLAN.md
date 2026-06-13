# TradeForge — Master Plan (v5)

Autonomous, self-improving systematic trading platform. Written to **maximize compounded growth** within a defined risk band, while deterministic code makes catastrophic loss structurally impossible.

> **Regulatory note (current as of June 2026):** The SEC approved amendments to FINRA Rule 4210 on April 14, 2026; the $25,000 minimum and the Pattern Day Trader designation were **eliminated effective June 4, 2026** (FINRA Regulatory Notice 26-10), replaced by a risk-based intraday margin system. Firms have until **October 20, 2027** to phase it in, so behavior varies by broker — **verify your broker's current status.** The margin-account minimum remains **$2,000**.

---

## 0. Objective & Operating Philosophy

**North star:** $1,000 + $50/week → $100,000, via validated, risk-capped, self-improving strategies. Milestones: `$1k → $2.5k → $5k → $10k → $25k → $50k → $100k`. Each triggers a review and a gain-ratchet (§2.4).

**Two engines, sequenced.** Early, the **$50/week deposit is the dominant force** (+5%/week on $1k, decaying to ~+0.5%/week at $10k); the job then is to compound the edge *without bleeding the deposits to costs*. Later, once the account is large enough that deposits are noise, the **trading edge becomes the growth engine.** Build accordingly: capital-preservation-plus-validation with aggressive-compounding.

**The maximization thesis.** Compounded (geometric) growth ≈ mean return − variance/2. You maximize it two ways: raise the edge, and **cut the variance for a given edge.** The highest-leverage move is therefore not bigger bets — it's *stacking uncorrelated edges* so the equity curve smooths, which lets you size larger at the same drawdown and compounds faster. Aggression and survival align. The system's durable moat is **the rate at which it discovers and validates new edges faster than old ones decay** — that is what "self-improving" means here.

**Core principle:** *LLM agents decide policy; deterministic code decides fast execution and protects capital.* The model sets the board (regime, arming, sizing policy, research); plain Python clicks the buttons and enforces the limits. The thing touching money has the least autonomy and zero latency.

---

## 1. The Growth Engine — How We Actually Maximize

Four lever groups. Each maps to concrete components later.

### A. Raise the edge (numerator)
- **Edge portfolio, not one strategy.** Run 3–5 validated strategies with *low mutual correlation* — e.g. breakout-retest (trend continuation), level mean-reversion (range/chop), and momentum-thrust — across timeframes and across equities + crypto. Each adds return; together they cut variance (group B).
- **The research conveyor (primary long-term lever).** A continuous pipeline mines hypotheses → backtests → OOS-gates → paper → live. The *throughput of validated edges* is the growth ceiling. Treat strategy decay as expected and replace from the conveyor (§6).
- **Conviction-tiered sizing.** Size proportional to expected edge (an approximation of optimal/Kelly sizing): B setups at RI 5, A at RI 6–7, A+ at RI 8. Capital concentrates in the best moments instead of spreading evenly across mediocre ones.
- **Regime-scaled exposure.** The regime layer is an *accelerator*, not just a filter: lean in during favorable, trending regimes; cut size or stand down in chop and around catalysts.

### B. Cut the variance (denominator) — this *raises* the growth rate
- **Uncorrelated edge-stacking** (the big one, above): lower curve volatility → higher safe size → faster compounding at the same risk of ruin.
- **Positive-skew trade structure:** partials + trailing runners (cut losers fast, let winners run) fatten the right tail, which compounds disproportionately.
- **Volatility targeting:** scale position size inversely to recent volatility so $-risk per trade stays constant across regimes — smooths the curve and stabilizes the compounding rate.

### C. Velocity & capital efficiency (more compounding cycles per unit time)
- **24/7 crypto compounding from day one** + (once on margin, post-PDT) **unrestricted equity day-trading** → more at-bats. *Double-edged:* more trades multiply both edge and cost drag, so this lever only pays once the edge clears realistic costs (§5). Throttle frequency to setup quality, not the reverse.
- **Capital-efficient instruments, sequenced.** After the base is validated, defined-risk options (e.g. debit spreads on liquid underlyings) give leverage with a capped, known downside — efficient for a small account. The cost is bid/ask spread and theta, so restrict to liquid names and never let one structure exceed the risk-index per-trade cap. *Do not build the options track first.*
- **Risk-based intraday margin (new post-PDT).** Once ≥$2k on margin, buying power is dynamic and concentration consumes it in real time — use it deliberately; the risk engine must read available buying power live, not assume it.

### D. Compounding mechanics & protecting gains
- **Reinvest everything, precisely.** Per-trade $-risk = (risk-index %) × **current equity**, recomputed each trade, so every realized profit and every deposit auto-compounds the next position's size.
- **Milestone ratchet (maximize *and* protect).** At each milestone, sweep a fixed fraction (default 25%) of gains into a **vault sleeve** the aggressive engine can't touch. You keep pressing with the rest while structurally protecting the milestone from giveback — "house money," systematized. A 35% drawdown needs a 54% gain to recover; the ratchet keeps you out of that hole.
- **Deposit timing as dry powder.** The weekly $50 tops up either the week's highest-conviction allocation or the vault, by rule, not by mood.
- **Tax-aware wrapper (consider, not advice).** Reinvesting in a taxable account generates short-term gains (ordinary income) — reserve for tax or you're compounding money you owe. A Roth IRA compounds tax-free (decisive if a 100x ever lands) but constrains: no margin, settled-cash only, annual contribution caps (~$7k), withdrawal limits. Worth weighing with a tax professional before P0.

---

## 2. Risk Index — the single dial (band 5–8, default 6)

One knob in `risk/limits.yaml` scales a coherent set of limits together. Per-trade risk is always **% of current equity**.

| RI | Per-trade | Max concurrent | Daily-loss halt | Weekly-loss halt | Portfolio heat | Leverage / options |
|----|---|---|---|---|---|---|
| 1–4 (below band, ref) | 0.25–0.75% | 1–2 | 1–1.5% | 3–4% | 1–2.5% | minimal / none |
| **5** | 1.0% | 2–3 | 2% | 5% | 3% | defined-risk options, small |
| **6 (default)** | 1.25% | 3 | 2.5% | 6% | 4% | options ok; ≤1.5× intraday |
| **7** | 1.5% | 3–4 | 3% | 7% | 5% | options ok; ≤2× intraday |
| **8** | 2.0% | 4 | 3.5% | 8% | 6% | options ok; up to broker intraday max |
| 9–10 (above band) | 2.5–3%+ | 5+ | 4%+ | 10%+ | 7%+ | max leverage — ruin risk climbs steeply |

**Conviction tiering (the dial flexes within the band):** setup grade maps to RI — B→5, A→6–7, A+→8 — so size tracks edge automatically. This is group-A sizing made operational.

**Catastrophe protections — NOT on the dial, always on.** These are not "risk appetite"; they're insurance against a bug, a disconnect, or a death spiral, and they're what let you run hot safely:
- Broker-vs-ledger **reconciliation halt** on any mismatch.
- **Dead-man's switch:** lost connection with an open position → cancel/flatten or alert-and-halt within a timeout.
- **Orphan-order recovery** on every startup before new orders.
- **Program-abort:** −20% in a calendar month → mandatory review; **−35% from all-time equity high → halt + manual restart.** Calibrated wide for a 5–8 band, but keep them — they prevent the unrecoverable.

---

## 3. Account & Market Structure (post-PDT reality)

- **Under $2k → cash account.** No PDT concern, but **T+1 settlement** means only settled cash trades; you can't freely round-trip the same dollars same day. With $50/week you cross $2k in ~20 weeks even with flat trading — sooner with gains — so this phase is short.
- **≥$2k → margin account.** PDT eliminated: day-trade equities freely, subject to **risk-based intraday margin** (concentration shrinks live buying power). Verify your broker has adopted the new framework (staggered through Oct 2027).
- **Crypto: the day-one velocity venue.** 24/7, no PDT, fast settlement → most compounding cycles per week. *Caveat:* Robinhood crypto spreads are wide (their revenue), so the cost model must be honest and the edge must be **re-validated on crypto** — the equity playbook is not assumed to transfer.
- **Small-account cost reality.** At $1k, 1% risk = $10/trade; a few dollars of round-trip friction can erase the trade's expectancy. Model **cost-as-%-of-equity/year** end-to-end; a PF-2.24 backtest on optimistic costs can be PF < 1 net of real frictions. This is why early phases preserve, and why velocity (C) is throttled to quality.
- **Broker capability to confirm before P3:** native server-side OCO brackets (if only locally simulated, the dead-man's switch becomes mandatory), real-time vs delayed quotes, and supported order/margin types.

---

## 4. Architecture (lean; deterministic fast loop, LLM slow loop)

```
~/projects/tradeforge/
├── CLAUDE.md                    # mission, reframed goal, conventions, risk band
├── .claude/
│   ├── settings.json            # PreToolUse hook: block live orders unless gated
│   ├── agents/                  # LEAN roster (~4 LLM agents to start)
│   └── skills/                  # ported skills (levels, sizing, IV read, journal, …)
├── orchestrator/
│   ├── main.py                  # CLI / long-running loop
│   ├── bus.py                   # event bus + replayable DuckDB event log
│   ├── events.py                # typed events
│   ├── fast_loop/               # DETERMINISTIC engine: entries/exits, no LLM, no MCP in hot path
│   ├── watchdog.py              # heartbeat + dead-man's switch
│   ├── hooks.py                 # gate live-order endpoints
│   ├── workflows/               # premarket / intraday-boot / eod / weekly
│   └── tools/                   # market_data, order_gateway (paper/live router), notify (iMessage)
├── strategies/
│   ├── registry.yaml            # status, edge stats, correlation, allocation
│   ├── breakout_retest/         # the PF-2.24 baseline + playbook
│   ├── level_meanrev/           # uncorrelated complement (chop regime)
│   ├── momentum_thrust/         # uncorrelated complement (trend regime)
│   └── _template/
├── backtest/
│   ├── engine/                  # vectorbt or backtesting.py + realistic cost model
│   ├── stats/                   # CIs, multiple-testing haircut, locked OOS vault, regime split
│   ├── walk_forward.py
│   └── reports/
├── watchlist/
│   ├── universe.duckdb          # symbols, tiers, per-strategy scores, correlation matrix
│   ├── criteria.yaml
│   ├── screeners/               # stocks.py, crypto.py, unusual_volume.py
│   └── level_respect.py         # per-strategy fit scoring (mini-backtest)
├── orderbook/
│   ├── orderbook.duckdb         # orders, brackets(OCO), positions(MFE/MAE), fills(slippage), recon
│   ├── state_machine.py         # STAGED→APPROVED→SUBMITTED→WORKING→{FILLED|PARTIAL|…}
│   └── reconcile.py             # runs FIRST on boot; paper AND live
├── paper/ledger.duckdb          # identical event path to live
├── risk/
│   ├── limits.yaml              # SINGLE SOURCE OF TRUTH — risk-index table, ratchet, abort
│   └── breakers.py              # deterministic, subscribes to bus, no LLM
├── data/duckdb/market.duckdb    # bars, levels, IV, corporate-action-adjusted
├── journal/                     # per-trade MD+JSON, digests, headless chart PNGs
├── reporting/charts.py          # headless render (journal/digest/backtest)
├── audit/guardrail_audit.py     # nightly deterministic audit
├── tests/ · scripts/cron/ · pyproject.toml · .env.example · .mcp.json · README.md
```

**Fast loop (deterministic):** evaluates pre-armed, pre-approved rules against the live feed and manages entry, stop, TP1 partial, breakeven move, trailing runner, time-stop, session-flatten. Native broker brackets where available. Logs a latency budget.

**Slow loop (LLM agents, ~4 to start):** `regime-reader` (daily regime/vol/IV tag → arms strategies, sets exposure scalar), `strategy-researcher` (offline mining, backtests, walk-forward), `journalist` (auto journals + premarket/EOD digests with chart PNGs), `performance-analyst` (equity curve, alpha-vs-SPY, decay detection, milestone + ratchet, risk-of-ruin estimate). Defer `catalyst-watcher`, `watchlist-curator` (start as a weekly script), `options-strategist` until capital/validation justify. Model tiers: Haiku for routine, Sonnet/Opus for research — track API cost as a P&L line.

**Deterministic services (no LLM):** fast loop, order state machine + brackets + reconciliation, risk gate + breakers + halts, guardrail auditor, watchdog + dead-man's switch.

**Event-driven core:** asyncio bus + persisted, replayable DuckDB `events` table.
Catalog: `PRICE_CROSS_LEVEL, LEVEL_BREAK_CONFIRMED, SETUP_FORMING/CONFIRMED/INVALIDATED, ORDER_INTENT, ORDER_APPROVED/VETOED, ORDER_FILLED/PARTIAL/REJECTED, TP1_HIT, STOP_HIT, POSITION_CLOSED, CIRCUIT_BREAKER_TRIPPED, COOLDOWN_STARTED, NO_TRADE_WINDOW, VOL_SPIKE, DATA_ANOMALY, STRATEGY_DEMOTED, MILESTONE_REACHED, RATCHET_SWEEP, DEPOSIT_LOGGED, WATCHLIST_UPDATED, SYMBOL_PROMOTED/DEMOTED`.

Key flows: **F1 Signal→Exec** `PRICE_CROSS_LEVEL → fast-loop trigger (NTZ? armed? regime?) → risk gate (size by conviction tier, heat, halts) → native bracket → fill`. **F2 Lifecycle** `FILLED → manage (TP1 50%, stop→BE, trail) → close → journalist + analyst + heat release`. **F3 Interrupts (preempt all)** daily/weekly/program halt, NO_TRADE_WINDOW, DATA_ANOMALY. **F4 Decay→demote** rolling PF(30) < threshold → auto LIVE→PAPER + researcher review. **F5 Growth** weekly DEPOSIT_LOGGED → resize; MILESTONE_REACHED → review + RATCHET_SWEEP.

**Order book / watchlist / journal:** order ledger with slippage capture feeding the cost model back into backtests (closes the sim-to-real gap). Universe tiers CORE (2–4, live) / ACTIVE (≤8, paper) / SCOUT (~25, observe), gated by a per-strategy **level-respect score** (replay 90 days; screening *is* a backtest) plus a **correlation check** so CORE stays genuinely diversified (don't fill it with three SPY proxies). Journal: frame card, intended-vs-actual slippage, MFE/MAE, deterministic plan-adherence flags, 3-line narrative, chart PNG.

---

## 5. Strategy & Validation

**The playbook (breakout_retest):** levels PDH/PDL/PMH/PML; battle zone = level ± 0.15×ATR; **NTZ = overlap of [PDL,PDH] and [PML,PMH]**, no entries inside; clean break = close beyond level by ≥ break_buffer; retest within window with a confirmation candle; stop beyond the retest extreme; exit fixed-2R or partial+runner. Ablation **V0→V4** (V0 = PDH/PDL + 2R, must reproduce the known PF 2.24 *net of realistic costs* as a sanity gate; add NTZ, then PMH/PML, then partial/runner, then full playbook). A component survives only if it improves expectancy in-sample **and** holds OOS **and** clears the multiple-testing haircut.

**Edge portfolio (group A/B):** build `level_meanrev` and `momentum_thrust` as deliberately low-correlation complements; track the live **correlation matrix** in `registry.yaml` and allocate to minimize curve variance, not to chase the single highest backtest PF.

**Validation rigor:**
- PF/expectancy with **confidence intervals + trade counts** (distrust point estimates under ~100–150 trades).
- **Multiple-testing correction** (deflated Sharpe / PF threshold that rises with the number of variants tried); **log every hypothesis**, not just survivors.
- **Locked OOS vault** — a history slice never touched in development and never reused.
- **Point-in-time + corporate-action adjustment** (splits/dividends) so levels stay meaningful and backtest = live (no train/serve skew).
- **Performance by regime** (trend / chop / vol-shock), not just aggregate.
- **Realistic small-account cost model** (commission + spread + slippage, honest for crypto/options).

**Promotion gates:** backtest→paper (≥100 trades, PF ≥ 1.5, OOS required, cost-realistic); paper→live (≥30 paper trades, PF ≥ 1.3, maxDD ≤ 10%, net of costs). Data: premarket-capable intraday bars 2–3 yrs back (Alpaca extended-hours free, or Polygon).

---

## 6. Self-Improvement Conveyor (safe by construction)

This is the engine, and the discipline *is* the feature — "self-improving trading AI" is otherwise the textbook path to overfitting and ruin.

**Allowed:** (1) researcher proposes new strategies/params → must clear the full backtest+OOS+walk-forward+paper gate + human confirm; (2) analyst auto-demotes decayed strategies LIVE→PAPER (deterministic); (3) scheduled walk-forward refits **within pre-registered sweep ranges**, each OOS-checked, human-confirmed; (4) cost-model updates from real slippage; (5) scheduled universe/correlation re-scoring.

**Forbidden:** tuning live parameters from live P&L; changing a strategy in reaction to a drawdown (you may only demote/halt); intra-week edits; any LLM **writing live config or `limits.yaml`**; promoting on in-sample results.

**Research firewall:** research agents read everything, write nothing live. Production changes only through the deterministic gate + human confirm. A **monthly research review** lets new ideas compete. Self-improvement upgrades the *pipeline of validated edges* — never the live knobs.

---

## 7. Ops & Resilience

- **Watchdog/heartbeat:** independent process; on silence → alert + safe-mode.
- **Dead-man's switch:** lost broker/data with an open position → cancel/flatten or alert-and-halt within a timeout (mandatory if brackets are local, §3).
- **Crash/orphan recovery:** on boot, reconcile with broker first — adopt/cancel orphans, rebuild from the event log, then resume.
- **Hosting:** live on a small always-on cloud host, not a sleeping laptop; durable storage for the event log + DBs. Laptop fine for backtest/paper.
- **API/infra cost vs P&L:** explicit line item — on $1k it can dominate; right-size models accordingly.

---

## 8. Build Phases (PDT-aware, with gates)

- **P0 — Pre-flight + scaffold.** Resolve open decisions (§10), choose account wrapper (taxable vs Roth), scaffold, venv (pip3 only), `CLAUDE.md`, `.mcp.json`, `limits.yaml` with the risk-index table + ratchet + abort, git. *Gate: decisions in CLAUDE.md.*
- **P1 — Data + levels.** Alpaca/Polygon → DuckDB bars (2m/5m, SPY/QQQ + 1–2 liquid crypto), point-in-time, split-adjusted levels (PDH/PDL/PMH/PML/NTZ) + ATR, nightly refresh. *Gate: eyeball-validate 10 sessions vs TradingView; levels-math tests pass.*
- **P2 — Backtest + ablation + edge portfolio.** Realistic costs; V0 reproduces PF 2.24 net of costs (else stop/diagnose); V1–V4 + OOS (locked vault) + walk-forward + multiple-testing haircut; build the two uncorrelated complements; report correlation + by-regime. *Gate: components earn their place per the rule; portfolio variance < best single strategy.*
- **P3 — Event core + order book + paper + resilience.** Bus, deterministic fast loop, state machine + native/hardened brackets, paper ledger on the identical event path, breakers, on-boot recovery, dead-man's switch. *Gate: F1+F2 replay test passes; simulated mid-trade crash recovers cleanly.*
- **P4 — Lean agents + digests.** The ~4 LLM agents; premarket/EOD digests with chart PNGs; alpha-vs-SPY + risk-of-ruin in the analyst; conviction-tier sizing wired into the risk gate. *Gate: a week of digests you'd act on.*
- **P5 — Paper trade.** Validated strategy(ies), paper only, ≥30 trades, weekly reconciliation, after-tax + cost tracking, watchlist script. *Gate: paper→live gate met net of costs.*
- **P6 — Controlled live.** Start in **crypto** (day-one venue) + cash-account equity; cross $2k → margin, open equity day-trading. RI per index, all halts + dead-man's switch armed, ratchet active. *Gate: program-abort not hit; alpha-vs-SPY ≥ 0 after costs.*
- **P7 — Optimization loop.** Research→backtest→paper→live through the same gates; monthly meta-review; per-trade $ scales with equity at each milestone; add options/extra agents only once the base earns its keep.

**Go-live checklist:** native/hardened brackets confirmed · dead-man's switch tested · orphan recovery tested · reconciliation halt wired · all halts firing · cost model realistic · ratchet + abort live · ≥1 strategy past paper→live · API+infra cost < expected edge.

---

## 9. Metrics That Matter

Alpha vs SPY after costs (the bar — if ≤ 0 over 6 months live, stop) · **geometric growth rate** and curve volatility (the maximization target) · expectancy in R, win rate, PF with CIs + counts · max/current drawdown vs abort thresholds · cost-as-%-of-equity/yr · API+infra cost vs P&L · after-tax equity · paper-vs-backtest reconciliation · estimated risk of ruin at current RI and measured edge · live strategy correlation matrix.

---

## 10. Open Decisions (to finalize P0)

1. **Account wrapper:** taxable (flexible, but tax drag) or Roth IRA (tax-free compounding, but no margin / settled-cash / contribution caps)? Materially changes leverage and velocity.
2. **Broker status:** has your broker adopted the post-PDT risk-based margin framework yet, and does its MCP support native OCO brackets?
3. **Starting RI:** open at 6, or 5 during validation then step toward 8 as paper proves out?
4. **Edge-portfolio scope at launch:** one validated strategy live + two in research, or hold live until ≥2 uncorrelated edges clear the gate (smoother curve, slower start)?

---

## 11. Locked-In Principles

LLM decides policy / deterministic code executes fast + protects · one event path for paper and live · everything replayable · execution stays dumb · maximize the *geometric* rate (cut variance, stack uncorrelated edges) · improve the pipeline, never live knobs · catastrophe protections are never on the risk dial · the system's first job is not dying; its second is beating SPY after costs.
