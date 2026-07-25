# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Full design rationale lives in [MASTER_PLAN.md](MASTER_PLAN.md). **This file is the operating contract** — mission, resolved P0 decisions, and the conventions every change must respect. MASTER_PLAN.md is the reasoning; this file is the rule.

## Mission

TradeForge is an autonomous, self-improving systematic trading platform whose job is to **maximize compounded growth inside a fixed risk band while deterministic code makes catastrophic loss structurally impossible.**

- **North star:** $1,000 + $50/week → $100,000.
- **Milestones:** `$1k → $2.5k → $5k → $10k → $25k → $50k → $100k`. Each triggers a review and a gain-ratchet (see below).

## Two engines, sequenced

- **Early (≈$1k–$10k): the $50/week deposit is the dominant force** (+5%/week on $1k, decaying to ~+0.5%/week by $10k). The job is to compound the edge *without bleeding deposits to costs* — preserve capital, validate, keep frictions honest.
- **Later (account large enough that deposits are noise): the trading edge becomes the growth engine.**

Build for both: capital-preservation-plus-validation early, aggressive-compounding once edges are proven. Never optimize the late-game engine at the cost of not-dying early.

## The maximization thesis

Compounded (geometric) growth ≈ **mean return − variance/2.** Raise it two ways: raise the edge, and **cut the variance for a given edge.** The highest-leverage move is **stacking uncorrelated edges** — it smooths the equity curve, which lets you size larger at the same drawdown, which compounds faster. Aggression and survival align. **Size by conviction** (best moments get the most capital), not evenly across mediocre ones. The durable moat is *the rate at which we discover and validate new edges faster than old ones decay.*

## Resolved P0 decisions (§10)

1. **Account wrapper — Taxable.** Keeps margin, unrestricted intraday day-trading, and the day-one crypto venue available (crypto is not IRA-eligible at Robinhood). *Consequence:* realized gains are short-term/ordinary income → carry a **tax-reserve ledger line** and track **after-tax equity** as a first-class metric; compounding uses after-tax dollars.
2. **Broker (Robinhood) post-PDT status — unverified.** *Consequence:* assume **legacy PDT until proven.** Build the **cash-account path first** (settled cash, T+1, no same-dollar same-day round-trips); keep margin day-trading behind a config flag that flips only after the broker's risk-based-margin adoption is confirmed (required before P6).
3. **Native OCO brackets — NOT available** via the Robinhood agentic API. `place_equity_order` is **single-leg** (market / limit / stop_market / stop_limit), has **no bracket parameter**, and is **equities-only** (no options, no crypto order endpoint). *Consequence:* brackets are **simulated locally by the deterministic fast loop**, the **dead-man's switch is mandatory**, and crypto + options each require a **separate execution path**.
4. **Starting risk index — RI 5, ramping to 8.** `risk/limits.yaml` opens with **floor = 5.** *Consequence:* the step-up toward 8 is a **manual, gated decision driven by proven paper/live performance — never auto-tuned from live P&L.** Conviction tiering flexes within `[current floor, 8]`.
5. **Launch scope — one live edge + two in research.** Take the single validated `breakout_retest` live as soon as it clears paper→live; carry `level_meanrev` + `momentum_thrust` as PAPER/RESEARCH in `strategies/registry.yaml`. *Consequence:* real-cost/slippage capture and the research conveyor start early, and the **live correlation matrix is tracked from day one** so the 2nd/3rd edge is admitted on *low correlation*, not raw PF.

## Conventions (every change respects these)

- **Paper-first.** Nothing reaches live until it clears the gates — backtest→paper (≥100 trades, PF ≥ 1.5, OOS required, cost-realistic) and paper→live (≥30 paper trades, PF ≥ 1.3, maxDD ≤ 10%, net of costs). Paper and live ride the **identical event path.**
- **LLM decides policy; deterministic code executes the fast loop and protects.** Agents set the board (regime, arming, sizing policy, research); plain Python clicks the buttons and enforces the limits. **No LLM and no MCP calls in the hot path.** The thing touching money has the least autonomy and zero latency.
- **No live orders without hook approval.** A `PreToolUse` hook in `.claude/settings.json` blocks live-order endpoints unless explicitly gated. Default to paper; route live through `order_gateway`. Before any real order: `review_equity_order` → confirm → `place_equity_order` (agentic_allowed account only). **Never bypass the hook.**
- **pip3 only** for Python packages (no conda, no bare `pip`).
- **Self-improvement upgrades the *pipeline*, never the live knobs.** Forbidden: tuning live params from live P&L; changing a strategy in reaction to a drawdown (you may only demote/halt); intra-week edits; any LLM writing live config or `limits.yaml`; promoting on in-sample results. Research agents read everything, write nothing live.
- **`risk/limits.yaml` is the single source of truth** for the risk-index table, ratchet, and abort thresholds. Edit it by hand, deliberately.

## Risk dial & protections

**Risk index — one knob (band 5–8, current floor 5).** Per-trade risk is always **% of current equity**, recomputed each trade. **Conviction tiering flexes the dial within the band: B → 5, A → 6–7, A+ → 8**, so size tracks edge automatically.

| RI | Per-trade | Concurrent | Daily halt | Weekly halt | Heat |
|----|-----------|------------|-----------|-------------|------|
| 5 (floor) | 1.0% | 2–3 | 2% | 5% | 3% |
| 6 | 1.25% | 3 | 2.5% | 6% | 4% |
| 7 | 1.5% | 3–4 | 3% | 7% | 5% |
| 8 | 2.0% | 4 | 3.5% | 8% | 6% |

**Milestone ratchet (maximize *and* protect).** At each milestone, sweep **25% of gains** into a **vault sleeve the aggressive engine cannot touch.** Press with the rest; structurally protect the milestone from giveback.

**Program-abort — NOT on the risk dial, always on.** **−20% in a calendar month → mandatory review; −35% from the all-time equity high → halt + manual restart.** These are insurance against bugs, disconnects, and death-spirals — not risk appetite — and are never adjusted by the dial. Also always-on alongside them: broker-vs-ledger reconciliation halt, dead-man's switch, and orphan-order recovery on every boot.

## Lean agent roster (~4 to start)

- **regime-reader** — daily regime/vol/IV tag → arms strategies, sets the exposure scalar.
- **strategy-researcher** — offline mining, backtests, walk-forward (reads everything, writes nothing live).
- **journalist** — auto per-trade journals + premarket/EOD digests with chart PNGs.
- **performance-analyst** — equity curve, alpha-vs-SPY, decay detection, milestone + ratchet, risk-of-ruin estimate.

Defer `catalyst-watcher`, `watchlist-curator` (start as a weekly script), and `options-strategist` until capital and validation justify them. Model tiers: Haiku for routine, Sonnet/Opus for research — track API cost as a P&L line.
