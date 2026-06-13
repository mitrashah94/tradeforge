# Agent roster — the lean slow loop

TradeForge runs **two engines** (MASTER_PLAN.md §4): a **deterministic fast loop**
(no LLM, no MCP in the hot path — it touches money and has zero autonomy) and an
**LLM slow loop** that *decides policy*. The agents below are the slow loop. They
set the board (regime, arming, sizing policy, research, journaling, analysis);
plain Python clicks the buttons and enforces the limits.

> **Core principle:** *LLM agents decide policy; deterministic code decides fast
> execution and protects capital.* The thing touching money has the least
> autonomy and zero latency.

## The 4 agents (lean by design)

| Agent | One line | Model tier |
|---|---|---|
| **regime-reader** | Daily policy-setter: tags regime + realized-vol, arms strategies for the session, sets the exposure scalar. | `haiku` (routine, daily) |
| **strategy-researcher** | Offline edge miner: proposes strategies/params, runs full backtest + OOS + walk-forward — never auto-promotes. | `opus` (research) |
| **journalist** | Auto per-trade journals + premarket/EOD digests with chart PNGs. | `haiku` (routine, high-frequency) |
| **performance-analyst** | Equity curve, alpha-vs-SPY, decay detection, milestone + ratchet, risk-of-ruin estimate. | `sonnet` (analysis) |

## Model-tier policy (track API cost as a P&L line — MASTER_PLAN.md §4, §7)

On a $1k account, API + infra cost can dominate, so the model tier is chosen
deliberately per agent and **API cost is an explicit P&L line item**:

- **Haiku** — routine, high-frequency, narrow-judgment work: the daily
  `regime-reader` read and the per-trade `journalist`. The heavy lifting is
  already deterministic Python (`orchestrator/agents/regime_reader.py`); the
  model only oversees/narrates, so a small model is right.
- **Sonnet** — periodic analysis with real judgment: `performance-analyst`
  (decay calls, alpha attribution, ratchet/risk-of-ruin reasoning).
- **Opus** — open-ended research where quality compounds: `strategy-researcher`
  mining hypotheses and reasoning about validation. Run infrequently (offline,
  batched), so the per-call cost is acceptable for the edge it can find.

Right-size aggressively: spend model dollars where they buy validated edge, not on
the hot path (which has none) or on routine narration (which needs little).

## Deferred agents — and why

The roster is intentionally small. These are **not built yet**; each waits for a
concrete trigger so we don't pay for capability the account can't justify:

- **catalyst-watcher** — *deferred until a news/catalyst feed exists AND capital
  justifies it.* There is no catalyst data source wired, and around-catalyst
  behavior today is handled defensively (the regime-reader cuts exposure in
  `vol_shock`, the breakers enforce `NO_TRADE_WINDOW`). Adding an agent before the
  feed exists is cost with no signal.
- **watchlist-curator** — *started this phase as a WEEKLY SCRIPT, not an agent.*
  Universe scoring + correlation re-ranking is deterministic and scheduled
  (MASTER_PLAN.md §6 "scheduled universe/correlation re-scoring"); it does not
  need an LLM yet. **Promote to an agent later** when the curation needs judgment
  (e.g. qualitative theme/sector calls) the script can't encode.
- **options-strategist** — *deferred until the base earns its keep AND an options
  data/exec path exists.* MASTER_PLAN.md §1.C is explicit: *"Do not build the
  options track first."* The agentic broker API is **single-leg, equities-only**
  (CLAUDE.md) — no options order endpoint and no options/IV feed — so this agent
  has nothing to read or trade yet. The `iv-rank-skew-read` and
  `options-greeks-framer` skills are ready for it when those preconditions land.

## Research firewall (MASTER_PLAN.md §6 — safe by construction)

**Agents read everything, write NOTHING live.** Production changes flow only
through the **deterministic gate + human confirm**, never an LLM write.

- **FORBIDDEN (any agent):** writing live config or `risk/limits.yaml`; tuning
  live parameters from live P&L; changing a strategy in reaction to a drawdown
  (you may only demote/halt); intra-week edits; promoting on in-sample results.
- **Protected (read-only for agents):** `risk/limits.yaml`,
  `strategies/registry.yaml` (live status/allocation), `.claude/settings.json`
  (the live-order hook), `backtest/stats/oos_vault.yaml` (the locked OOS vault),
  and the rest of `risk/`, `.claude/`, `orderbook/`, `paper/`.
- **Agent-writable (research/journal/report only):** `research/`, `journal/`,
  `reports/`, rendered artifacts. Nothing here touches money until a human
  promotes it through the gate.

This is enforced **two ways**: culturally (this file + each agent's `.md` +
CLAUDE.md) and **programmatically** by `orchestrator/agents/firewall.py`
(`assert_not_live_config(path)` raises a `FirewallViolation` on any agent write to
a protected path; `agent_writable(path)` is the positive check). The cultural rule
is the primary guard; the module is the backstop against a misbehaving or
prompt-injected agent. Self-improvement upgrades the *pipeline of validated
edges* — never the live knobs.
