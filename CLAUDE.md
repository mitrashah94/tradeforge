# CLAUDE.md — DayTrading workspace

Decision-support system for a small-account asymmetric options campaign.
TradingView (analysis + signals) -> Claude (validation + math + journaling) ->
Robinhood Legend (manual execution by the trader).

**At session start, read [`daytrading_memory.md`](daytrading_memory.md)** — the running
state, decisions, account status, and hard-won gotchas — then this file and `strategy.md`.
Keep `daytrading_memory.md` updated as things change (it replaces the old auto-memory).

## Hard rules — non-negotiable

1. **Never place, modify, or cancel any order.** Robinhood MCP write tools
   (`place_*`, `cancel_*`, `review_*` used to stage orders) are off-limits.
   Read-only tools (quotes, chains, positions, portfolio) are fine.
2. **Recommend the user to enter a trade.** Verify their written rules
   against observable data; the decision and the submit click are theirs.
3. **No other strategies.** The only strategy is [strategy.md](strategy.md).
   Do not import ideas from other trading skills or memory.
4. **Not a licensed advisor.** Process and math support only.
5. Nothing in `signals/` may import, call, or reference a brokerage API.

## File map

| Path | What it is |
|---|---|
| `daytrading_memory.md` | Running memory: state, decisions, account, gotchas, session log (read first) |
| `NEXT_SESSION_PLAN.md` | Handoff + the step-by-step options **Order Playbook** |
| `strategy.md` | The operating plan (authoritative) |
| `AGENTS.md` | Rules and schema for subagents |
| `Trading_Journal.xlsx` | Trade log, weekly metrics, sim-qualification tracker (R unit lives in Settings!B4) |
| `pine/asymmetric_live_signal.pine` | indicator() — levels + WATCH/QUALIFIED/REJECT/INVALIDATED/EXPIRED JSON alerts. No orders, no sizing. |
| `pine/asymmetric_backtest_strategy.pine` | strategy() — underlying backtest, $1,100 capital, equity-capped sizing, R-based reporting |
| `signals/receiver.py` | Local pipeline: parse -> validate -> dedupe -> decision card |
| `signals/fixtures/` | Sample payloads for offline testing |
| `signals/test_receiver.py` | Self-contained tests (plain asserts) |
| `tradeforge/` | Dormant autonomous-platform repo (own CLAUDE.md, own venv), kept for later integration. NOT an active strategy source — hard rule 3 applies; nothing in it overrides strategy.md. |
| `FEATURE_MAP.md` | Functional comparison of DayTrading vs tradeforge by pipeline stage: what lives where, what's integrated, what needs updating |

## Signal pipeline

The Pine indicator emits single-line JSON on confirmed 5-minute closes only
(`barstate.isconfirmed` + `alert.freq_once_per_bar_close`), one event per
side per day, deduplicated by `event_id`. Canonical schema: see AGENTS.md.

Local verification (no TradingView needed):

```bash
python3 signals/receiver.py --all      # run every fixture
python3 signals/receiver.py --demo     # built-in samples through full pipeline
python3 signals/test_receiver.py       # assertions
```

A QUALIFIED card is an *inspection prompt*, never an entry instruction.

## TradingView MCP — hard-won quirks

- `pine_set_source` writes a **hidden headless editor**, not the visible one.
  Compiling there works and `pine_get_source` reads it back, but **it cannot
  be saved to the script library**. The visible editor rejects synthetic
  paste and auto-indent mangles synthetic typing. **Source injection into a
  library script requires the user's hands**: put the source on the system
  clipboard (`pbcopy < file`), then have the user Cmd+A / Cmd+V / Cmd+S in
  the Pine editor. Verify afterwards via `pine_list_scripts` (title +
  modified timestamp change).
- `layout_switch` reports success without switching; use the Manage-layouts
  menu (`ui_click` data-name `save-load-menu`, then click the layout link
  via `ui_evaluate`).
- "Make a copy" in the editor copies the **visible** editor's script.
- Strategy Tester **initial capital override** silently rejects orders the
  capital cannot fund (this produced "no trade data" once). The backtest
  script itself now sets 1100 and sizes affordably — do not override it.
- Alerts must be recreated per chart/symbol. Each morning, on the chosen
  candidate's chart: alert condition = "Asymmetric Live Signal v2" +
  **"QUALIFIED (any)"**.
  **NEVER use "Any alert() function call".** That pipes **WATCH** to the
  trader's screen with the same weight as QUALIFIED. On 2026-07-13 he mistook
  WATCH for QUALIFIED and believed he had two setups when the engine produced
  **zero**. WATCH is not a setup and not an entry. Only QUALIFIED may ever
  interrupt him. Add the "INVALIDATED" alert only once he is actually in a
  position. Claude still consumes the full event stream for context.
- **`pine_set_source` + save writes to whichever script is OPEN in the VISIBLE
  Pine editor.** `pine_open` does NOT reliably switch it. This has clobbered a
  library script. Before any write: `pine_list_scripts`, confirm the target is
  the open one, and verify title + modified timestamp afterwards. The on-disk
  `pine/*.pine` files are the source of truth and make any clobber recoverable.
  Safest path for a library install is still clipboard + the user's Cmd+A/V/S.
- Pane indexes in 2x2 layouts do not map left-right/top-bottom; verify with
  a screenshot after `pane_set_symbol`.

## Daily assistant workflow (CT)

1. **07:45–08:15** — pull premarket data for SPY/QQQ/XLF/XLE/IWM
   (TradingView MCP + Robinhood read-only), fill the scorecard from
   strategy.md section 9, check the economic calendar, check option-chain
   liquidity on the top two candidates.
2. **08:30** — confirm candidate chart loaded, indicator running, alert
   created, Robinhood buying power + account type confirmed by user.
3. **08:45–10:30** — entry window. On QUALIFIED: run the card through
   `signals/receiver.py`, verify plan gates, compute the option-contract
   feasibility check (see strategy.md section 7), read out passing
   contracts. User decides.
4. After any trade: journal row in Trading_Journal.xlsx (both before-entry
   and after-exit fields), screenshots via `capture_screenshot`.
5. **14:55** — remind user to be flat.
6. **After the close — RECORD THE SESSION. Every day, including no-trade days.**
   See the learning loop below. This is not optional; a no-trade day that goes
   unrecorded is the single best-behaved day in the system's history vanishing.

## The learning loop (run it every session, trade or no trade)

The journal is one row per **trade**. That means a disciplined **no-trade day
produced zero rows anywhere** — the system could not see its own best behavior,
and strategy.md §13 requires *"3 intentional no-trade sessions"* it had no way to
evidence. `backtests/sessions.jsonl` is one row per **day** and closes that hole.

**Daily, after the close:**

1. Save the day's 5m bars + `levels.json` for the watched tickers into
   `backtests/session_YYYY-MM-DD/` (SPY/QQQ too — "candidate fights SPY/QQQ" is a
   §5 no-trade condition, so auditing a no-trade day sometimes *requires* them).
2. `python3 analysis/replay_session.py --session backtests/session_YYYY-MM-DD`
   — reruns the exact four-track engine offline and writes `events.json`.
3. `python3 analysis/record_session.py` — appends one row to
   `backtests/sessions.jsonl`. The **verdict is derived, never asserted**: it is the
   2x2 of *(engine had a QUALIFIED?)* x *(trader acted?)* →
   `CORRECT_TRADE` | `CORRECT_NO_TRADE` | `MISSED_SIGNAL` | `OFF_PLAN_TRADE`.
   The trader does not get to grade his own homework.
4. One line in `daytrading_memory.md`'s session log.

**Weekly (Friday):** regenerate `reports/learning_report.md` and answer, verbatim:

> *"Did every session get recorded and replayed, and did my action match the
> engine's state in ≥80% of sessions? If not, which cell of the matrix am I
> living in?"*

**What the loop actually learns — two numbers, nothing else:**

- **Decision accuracy** = (CORRECT_TRADE + CORRECT_NO_TRADE) / sessions. Target ≥80%,
  mirroring the §13 adherence gate. *Currently 50% (n=2).*
- **QUALIFIED frequency.** If the strategy almost never fires, the edge is
  **unvalidatable no matter how disciplined he is** — a completely different problem
  from adherence, and one that would otherwise be invisible. *Currently 0.00/session.*

**Standing caveats, restated in every weekly review:**

- **ORB same-bar priority remains UNVALIDATED** (6-trade sample).
- **Open question — the pre-window REJECT.** On 2026-07-13 XLF's PDH track broke at
  08:30 and REJECTed at **08:40, five minutes before the 08:45 entry window opened** —
  burned for the day before the trader was even allowed to trade. ORB tracks are
  structurally immune (the OR completes at 08:45), so only PD tracks die this way.
  Decide it with `replay_session.py` over ~20 accumulated sessions, **not with priors**.
  Do NOT gate the *break* to 08:45 — that would kill the good 08:35-break/08:50-retest.

**Hypotheses are tested by rerunning `replay_session.py` variants over the accumulated
session dirs.** The sessions are the corpus. Do not add a database, a dashboard, or an
"insights engine" — this is a one-person system and an elaborate pipeline will rot.

**R does not move on deposits.** strategy.md §1 raises R only after **five correctly
executed trades** — execution quality, not P&L, and not account size. That counter is
still at **zero** (Day 1 was off-plan). A $200 deposit does not change 1R = $25.

## Journal conventions

One row per trade or sim. Realized R = net option P&L / Settings!B4.
Never overwrite formula columns (J–O, R, W, AA–AB, AF–AG, AP).
Weekly metrics compute themselves; the metric that matters most is
rule-adherence %.

## Model workflow

The main Claude Code session is the orchestrator.

For non-trivial tasks:

1. Inspect the relevant files and repository state.
2. Consult the configured Fable advisor before committing to an
   implementation approach.
3. Create a bounded implementation brief.
4. Delegate code implementation to the `sonnet-coder` subagent.
5. Inspect the resulting diff and test output.
6. Consult the Fable advisor again before declaring a consequential
   change complete.

Use the advisor when requirements are ambiguous, architecture decisions
are involved, an implementation fails, or a consequential change appears
ready for completion.

Do not use Fable as a file-editing agent. Fable is the strategic advisor.
Use `sonnet-coder` for implementation.