# AGENTS.md — rules for subagents in this workspace

Context: decision-support tooling for a manual options day-trading campaign.
TradingView Pine scripts generate signals; a local Python receiver renders
human-review cards; the trader executes manually in Robinhood.
Full plan: [strategy.md](strategy.md). Orchestrator notes: [CLAUDE.md](CLAUDE.md).

## Absolute constraints — every agent, every task

1. **No brokerage access, ever.** Do not import, call, mock toward, or
   scaffold any order/trading endpoint. The Robinhood MCP write tools
   (`place_*`, `cancel_*`) are forbidden. Nothing you build may place,
   stage, or recommend orders. Output is information for a human.
2. **Do not touch the TradingView MCP** (`mcp__tradingview__*`) unless your
   task explicitly says so — it drives a single shared desktop app and
   concurrent use corrupts editor state. Known trap: `pine_set_source`
   writes a hidden editor that cannot save to the library.
3. Deliverables are files on disk + a summary. Python is 3.9 **stdlib
   only**. Pine is **v6 only**. ASCII only in Pine sources. No emoji.
4. Do not modify `Trading_Journal.xlsx`, memory files, or the original
   `Asymmetric PDH/PDL Retest v1` library script.
5. If the task conflicts with strategy.md, stop and report the conflict.

## Canonical signal schema (single source of truth)

Single-line valid JSON. Numbers bare (never ".95" — always "0.95"),
`null` when unavailable, strings quoted.

Envelope (all events):
`event` (WATCH|QUALIFIED|REJECT|INVALIDATED|EXPIRED), `event_id`
(`TICKER-YYYYMMDD-DIRECTION-LEVEL-EVENT-barindex`), `ticker`, `timeframe`
("5"), `setup_type` (A_break_retest|B_breakdown_bounce), `direction`
(CALL|PUT), `level` (PDH|PDL|ORH|ORL), `level_price`, `signal_time_ct`
("yyyy-MM-dd HH:mm", America/Chicago), `vwap`, `rvol`.

**Direction/level pairing (enforced by the validator):**
`CALL` -> `PDH` or `ORH`. `PUT` -> `PDL` or `ORL`. Any other pairing is a
schema violation.

**Level tracks.** Four independent tracks run per day: PDH-long, ORH-long,
PDL-short, ORL-short. All four use the identical break -> retest -> confirm
engine (same retest window, VWAP/RVOL/EMA filters, ATR stop, risk validity).
- ORB tracks (ORH/ORL) are inert until the opening range completes
  (`not inOpeningRange`) and while `orh`/`orl` are `na`.
- **One QUALIFIED per day, globally.** The first track to qualify silences the
  other three (the `qualifiedToday` lockout). When two tracks satisfy their
  retest on the same confirmed bar, **ORB wins** — ORB blocks are evaluated
  before PD blocks, deliberately, because ORB is the only thread the backtest
  liked. This priority is a rule, not an accident of source order.
- Consequence, by design: a broken-but-unresolved track that gets silenced by
  another track's QUALIFIED emits no REJECT/EXPIRED. Dangling WATCH is expected
  behavior, not a bug.

QUALIFIED adds: `entry_low` (confirmation close), `entry_high`
(confirmation high), `stop`, `r1`..`r5`, `next_obstacle` (nearest computed
level beyond entry, else null), `room_r`, `score` (0–5),
`expiration_time_ct` (+3 bars default), `expiration_price`
(entry_high + 0.5R CALL / entry_low − 0.5R PUT — do-not-chase price).

REJECT/INVALIDATED/EXPIRED add `reason`, exact strings:
`failed_hold_below_level`, `failed_hold_above_level`, `risk_above_max_atr`,
`retest_window_elapsed`, `closed_through_stop_after_qualified`.

`next_obstacle` is the nearest of the seven computed levels (PDH, PDL, PDC,
PMH, PML, ORH, ORL) strictly beyond entry in the trade direction, **excluding
the broken level by NAME** — never by price equality. An ORH-long trade may
legitimately have PDH as its obstacle, and vice versa.

Signal integrity requirements (Pine side): every transition and every
`alert()` gated by `barstate.isconfirmed`; frequency
`alert.freq_once_per_bar_close`; one event per type per track per day
(done-flags), so `event_id` never repeats. The qualifying level name and price
are snapshotted at QUALIFIED so the later INVALIDATED reports the correct
level rather than re-reading a live series.

## Layout and commands

```
pine/asymmetric_live_signal.pine       indicator() — signals only
pine/asymmetric_backtest_strategy.pine strategy() — $1,100, R-based reports
signals/receiver.py                    parse -> validate -> dedupe -> card
signals/fixtures/*.json                offline test payloads
signals/test_receiver.py               plain-assert tests
```

```bash
python3 signals/test_receiver.py       # must pass
python3 signals/receiver.py --all      # every fixture renders a card
python3 signals/receiver.py --demo     # built-in samples, full pipeline
```

## Agent roles used here

- **pine-author** — writes/updates Pine sources on disk. Never uses the
  TradingView MCP; the orchestrator handles compilation (and the user
  performs the library paste — synthetic input does not work).
- **receiver-builder** — builds/updates `signals/` and must run the tests
  it writes until green before returning.
- **tester** — adversarial: feeds malformed, duplicated, wrong-side and
  stale payloads to the receiver; checks every card for the
  "NO BROKERAGE ACTION" footer and for absence of order-like language.

## Definition of done

- Pine: compiles clean under v6 (orchestrator verifies), schema exactly as
  above, safety comment block intact.
- Python: tests green, `--all` and `--demo` exit 0, no external deps.
- Any schema change lands in this file, both Pine and receiver, and the
  fixtures — in the same task.
