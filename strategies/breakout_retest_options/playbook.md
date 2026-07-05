# breakout_retest_options — the IV-rank options overlay (SPY / QQQ)

**What it is.** An *execution overlay*, not a new edge. It takes the same
break-and-retest directional trigger the `breakout_retest` equity strategy fires
(PDH/PDL break → retest → confirm; direction, spot, protective stop, fixed-2R
target, conviction grade) and decides **how to express that view in listed
options** on SPY / QQQ. It is pure, deterministic, and **RESEARCH-only** — it
emits a decision to inspect on paper. It never places orders: options are not on
the sanctioned agentic order path (CLAUDE.md P0 #3).

## The four inputs, and what each one decides

| Input | Decides | Rule |
|-------|---------|------|
| **IV rank** (0–1) | the **structure** | low → buy premium; mid → debit vertical; high → sell premium |
| **delta** | the **strikes** | pick the contract whose \|delta\| is nearest each leg's target |
| **theta** | a **gate** | reject a long-premium ticket whose decay over the hold eats the edge |
| **RI options policy** | **permission + size** | `none` blocks; `minimal` = long-only; `defined_risk_small`/`ok` = all three, size-capped |

### IV rank → structure

- **Low IVR (`< iv_rank_low`, default 0.30) → long single option.** Premium is
  cheap, so buy it outright at ~0.55 delta (band 0.45–0.60). Max loss = the debit.
  Full theta exposure — that's the price of cheap, high-gamma directional convexity.
- **Mid IVR → debit vertical.** Buy the ~0.55-delta leg, sell a ~0.30-delta leg
  further OTM. Cuts the cost *and the net theta* vs the outright; caps the upside
  at the short strike.
- **High IVR → credit spread (theta-positive).** Premium is rich, so **sell** it:
  a bull-put credit spread for a long view (sell ~0.32-delta put, buy a
  ~0.16-delta put for defined risk), a bear-call for a short view. Net theta is
  **positive** — decay now works *for* the position, which is the whole point of
  "cut theta bleed when IV is high."

### delta → strikes

Every leg is chosen by nearest-\|delta\| to its target (`long_delta_target`,
`debit_short_delta_target`, `credit_short_delta_target`, `credit_long_delta_target`).
Strike ordering is enforced per structure (bull-call short strike above the long,
bull-put long strike below the short, etc.).

### theta → gate

For long-premium tickets (long option, debit vertical) the overlay estimates
theta over the intended intraday hold (`intraday_hold_fraction` of a day) and
compares it to the first-order move-to-target profit. If
`theta_cost / expected_gross_edge > max_theta_to_edge` (default 0.35) the ticket
is rejected (`reject_on_theta_gate: true`) or flagged. Credit spreads skip the
gate — their theta is a tailwind.

### RI options policy → permission + size

Read straight from `risk/limits.yaml → level(ri).options`:

| Policy | RI | Effect |
|--------|----|--------|
| `none` | 1–3 | options blocked entirely |
| `minimal` | 4 | **long options only** (no short legs), tightest premium cap |
| `defined_risk_small` | 5 | all three structures (all are defined-risk), small size cap |
| `ok` | 6–8 | all three, normal size cap |

## Two sizing modes

### `vol_target` (default) — the equity-strategy discipline

Contracts = `floor(min(vol_target, max_loss, premium))`:

1. **vol-target** — contracts so the first-order $-loss at the *underlying* stop
   (`|delta| · stop_distance · 100`) equals the per-trade `dollar_risk`.
2. **max-loss** — worst-case *defined* loss ≤ `max_loss_mult · dollar_risk`.
3. **premium** — cash outlay / collateral ≤ `max_premium_pct · equity`.

On a small account this rounds to **zero** SPY/QQQ contracts and the overlay
honestly **skips**, reporting `min_viable_equity`:

| Account | RI / policy | Per-trade risk | 1× SPY 500c (debit ~$310) | Verdict |
|---------|-------------|----------------|---------------------------|---------|
| $1,000 | RI-6 `ok` (1.25%) | $12.50 | worst-case $310 = 24.8× budget | SKIP — needs ~$24,800 |
| $1,000 | RI-4 `minimal` (0.75%) | $7.50 | $310 = 41× budget | SKIP — needs ~$41,300 |
| $100,000 | RI-6 `ok` (1.25%) | $1,250 | 4 contracts | OK |

This is the CLAUDE.md cost-viability floor: *raise selectivity, never widen
risk.* But it answers "can I 1%-risk SPY options on $1k?" — **no** — not "can I
trade options on $1k at all."

### `premium_risk` — the small-account options unit (the `small_account_first90` profile)

The honest way a $1,000 account actually trades options: the risk unit is the
**premium you can lose** on a *defined-risk* ticket, sized as a fixed % of
equity — not 1%-at-the-underlying-stop. The profile:

- **trades only the opening 90 minutes** (`session_first_n_minutes: 90`) — the
  high-liquidity, high-participation window break-and-retest lives in; the signal
  must carry `time_et`;
- **allows 0DTE** (`min_dte: 0`) — the cheap intraday vehicle — with a ~90-minute
  theta hold (`intraday_hold_fraction: 0.20`), so decay drag is modest early;
- **sizes by premium-at-risk**: `contracts = floor(max_trade_risk_pct·equity /
  ticket_max_loss)`, with a hard ceiling `hard_max_trade_risk_pct·equity`;
- **downgrades to the cheapest defined-risk ticket** when the IV-preferred
  structure is unaffordable: ATM outright → 1-wide debit vertical → low-delta
  long → 1-wide credit spread. It will **not** let you nuke 31% on one ATM call.

Worked $1,000 example (low IV, long signal at 10:05):

| Step | Result |
|------|--------|
| IV-preferred | long ATM 500c — **$310 debit = 31%** ✗ over ceiling |
| downgraded to | **1-wide 500/501 debit vertical** — $65 max loss |
| size | **1 contract**, position max loss **$65 = 6.5%** of $1k |
| flags | `downgraded_to:debit_vertical`, `ticket_risk_exceeds_daily_halt:6.5%>2.5%` |

**The unavoidable truth, stated plainly:** the cheapest sane SPY/QQQ
defined-risk ticket risks **~6–12% of a $1,000 account** — several times the RI
daily-halt (2.5%). The overlay does not hide this; it **widens risk loudly**
(`risk_pct_of_equity` in diagnostics, `ticket_risk_exceeds_daily_halt` warning).
Trading options on $1k is only defensible with the guardrails baked into the
profile: **defined-risk only, opening-90-min only, 1–2 tickets/day, and a hard
daily stop after the first loser** (one $65 loss ≈ a normal day's halt). If even
the cheapest ticket exceeds the hard ceiling, the overlay still skips with
`no_defined_risk_ticket_under_ceiling` + `min_viable_equity`.

> Alternative worth weighing: the reason a ticket is ~7% of equity is that
> SPY/QQQ are ~$500–600 underlyings. A cheaper liquid optionable underlying
> makes each contract a smaller slice of $1k and lets the RI halts breathe — at
> the cost of leaving the SPY/QQQ break-retest universe.

## Live evaluation runner (`evaluate.py`)

`python3 -m strategies.breakout_retest_options.evaluate --side long --equity 1000 --time 10:05`
pulls a live SPY/QQQ chain + IV-rank estimate from **Polygon** and prints the
overlay's decision. It is a RESEARCH aid — it prints, it never orders.

Built around the free-tier constraints:

- **5 calls/min** — every network call is throttled by a `RateLimiter` (12s
  spacing). One evaluation ≈ 2 calls/underlying (chain snapshot + 1y daily bars),
  so SPY+QQQ ≈ 4 calls, inside a minute.
- **Greeks may be missing** on the free snapshot → filled locally with
  **Black-Scholes** from IV (`bs_greeks`); 0DTE time-to-expiry is floored so
  greeks don't blow up.
- **IV rank is a PROXY** (`iv_rank_proxy`): the current ATM IV ranked inside the
  trailing 1-year envelope of 20-day realized vol — no paid IV-history feed
  needed. Labelled a proxy everywhere it prints. Swap in a true IV-history source
  when available.
- **`--offline`** runs the whole path on a built-in fixture (no key/SDK/network) —
  the mode the tests use.

Set `POLYGON_API_KEY` (see `.env.example`). If the key's Options tier lacks IV on
the snapshot, the runner can't derive greeks and drops those contracts.

## Status

**RESEARCH.** No independent edge stats — it inherits `breakout_retest`'s trigger
and adds an options-expression layer whose real cost (options slippage, assignment,
early-exercise, the theta path) is not yet paper-validated. It must clear the same
paper→live gates as any edge, on options-realistic costs, before anything goes live.
Until then it is a decision aid, not an order source.
