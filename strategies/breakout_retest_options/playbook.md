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

## Sizing — three caps, take the minimum

Contracts = `floor(min(vol_target, max_loss, premium))`:

1. **vol-target** — contracts so the first-order $-loss at the *underlying* stop
   (`|delta| · stop_distance · 100`) equals the per-trade `dollar_risk`. Same
   risk discipline as the equity strategy: you exit at the technical stop.
2. **max-loss** — worst-case *defined* loss ≤ `max_loss_mult · dollar_risk`
   (options gap; the stop may not fill at the modeled price).
3. **premium** — cash outlay / collateral ≤ `max_premium_pct · equity`.

## The $1,000 reality (why this overlay mostly says "skip")

This is the honest, load-bearing finding for a small account — the overlay
surfaces it instead of hiding it behind a rounded-up 1-lot:

| Account | RI / policy | Per-trade risk | 1× SPY 500c (debit ~$310) | Verdict |
|---------|-------------|----------------|---------------------------|---------|
| **$1,000** | RI-6 `ok` (1.25%) | **$12.50** | worst-case loss $310 = **24.8× budget** | **SKIP** — needs ~**$24,800** for 1 contract |
| **$1,000** | RI-4 `minimal` (0.75%) | **$7.50** | worst-case loss $310 = **41× budget** | **SKIP** — needs ~**$41,300** |
| **$100,000** | RI-6 `ok` (1.25%) | $1,250 | 4 contracts within all caps | **OK** |

One SPY/QQQ contract's worst-case loss ($30–$300+) dwarfs a disciplined 1%-of-$1k
per-trade budget. So on a $1,000 account the overlay **skips and reports
`min_viable_equity`** plus the binding constraint. This is the cost-viability
floor from CLAUDE.md in action: *raise selectivity, never widen risk.* The
mitigations it points at — cheaper defined-risk spreads (lower per-contract max
loss), lower-delta strikes, or simply waiting until equity compounds past the
`min_viable_equity` — are all "trade less / trade smaller," never "risk more."

`allow_min_ticket: true` (default **false**) overrides the skip to take exactly
one defined-risk contract, stamped `over_budget` with the actual risk %, for
explicit paper study only.

## Status

**RESEARCH.** No independent edge stats — it inherits `breakout_retest`'s trigger
and adds an options-expression layer whose real cost (options slippage, assignment,
early-exercise, the theta path) is not yet paper-validated. It must clear the same
paper→live gates as any edge, on options-realistic costs, before anything goes live.
Until then it is a decision aid, not an order source.
