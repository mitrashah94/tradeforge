"""prop/campaign.py — the FUNDED-ACCOUNT SCALING CAMPAIGN (the gradual $100k path).

One eval never pays $100k. The realistic route is a CAMPAIGN: start with a small
cash budget, buy evaluations, pass some, harvest funded payouts, REINVEST payouts
into more evals, and scale the count of concurrent funded accounts — while account
churn (funded accounts breaching the drawdown) constantly eats the fleet. This
module Monte-Carlos that whole loop end-to-end so "can $1k gradually become $100k
of payouts in 12–18 months?" gets a number instead of a vibe.

HONESTY DECISIONS (each one matters, each is stated):
  * PERFECT CORRELATION: every account (eval or funded) trades the SAME strategy
    on the SAME market days, so on a given day they all take the same fractional
    return — one bad day damages the whole fleet at once. Modeling accounts as
    independent would fake diversification that does not exist.
  * Eval accounts run the full rule set via the TESTED state machine
    (``prop.account.step_day``: daily loss, trailing/static DD, min days,
    consistency). Funded accounts mirror ``simulate.funded_year_payout``'s
    mechanics (monthly payout cycle, split, balance reset, still-live DD + daily
    loss). Payout-time consistency rules some firms apply are NOT modeled
    (slightly optimistic).
  * ``max_concurrent`` caps the fleet (real firms cap accounts per trader,
    typically ~5–20). This cap — not the edge — often binds the ceiling.
  * EOD-only P&L (the same optimistic-on-intraday caveat as ``prop/account.py``).

PURE / DETERMINISTIC: seeded RNG, no I/O.
"""

from __future__ import annotations

import numpy as np

from prop.account import IN_PROGRESS, PASSED, new_account, step_day
from prop.rules import PropRules
from prop.simulate import block_bootstrap

_PAYOUT_CYCLE_DAYS = 21   # ~monthly funded payout cadence (mirrors simulate.py)


def run_campaign(
    returns,
    rules: PropRules,
    *,
    leverage: float,
    initial_cash: float = 1000.0,
    horizon_days: int = 378,
    max_concurrent: int = 10,
    target_payout: float = 100_000.0,
    checkpoints: tuple = (252, 378),
    stagger_days: int = 0,
    n_paths: int = 500,
    block: int = 10,
    seed: int = 5,
) -> dict:
    """MC the eval->funded->reinvest campaign; report P(target) at each checkpoint.

    Each path is one WORLD: a single block-bootstrapped daily-return sequence that
    every account in the fleet experiences simultaneously. Cash policy: every day,
    while ``cash >= eval_fee`` and the fleet is below ``max_concurrent``, buy a new
    eval. Funded payouts add to cash (and to the gross-payout total — the metric
    the $100k goal is scored on). Returns P(gross payouts >= target) at each
    checkpoint day, plus payout / fee / fleet statistics at the horizon.

    ``stagger_days`` spaces eval purchases at most one per that many days.
    Same-day evals are perfect CLONES here (identical rules, identical P&L —
    they pass or bust together, one correlated coin flip), so staggering starts
    is the campaign's only real diversifier: accounts at different points in
    their eval when a bad stretch hits resolve differently. 0 keeps the naive
    buy-everything-now policy.
    """
    horizon = int(horizon_days)
    cps = tuple(int(c) for c in checkpoints if int(c) <= horizon)
    paths = block_bootstrap(returns, horizon, n_paths, block, seed)
    m = paths.shape[0]

    totals = np.zeros(m, dtype="float64")          # gross payouts at horizon
    fees = np.zeros(m, dtype="float64")
    peak_fleet = np.zeros(m, dtype="float64")
    passes = np.zeros(m, dtype="float64")          # evals passed per world
    cp_totals = {cp: np.zeros(m, dtype="float64") for cp in cps}

    fee = float(rules.eval_fee)
    static_floor = rules.account_size - rules.max_drawdown
    dl = rules.daily_loss

    stagger = max(0, int(stagger_days))

    for i in range(m):
        cash = float(initial_cash)
        gross = 0.0
        fee_spent = 0.0
        n_passed = 0
        evals: list = []      # AccountState (the tested eval state machine)
        funded: list = []     # [balance, high_water, funded_days]
        peak = 0
        last_buy_day = -10**9

        for d in range(horizon):
            # --- buy evals while cash + slots allow (fee<=0 bounded by slots);
            #     at most one purchase per `stagger` days when staggering ---
            while cash >= fee and (len(evals) + len(funded)) < max_concurrent:
                if stagger > 0 and (d - last_buy_day) < stagger:
                    break
                cash -= fee
                fee_spent += fee
                evals.append(new_account(rules))
                last_buy_day = d
                if stagger > 0:
                    break  # one per stagger window
                if fee <= 0:
                    # self-funded profile: one "account" per slot, no infinite loop
                    if (len(evals) + len(funded)) >= max_concurrent:
                        break
            peak = max(peak, len(evals) + len(funded))

            # the day's dollar P&L — identical across the fleet (perfect correlation)
            pnl = rules.account_size * float(leverage) * float(paths[i, d])

            # --- step evals through the tested state machine ---
            still: list = []
            for st in evals:
                step_day(st, pnl, rules)
                if st.status == PASSED:
                    n_passed += 1
                    funded.append([rules.account_size, rules.account_size, 0])
                elif st.status == IN_PROGRESS:
                    still.append(st)
                # FAILED -> dropped (the fee is already spent)
            evals = still

            # --- step funded accounts (mirrors funded_year_payout) ---
            alive: list = []
            for acct in funded:
                # daily-loss rule still applies when the firm has one
                if dl is not None and pnl < 0 and -pnl > dl + 1e-9:
                    continue  # account terminated
                acct[0] += pnl
                acct[1] = max(acct[1], acct[0])
                floor = (acct[1] - rules.max_drawdown) if rules.trailing else static_floor
                if acct[0] <= floor + 1e-9:
                    continue  # drawdown breach -> terminated
                acct[2] += 1
                if acct[2] % _PAYOUT_CYCLE_DAYS == 0:
                    profit = acct[0] - rules.account_size
                    if profit > rules.payout_min_profit and profit > 0:
                        pay = rules.profit_split * profit
                        gross += pay
                        cash += pay
                        acct[0] = rules.account_size   # withdraw; buffer thins
                        acct[1] = rules.account_size
                alive.append(acct)
            funded = alive

            if (d + 1) in cp_totals:
                cp_totals[d + 1][i] = gross

        totals[i] = gross
        fees[i] = fee_spent
        peak_fleet[i] = peak
        passes[i] = n_passed

    out = {
        "n_paths": m,
        "leverage": float(leverage),
        "initial_cash": float(initial_cash),
        "max_concurrent": int(max_concurrent),
        "target_payout": float(target_payout),
        "mean_gross_payout": float(totals.mean()),
        "median_gross_payout": float(np.median(totals)),
        "p90_gross_payout": float(np.percentile(totals, 90)),
        "mean_fees": float(fees.mean()),
        "mean_net_payout": float((totals - fees).mean()),
        "mean_evals_passed": float(passes.mean()),
        "mean_peak_fleet": float(peak_fleet.mean()),
    }
    for cp in cps:
        out[f"p_target_{cp}d"] = float((cp_totals[cp] >= target_payout).mean())
        out[f"median_payout_{cp}d"] = float(np.median(cp_totals[cp]))
    return out
