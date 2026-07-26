"""prop/simulate.py — P(pass), payout distribution, and end-to-end EV of the prop path.

Turns a strategy's fractional daily-return stream into the money question:
  * scale returns to dollar P&L on the funded account (a LEVERAGE knob — how hard
    you press the edge to reach the target without breaching the drawdown);
  * run the eval STATE MACHINE historically (rolling real windows) AND by
    block-bootstrap Monte-Carlo (preserving vol clustering) -> P(pass) / P(bust) /
    time-to-pass;
  * model the FUNDED phase (monthly payout cycles under the same DD rule that can
    still kill the account) -> the annual-payout distribution per account;
  * combine into the END-TO-END EV net of the eval fee, and answer "how many
    funded accounts to reach a $100k payout year" — the honest comparison against
    a $1k moonshot.

The LEVERAGE SWEEP is the maximization thesis applied to the eval: too little and
you never hit the target; too much and you breach the drawdown — there is an
interior size that maximizes P(pass) / EV. PURE / DETERMINISTIC: seeded RNG, no
I/O, no scipy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from prop.account import PASSED, run_eval
from prop.rules import PropRules

TRADING_DAYS_PER_YEAR = 252
_PAYOUT_CYCLE_DAYS = 21   # ~monthly funded payout cadence


# --------------------------------------------------------------------------- #
# returns -> dollar P&L
# --------------------------------------------------------------------------- #
def returns_to_pnl(returns: Sequence[float], account_size: float, leverage: float) -> np.ndarray:
    """Daily dollar P&L = ``account_size * leverage * return`` per day."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    return account_size * float(leverage) * r


# --------------------------------------------------------------------------- #
# block bootstrap (preserves short-run autocorrelation / vol clustering)
# --------------------------------------------------------------------------- #
def block_bootstrap(
    returns: Sequence[float], horizon: int, n_paths: int, block: int, seed: int
) -> np.ndarray:
    """``(n_paths, horizon)`` array of resampled returns via the moving-block bootstrap."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return np.zeros((n_paths, horizon))
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    out = np.empty((n_paths, n_blocks * block), dtype="float64")
    max_start = max(1, n - block + 1)
    for p in range(n_paths):
        starts = rng.integers(0, max_start, size=n_blocks)
        chunks = [r[s:s + block] for s in starts]
        # a block near the tail may be short; pad by wrapping.
        flat = np.concatenate([
            c if c.size == block else np.resize(c, block) for c in chunks
        ])
        out[p] = flat
    return out[:, :horizon]


# --------------------------------------------------------------------------- #
# eval: historical rolling windows + Monte-Carlo
# --------------------------------------------------------------------------- #
def historical_eval(
    returns: Sequence[float], rules: PropRules, *, leverage: float, horizon: int = 60,
) -> dict:
    """Roll every start offset of the REAL return stream through the eval.

    For each start index we feed the next ``horizon`` days (or until the account
    resolves) into the state machine. Reports the empirical pass / fail / unresolved
    fractions over all real windows — the honest, assumption-free view (limited by
    how many independent windows the history holds).
    """
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"n_windows": 0, "p_pass": float("nan"), "p_fail": float("nan"),
                "p_unresolved": float("nan"), "median_days_to_pass": float("nan")}
    starts = range(0, max(1, n - horizon + 1))
    passes = fails = unresolved = 0
    days_to_pass: list[int] = []
    n_windows = 0
    for s in starts:
        pnl = returns_to_pnl(r[s:s + horizon], rules.account_size, leverage)
        out = run_eval(pnl, rules)
        n_windows += 1
        if out.passed:
            passes += 1
            days_to_pass.append(out.days)
        elif out.failed:
            fails += 1
        else:
            unresolved += 1
    return {
        "n_windows": n_windows,
        "p_pass": passes / n_windows,
        "p_fail": fails / n_windows,
        "p_unresolved": unresolved / n_windows,
        "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else float("nan"),
    }


def monte_carlo_eval(
    returns: Sequence[float], rules: PropRules, *, leverage: float,
    horizon: int = 60, n_paths: int = 2000, block: int = 5, seed: int = 0,
) -> dict:
    """Block-bootstrap MC of the eval -> P(pass) / P(bust) / days-to-pass distribution."""
    paths = block_bootstrap(returns, horizon, n_paths, block, seed)
    passes = fails = unresolved = 0
    days_to_pass: list[int] = []
    for i in range(paths.shape[0]):
        pnl = returns_to_pnl(paths[i], rules.account_size, leverage)
        out = run_eval(pnl, rules)
        if out.passed:
            passes += 1
            days_to_pass.append(out.days)
        elif out.failed:
            fails += 1
        else:
            unresolved += 1
    m = paths.shape[0]
    return {
        "n_paths": m,
        "p_pass": passes / m,
        "p_fail": fails / m,
        "p_unresolved": unresolved / m,
        "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else float("nan"),
    }


# --------------------------------------------------------------------------- #
# funded phase: monthly payout cycles under the still-live DD rule
# --------------------------------------------------------------------------- #
def funded_year_payout(
    returns: Sequence[float], rules: PropRules, *, leverage: float,
    days: int = TRADING_DAYS_PER_YEAR, n_paths: int = 2000, block: int = 5, seed: int = 1,
) -> dict:
    """MC of the FUNDED year -> the annual-payout distribution per account.

    Each path trades ``days`` funded days at ``leverage``. Every ``_PAYOUT_CYCLE_DAYS``
    the profit above the starting balance (if it clears ``payout_min_profit``) is
    WITHDRAWN — the trader keeps ``profit_split`` of it — and the balance resets to
    the account size (so the trailing floor stays near the start; withdrawing thins
    the buffer, exactly as in reality). If the (trailing or static) drawdown floor
    is breached the account is TERMINATED and payouts stop. Reports mean / median /
    p10 / p90 annual payout and the account survival rate.
    """
    paths = block_bootstrap(returns, days, n_paths, block, seed)
    payouts = np.zeros(paths.shape[0], dtype="float64")
    survived = 0
    floor_static = rules.account_size - rules.max_drawdown
    for i in range(paths.shape[0]):
        pnl = returns_to_pnl(paths[i], rules.account_size, leverage)
        balance = rules.account_size
        high_water = rules.account_size
        total_payout = 0.0
        alive = True
        for d in range(pnl.size):
            balance += pnl[d]
            high_water = max(high_water, balance)
            floor = (high_water - rules.max_drawdown) if rules.trailing else floor_static
            if balance <= floor + 1e-9:
                alive = False
                break
            if (d + 1) % _PAYOUT_CYCLE_DAYS == 0:
                profit = balance - rules.account_size
                if profit > rules.payout_min_profit and profit > 0:
                    total_payout += rules.profit_split * profit
                    balance = rules.account_size  # withdraw profit; reset buffer
                    high_water = rules.account_size
        payouts[i] = total_payout
        if alive:
            survived += 1
    return {
        "n_paths": paths.shape[0],
        "mean_payout": float(payouts.mean()),
        "median_payout": float(np.median(payouts)),
        "p10_payout": float(np.percentile(payouts, 10)),
        "p90_payout": float(np.percentile(payouts, 90)),
        "survival_rate": survived / paths.shape[0],
    }


# --------------------------------------------------------------------------- #
# end-to-end EV + the $100k-payout scaling answer
# --------------------------------------------------------------------------- #
def expected_value(
    returns: Sequence[float], rules: PropRules, *, leverage: float,
    horizon: int = 60, funded_days: int = TRADING_DAYS_PER_YEAR,
    n_paths: int = 2000, block: int = 5, seed: int = 0,
    target_payout: float = 100_000.0,
) -> dict:
    """Combine eval P(pass) + funded payout into the honest EV of ONE eval attempt.

    ``ev_one_attempt = p_pass * E[annual payout | funded] - eval_fee`` (a single
    paid attempt). Also reports the expected fee to eventually pass (geometric,
    ``eval_fee / p_pass``) and how many CONCURRENT funded accounts the mean payout
    implies to reach ``target_payout`` in a year — the direct comparison against a
    $1k moonshot.
    """
    ev = monte_carlo_eval(returns, rules, leverage=leverage, horizon=horizon,
                          n_paths=n_paths, block=block, seed=seed)
    fund = funded_year_payout(returns, rules, leverage=leverage, days=funded_days,
                              n_paths=n_paths, block=block, seed=seed + 1)
    p_pass = ev["p_pass"]
    mean_payout = fund["mean_payout"]
    ev_one_attempt = p_pass * mean_payout - rules.eval_fee
    expected_fee_to_pass = (rules.eval_fee / p_pass) if p_pass > 0 else float("inf")
    accounts_for_target = (
        int(np.ceil(target_payout / mean_payout)) if mean_payout > 0 else None
    )
    return {
        "firm": rules.name,
        "leverage": float(leverage),
        "p_pass": p_pass,
        "p_fail": ev["p_fail"],
        "median_days_to_pass": ev["median_days_to_pass"],
        "mean_annual_payout_per_account": mean_payout,
        "median_annual_payout_per_account": fund["median_payout"],
        "funded_survival_rate": fund["survival_rate"],
        "eval_fee": rules.eval_fee,
        "ev_one_attempt": ev_one_attempt,
        "expected_fee_to_pass": expected_fee_to_pass,
        "accounts_for_target_payout": accounts_for_target,
        "target_payout": target_payout,
    }


def compound_to_target(
    returns: Sequence[float],
    *,
    initial: float = 1000.0,
    target: float = 100_000.0,
    leverage: float = 1.0,
    days: int = 378,
    n_paths: int = 2000,
    block: int = 5,
    seed: int = 2,
    ruin_frac: float = 0.0,
) -> dict:
    """MC of compounding a PERSONAL account toward ``target`` (the moonshot math).

    Each path compounds ``initial`` at ``equity *= 1 + leverage*r`` day by day.
    A path RESOLVES when equity reaches ``target`` (hit) or falls to/below
    ``ruin_frac * initial`` (ruin; the default 0.0 means only a true wipeout stops
    it — the most charitable "ride it to zero" assumption). ``days=378`` ≈ 1.5
    trading years. Reports P(hit target), P(ruin), median time-to-target among
    hitters, and the terminal-wealth distribution — the honest read on
    "$1k -> $100k" at any aggression. Margin/assignment mechanics are ignored
    (charitable again): this is an UPPER BOUND on the real probability.
    """
    paths = block_bootstrap(returns, days, n_paths, block, seed)
    ruin_level = max(0.0, float(ruin_frac) * float(initial))
    hits = ruins = 0
    t_hit: list[int] = []
    terminals = np.empty(paths.shape[0], dtype="float64")
    for i in range(paths.shape[0]):
        eq = float(initial)
        resolved = False
        for d in range(paths.shape[1]):
            eq *= (1.0 + float(leverage) * paths[i, d])
            if eq <= ruin_level or eq <= 0.0:
                ruins += 1
                eq = max(eq, 0.0)
                resolved = True
                break
            if eq >= target:
                hits += 1
                t_hit.append(d + 1)
                resolved = True
                break
        terminals[i] = eq
    m = paths.shape[0]
    return {
        "initial": float(initial),
        "target": float(target),
        "leverage": float(leverage),
        "days": int(days),
        "p_target": hits / m,
        "p_ruin": ruins / m,
        "median_days_to_target": float(np.median(t_hit)) if t_hit else float("nan"),
        "median_terminal": float(np.median(terminals)),
        "p90_terminal": float(np.percentile(terminals, 90)),
        "p99_terminal": float(np.percentile(terminals, 99)),
    }


def sweep_leverage(
    returns: Sequence[float], rules: PropRules, *,
    leverages: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
    horizon: int = 60, n_paths: int = 2000, block: int = 5, seed: int = 0,
) -> list[dict]:
    """P(pass) / P(bust) / EV across leverages -> the interior best size.

    The maximization sweep: returns one row per leverage so the caller can pick the
    size that maximizes P(pass) (or EV). Sorted by leverage ascending.
    """
    rows: list[dict] = []
    for lev in leverages:
        e = expected_value(returns, rules, leverage=lev, horizon=horizon,
                           n_paths=n_paths, block=block, seed=seed)
        rows.append(e)
    return rows
