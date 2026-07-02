"""prop/account.py — the deterministic funded-account STATE MACHINE.

Steps an account day-by-day through a sequence of daily P&L (dollars), enforcing a
:class:`~prop.rules.PropRules` exactly as a prop firm would, and resolving to
PASSED / FAILED / IN_PROGRESS. This is the honest core: the firm's rules are hard
breakers (the same shape as ``risk/limits.yaml``'s daily-halt + program-abort),
and a strategy either survives them to the target or it doesn't.

DAILY-DATA CAVEAT (stated, not hidden): a firm's daily-loss and trailing-DD limits
are INTRADAY (they can trip mid-session even if the day closes flat). With
end-of-day P&L we only see the net, so the EOD checks here are OPTIMISTIC on those
intraday breaches — the real P(pass) is somewhat lower than an EOD sim implies.
Pass an intraday MAE/low series (``day_low_pnl``) to :func:`step_day` to tighten
this when the data supports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from prop.rules import PropRules

IN_PROGRESS = "IN_PROGRESS"
PASSED = "PASSED"
FAILED = "FAILED"


@dataclass
class AccountState:
    """The account's running state as it steps through days.

    ``balance`` is realized equity (we treat each day's P&L as realized EOD).
    ``high_water`` tracks the peak balance for a TRAILING drawdown threshold.
    ``max_single_day_profit`` feeds the consistency rule checked at pass time.
    """

    balance: float
    high_water: float
    day: int = 0
    trading_days: int = 0
    max_single_day_profit: float = 0.0
    status: str = IN_PROGRESS
    fail_reason: str = ""
    resolved_day: Optional[int] = None

    @property
    def profit(self) -> float:
        return self.balance - self._start

    _start: float = field(default=0.0, repr=False)


def new_account(rules: PropRules) -> AccountState:
    """A fresh eval account at the firm's starting balance."""
    a = AccountState(balance=rules.account_size, high_water=rules.account_size)
    a._start = rules.account_size
    return a


def _drawdown_floor(state: AccountState, rules: PropRules) -> float:
    """The balance level at/below which the account is BLOWN.

    Trailing: ``high_water - max_drawdown`` (follows the peak up, never down).
    Static:   ``account_size - max_drawdown`` (fixed from the start).
    """
    if rules.trailing:
        return state.high_water - rules.max_drawdown
    return rules.account_size - rules.max_drawdown


def step_day(
    state: AccountState,
    pnl: float,
    rules: PropRules,
    *,
    is_trading_day: bool = True,
    day_low_pnl: Optional[float] = None,
) -> AccountState:
    """Advance ``state`` by one day of P&L, enforcing every rule. Idempotent once resolved.

    Order of checks (fail conditions win over a same-day pass — conservative):
      1. DAILY-LOSS breach — the day's loss (or intraday low ``day_low_pnl`` if
         supplied) beyond ``daily_loss`` -> FAILED;
      2. MAX-DRAWDOWN breach — balance at/below the (trailing or static) floor
         (checked on the intraday low when supplied) -> FAILED;
      3. PROFIT TARGET — balance >= target AND enough trading days AND consistency
         holds -> PASSED.
    A resolved account is returned unchanged.
    """
    if state.status != IN_PROGRESS:
        return state
    state.day += 1
    if is_trading_day:
        state.trading_days += 1

    # 1. daily-loss breach (use the intraday low if given, else the EOD net).
    dl = rules.daily_loss
    if dl is not None:
        worst = day_low_pnl if day_low_pnl is not None else pnl
        if worst < 0 and -worst > dl + 1e-9:
            state.status = FAILED
            state.fail_reason = f"daily-loss breach: {worst:.2f} beyond -{dl:.2f}"
            state.resolved_day = state.day
            return state

    # intraday drawdown breach (before booking the EOD balance) if a low is given.
    if day_low_pnl is not None:
        intraday_balance = state.balance + day_low_pnl
        if intraday_balance <= _drawdown_floor(state, rules) + 1e-9:
            state.status = FAILED
            state.fail_reason = (
                f"intraday max-drawdown breach: {intraday_balance:.2f} "
                f"<= floor {_drawdown_floor(state, rules):.2f}"
            )
            state.resolved_day = state.day
            return state

    # book the day's realized P&L.
    state.balance += pnl
    if pnl > state.max_single_day_profit:
        state.max_single_day_profit = pnl
    if state.balance > state.high_water:
        state.high_water = state.balance

    # 2. end-of-day max-drawdown breach.
    if state.balance <= _drawdown_floor(state, rules) + 1e-9:
        state.status = FAILED
        state.fail_reason = (
            f"max-drawdown breach: balance {state.balance:.2f} "
            f"<= floor {_drawdown_floor(state, rules):.2f}"
        )
        state.resolved_day = state.day
        return state

    # 3. profit target (+ min days + consistency).
    if state.balance >= rules.target_balance - 1e-9:
        if state.trading_days >= rules.min_trading_days and _consistency_ok(state, rules):
            state.status = PASSED
            state.resolved_day = state.day
    return state


def _consistency_ok(state: AccountState, rules: PropRules) -> bool:
    """True if no single day's profit dominates the total beyond the consistency cap."""
    if rules.consistency_pct is None:
        return True
    total = state.balance - rules.account_size
    if total <= 0:
        return True
    return state.max_single_day_profit <= rules.consistency_pct * total + 1e-9


@dataclass
class EvalOutcome:
    """The result of running one daily-P&L path through the eval state machine."""

    status: str                 # PASSED | FAILED | IN_PROGRESS
    days: int                   # days elapsed at resolution (or the full path)
    trading_days: int
    final_balance: float
    profit: float
    fail_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASSED

    @property
    def failed(self) -> bool:
        return self.status == FAILED


def run_eval(
    daily_pnl: Sequence[float],
    rules: PropRules,
    *,
    day_lows: Optional[Sequence[float]] = None,
) -> EvalOutcome:
    """Run one full daily-P&L path through the eval and return the :class:`EvalOutcome`.

    Stops early on the resolving day (pass or fail). ``day_lows`` optionally supplies
    each day's intraday-low P&L for the stricter daily-loss / trailing-DD checks.
    """
    state = new_account(rules)
    for i, pnl in enumerate(daily_pnl):
        low = day_lows[i] if day_lows is not None else None
        step_day(state, float(pnl), rules, day_low_pnl=(None if low is None else float(low)))
        if state.status != IN_PROGRESS:
            break
    return EvalOutcome(
        status=state.status,
        days=state.resolved_day if state.resolved_day is not None else state.day,
        trading_days=state.trading_days,
        final_balance=state.balance,
        profit=state.balance - rules.account_size,
        fail_reason=state.fail_reason,
    )
