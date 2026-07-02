"""tests/test_prop_account.py — prop rules loader + the eval STATE MACHINE.

Deterministic, offline, hand-computed. Asserts the firm rules resolve to the right
dollar levels and that the account state machine enforces each rule exactly: pass
on the profit target (with min-days + consistency), fail on a daily-loss breach,
fail on a static AND a trailing max-drawdown breach, and the consistency rule
blocking a one-lucky-day pass.
"""

from __future__ import annotations

import pytest

from prop.account import FAILED, IN_PROGRESS, PASSED, new_account, run_eval, step_day
from prop.rules import PropRules, list_firms, load_firm


# --------------------------------------------------------------------------- #
# rules loader + derived dollar levels
# --------------------------------------------------------------------------- #
def test_load_firm_derived_levels():
    r = load_firm("futures_50k")
    assert r.account_size == 50_000
    assert r.profit_target == pytest.approx(3_000)     # 6%
    assert r.max_drawdown == pytest.approx(2_500)      # 5%
    assert r.target_balance == pytest.approx(53_000)
    assert r.trailing is True
    assert r.daily_loss is None                        # futures preset: none
    assert "fx_10k" in list_firms() and "equities_25k" in list_firms()


def test_fx_daily_loss_level():
    r = load_firm("fx_10k")
    assert r.daily_loss == pytest.approx(500)          # 5% of 10k
    assert r.trailing is False


# --------------------------------------------------------------------------- #
# the state machine
# --------------------------------------------------------------------------- #
def _futures():
    return load_firm("futures_50k")


def test_pass_requires_target_and_min_days():
    r = _futures()  # target +3000, min 7 days, consistency 0.30
    # +500/day: balance hits 53000 at day 6 (< 7 days) -> NOT passed yet.
    st = new_account(r)
    for _ in range(6):
        step_day(st, 500.0, r)
    assert st.balance == pytest.approx(53_000)
    assert st.status == IN_PROGRESS          # target hit but only 6 trading days
    # day 7 clears the min-days gate -> PASSED.
    step_day(st, 500.0, r)
    assert st.status == PASSED and st.resolved_day == 7


def test_fail_on_trailing_drawdown():
    r = _futures()  # trailing, maxDD 2500
    st = new_account(r)
    step_day(st, 2000.0, r)                  # balance 52000, HWM 52000, floor 49500
    assert st.high_water == pytest.approx(52_000)
    step_day(st, -2600.0, r)                 # 49400 <= 49500 -> blown
    assert st.status == FAILED and "max-drawdown" in st.fail_reason


def test_fail_on_static_drawdown_accumulated():
    r = load_firm("equities_25k")            # static maxDD 2000 (floor 23000), daily 1000
    st = new_account(r)
    for pnl in (-900.0, -900.0, -300.0):     # each within the 1000 daily cap
        step_day(st, pnl, r)
    assert st.balance == pytest.approx(22_900)   # 25000 - 2100 <= 23000
    assert st.status == FAILED and "max-drawdown" in st.fail_reason


def test_fail_on_daily_loss():
    r = load_firm("fx_10k")                  # daily loss 500
    st = new_account(r)
    step_day(st, -600.0, r)                  # single-day loss beyond the cap
    assert st.status == FAILED and "daily-loss" in st.fail_reason


def test_consistency_blocks_one_lucky_day():
    r = _futures()  # consistency 0.30
    st = new_account(r)
    step_day(st, 2000.0, r)                  # one dominant day
    for _ in range(6):
        step_day(st, 200.0, r)               # balance 53200, profit 3200, 7 days
    # max single day 2000 > 0.30 * 3200 = 960 -> consistency fails the pass.
    assert st.balance == pytest.approx(53_200)
    assert st.status == IN_PROGRESS


def test_intraday_low_can_fail_even_if_eod_flat():
    r = load_firm("fx_10k")                  # daily loss 500
    st = new_account(r)
    # EOD net 0 but the intraday low dipped -700 -> the honest check fails it.
    step_day(st, 0.0, r, day_low_pnl=-700.0)
    assert st.status == FAILED and "daily-loss" in st.fail_reason


def test_run_eval_stops_on_resolution():
    r = _futures()
    # +500 x 7 passes at day 7; extra days after are ignored.
    out = run_eval([500.0] * 20, r)
    assert out.passed and out.days == 7 and out.trading_days == 7
    assert out.profit == pytest.approx(3_500)
