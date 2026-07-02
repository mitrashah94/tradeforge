"""tests/test_prop_campaign.py — compound-to-target MC + the scaling campaign.

Deterministic, offline, hand-computed on CONSTANT return streams (the block
bootstrap is degenerate there, so every path is identical and the arithmetic is
exact): compound growth timing, ruin, the campaign's eval->funded->payout->
reinvest loop, the concurrency cap, and the $1k-account scale-invariance
(P(pass) identical to the $25k profile; dollars 1/25th).
"""

from __future__ import annotations

import numpy as np
import pytest

from prop.campaign import run_campaign
from prop.rules import load_firm
from prop.simulate import compound_to_target, monte_carlo_eval


# --------------------------------------------------------------------------- #
# compound_to_target — the personal moonshot math
# --------------------------------------------------------------------------- #
def test_compound_hits_target_at_sufficient_growth():
    # 2%/day compounded: 1000 * 1.02^d >= 100000 -> d >= ln(100)/ln(1.02) = 232.6 -> 233.
    res = compound_to_target([0.02], initial=1000, target=100_000, leverage=1.0,
                             days=378, n_paths=10, seed=0)
    assert res["p_target"] == pytest.approx(1.0)
    assert res["median_days_to_target"] == pytest.approx(233)


def test_compound_misses_target_when_growth_too_slow():
    # 1%/day needs 463 days > 378 -> never hits; terminal = 1000 * 1.01^378.
    res = compound_to_target([0.01], initial=1000, target=100_000, leverage=1.0,
                             days=378, n_paths=10, seed=0)
    assert res["p_target"] == pytest.approx(0.0)
    assert res["median_terminal"] == pytest.approx(1000 * 1.01 ** 378, rel=1e-9)


def test_compound_ruin_on_wipeout():
    # -60% day at 2x leverage = -120% -> equity <= 0 -> ruin on day 1.
    res = compound_to_target([-0.60], initial=1000, target=100_000, leverage=2.0,
                             days=10, n_paths=5, seed=0)
    assert res["p_ruin"] == pytest.approx(1.0)
    assert res["median_terminal"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# personal_1k profile — scale invariance
# --------------------------------------------------------------------------- #
def test_personal_1k_scale_invariance_of_p_pass():
    # Same percentage rules as equities_25k -> identical P(pass) on the same
    # stream/seed; only the dollar levels differ (1/25th).
    p1k = load_firm("personal_1k")
    p25k = load_firm("equities_25k")
    assert p1k.profit_target == pytest.approx(80.0)
    assert p1k.max_drawdown == pytest.approx(80.0)
    r = [0.002, -0.001, 0.003, -0.002, 0.004, 0.001] * 10
    a = monte_carlo_eval(r, p1k, leverage=4.0, horizon=40, n_paths=200, seed=9)
    b = monte_carlo_eval(r, p25k, leverage=4.0, horizon=40, n_paths=200, seed=9)
    assert a["p_pass"] == pytest.approx(b["p_pass"])
    assert a["p_fail"] == pytest.approx(b["p_fail"])


# --------------------------------------------------------------------------- #
# the campaign — exact arithmetic on a constant winning stream
# --------------------------------------------------------------------------- #
def test_campaign_exact_arithmetic_constant_winner():
    r = load_firm("futures_50k")   # fee 150, target +3000, min 7d, split 0.90, trail 2500
    # +0.004/day at 3x -> +600/day/account. $1000 buys 6 evals on day 0 (cash 100).
    # Each eval: +600*5=3000 at day 5 but min_trading_days=7 -> PASSES day 7
    # (consistency: 600 <= 0.30*4200). Funded from day 8: +600/day; payout cycle
    # at 21 funded days -> profit 12600 -> pay 0.90*12600 = 11340/account/cycle.
    # Horizon 49: funded days 8..49 = 42 -> exactly 2 cycles/account.
    # BUT the first payout (day ~28) refills cash (11340*6) -> 4 more slots bought
    # (cap 10) which ALSO pass and pay by day 49? Their eval starts day 29-ish...
    # keep it simple: cap the fleet at 6 so no reinvestment expansion is possible.
    res = run_campaign([0.004] * 8, r, leverage=3.0, initial_cash=1000.0,
                       horizon_days=49, max_concurrent=6, target_payout=100_000.0,
                       checkpoints=(49,), n_paths=4, block=5, seed=1)
    assert res["mean_evals_passed"] == pytest.approx(6.0)
    assert res["mean_fees"] == pytest.approx(900.0)
    assert res["mean_gross_payout"] == pytest.approx(2 * 11_340.0 * 6, rel=1e-9)  # 136,080
    assert res["p_target_49d"] == pytest.approx(1.0)   # 136k >= 100k
    assert res["mean_peak_fleet"] == pytest.approx(6.0)


def test_campaign_losing_stream_pays_nothing():
    r = load_firm("futures_50k")
    res = run_campaign([-0.004] * 8, r, leverage=3.0, initial_cash=1000.0,
                       horizon_days=40, max_concurrent=6, checkpoints=(40,),
                       n_paths=4, block=5, seed=1)
    assert res["mean_gross_payout"] == pytest.approx(0.0)
    assert res["p_target_40d"] == pytest.approx(0.0)
    # every eval busts on the trailing floor; fees are the only cash out the door.
    assert res["mean_fees"] >= 900.0


def test_campaign_stagger_spaces_purchases():
    r = load_firm("futures_50k")
    # stagger 5: purchases land on days 0, 5, 10, ... -> after 12 days only 3
    # evals exist (cash allows 6). Constant winner so nothing busts meanwhile.
    res = run_campaign([0.004] * 8, r, leverage=3.0, initial_cash=1000.0,
                       horizon_days=12, max_concurrent=6, checkpoints=(12,),
                       stagger_days=5, n_paths=2, block=5, seed=1)
    assert res["mean_fees"] == pytest.approx(3 * 150.0)
    assert res["mean_peak_fleet"] == pytest.approx(3.0)


def test_campaign_reinvests_payouts_into_more_evals():
    r = load_firm("futures_50k")
    # Same winner but cap 10: the first payout wave (day ~28) buys 4 more evals.
    res = run_campaign([0.004] * 8, r, leverage=3.0, initial_cash=1000.0,
                       horizon_days=80, max_concurrent=10, checkpoints=(80,),
                       n_paths=4, block=5, seed=1)
    assert res["mean_evals_passed"] == pytest.approx(10.0)   # 6 initial + 4 reinvested
    assert res["mean_peak_fleet"] == pytest.approx(10.0)
    assert res["mean_gross_payout"] > 2 * 11_340.0 * 6       # more than the capped case
