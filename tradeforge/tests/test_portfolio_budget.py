"""tests/test_portfolio_budget.py — pure-math: sizing, halts, heat/family admission.

Exercises the book-level budgeter directly: the synthetic-stop band, per-candidate
sizing for both shapes, the program-abort + dial halts, and the heat-admission
walk (book cap, per-family cap, concurrency, cash), all against the real
``risk/limits.yaml``.
"""

from __future__ import annotations

import pytest

from portfolio.budget import (
    admit_candidates,
    dial_halt,
    program_abort_halt,
    score_dollar_risk,
    size_candidate,
    synthetic_stop,
)
from portfolio.config import load_portfolio_config
from portfolio.model import BookState, Candidate
from risk.config import load_limits
from risk.sizing import per_trade_dollar_risk, resolve_ri

LIMITS = load_limits()
PCFG = load_portfolio_config()


# --------------------------------------------------------------------------- #
# synthetic stop band + sizing
# --------------------------------------------------------------------------- #
def test_synthetic_stop_uses_atr_when_available():
    assert synthetic_stop(100.0, 2.0, k=2.5, fallback_frac=0.15) == pytest.approx(95.0)


def test_synthetic_stop_falls_back_without_atr():
    assert synthetic_stop(100.0, None, k=2.5, fallback_frac=0.15) == pytest.approx(85.0)


def test_score_candidate_sizes_by_risk():
    # grade B -> resolve_ri clamps it to the operator's current floor (or stays
    # at its base tier if the floor is lower); dollar risk = that RI row's
    # per_trade_pct of equity. entry 100, stop 95 -> rps 5 -> shares = risk/rps.
    equity = 100_000.0
    entry, stop = 100.0, 95.0
    rps = entry - stop
    expected_dollar_risk = per_trade_dollar_risk(equity, resolve_ri("B", LIMITS), LIMITS)
    c = Candidate(sleeve="s", symbol="X", kind="score", grade="B",
                  score=1.0, entry_price=entry, stop=stop, atr=2.0)
    p = size_candidate(c, equity, LIMITS)
    assert p.shares == pytest.approx(expected_dollar_risk / rps)
    assert p.dollar_risk == pytest.approx(expected_dollar_risk)


def test_weight_candidate_sizes_by_target_and_synthetic_heat():
    # weight 0.2 of 100k at price 500 -> notional 20000 -> 40 shares.
    # synthetic band = 2.5 * ATR(1.0) = 2.5 -> heat = 40 * 2.5 = 100.
    c = Candidate(sleeve="s", symbol="X", kind="weight", grade="B",
                  target_weight=0.2, entry_price=500.0, atr=1.0)
    p = size_candidate(c, 100_000.0, LIMITS, synthetic_k=2.5)
    assert p.shares == pytest.approx(40.0)
    assert p.dollar_risk == pytest.approx(100.0)


def test_cash_clip_limits_notional():
    c = Candidate(sleeve="s", symbol="X", kind="weight", grade="B",
                  target_weight=1.0, entry_price=100.0, atr=1.0)
    p = size_candidate(c, 100_000.0, LIMITS, available_cash=5000.0)
    assert p.shares == pytest.approx(50.0)  # clipped to $5000 / $100


# --------------------------------------------------------------------------- #
# halts
# --------------------------------------------------------------------------- #
def test_program_abort_peak_halt():
    book = BookState(cash=0.0, peak_equity=100_000.0)
    assert program_abort_halt(64_000.0, book, LIMITS) is not None   # -36% from peak
    assert program_abort_halt(70_000.0, book, LIMITS) is None       # -30% (no peak halt)


def test_program_abort_monthly_review():
    book = BookState(cash=0.0, peak_equity=100_000.0, month_start_equity=100_000.0)
    # -22% in month triggers (>= 20% monthly review) even though < 35% peak halt.
    assert program_abort_halt(78_000.0, book, LIMITS) is not None


def test_dial_daily_halt():
    book = BookState(cash=0.0, prev_nav=100_000.0)
    # RI6 daily_halt_pct = 2.5%. -3% vs prior close -> halt.
    assert dial_halt(97_000.0, book, LIMITS) is not None
    assert dial_halt(99_000.0, book, LIMITS) is None


# --------------------------------------------------------------------------- #
# heat-admission walk
# --------------------------------------------------------------------------- #
def _score_cand(sym, grade="A+", score=1.0, family="trend"):
    # grade A+'s base tier (8) equals band_high, so it always resolves to RI8
    # regardless of the operator's floor -> its per-trade dollar risk is fixed
    # at RI8's per_trade_pct of equity; entry 100, stop 95.
    c = Candidate(sleeve="s", symbol=sym, kind="score", grade=grade,
                  score=score, family=family, entry_price=100.0, stop=95.0, atr=2.0)
    c.norm_score = score
    return c


def test_heat_admission_stops_at_portfolio_heat_pct():
    # Book heat cap comes from the operator's current floor row
    # (limits.level(limits.default_ri).portfolio_heat_pct); each A+ candidate's
    # dollar risk is fixed at RI8's per_trade_pct of equity (see _score_cand).
    # Work out how many fit under the book cap (also clamped by max_concurrent)
    # for WHATEVER floor is currently configured, then build one extra candidate
    # so the walk is guaranteed to reject something.
    equity = 100_000.0
    row = LIMITS.level(LIMITS.default_ri)
    heat_cap = row.portfolio_heat_pct / 100.0 * equity
    cand_risk = per_trade_dollar_risk(equity, resolve_ri("A+", LIMITS), LIMITS)
    n_by_heat = int(heat_cap // cand_risk)
    n_admit = min(n_by_heat, row.max_concurrent)
    assert n_admit >= 1, "test setup: no A+ candidate fits under this config's heat cap"

    names = ["A", "B", "C", "D", "E"][: n_admit + 1]
    scores = [0.9 - 0.1 * i for i in range(len(names))]
    cands = [_score_cand(sym, score=s) for sym, s in zip(names, scores)]
    opens, rejected = admit_candidates(cands, equity, LIMITS, PCFG, available_cash=1e9)
    assert [o.symbol for o in opens] == names[:n_admit]
    binding = "heat cap" if n_by_heat <= row.max_concurrent else "concurrency"
    assert any(binding in why for _c, why in rejected)


def test_family_cap_binds_below_book_cap():
    # Use grade B (not A+) so the two candidates' combined dollar risk fits
    # comfortably under the book-wide heat cap for the operator's current
    # floor -- otherwise the book cap (not the family cap under test) would be
    # the one that binds, as it does for A+ at a tight floor.
    equity = 100_000.0
    cand_risk = per_trade_dollar_risk(equity, resolve_ri("B", LIMITS), LIMITS)
    row = LIMITS.level(LIMITS.default_ri)
    book_heat_cap = row.portfolio_heat_pct / 100.0 * equity
    assert 2 * cand_risk <= book_heat_cap, (
        "test setup: the book heat cap would bind before the family cap for "
        "this config -- pick a grade with a smaller resolved dollar risk"
    )
    # a family cap between 1x and 2x a single candidate's risk admits exactly
    # one and blocks the second.
    family_cap_frac = (1.5 * cand_risk) / equity
    pcfg = load_portfolio_config()
    object.__setattr__(pcfg, "family_caps", {"trend": family_cap_frac})
    cands = [_score_cand("A", grade="B", score=0.9, family="trend"),
             _score_cand("B", grade="B", score=0.8, family="trend")]
    opens, rejected = admit_candidates(cands, equity, LIMITS, pcfg, available_cash=1e9)
    assert [o.symbol for o in opens] == ["A"]
    assert any("family" in why for _c, why in rejected)


def test_concurrency_cap_stops_walk():
    # Enough small (grade B) candidates that the book heat cap never binds, so
    # whatever admits fewer than all of them must be the concurrency cap —
    # letting the expected admitted count track the operator's current
    # max_concurrent for this RI row instead of a hardcoded number.
    equity = 100_000.0
    row = LIMITS.level(LIMITS.default_ri)
    cand_risk = per_trade_dollar_risk(equity, resolve_ri("B", LIMITS), LIMITS)
    heat_cap = row.portfolio_heat_pct / 100.0 * equity
    n = row.max_concurrent + 2
    assert row.max_concurrent * cand_risk <= heat_cap + 1e-9, (
        "test setup: the heat cap would bind before concurrency for this config"
    )
    cands = []
    for i in range(n):
        sym = chr(ord("A") + i)
        c = Candidate(sleeve="s", symbol=sym, kind="score", grade="B", score=1.0 - i * 0.1,
                      entry_price=1000.0, stop=995.0, atr=2.0, family=sym)
        c.norm_score = 1.0 - i * 0.1
        cands.append(c)
    opens, rejected = admit_candidates(cands, equity, LIMITS, PCFG, available_cash=1e9)
    assert len(opens) == row.max_concurrent
    assert any("concurrency" in why for _c, why in rejected)
