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
    # grade B -> RI clamps to floor 6 -> 1.25% of 100k = $1250 risk.
    # entry 100, stop 95 -> rps 5 -> shares 250.
    c = Candidate(sleeve="s", symbol="X", kind="score", grade="B",
                  score=1.0, entry_price=100.0, stop=95.0, atr=2.0)
    p = size_candidate(c, 100_000.0, LIMITS)
    assert p.shares == pytest.approx(250.0)
    assert p.dollar_risk == pytest.approx(1250.0)


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
    # grade A+ -> RI8 -> 2.0% of 100k = $2000 risk each; entry 100, stop 95.
    c = Candidate(sleeve="s", symbol=sym, kind="score", grade=grade,
                  score=score, family=family, entry_price=100.0, stop=95.0, atr=2.0)
    c.norm_score = score
    return c


def test_heat_admission_stops_at_portfolio_heat_pct():
    # Book RI = default (6): portfolio_heat_pct = 4.0% -> $4000 cap on $100k.
    # Each A+ candidate is $2000 risk -> exactly 2 admitted, the 3rd rejected.
    cands = [_score_cand("A", score=0.9), _score_cand("B", score=0.8), _score_cand("C", score=0.7)]
    opens, rejected = admit_candidates(cands, 100_000.0, LIMITS, PCFG, available_cash=1e9)
    assert [o.symbol for o in opens] == ["A", "B"]
    assert any("heat cap" in why for _c, why in rejected)


def test_family_cap_binds_below_book_cap():
    pcfg = load_portfolio_config()
    object.__setattr__(pcfg, "family_caps", {"trend": 0.02})  # 2% family cap = $2000
    cands = [_score_cand("A", score=0.9, family="trend"),
             _score_cand("B", score=0.8, family="trend")]
    opens, rejected = admit_candidates(cands, 100_000.0, LIMITS, pcfg, available_cash=1e9)
    # family 'trend' cap $2000 admits exactly one $2000 candidate.
    assert [o.symbol for o in opens] == ["A"]
    assert any("family" in why for _c, why in rejected)


def test_concurrency_cap_stops_walk():
    # 5 small candidates, grade B ($1250 each) — RI6 max_concurrent = 3.
    cands = []
    for i, sym in enumerate(["A", "B", "C", "D", "E"]):
        c = Candidate(sleeve="s", symbol=sym, kind="score", grade="B", score=1.0 - i * 0.1,
                      entry_price=1000.0, stop=995.0, atr=2.0, family=sym)
        c.norm_score = 1.0 - i * 0.1
        cands.append(c)
    opens, rejected = admit_candidates(cands, 100_000.0, LIMITS, PCFG, available_cash=1e9)
    assert len(opens) == 3
    assert any("concurrency" in why for _c, why in rejected)
