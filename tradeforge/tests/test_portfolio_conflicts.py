"""tests/test_portfolio_conflicts.py — pure-math: clustering, grade order, dedup, resolve.

No DB, no engine — exercises the correlation-clustering union-find (the primitive
shared with the watchlist), the conviction-grade ordering, the cluster dedup, and
the same-symbol conflict resolution directly on hand-built inputs.
"""

from __future__ import annotations

import pandas as pd

from portfolio.conflicts import (
    cluster_dedup,
    correlation_clusters,
    grade_rank,
    resolve_conflicts,
)
from portfolio.model import Candidate


def _corr(mapping):
    """Build a square correlation DataFrame from {a: {b: rho}}."""
    syms = list(mapping.keys())
    df = pd.DataFrame(index=syms, columns=syms, dtype="float64")
    for a in syms:
        for b in syms:
            df.at[a, b] = mapping[a].get(b, 0.0)
    for s in syms:
        df.at[s, s] = 1.0
    return df


# --------------------------------------------------------------------------- #
# union-find clustering
# --------------------------------------------------------------------------- #
def test_clusters_link_high_correlation():
    corr = _corr({
        "AAA": {"BBB": 0.95, "CCC": 0.1},
        "BBB": {"AAA": 0.95, "CCC": 0.1},
        "CCC": {"AAA": 0.1, "BBB": 0.1},
    })
    clusters = correlation_clusters(["AAA", "BBB", "CCC"], corr, threshold=0.8)
    # AAA & BBB collapse (0.95 >= 0.8); CCC is its own cluster.
    assert clusters["AAA"] == clusters["BBB"]
    assert clusters["CCC"] != clusters["AAA"]


def test_clusters_absolute_value_links_inverse_pair():
    # A strong NEGATIVE correlation is the same risk factor inverted -> one cluster.
    corr = _corr({"SPY": {"SH": -0.98}, "SH": {"SPY": -0.98}})
    clusters = correlation_clusters(["SPY", "SH"], corr, threshold=0.8)
    assert clusters["SPY"] == clusters["SH"]


def test_clusters_transitive_single_linkage():
    # A~B and B~C (single linkage) -> A,B,C all one cluster even if A~C is low.
    corr = _corr({
        "A": {"B": 0.9, "C": 0.2},
        "B": {"A": 0.9, "C": 0.9},
        "C": {"A": 0.2, "B": 0.9},
    })
    clusters = correlation_clusters(["A", "B", "C"], corr, threshold=0.8)
    assert clusters["A"] == clusters["B"] == clusters["C"]


def test_clusters_absent_symbols_are_singletons():
    clusters = correlation_clusters(["X", "Y"], None, threshold=0.8)
    assert clusters["X"] != clusters["Y"]


# --------------------------------------------------------------------------- #
# grade ordering
# --------------------------------------------------------------------------- #
def test_grade_rank_order():
    assert grade_rank("A+") > grade_rank("A") > grade_rank("B") > grade_rank("?")


# --------------------------------------------------------------------------- #
# same-symbol resolve (#6)
# --------------------------------------------------------------------------- #
def test_resolve_same_symbol_higher_grade_wins():
    a = Candidate(sleeve="brk", symbol="SPY", kind="score", grade="A+", score=0.1, norm_score=0.1)
    b = Candidate(sleeve="rot", symbol="SPY", kind="weight", grade="B", target_weight=0.5, norm_score=0.9)
    kept, rejected = resolve_conflicts([a, b])
    assert len(kept) == 1 and kept[0].grade == "A+" and kept[0].sleeve == "brk"
    assert len(rejected) == 1 and rejected[0][0].sleeve == "rot"


def test_resolve_same_symbol_tie_grade_breaks_on_score():
    a = Candidate(sleeve="s1", symbol="X", kind="score", grade="A", score=0.2, norm_score=0.2)
    b = Candidate(sleeve="s2", symbol="X", kind="score", grade="A", score=0.8, norm_score=0.8)
    kept, rejected = resolve_conflicts([a, b])
    assert kept[0].sleeve == "s2"  # higher norm_score wins the grade tie


def test_resolve_distinct_symbols_all_kept():
    a = Candidate(sleeve="s", symbol="A", kind="score", grade="B", score=1.0)
    b = Candidate(sleeve="s", symbol="B", kind="score", grade="B", score=1.0)
    kept, rejected = resolve_conflicts([a, b])
    assert {c.symbol for c in kept} == {"A", "B"} and not rejected


# --------------------------------------------------------------------------- #
# cluster dedup (#3)
# --------------------------------------------------------------------------- #
def test_cluster_dedup_keeps_top_ranked_per_cluster():
    corr = _corr({"AAA": {"ZZZ": 0.99}, "ZZZ": {"AAA": 0.99}})
    # ranked best-first: AAA (norm 0.9) before ZZZ (norm 0.5); same cluster.
    top = Candidate(sleeve="s", symbol="AAA", kind="score", grade="B", score=2.0, norm_score=0.9)
    dup = Candidate(sleeve="s", symbol="ZZZ", kind="score", grade="B", score=1.0, norm_score=0.5)
    kept, rejected, clusters = cluster_dedup([top, dup], corr, threshold=0.8)
    assert [c.symbol for c in kept] == ["AAA"]
    assert rejected[0][0].symbol == "ZZZ"


def test_cluster_dedup_held_cluster_blocks_new():
    corr = _corr({"AAA": {"ZZZ": 0.99}, "ZZZ": {"AAA": 0.99}})
    new = Candidate(sleeve="s", symbol="ZZZ", kind="score", grade="B", score=1.0, norm_score=0.5)
    # the book already holds AAA, which is ~1.0 correlated with ZZZ -> blocked.
    kept, rejected, clusters = cluster_dedup(
        [new], corr, threshold=0.8, held_symbols={"AAA"}
    )
    assert not kept and rejected[0][0].symbol == "ZZZ"
