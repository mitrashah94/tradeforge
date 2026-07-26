"""portfolio/conflicts.py — correlation clustering + dedup + same-symbol resolve.

Two distinct jobs the engine runs back-to-back (capabilities #3 and #6):

  (#3) CLUSTER DEDUP — keep only ONE position per correlation CLUSTER. Highly
       correlated names (two ~1.0-corr ETFs, a sector and its 3x cousin) are the
       same bet wearing two tickers; funding both doubles the risk the heat budget
       thinks it is taking. We single-linkage-cluster the symbols off the measured
       ``correlation_matrix`` (union-find on the thresholded graph, no scipy) and,
       within each cluster, keep the highest-ranked candidate and reject the rest.

  (#6) SAME-SYMBOL RESOLVE — when two sleeves both want the SAME symbol: same side
       → keep one (charge heat once, attribute to the higher conviction grade);
       opposite intent → the higher conviction grade wins, ties broken by score.

THE SHARED CLUSTERING PRIMITIVE
-------------------------------
:func:`correlation_clusters` (the union-find) is the single factored
implementation the watchlist's ``watchlist/clustering.py`` (Phase 2) imports too —
the cross-cutting "factor correlation-clustering once" rule. It operates on a
correlation matrix shaped like ``backtest.portfolio.correlation_matrix``'s output
(a square DataFrame or a plain ``{a: {b: rho}}`` mapping).

PURE / DETERMINISTIC / OFFLINE: no scipy, no I/O. Grade order is fixed
(``A+`` > ``A`` > ``B``); ties everywhere break on a deterministic key so the same
inputs always yield the same book.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# Conviction grade ordering (higher wins a same-symbol conflict). Unknown grades
# sort below every known grade (rank 0) so a typo never silently outranks A+.
_GRADE_RANK = {"B": 1, "A": 2, "A+": 3}


def grade_rank(grade: str) -> int:
    """Integer rank of a conviction grade (``A+`` > ``A`` > ``B`` > unknown)."""
    return _GRADE_RANK.get(str(grade), 0)


# --------------------------------------------------------------------------- #
# Union-find (disjoint set) — the clustering primitive
# --------------------------------------------------------------------------- #
class _UnionFind:
    """Minimal union-find over hashable items with path-compression + union-by-rank."""

    def __init__(self, items: Iterable):
        self.parent = {it: it for it in items}
        self.rank = {it: 0 for it in self.parent}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _corr_lookup(corr, a: str, b: str):
    """Read ``corr[a][b]`` from a DataFrame-like or nested-mapping correlation matrix."""
    # pandas DataFrame: use .at; plain mapping: nested dict.
    try:
        loc = corr.at  # type: ignore[attr-defined]
    except AttributeError:
        row = corr.get(a) if hasattr(corr, "get") else None
        if row is None:
            return None
        return row.get(b)
    try:
        return loc[a, b]
    except (KeyError, IndexError):
        return None


def _corr_symbols(corr) -> list:
    """The symbol list (index/columns) of a correlation matrix, DataFrame or mapping."""
    cols = getattr(corr, "columns", None)
    if cols is not None:
        return list(cols)
    if isinstance(corr, Mapping):
        return list(corr.keys())
    return []


def correlation_clusters(
    symbols: Sequence[str], corr, threshold: float = 0.8
) -> dict:
    """Single-linkage clusters of ``symbols`` at ``|rho| >= threshold`` (union-find).

    Two symbols are linked when the ABSOLUTE value of their correlation reaches
    ``threshold`` (a strong positive OR a strong negative co-move is the same risk
    factor in opposite signs — an inverse ETF pair is as redundant as a duplicate).
    Symbols absent from ``corr`` form singleton clusters (no info → assume distinct).

    Returns ``{symbol: cluster_id}`` where ``cluster_id`` is the lexicographically
    smallest symbol in the cluster (a stable, deterministic representative).
    """
    syms = list(dict.fromkeys(symbols))  # de-dup, preserve order
    uf = _UnionFind(syms)
    present = [s for s in syms if s in set(_corr_symbols(corr))]
    thr = abs(float(threshold))
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            rho = _corr_lookup(corr, a, b)
            if rho is None:
                continue
            try:
                if abs(float(rho)) >= thr:
                    uf.union(a, b)
            except (TypeError, ValueError):
                continue
    # Representative = lexicographically smallest member of each set.
    members: dict = {}
    for s in syms:
        members.setdefault(uf.find(s), []).append(s)
    out: dict = {}
    for root, group in members.items():
        rep = min(group)
        for s in group:
            out[s] = rep
    return out


# --------------------------------------------------------------------------- #
# (#6) same-symbol conflict resolution
# --------------------------------------------------------------------------- #
def _candidate_key(c) -> tuple:
    """Deterministic priority key for a candidate: (grade, rank_value, -symbol)."""
    # Higher grade first, then higher rank value, then symbol asc (so the tuple is
    # sorted DESC for "best"). We negate via reverse in callers; expose the raw key.
    return (grade_rank(c.grade), c.rank_value(), c.symbol)


def resolve_conflicts(candidates: Sequence) -> tuple:
    """Resolve same-symbol conflicts → (kept, rejected) lists. (#6)

    Groups candidates by symbol. Within a symbol:
      * one candidate → kept as-is;
      * many → the highest conviction grade wins; ties broken by the higher rank
        value (``norm_score``/``score``), then by sleeve name for determinism. The
        losers are rejected with a reason naming the winner (so the same symbol is
        funded ONCE — heat charged once, attributed to the winning grade). This
        covers both the "same-side merge" and the "opposite-intent higher grade
        wins" cases of the spec; the book is long-only so the surviving side is
        always long.

    Order of the kept list follows the input order of each symbol's winner, so a
    downstream stable sort by rank still behaves predictably.
    """
    by_symbol: dict = {}
    order: list = []
    for c in candidates:
        if c.symbol not in by_symbol:
            by_symbol[c.symbol] = []
            order.append(c.symbol)
        by_symbol[c.symbol].append(c)

    kept: list = []
    rejected: list = []
    for sym in order:
        group = by_symbol[sym]
        if len(group) == 1:
            kept.append(group[0])
            continue
        # winner: max grade, then rank value, then sleeve name (asc) for stability.
        winner = max(
            group,
            key=lambda c: (grade_rank(c.grade), c.rank_value(), _neg_str(c.sleeve)),
        )
        kept.append(winner)
        for c in group:
            if c is winner:
                continue
            rejected.append(
                (c, f"same-symbol conflict on {sym}: {winner.sleeve}/{winner.grade} wins")
            )
    return kept, rejected


def _neg_str(s: str):
    """A key that orders strings ascending inside a ``max`` (so 'a' beats 'z')."""
    # max() wants the smallest sleeve name to win ties; invert by code points.
    return tuple(-ord(ch) for ch in s)


# --------------------------------------------------------------------------- #
# (#3) cluster dedup — one position per correlation cluster
# --------------------------------------------------------------------------- #
def cluster_dedup(
    ranked_candidates: Sequence,
    corr,
    threshold: float = 0.8,
    held_symbols: Iterable | None = None,
) -> tuple:
    """Keep one candidate per correlation cluster → (kept, rejected, clusters). (#3)

    ``ranked_candidates`` MUST already be in cross-sleeve priority order (best
    first; see :mod:`portfolio.rank`). Walking that order, the FIRST candidate seen
    for a cluster is kept and every later candidate in the same cluster is rejected
    — "keep the highest-ranked, reject the rest". ``held_symbols`` pre-seeds the
    taken set with the clusters the book ALREADY holds, so a new candidate
    correlated to an existing position is also rejected (no doubling an existing
    bet).

    Clusters are computed over the UNION of the candidates' AND the held symbols,
    so a candidate correlated to a held name (different ticker) collapses into the
    same cluster id and is correctly blocked.
    """
    cands = list(ranked_candidates)
    held = list(held_symbols or [])
    universe = list(dict.fromkeys([*[c.symbol for c in cands], *held]))
    clusters = correlation_clusters(universe, corr, threshold=threshold)

    taken = {clusters.get(s, s) for s in held}
    kept: list = []
    rejected: list = []
    for c in cands:
        cid = clusters.get(c.symbol, c.symbol)
        if cid in taken:
            rejected.append((c, f"cluster dedup: cluster {cid} already taken"))
            continue
        taken.add(cid)
        kept.append(c)
    return kept, rejected, clusters
