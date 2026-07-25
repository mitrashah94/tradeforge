"""portfolio/rank.py — one cross-sleeve priority order. (#2)

A breakout momentum number (e.g. +0.18 trailing return) and a rotation target
weight (e.g. 0.25 of the book) live on totally different scales — comparing them
raw would let one sleeve's units dominate the book. So we normalize each
candidate's signal WITHIN its own sleeve to a common [0, 1] percentile, then lay
all sleeves' candidates on ONE list ordered by that normalized score. The best
opportunity in each sleeve lands near 1.0; the cross-sleeve walk (used by cluster
dedup and the heat-admission budget) then funds genuinely-best-first.

The percentile is rank-based (robust to a single outlier blowing out a min-max
range): within a sleeve, a candidate's ``norm_score`` is the average-rank
percentile of its raw signal among that sleeve's candidates (a lone candidate →
1.0, the sleeve's top). Ties share the average percentile, so an equal-weight
basket's members rank equally — exactly right.

An OPTIONAL Kronos blend (Phase 3) nudges the normalized score by the candidate's
forecast ``exp_return`` when ``kronos_blend > 0``; off by default so the engine
ranks identically with or without torch.
"""

from __future__ import annotations

from typing import Sequence


def _raw_signal(c) -> float:
    """The raw within-sleeve ranking signal: score for score sleeves, weight for weight."""
    if c.score is not None:
        return float(c.score)
    if c.target_weight is not None:
        return float(c.target_weight)
    return float("-inf")


def _percentiles(values: Sequence[float]) -> list:
    """Average-rank percentile of each value in [0, 1] (lone value → 1.0).

    Ties get the mean of the ranks they span, so duplicates rank equally. With a
    single value there is no spread to normalize against → it is the sleeve top.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda i: values[i])
    # average-rank: group equal values, assign the mean ordinal rank.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0           # 0-based mean ordinal of the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return [r / (n - 1) for r in ranks]    # map [0, n-1] -> [0, 1]


def normalize_scores(candidates: Sequence) -> list:
    """Fill each candidate's ``norm_score`` with its within-sleeve percentile. (#2)

    Mutates the candidates in place (sets ``norm_score``) and returns the same
    list for chaining. Grouping is by ``sleeve`` so each sleeve normalizes against
    only its own candidates.
    """
    cands = list(candidates)
    by_sleeve: dict = {}
    for c in cands:
        by_sleeve.setdefault(c.sleeve, []).append(c)
    for group in by_sleeve.values():
        vals = [_raw_signal(c) for c in group]
        pcts = _percentiles(vals)
        for c, p in zip(group, pcts):
            c.norm_score = float(p)
    return cands


def _kronos_term(c) -> float:
    """The Kronos blend term: the forecast ``exp_return`` (0.0 when absent)."""
    f = getattr(c, "forecast", None)
    if not f:
        return 0.0
    try:
        return float(f.get("exp_return", 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def rank_candidates(candidates: Sequence, kronos_blend: float = 0.0) -> list:
    """Return candidates in ONE cross-sleeve priority order (best first). (#2)

    Normalizes within sleeve, optionally blends the Kronos ``exp_return``, then
    sorts by ``(normalized score desc, family, symbol, sleeve)`` — the plan's
    explicit order, with the trailing keys making the sort fully deterministic.
    """
    cands = normalize_scores(candidates)
    blend = float(kronos_blend or 0.0)

    def key(c):
        base = float(c.norm_score if c.norm_score is not None else 0.0)
        if blend > 0:
            base = base + blend * _kronos_term(c)
        # sort: norm desc (so negate), then family/symbol/sleeve asc.
        return (-base, c.family, c.symbol, c.sleeve)

    return sorted(cands, key=key)
