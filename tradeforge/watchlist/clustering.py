"""watchlist/clustering.py — correlation clustering for the ticker universe.

The watchlist's CORE-diversification check ("don't fill CORE with three SPY
proxies") and the Phase-1 portfolio engine's cluster dedup ("one position per
correlation cluster") are the SAME operation on the SAME math. Per the
cross-cutting "factor correlation-clustering once" rule, the union-find primitive
lives in :mod:`portfolio.conflicts` and is imported here — there is exactly one
implementation of single-linkage correlation clustering in the codebase.

This module adds the watchlist-flavored convenience: build the candidate
correlation matrix from their daily-return streams (via
``backtest.portfolio.correlation_matrix``) and cluster it, so ``assign_tiers`` can
admit ≤ 1 CORE name per cluster instead of an O(n²) pairwise sweep.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from backtest.portfolio import correlation_matrix as _returns_correlation_matrix
from portfolio.conflicts import correlation_clusters

__all__ = ["correlation_clusters", "cluster_symbols", "clusters_from_return_matrix"]


def cluster_symbols(symbols: Sequence[str], corr, threshold: float = 0.85) -> dict:
    """``{symbol: cluster_id}`` from a correlation matrix (the shared union-find).

    Thin pass-through to :func:`portfolio.conflicts.correlation_clusters` so the
    watchlist imports the ONE clustering implementation. ``corr`` is a square
    correlation DataFrame (or nested mapping); ``threshold`` is the ``|rho|`` at /
    above which two names collapse into one cluster.
    """
    return correlation_clusters(symbols, corr, threshold=threshold)


def clusters_from_return_matrix(
    return_matrix: pd.DataFrame, threshold: float = 0.85
) -> dict:
    """Cluster the columns of an aligned daily-RETURN matrix.

    Computes the Pearson correlation with ``backtest.portfolio.correlation_matrix``
    (flat days already aligned/filled by the caller) then single-linkage clusters
    it. Returns ``{symbol: cluster_id}``.
    """
    if return_matrix is None or return_matrix.shape[1] == 0:
        return {}
    corr = _returns_correlation_matrix(return_matrix)
    return correlation_clusters(list(return_matrix.columns), corr, threshold=threshold)
