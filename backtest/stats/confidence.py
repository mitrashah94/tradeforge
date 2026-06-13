"""backtest/stats/confidence.py — bootstrap confidence intervals.

MASTER_PLAN.md §5: *"distrust point estimates under ~100–150 trades."* Point
estimates of profit factor and expectancy are noisy on small samples, so every
reported edge gets a confidence interval AND its trade count, and an
``is_underpowered`` flag for the <100-trade danger zone.

Method: the **percentile bootstrap**. We resample the per-trade pnls (or
R-multiples) WITH REPLACEMENT ``n_boot`` times, recompute the statistic on each
resample, and take the empirical ``[alpha/2, 1-alpha/2]`` percentiles as the
CI. This makes no distributional assumption — appropriate for fat-tailed,
skewed trade-pnl distributions.

Determinism: every function takes a ``seed`` (default 0) so tests and the
research log are reproducible. No wall-clock, no global RNG state.

Public API
----------
    pf_ci(trade_pnls, n_boot=2000, alpha=0.05, seed=0) -> (pf, lo, hi, n)
    expectancy_ci(values, kind='dollar'|'r', ...)       -> (mean, lo, hi, n)
    is_underpowered(n, threshold=100)                   -> bool

Each CI function ALWAYS returns ``n`` (the trade count) alongside the interval,
so callers cannot report a CI without also surfacing how many trades back it.
"""

from __future__ import annotations

import numpy as np

from backtest.stats.metrics import _to_array, profit_factor

# The trade-count floor below which point estimates are not trustworthy.
UNDERPOWERED_THRESHOLD = 100


def is_underpowered(n: int, threshold: int = UNDERPOWERED_THRESHOLD) -> bool:
    """True if a sample of ``n`` trades is too small to trust a point estimate.

    Per MASTER_PLAN §5 the danger zone is under ~100–150 trades; we draw the
    hard line at ``threshold`` (default 100). Callers should widen their skepticism
    (and lean on the CI, not the point estimate) when this is True.
    """
    return int(n) < int(threshold)


def _bootstrap_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """An (n_boot, n) array of resample indices drawn with replacement."""
    return rng.integers(0, n, size=(n_boot, n))


def pf_ci(
    trade_pnls,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Bootstrap CI for profit factor.

    Resamples ``trade_pnls`` (net $ per trade) with replacement ``n_boot`` times
    and returns ``(pf, lo, hi, n)`` where ``pf`` is the point estimate on the
    full sample and ``(lo, hi)`` are the ``alpha/2`` / ``1-alpha/2`` percentiles
    of the bootstrap PF distribution.

    Edge cases:
      - n == 0  -> ``(nan, nan, nan, 0)``.
      - A resample with no losers yields ``inf`` PF; such draws are dropped from
        the percentile computation (they would dominate the upper tail with a
        meaningless value). If ALL draws are degenerate the bounds are ``inf``.
    """
    arr = _to_array(trade_pnls)
    n = int(arr.size)
    point = profit_factor(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    idx = _bootstrap_indices(n, n_boot, rng)
    samples = arr[idx]  # (n_boot, n)

    pos = np.where(samples > 0, samples, 0.0).sum(axis=1)
    neg = -np.where(samples < 0, samples, 0.0).sum(axis=1)  # positive magnitudes

    with np.errstate(divide="ignore", invalid="ignore"):
        pfs = np.where(neg > 0, pos / neg, np.inf)

    finite = pfs[np.isfinite(pfs)]
    if finite.size == 0:
        return (point, float("inf"), float("inf"), n)

    lo = float(np.percentile(finite, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return (point, lo, hi, n)


def expectancy_ci(
    values,
    kind: str = "dollar",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Bootstrap CI for expectancy (mean per-trade value).

    ``values`` are per-trade pnls ($) when ``kind == 'dollar'`` or per-trade
    R-multiples when ``kind == 'r'`` — the math is identical (a mean), the
    ``kind`` is accepted for call-site clarity and validated. Returns
    ``(mean, lo, hi, n)``; ``(nan, nan, nan, 0)`` when empty.
    """
    if kind not in ("dollar", "r"):
        raise ValueError(f"kind must be 'dollar' or 'r', got {kind!r}")

    arr = _to_array(values)
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)

    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = _bootstrap_indices(n, n_boot, rng)
    means = arr[idx].mean(axis=1)

    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (point, lo, hi, n)


def ci_width(ci: tuple[float, float, float, int]) -> float:
    """Width (hi - lo) of a ``(point, lo, hi, n)`` CI tuple; ``nan`` if unbounded."""
    _point, lo, hi, _n = ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    return float(hi - lo)
