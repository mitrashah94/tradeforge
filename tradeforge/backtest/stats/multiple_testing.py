"""backtest/stats/multiple_testing.py — guards against data-mining.

MASTER_PLAN.md §5/§6: *"Multiple-testing correction (deflated Sharpe / PF
threshold that rises with the number of variants tried); log every hypothesis,
not just survivors."* The more variants you try, the more likely one looks good
by chance — so the bar to clear must rise with the number of trials, and every
hypothesis (survivor or not) must be recorded so the trial count is honest.

This module provides three things:

1. ``deflated_sharpe(sharpe, n_trials, n_obs, skew, kurt) -> float``
   The Bailey & López de Prado **Deflated Sharpe Ratio** probability — the
   probability that the observed (non-annualized, per-observation) Sharpe is
   genuinely > 0 AFTER accounting for (a) the number of trials, (b) the sample
   length, and (c) the non-normality (skew/kurtosis) of the returns. Always in
   ``[0, 1]``. Higher is better; treat <0.95 as "did not clear".

2. ``min_pf_threshold(n_trials) -> float``
   The minimum profit factor a variant must clear, RISING monotonically with the
   number of variants tried. Base 1.3 (the MASTER_PLAN paper->live floor),
   scaled up by a slowly-growing function of ``n_trials`` so the 50th variant
   has to clear a meaningfully higher bar than the 1st.

3. ``HypothesisLog`` — an append-only JSONL ledger at
   ``backtest/stats/hypothesis_log.jsonl`` recording EVERY hypothesis tried
   (name, params, n_trades, pf, expectancy_r, sharpe, timestamp, passed). The
   timestamp is passed in (never ``datetime.now`` in the library path) so the
   log is deterministic and replayable.

No scipy: the standard-normal CDF/PPF are implemented here (Abramowitz-Stegun
rational approximations) so tests stay offline and dependency-free.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "hypothesis_log.jsonl"

# The base PF floor (MASTER_PLAN paper->live gate) before any multiple-testing
# inflation, and how aggressively the threshold climbs with more trials.
BASE_PF_THRESHOLD = 1.3
PF_TRIAL_SCALE = 0.15  # bigger -> threshold rises faster with n_trials


# --------------------------------------------------------------------------- #
# Standard normal CDF / PPF (no scipy)
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    """Standard-normal CDF via the erf identity (math.erf is stdlib, exact-ish)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF (quantile).

    Acklam's rational approximation; abs error < ~1e-9 over (0,1). Used to turn
    a target confidence (e.g. the expected max of ``n_trials`` draws) into a
    Sharpe benchmark for the deflated Sharpe.
    """
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        raise ValueError(f"p must be in (0,1), got {p!r}")

    # Coefficients (Acklam).
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
# --------------------------------------------------------------------------- #
EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_trials_sharpe: float = 1.0) -> float:
    """Expected MAX Sharpe across ``n_trials`` independent strategies, all with

    true Sharpe 0 (the "false-discovery benchmark"). This is the
    ``E[max]`` term in the deflated Sharpe — the Sharpe a winner must beat just
    to look special among ``n_trials`` random tries.

    Formula (Bailey & López de Prado): with V = variance of the trial Sharpes,
        E[max] ≈ sqrt(V) * ( (1-γ) * Z⁻¹(1 - 1/N) + γ * Z⁻¹(1 - 1/(N·e)) )
    where γ is the Euler-Mascheroni constant and Z⁻¹ is the normal quantile.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    sd = math.sqrt(max(var_trials_sharpe, 0.0))
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
    return sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    var_trials_sharpe: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio probability (Bailey & López de Prado).

    Parameters
    ----------
    sharpe
        The observed **per-observation** (non-annualized) Sharpe ratio of the
        return series (e.g. ``metrics.sharpe_per_period(daily_returns)``).
    n_trials
        Number of independent strategy configurations tried (the multiple-testing
        count — feed it from ``HypothesisLog.count()`` or the sweep size).
    n_obs
        Number of return observations the Sharpe was computed on.
    skew, kurt
        Skewness and (non-excess) kurtosis of the returns. Normal => skew 0,
        kurt 3. Fat tails (kurt > 3) and negative skew SHRINK the DSR.
    var_trials_sharpe
        Variance of the Sharpe ratios across the trials (defaults to 1, the
        standard assumption when only the count is known).

    Returns
    -------
    A probability in ``[0, 1]``: the probability the strategy's TRUE Sharpe
    exceeds the expected-max false-discovery benchmark, given the trial count
    and the higher moments. Treat ``>= 0.95`` as "clears multiple testing".
    """
    if n_obs is None or n_obs < 2:
        return 0.0

    sr0 = expected_max_sharpe(n_trials, var_trials_sharpe=var_trials_sharpe)

    # Standard error of the Sharpe estimator under non-normal returns
    # (Mertens / Lo): SE = sqrt[ (1 - γ3·SR + (γ4-1)/4 · SR²) / (T-1) ].
    sr = float(sharpe)
    if not math.isfinite(sr):
        return 0.0
    denom = float(n_obs - 1)
    var_sr = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / denom
    if var_sr <= 0.0:
        # Degenerate adjusted variance; fall back to a tiny positive SE so the
        # function stays in [0,1] rather than blowing up.
        var_sr = 1e-12
    se = math.sqrt(var_sr)

    z = (sr - sr0) / se
    p = norm_cdf(z)
    # Clamp into [0,1] (norm_cdf already is, but be defensive about fp edges).
    return float(min(1.0, max(0.0, p)))


# --------------------------------------------------------------------------- #
# Rising PF threshold
# --------------------------------------------------------------------------- #
def min_pf_threshold(
    n_trials: int,
    base: float = BASE_PF_THRESHOLD,
    scale: float = PF_TRIAL_SCALE,
) -> float:
    """Minimum profit factor a variant must clear, rising with ``n_trials``.

    Strictly monotonically INCREASING in ``n_trials`` (each additional variant
    raises the bar a little). The growth is logarithmic so it climbs fast over
    the first handful of trials and then decelerates — you pay a real haircut for
    trying 10 things, a bigger one for 100, but it never runs away to absurdity.

        threshold(n) = base + scale * ln(n)

    ``n_trials`` is clamped to >= 1 (the first try has threshold == base).
    """
    n = max(int(n_trials), 1)
    return float(base + scale * math.log(n))


# --------------------------------------------------------------------------- #
# Persistent hypothesis log
# --------------------------------------------------------------------------- #
@dataclass
class HypothesisRecord:
    """One logged hypothesis (a row in the JSONL ledger)."""

    name: str
    params: dict
    n_trades: int
    pf: float
    expectancy_r: float
    sharpe: float
    timestamp: str          # ISO-8601 string, PASSED IN (never datetime.now here)
    passed: bool
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=_json_default)


def _json_default(o):
    # Make numpy scalars / non-finite floats survive a round trip.
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _jsonify_floats(d: dict) -> dict:
    """Replace non-JSON-finite floats (inf/nan) with strings so JSONL is valid."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = "Infinity" if v > 0 else ("-Infinity" if v < 0 else "NaN")
        else:
            out[k] = v
    return out


class HypothesisLog:
    """Append-only JSONL ledger of every hypothesis tried.

    The discipline (MASTER_PLAN §5/§6): *log every hypothesis, not just
    survivors.* Each ``append`` writes one line; ``read_all`` round-trips them
    back; ``count`` gives the honest trial count to feed into
    ``deflated_sharpe`` / ``min_pf_threshold``.

    The timestamp is supplied by the caller (an ISO string) so the log is
    deterministic — no wall-clock in the library path.
    """

    def __init__(self, path: str | Path = DEFAULT_LOG_PATH):
        self.path = Path(path)

    # ---------------------------------------------------------------- write
    def append(
        self,
        name: str,
        params: dict,
        n_trades: int,
        pf: float,
        expectancy_r: float,
        sharpe: float,
        timestamp: str,
        passed: bool,
        note: str = "",
    ) -> HypothesisRecord:
        """Append one hypothesis record to the JSONL log and return it.

        ``timestamp`` must be a caller-provided ISO-8601 string (determinism).
        Non-finite floats (inf/nan PFs) are serialized as the JSON-spec strings
        ``"Infinity"``/``"-Infinity"``/``"NaN"`` and decoded back on read.
        """
        rec = HypothesisRecord(
            name=str(name),
            params=dict(params or {}),
            n_trades=int(n_trades),
            pf=float(pf),
            expectancy_r=float(expectancy_r),
            sharpe=float(sharpe),
            timestamp=str(timestamp),
            passed=bool(passed),
            note=str(note),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _jsonify_floats(asdict(rec))
        line = json.dumps(payload, sort_keys=True, default=_json_default)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return rec

    def append_record(self, rec: HypothesisRecord) -> HypothesisRecord:
        """Append a pre-built :class:`HypothesisRecord`."""
        return self.append(
            name=rec.name,
            params=rec.params,
            n_trades=rec.n_trades,
            pf=rec.pf,
            expectancy_r=rec.expectancy_r,
            sharpe=rec.sharpe,
            timestamp=rec.timestamp,
            passed=rec.passed,
            note=rec.note,
        )

    # ----------------------------------------------------------------- read
    def read_all(self) -> list[HypothesisRecord]:
        """Read every logged record back (in append order)."""
        if not self.path.exists():
            return []
        out: list[HypothesisRecord] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                d = json.loads(raw)
                # Decode the spec floats back to Python floats.
                for k in ("pf", "expectancy_r", "sharpe"):
                    v = d.get(k)
                    if v == "Infinity":
                        d[k] = float("inf")
                    elif v == "-Infinity":
                        d[k] = float("-inf")
                    elif v == "NaN":
                        d[k] = float("nan")
                out.append(
                    HypothesisRecord(
                        name=d["name"],
                        params=d.get("params", {}),
                        n_trades=int(d["n_trades"]),
                        pf=float(d["pf"]),
                        expectancy_r=float(d["expectancy_r"]),
                        sharpe=float(d["sharpe"]),
                        timestamp=d["timestamp"],
                        passed=bool(d["passed"]),
                        note=d.get("note", ""),
                    )
                )
        return out

    def count(self) -> int:
        """Number of hypotheses logged so far (the honest trial count)."""
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for raw in fh:
                if raw.strip():
                    n += 1
        return n

    def clear(self) -> None:
        """Delete the log file (used by tests; not for production paths)."""
        if self.path.exists():
            os.remove(self.path)
