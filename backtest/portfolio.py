"""backtest/portfolio.py — the edge-portfolio / maximization analysis.

This is where the MASTER_PLAN §0/§1.B *maximization thesis* is put to the test:

    geometric growth  g  ≈  mean_return − ½·variance

Stacking UNCORRELATED edges lowers the *blend* variance for a given mean, which
RAISES g — so a decorrelated blend can compound faster than the best single
strategy even when each individual edge is weak. This module computes that
honestly:

  1. Run a chosen edge set under the **realistic** cost profile (the honest
     small-account picture — never the optimistic tv_style here).
  2. Build the aligned per-session fractional-return matrix (outer-join on the
     ET session_date, fill 0 where a strategy did not trade that day — a flat day
     is a real 0 return for that sleeve).
  3. Compute the live correlation matrix.
  4. Form two blends: EQUAL-weight and long-only MIN-VARIANCE (solve
     min wᵀΣw s.t. Σw = 1, w ≥ 0 via a projected-gradient solver — no scipy).
  5. For every single AND every blend compute mean_daily, var_daily, annualized
     Sharpe, and g = mean − ½·var.
  6. State the explicit verdict: does the best blend's g beat the best single's
     g? Is the blend's variance below the best single's variance? Be honest if
     not.

Everything keys off ONE shared DuckDB connection (``data.schema.connect``)
passed down to every run — opening a connection per call trips DuckDB's file
lock in loops.

CLI:  PYTHONPATH=. .venv/bin/python backtest/portfolio.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.runner import daily_returns_pct, run_strategy
from backtest.stats.metrics import sharpe
from data.schema import DEFAULT_DB_PATH, connect

# Annualization for the Sharpe column (daily series).
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Edge-set specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EdgeSpec:
    """One sleeve of the portfolio: a named, instantiated strategy + market."""

    name: str               # short label used in tables / charts / registry
    factory: object         # callable(params) -> engine Strategy
    params: dict            # resolved params dict (already variant-merged)
    symbol: str
    timeframe: str

    def build(self):
        return self.factory(self.params)


def default_edge_set() -> list[EdgeSpec]:
    """The blend's edge set: the three strategies, each its headline variant.

    breakout_retest uses **v0_atr_stop** here, not raw V0 — v0_atr_stop is the
    realistic-cost survivor (the tight 1-tick stop blows up under slippage), so
    the blend is built from each strategy's best realistic-cost config. The two
    complements use their DEFAULT variant. All on QQQ 5m (the one symbol with the
    full ~2y history + levels), which is the conservative apples-to-apples choice.
    """
    from strategies.breakout_retest.strategy import (
        BreakoutRetestStrategy,
        load_params as br_load,
    )
    from strategies.level_meanrev.strategy import (
        LevelMeanRevStrategy,
        load_params as lmr_load,
    )
    from strategies.momentum_thrust.strategy import (
        MomentumThrustStrategy,
        load_params as mt_load,
    )

    return [
        EdgeSpec(
            name="breakout_retest/v0_atr_stop",
            factory=lambda p: BreakoutRetestStrategy(params=p),
            params=br_load("v0_atr_stop"),
            symbol="QQQ",
            timeframe="5m",
        ),
        EdgeSpec(
            name="level_meanrev/DEFAULT",
            factory=lambda p: LevelMeanRevStrategy(params=p),
            params=lmr_load("DEFAULT"),
            symbol="QQQ",
            timeframe="5m",
        ),
        EdgeSpec(
            name="momentum_thrust/DEFAULT",
            factory=lambda p: MomentumThrustStrategy(params=p),
            params=mt_load("DEFAULT"),
            symbol="QQQ",
            timeframe="5m",
        ),
    ]


# --------------------------------------------------------------------------- #
# Build the aligned return matrix
# --------------------------------------------------------------------------- #
def returns_matrix(
    edges: list[EdgeSpec],
    start=None,
    end=None,
    cost_profile: str = "realistic",
    con=None,
    db_path: str = DEFAULT_DB_PATH,
    base_equity: float = 100_000.0,
) -> pd.DataFrame:
    """Run each edge and return the aligned per-session fractional-return matrix.

    Columns are edge names; the index is the union of every sleeve's exit
    session_dates (outer join). A cell is the strategy's realized fractional
    return that session, or **0.0** where it did not close a trade — a flat day
    is a genuine 0 for that sleeve, which is the correct fill for the blend math
    (the alternative, dropping the day, would silently overweight active days).

    Uses ONE DuckDB connection for every run (opens one if ``con`` is None).
    """
    own_con = con is None
    if own_con:
        con = connect(db_path)
    try:
        series: dict[str, pd.Series] = {}
        for spec in edges:
            res = run_strategy(
                spec.build(),
                spec.symbol,
                spec.timeframe,
                start=start,
                end=end,
                cost_profile=cost_profile,
                con=con,
                initial_equity=base_equity,
            )
            s = daily_returns_pct(res, base_equity=base_equity)
            # Normalize index to plain dates so the outer-join aligns cleanly.
            s.index = [d.date() if hasattr(d, "date") else d for d in s.index]
            series[spec.name] = s
    finally:
        if own_con:
            con.close()

    if not series:
        return pd.DataFrame()

    mat = pd.DataFrame(series)
    mat = mat.sort_index()
    mat = mat.fillna(0.0)  # flat day == 0 return for that sleeve
    return mat


# --------------------------------------------------------------------------- #
# Correlation matrix
# --------------------------------------------------------------------------- #
def correlation_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of the aligned daily-return columns.

    Computed on the union calendar with flat days = 0 (so the correlation
    reflects how the sleeves co-move across ALL sessions, not just the rare days
    both happened to trade — that is the correlation that drives blend variance).
    """
    if mat is None or mat.shape[1] == 0:
        return pd.DataFrame()
    return mat.corr()


# --------------------------------------------------------------------------- #
# Min-variance long-only weights (projected gradient, no scipy)
# --------------------------------------------------------------------------- #
def min_variance_weights(
    cov: np.ndarray,
    max_iter: int = 20_000,
    lr: float | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Long-only minimum-variance weights: min wᵀΣw s.t. Σw=1, w≥0.

    Projected-gradient descent on the simplex. Each step takes a gradient step
    ``w -= lr · 2Σw`` then projects back onto the probability simplex (Euclidean
    projection, Wang & Carreira-Perpiñán). Deterministic, dependency-free, and
    exact enough for a 3–5 asset blend. Falls back to equal weights on a
    degenerate covariance.
    """
    Sigma = np.asarray(cov, dtype="float64")
    n = Sigma.shape[0]
    if n == 0:
        return np.asarray([], dtype="float64")
    if n == 1:
        return np.asarray([1.0])

    # Symmetrize defensively.
    Sigma = 0.5 * (Sigma + Sigma.T)
    if not np.all(np.isfinite(Sigma)):
        return np.full(n, 1.0 / n)

    # Step size from the spectral norm (1/L for L = 2·λmax of Σ).
    if lr is None:
        try:
            lam_max = float(np.max(np.linalg.eigvalsh(Sigma)))
        except np.linalg.LinAlgError:
            lam_max = float(np.trace(Sigma)) or 1.0
        L = 2.0 * max(lam_max, 1e-18)
        lr = 1.0 / L if L > 0 else 1.0

    w = np.full(n, 1.0 / n)
    prev_obj = float("inf")
    for _ in range(max_iter):
        grad = 2.0 * Sigma @ w
        w = _project_simplex(w - lr * grad)
        obj = float(w @ Sigma @ w)
        if abs(prev_obj - obj) < tol:
            break
        prev_obj = obj
    return w


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto {w : Σw = 1, w ≥ 0}."""
    v = np.asarray(v, dtype="float64")
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    if not cond.any():
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


# --------------------------------------------------------------------------- #
# Blends
# --------------------------------------------------------------------------- #
def blend_returns(mat: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Weighted daily-return series of the blend (Σ wᵢ · rᵢ per session)."""
    w = np.asarray(weights, dtype="float64")
    out = mat.to_numpy(dtype="float64") @ w
    return pd.Series(out, index=mat.index, name="blend")


def _g(mean: float, var: float) -> float:
    """Geometric-growth proxy g = mean − ½·var."""
    return float(mean - 0.5 * var)


def _vol_targeted_g(g_table: pd.DataFrame, target_vol: float) -> pd.Series:
    """g of each row AFTER scaling it to a common ``target_vol`` (daily sd).

    The risk engine sizes every sleeve to a fixed $-risk, i.e. it levers each
    return stream to the same volatility. Scaling a stream by k multiplies its
    mean by k and its variance by k²; choosing ``k = target_vol / sd`` puts every
    sleeve (and the blend) on the SAME risk budget. Under that common budget,
    g = k·mean − ½·k²·var, and a higher Sharpe = a higher vol-targeted g. This is
    the scale-invariant statement of the maximization thesis (cut variance per
    unit edge -> compound faster at the same risk of ruin).
    """
    out = {}
    for name, r in g_table.iterrows():
        var = float(r["var_daily"])
        mean = float(r["mean_daily"])
        sd = float(np.sqrt(var)) if var > 0 else 0.0
        if sd <= 0:
            out[name] = float("nan")
            continue
        k = target_vol / sd
        out[name] = _g(k * mean, k * k * var)
    return pd.Series(out, dtype="float64")


def _series_stats(s: pd.Series) -> dict:
    """mean_daily, var_daily, sharpe (annualized), and g for a return series."""
    arr = s.astype("float64").to_numpy()
    if arr.size == 0:
        return {"mean_daily": float("nan"), "var_daily": float("nan"),
                "sharpe": float("nan"), "g": float("nan"), "n_days": 0}
    mean = float(arr.mean())
    var = float(arr.var(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean_daily": mean,
        "var_daily": var,
        "sharpe": sharpe(arr) if arr.size > 1 else float("nan"),
        "g": _g(mean, var),
        "n_days": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# The full analysis
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioAnalysis:
    """The bundled result of the blend / maximization analysis."""

    returns: pd.DataFrame                 # aligned per-session return matrix
    corr: pd.DataFrame                    # correlation matrix
    equal_weights: np.ndarray
    minvar_weights: np.ndarray
    g_table: pd.DataFrame                 # singles + blends: mean/var/sharpe/g
    verdict: dict                         # explicit maximization verdict
    names: list[str] = field(default_factory=list)


def analyze(
    edges: list[EdgeSpec] | None = None,
    start=None,
    end=None,
    cost_profile: str = "realistic",
    con=None,
    db_path: str = DEFAULT_DB_PATH,
    base_equity: float = 100_000.0,
) -> PortfolioAnalysis:
    """Run the full edge-portfolio / maximization analysis end-to-end."""
    edges = edges or default_edge_set()
    mat = returns_matrix(
        edges, start=start, end=end, cost_profile=cost_profile,
        con=con, db_path=db_path, base_equity=base_equity,
    )
    names = list(mat.columns)
    corr = correlation_matrix(mat)

    cov = np.cov(mat.to_numpy(dtype="float64").T, ddof=1) if mat.shape[1] > 1 else \
        np.asarray([[float(mat.iloc[:, 0].var(ddof=1))]])
    cov = np.atleast_2d(cov)

    n = len(names)
    eq_w = np.full(n, 1.0 / n) if n else np.asarray([])
    mv_w = min_variance_weights(cov)

    eq_blend = blend_returns(mat, eq_w)
    mv_blend = blend_returns(mat, mv_w)

    # ---- g-table: one row per single, plus the two blends ----
    rows: dict[str, dict] = {}
    for name in names:
        st = _series_stats(mat[name])
        st["is_blend"] = False
        rows[name] = st
    rows["BLEND_equal"] = {**_series_stats(eq_blend), "is_blend": True}
    rows["BLEND_minvar"] = {**_series_stats(mv_blend), "is_blend": True}

    g_table = pd.DataFrame(rows).T
    for c in ("mean_daily", "var_daily", "sharpe", "g"):
        g_table[c] = g_table[c].astype("float64")
    g_table["is_blend"] = g_table["is_blend"].astype(bool)
    g_table["n_days"] = g_table["n_days"].astype(int)

    # ---- the explicit verdict ----
    singles = g_table[~g_table["is_blend"]]
    blends = g_table[g_table["is_blend"]]
    best_single_name = singles["g"].idxmax()
    best_blend_name = blends["g"].idxmax()
    best_single_g = float(singles.loc[best_single_name, "g"])
    best_blend_g = float(blends.loc[best_blend_name, "g"])
    best_single_var = float(singles.loc[best_single_name, "var_daily"])
    min_single_var = float(singles["var_daily"].min())
    best_blend_var = float(blends.loc[best_blend_name, "var_daily"])

    beats_on_g = best_blend_g > best_single_g
    lowers_variance = best_blend_var < best_single_var
    lowers_vs_min_single = best_blend_var < min_single_var

    # ---- scale-invariant (vol-targeted) reading of the SAME thesis ----
    # The raw g = mean - 0.5*var penalty is tiny at these per-session
    # magnitudes (var ~1e-5, so 0.5*var ~ a fraction of a bp), so a single with a
    # much higher mean wins g even when it is far more volatile. But the
    # maximization MECHANISM (MASTER_PLAN §0/§1.B) is "lower curve volatility ->
    # higher SAFE SIZE at the same risk -> faster compounding". Make that
    # explicit: scale every sleeve AND the blend to a common target volatility
    # (the leverage the risk engine would actually apply) and recompute g. Under
    # a fixed risk budget the decorrelated blend's higher Sharpe converts
    # directly into a higher g. The Sharpe column is itself the cleanest
    # scale-invariant signal, so we surface "highest Sharpe" too.
    best_sharpe_name = g_table["sharpe"].idxmax()
    best_single_sharpe = float(singles["sharpe"].max())
    best_blend_sharpe = float(blends["sharpe"].max())
    blend_best_sharpe = bool(g_table.loc[best_sharpe_name, "is_blend"])

    target_vol = float(np.sqrt(min_single_var))  # common risk budget (daily sd)
    vt_g = _vol_targeted_g(g_table, target_vol)
    vt_singles = vt_g[~g_table["is_blend"]]
    vt_blends = vt_g[g_table["is_blend"]]
    vt_best_single_name = vt_singles.idxmax()
    vt_best_blend_name = vt_blends.idxmax()
    vt_best_single_g = float(vt_singles.max())
    vt_best_blend_g = float(vt_blends.max())
    vt_beats = vt_best_blend_g > vt_best_single_g

    verdict = {
        "best_single": best_single_name,
        "best_single_g": best_single_g,
        "best_single_var": best_single_var,
        "min_single_var": min_single_var,
        "best_blend": best_blend_name,
        "best_blend_g": best_blend_g,
        "best_blend_var": best_blend_var,
        "blend_beats_single_on_g": bool(beats_on_g),
        "blend_lowers_variance_vs_best_single": bool(lowers_variance),
        "blend_lowers_variance_vs_min_single": bool(lowers_vs_min_single),
        "g_uplift": best_blend_g - best_single_g,
        # Sharpe (scale-invariant edge quality)
        "best_sharpe": best_sharpe_name,
        "best_single_sharpe": best_single_sharpe,
        "best_blend_sharpe": best_blend_sharpe,
        "blend_has_best_sharpe": blend_best_sharpe,
        # vol-targeted g (the risk-budget reading of the thesis)
        "target_vol_daily": target_vol,
        "vt_best_single": vt_best_single_name,
        "vt_best_single_g": vt_best_single_g,
        "vt_best_blend": vt_best_blend_name,
        "vt_best_blend_g": vt_best_blend_g,
        "vt_blend_beats_single_on_g": bool(vt_beats),
        "vt_g_uplift": vt_best_blend_g - vt_best_single_g,
    }

    return PortfolioAnalysis(
        returns=mat, corr=corr, equal_weights=eq_w, minvar_weights=mv_w,
        g_table=g_table, verdict=verdict, names=names,
    )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def _fmt_pf(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{x:.3f}"


def format_g_table(g_table: pd.DataFrame) -> str:
    """Render the g-table as a fixed-width text block (mean/var in bps)."""
    lines = []
    hdr = (f"  {'strategy / blend':<34s} {'n':>4s} {'mean(bps)':>10s} "
           f"{'var(1e-4)':>10s} {'sharpe':>8s} {'g(bps)':>9s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for name, r in g_table.iterrows():
        lines.append(
            f"  {name:<34s} {int(r['n_days']):>4d} "
            f"{r['mean_daily']*1e4:>10.3f} {r['var_daily']*1e4:>10.4f} "
            f"{r['sharpe']:>8.2f} {r['g']*1e4:>9.3f}"
            + ("  <- blend" if r["is_blend"] else "")
        )
    return "\n".join(lines)


def format_corr(corr: pd.DataFrame) -> str:
    """Render the correlation matrix as a fixed-width text block."""
    if corr is None or corr.shape[1] == 0:
        return "  (no correlation matrix)"
    names = list(corr.columns)
    short = {n: n.split("/")[0][:10] for n in names}
    lines = []
    hdr = "  " + " " * 14 + "".join(f"{short[n]:>12s}" for n in names)
    lines.append(hdr)
    for rn in names:
        row = "  " + f"{short[rn]:<14s}" + "".join(
            f"{corr.loc[rn, cn]:>+12.3f}" for cn in names
        )
        lines.append(row)
    return "\n".join(lines)


def print_report(an: PortfolioAnalysis) -> None:
    """Print the correlation matrix, g-table, weights, and the verdict."""
    print("TradeForge — edge-portfolio / maximization analysis (realistic costs)")
    print(f"sessions in union calendar: {len(an.returns)}")
    print(f"strategies: {', '.join(an.names)}")

    print("\n===== correlation matrix (daily fractional returns) =====")
    print(format_corr(an.corr))

    print("\n===== min-variance long-only weights =====")
    for name, w in zip(an.names, an.minvar_weights):
        print(f"  {name:<34s} {w:6.3f}")
    print("  (equal weights: " +
          ", ".join(f"{w:.3f}" for w in an.equal_weights) + ")")

    print("\n===== g-table: singles vs blends (g = mean - 0.5*var) =====")
    print(format_g_table(an.g_table))

    v = an.verdict
    print("\n===== MAXIMIZATION VERDICT =====")
    print(f"  best single        : {v['best_single']}  "
          f"(g = {v['best_single_g']*1e4:+.3f} bps, var = {v['best_single_var']*1e4:.4f}e-4)")
    print(f"  best blend         : {v['best_blend']}  "
          f"(g = {v['best_blend_g']*1e4:+.3f} bps, var = {v['best_blend_var']*1e4:.4f}e-4)")
    print(f"  g uplift (blend - best single) : {v['g_uplift']*1e4:+.4f} bps")
    print(f"  blend beats best single on g?  : {v['blend_beats_single_on_g']}")
    print(f"  blend variance < best single's : {v['blend_lowers_variance_vs_best_single']}")
    print(f"  blend variance < MIN single's  : {v['blend_lowers_variance_vs_min_single']}")

    print("\n  -- scale-invariant reading (the risk-budget mechanism) --")
    print(f"  highest Sharpe     : {v['best_sharpe']} "
          f"({'BLEND' if v['blend_has_best_sharpe'] else 'single'}; "
          f"blend {v['best_blend_sharpe']:.2f} vs best single {v['best_single_sharpe']:.2f})")
    print(f"  vol-targeted g (all scaled to daily sd "
          f"{v['target_vol_daily']*1e4:.2f} bps):")
    print(f"    best single  : {v['vt_best_single']}  "
          f"g = {v['vt_best_single_g']*1e4:+.3f} bps")
    print(f"    best blend   : {v['vt_best_blend']}  "
          f"g = {v['vt_best_blend_g']*1e4:+.3f} bps")
    print(f"    blend beats single on vol-targeted g? : "
          f"{v['vt_blend_beats_single_on_g']}  "
          f"(uplift {v['vt_g_uplift']*1e4:+.4f} bps)")

    if v["blend_beats_single_on_g"]:
        print("\n  VERDICT: the decorrelated blend RAISES the geometric-growth "
              "proxy g above the best single strategy — the maximization thesis "
              "(cut variance for a given edge -> higher compounded growth) HOLDS "
              "on this edge set net of realistic costs.")
    elif v["blend_lowers_variance_vs_best_single"]:
        extra = ""
        if v["blend_has_best_sharpe"] and v["vt_blend_beats_single_on_g"]:
            extra = (" CRUCIALLY, on the scale-invariant reading — the actual "
                     "maximization mechanism — the min-variance blend has the "
                     "HIGHEST Sharpe and, once every sleeve is levered to a common "
                     "risk budget, the HIGHEST vol-targeted g. So the thesis holds "
                     "in the form that matters operationally (size to a fixed risk, "
                     "compound faster); it only fails on the raw unscaled g because "
                     "at these tiny daily magnitudes the ½·var penalty is "
                     "negligible and a high-mean/high-vol single wins the unscaled "
                     "number it would never be allowed to run at full size.")
        print("\n  VERDICT (partial): the blend does NOT beat the best single on "
              "the RAW g, but it DOES cut variance below the best single's." + extra)
    else:
        print("\n  VERDICT (negative): the blend neither beats the best single on g "
              "nor cuts variance below it. The maximization benefit did not "
              "materialize on this edge set — the single dominates on both mean "
              "and variance. Honest read: these are not the uncorrelated, "
              "comparably-strong edges the thesis needs.")


def main() -> int:
    con = connect(DEFAULT_DB_PATH)
    try:
        an = analyze(con=con)
        print_report(an)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
