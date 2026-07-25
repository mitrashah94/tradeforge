"""reporting/charts.py — headless tear-sheet + portfolio chart rendering.

Generates journal / digest / backtest PNGs without a display (MASTER_PLAN.md §4,
§9). Everything renders through the matplotlib **Agg** backend so it works on a
headless cloud host with no X server. Heavy plotting imports stay inside the
functions (the module imports cheaply; nothing happens until you render).

Public API
----------
    tear_sheet(result_or_returns, title, out_path, ...) -> str
        Per-strategy tear sheet: equity curve, drawdown underwater, return
        distribution, and a stats box (PF, expectancy_R, win%, n, Sharpe, maxDD).
        Accepts a :class:`BacktestResult` (richest — full trade ledger + equity
        curve drives every panel) OR a bare daily-return Series (e.g. a blend,
        which has no trade ledger) — in that case the panels are built from the
        synthetic equity curve implied by the returns.

    correlation_heatmap(corr_df, out_path, ...) -> str
        Annotated correlation-matrix heatmap (the live strategy correlation
        matrix from ``backtest.portfolio``).

    blend_vs_singles(g_table, out_path, ...) -> str
        Grouped bar chart of the geometric-growth proxy g = mean - 0.5*var for
        every single strategy and every blend, with the best single highlighted
        — the visual of the maximization verdict.

All renderers return the output path (str) and write a PNG under whatever
``out_path`` directory the caller chooses (the integration stage uses
``backtest/reports/``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Backend: force Agg BEFORE importing pyplot so we never need a display.
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent.parent / "backtest" / "reports"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _ensure_parent(out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _fmt(x: float, nd: int = 3, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "inf" if xf > 0 else ("-inf" if xf < 0 else "n/a")
    if pct:
        return f"{xf * 100:.1f}%"
    return f"{xf:.{nd}f}"


def _is_result(obj) -> bool:
    """Duck-type a BacktestResult (has .trades + .equity_curve + summary())."""
    return (
        hasattr(obj, "trades")
        and hasattr(obj, "equity_curve")
        and hasattr(obj, "summary")
    )


def _stats_from_result(result) -> dict:
    """Pull the tear-sheet stats bundle from a BacktestResult."""
    from backtest.runner import daily_returns_pct
    from backtest.stats.metrics import sharpe

    s = result.summary()
    dr = daily_returns_pct(result)
    shp = sharpe(dr) if len(dr) >= 2 else float("nan")
    return {
        "n_trades": s["n_trades"],
        "profit_factor": s["profit_factor"],
        "expectancy_r": s["expectancy_R"],
        "win_rate": s["win_rate"],
        "sharpe": shp,
        "max_drawdown_pct": s["max_drawdown_pct"],
        "net_profit": s["net_profit"],
    }


def _equity_from_returns(returns: pd.Series, base: float = 100_000.0) -> pd.Series:
    """Synthetic equity curve from a daily $ or fractional return series.

    If the magnitudes look like fractions (|values| < 1), they are treated as
    fractional daily returns compounded onto ``base``; otherwise they are treated
    as $ PnL added to ``base``. The index is preserved.
    """
    if returns is None or len(returns) == 0:
        return pd.Series([base], dtype="float64")
    vals = returns.astype("float64")
    looks_fractional = float(vals.abs().max()) < 1.0
    if looks_fractional:
        eq = base * (1.0 + vals).cumprod()
    else:
        eq = base + vals.cumsum()
    return eq


def _stats_from_returns(returns: pd.Series, base: float = 100_000.0) -> dict:
    """Tear-sheet stats from a bare daily-return Series (no trade ledger).

    PF / expectancy_R / win_rate here are DAILY-bucket analogues (computed over
    sessions, not individual trades) — the only thing a blend's return stream can
    support. The stats box labels them as daily where that matters.
    """
    from backtest.stats.metrics import max_drawdown, sharpe

    vals = returns.astype("float64").to_numpy() if len(returns) else np.asarray([])
    if vals.size == 0:
        return {
            "n_days": 0,
            "profit_factor": float("nan"),
            "win_rate": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown_pct": 0.0,
            "mean": float("nan"),
            "net": 0.0,
        }
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals < 0].sum())
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else float("nan"))
    eq = _equity_from_returns(returns, base=base)
    return {
        "n_days": int(vals.size),
        "profit_factor": pf,
        "win_rate": float((vals > 0).mean()),
        "sharpe": sharpe(vals) if vals.size >= 2 else float("nan"),
        "max_drawdown_pct": max_drawdown(eq) * 100.0,
        "mean": float(vals.mean()),
        "net": float(vals.sum()),
    }


# --------------------------------------------------------------------------- #
# Tear sheet
# --------------------------------------------------------------------------- #
def tear_sheet(
    result_or_returns,
    title: str,
    out_path: str | Path,
    base_equity: float = 100_000.0,
) -> str:
    """Render a per-strategy tear sheet PNG and return ``out_path``.

    Panels:
      * equity curve (top, full width),
      * drawdown underwater (middle-left),
      * per-session return distribution (middle-right),
      * a stats box (bottom): PF, expectancy_R, win%, n, Sharpe, maxDD.

    ``result_or_returns`` is a :class:`BacktestResult` (preferred — full ledger)
    or a daily-return :class:`pandas.Series` (a blend has no ledger). ``title``
    is the figure suptitle; ``out_path`` is where the PNG is written.
    """
    out = _ensure_parent(out_path)

    if _is_result(result_or_returns):
        from backtest.runner import daily_returns_pct

        result = result_or_returns
        eq = result.equity_curve
        if eq is None or len(eq) == 0:
            eq = _equity_from_returns(pd.Series(dtype="float64"), base=base_equity)
        rets = daily_returns_pct(result)
        stats = _stats_from_result(result)
        is_result = True
    else:
        rets = pd.Series(result_or_returns).astype("float64")
        eq = _equity_from_returns(rets, base=base_equity)
        stats = _stats_from_returns(rets, base=base_equity)
        is_result = False

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.0, 1.6, 0.9], hspace=0.42, wspace=0.22)

    # ---- equity curve (top, full width) ----
    ax_eq = fig.add_subplot(gs[0, :])
    eq_vals = np.asarray(eq, dtype="float64")
    ax_eq.plot(range(len(eq_vals)), eq_vals, color="#1f77b4", lw=1.4)
    ax_eq.axhline(base_equity, color="#888", lw=0.8, ls="--", alpha=0.6)
    ax_eq.set_title("Equity curve", fontsize=10, loc="left")
    ax_eq.set_ylabel("equity ($)")
    ax_eq.grid(alpha=0.25)

    # ---- drawdown underwater (middle-left) ----
    ax_dd = fig.add_subplot(gs[1, 0])
    if eq_vals.size:
        peak = np.maximum.accumulate(eq_vals)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (eq_vals - peak) / peak, 0.0) * 100.0
        ax_dd.fill_between(range(len(dd)), dd, 0.0, color="#d62728", alpha=0.45)
        ax_dd.plot(range(len(dd)), dd, color="#d62728", lw=0.8)
    ax_dd.set_title("Drawdown (underwater)", fontsize=10, loc="left")
    ax_dd.set_ylabel("drawdown (%)")
    ax_dd.grid(alpha=0.25)

    # ---- return distribution (middle-right) ----
    ax_h = fig.add_subplot(gs[1, 1])
    rv = rets.astype("float64").to_numpy() if len(rets) else np.asarray([])
    if rv.size:
        ax_h.hist(rv * 100.0, bins=min(40, max(8, rv.size // 3)),
                  color="#2ca02c", alpha=0.7, edgecolor="white", lw=0.3)
        ax_h.axvline(0.0, color="#444", lw=0.9, ls="--")
        ax_h.axvline(rv.mean() * 100.0, color="#1f77b4", lw=1.2,
                     label=f"mean {rv.mean()*100:+.3f}%")
        ax_h.legend(fontsize=7, loc="upper right")
    ax_h.set_title("Per-session return distribution", fontsize=10, loc="left")
    ax_h.set_xlabel("session return (%)")
    ax_h.grid(alpha=0.25)

    # ---- stats box (bottom, full width) ----
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis("off")
    if is_result:
        rows = [
            ("Profit factor", _fmt(stats["profit_factor"])),
            ("Expectancy (R)", _fmt(stats["expectancy_r"])),
            ("Win rate", _fmt(stats["win_rate"], pct=True)),
            ("Trades (n)", str(stats["n_trades"])),
            ("Sharpe (ann.)", _fmt(stats["sharpe"], nd=2)),
            ("Max drawdown", _fmt(stats["max_drawdown_pct"], nd=2) + "%"),
            ("Net profit", f"${stats['net_profit']:,.0f}"),
        ]
    else:
        rows = [
            ("Profit factor (daily)", _fmt(stats["profit_factor"])),
            ("Win rate (daily)", _fmt(stats["win_rate"], pct=True)),
            ("Sessions (n)", str(stats["n_days"])),
            ("Mean daily return", _fmt(stats["mean"] * 100, nd=4) + "%"
             if math.isfinite(stats["mean"]) else "n/a"),
            ("Sharpe (ann.)", _fmt(stats["sharpe"], nd=2)),
            ("Max drawdown", _fmt(stats["max_drawdown_pct"], nd=2) + "%"),
            ("Net return", _fmt(stats["net"] * 100, nd=2) + "%"
             if math.isfinite(stats["net"]) else "n/a"),
        ]
    # Lay the stats out as two columns of label: value pairs.
    n = len(rows)
    half = (n + 1) // 2
    col_x = [0.02, 0.52]
    for ci, chunk in enumerate((rows[:half], rows[half:])):
        y = 0.85
        for label, val in chunk:
            ax_t.text(col_x[ci], y, f"{label}:", fontsize=10, fontweight="bold",
                      transform=ax_t.transAxes, va="top")
            ax_t.text(col_x[ci] + 0.30, y, val, fontsize=10,
                      transform=ax_t.transAxes, va="top")
            y -= 0.26
    ax_t.set_title("Stats", fontsize=10, loc="left")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Correlation heatmap
# --------------------------------------------------------------------------- #
def correlation_heatmap(
    corr_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Strategy daily-return correlation matrix",
) -> str:
    """Render an annotated correlation-matrix heatmap PNG and return ``out_path``.

    ``corr_df`` is a square DataFrame of pairwise correlations (rows == cols ==
    strategy names), as produced by ``backtest.portfolio.correlation_matrix``.
    """
    out = _ensure_parent(out_path)
    labels = list(corr_df.columns)
    m = corr_df.to_numpy(dtype="float64")
    nlab = len(labels)

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * nlab, 1.6 + 1.0 * nlab))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(nlab))
    ax.set_yticks(range(nlab))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(nlab):
        for j in range(nlab):
            v = m[i, j]
            txtcolor = "white" if abs(v) > 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=txtcolor, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Per-trade chart (the journalist's chart PNG, MASTER_PLAN §4)
# --------------------------------------------------------------------------- #
def trade_chart(
    bars,
    out_path: str | Path,
    *,
    title: str = "trade",
    levels: dict | None = None,
    entry: dict | None = None,
    exit: dict | None = None,
    stop: float | None = None,
    target: float | None = None,
) -> str:
    """Render a per-trade candle chart with levels + fill markers; return path.

    The journalist's chart (MASTER_PLAN §4): the session's price candles, the
    strategy levels (PDH/PDL/PMH/PML), the planned stop/target, and the entry /
    exit fill markers, all on one headless (Agg) PNG.

    Args:
        bars: the session OHLC bars. Either a pandas DataFrame with columns
            ``open/high/low/close`` (optionally ``ts_utc``) or a list of dicts
            with those keys. An empty/None ``bars`` still renders a valid PNG
            (levels + markers only) so a chart is always produced.
        out_path: where to write the PNG.
        title: figure title (e.g. ``"QQQ breakout_retest 2026-06-01"``).
        levels: optional dict of horizontal levels to draw, e.g.
            ``{"PDH": 101.2, "PDL": 99.1, "PMH": ..., "PML": ...}`` — any subset.
        entry: optional ``{"price": float, "index": int|None}`` entry-fill marker.
        exit: optional ``{"price": float, "index": int|None}`` exit-fill marker.
        stop / target: optional planned protective/target price lines.

    Returns:
        The output path (str).
    """
    out = _ensure_parent(out_path)

    # ---- normalize bars to numpy OHLC arrays ----
    o = h = l = c = np.asarray([], dtype="float64")
    n = 0
    if bars is not None:
        if isinstance(bars, pd.DataFrame):
            if len(bars):
                o = bars["open"].to_numpy(dtype="float64")
                h = bars["high"].to_numpy(dtype="float64")
                l = bars["low"].to_numpy(dtype="float64")
                c = bars["close"].to_numpy(dtype="float64")
        else:
            rows = list(bars)
            if rows:
                o = np.asarray([float(r["open"]) for r in rows], dtype="float64")
                h = np.asarray([float(r["high"]) for r in rows], dtype="float64")
                l = np.asarray([float(r["low"]) for r in rows], dtype="float64")
                c = np.asarray([float(r["close"]) for r in rows], dtype="float64")
        n = len(c)

    fig, ax = plt.subplots(figsize=(11, 6))

    # ---- candles ----
    for i in range(n):
        color = "#2ca02c" if c[i] >= o[i] else "#d62728"
        # wick
        ax.plot([i, i], [l[i], h[i]], color=color, lw=0.8, zorder=2)
        # body
        lo, hi = min(o[i], c[i]), max(o[i], c[i])
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, lo), 0.6, max(hi - lo, 1e-9),
                facecolor=color, edgecolor=color, lw=0.5, zorder=3, alpha=0.9,
            )
        )

    # ---- horizontal strategy levels ----
    level_colors = {
        "PDH": "#1f77b4", "PDL": "#1f77b4",
        "PMH": "#9467bd", "PML": "#9467bd",
    }
    if levels:
        for name, price in levels.items():
            if price is None:
                continue
            col = level_colors.get(str(name).upper(), "#7f7f7f")
            ax.axhline(float(price), color=col, lw=1.0, ls="--", alpha=0.7, zorder=1)
            ax.text(0.002, float(price), f" {name}", color=col, fontsize=8,
                    va="bottom", ha="left", transform=ax.get_yaxis_transform())

    # ---- planned stop / target lines ----
    if stop is not None:
        ax.axhline(float(stop), color="#d62728", lw=1.1, ls=":", alpha=0.8, zorder=1)
        ax.text(0.998, float(stop), "stop ", color="#d62728", fontsize=8,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())
    if target is not None:
        ax.axhline(float(target), color="#2ca02c", lw=1.1, ls=":", alpha=0.8, zorder=1)
        ax.text(0.998, float(target), "target ", color="#2ca02c", fontsize=8,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())

    # ---- entry / exit fill markers ----
    def _xy(marker, default_idx):
        if not marker or marker.get("price") is None:
            return None
        idx = marker.get("index")
        if idx is None:
            idx = default_idx
        # Clamp into the drawable range so markers always land on the axes.
        if n:
            idx = max(0, min(int(idx), n - 1))
        else:
            idx = 0
        return (idx, float(marker["price"]))

    ex = _xy(entry, 0)
    xx = _xy(exit, (n - 1) if n else 0)
    if ex is not None:
        ax.scatter([ex[0]], [ex[1]], marker="^", s=140, color="#0b6e0b",
                   edgecolor="white", lw=0.8, zorder=5, label="entry")
        ax.annotate(f"entry {ex[1]:.2f}", ex, textcoords="offset points",
                    xytext=(6, 8), fontsize=8, color="#0b6e0b")
    if xx is not None:
        ax.scatter([xx[0]], [xx[1]], marker="v", s=140, color="#7a0b0b",
                   edgecolor="white", lw=0.8, zorder=5, label="exit")
        ax.annotate(f"exit {xx[1]:.2f}", xx, textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color="#7a0b0b")

    # If there are no bars, set a sane y-range from whatever prices we have.
    if n == 0:
        ys = [v for v in (
            stop, target,
            ex[1] if ex else None, xx[1] if xx else None,
            *([float(p) for p in levels.values() if p is not None] if levels else []),
        ) if v is not None]
        if ys:
            lo, hi = min(ys), max(ys)
            pad = max((hi - lo) * 0.1, 0.5)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-1, 1)

    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("bar index (session)")
    ax.set_ylabel("price")
    ax.grid(alpha=0.25)
    if ex is not None or xx is not None:
        ax.legend(fontsize=8, loc="best")

    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------- #
# Blend vs singles (the maximization chart)
# --------------------------------------------------------------------------- #
def blend_vs_singles(
    g_table: pd.DataFrame,
    out_path: str | Path,
    title: str = "Geometric-growth proxy g = mean − ½·var: singles vs blends",
) -> str:
    """Bar chart of g (and a variance sub-panel) for every single + blend.

    ``g_table`` is a DataFrame indexed by entity name with at least columns
    ``g``, ``var_daily`` and a boolean ``is_blend`` flag (as produced by
    ``backtest.portfolio.g_table``). Singles are drawn in one color, blends in
    another, and the best single's g is marked with a reference line so the
    "does a blend beat the best single?" verdict reads off the chart.
    """
    out = _ensure_parent(out_path)
    df = g_table.copy()
    names = list(df.index)
    g = df["g"].to_numpy(dtype="float64")
    var = df["var_daily"].to_numpy(dtype="float64")
    is_blend = (
        df["is_blend"].to_numpy(dtype=bool)
        if "is_blend" in df.columns
        else np.zeros(len(df), dtype=bool)
    )

    # Best SINGLE g (for the reference line).
    single_g = g[~is_blend]
    best_single = float(np.nanmax(single_g)) if single_g.size else float("nan")

    colors = ["#ff7f0e" if b else "#1f77b4" for b in is_blend]

    fig, (ax_g, ax_v) = plt.subplots(
        2, 1, figsize=(max(7, 1.1 * len(names)), 7.5),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.45},
    )

    x = np.arange(len(names))
    ax_g.bar(x, g * 1e4, color=colors, edgecolor="white", lw=0.5)  # g in bps
    if math.isfinite(best_single):
        ax_g.axhline(best_single * 1e4, color="#d62728", lw=1.2, ls="--",
                     label=f"best single g = {best_single*1e4:+.2f} bps")
        ax_g.legend(fontsize=8, loc="best")
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_g.set_ylabel("g  (bps/session)")
    ax_g.set_title(title, fontsize=11, fontweight="bold")
    ax_g.grid(alpha=0.25, axis="y")
    ax_g.axhline(0.0, color="#444", lw=0.8)
    for xi, gv in zip(x, g):
        ax_g.text(xi, gv * 1e4, f"{gv*1e4:+.2f}", ha="center",
                  va="bottom" if gv >= 0 else "top", fontsize=7)

    # Variance sub-panel (lower is better — the maximization denominator).
    ax_v.bar(x, var * 1e4, color=colors, edgecolor="white", lw=0.5)
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax_v.set_ylabel("var  (×1e-4)")
    ax_v.set_title("Daily-return variance (lower → higher g)", fontsize=10, loc="left")
    ax_v.grid(alpha=0.25, axis="y")

    # Legend for the color encoding.
    from matplotlib.patches import Patch

    handles = [
        Patch(color="#1f77b4", label="single strategy"),
        Patch(color="#ff7f0e", label="blend"),
    ]
    ax_v.legend(handles=handles, fontsize=8, loc="best")

    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)
