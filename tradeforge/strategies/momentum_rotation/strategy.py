"""strategies/momentum_rotation/strategy.py — the MOMENTUM / DUAL-MOMENTUM
ROTATION sleeve (the slow, cross-asset, trend-filtered allocation edge).

This is the daily-NAV sibling of the intraday breakout/fade edges: a
cross-sectional, LONG-ONLY, monthly-rebalanced target-weight allocator that
implements the :class:`~backtest.daily.engine.DailyStrategy` interface
(``target_weights(asof_date, history) -> {symbol: fraction}``). It is the kind of
edge the deep daily Yahoo data layer exists to feed — decades of total-return
history across a broad ETF universe (MASTER_PLAN.md §1.A/B edge-stacking, §5
"slow cross-asset momentum / trend / rotation").

WHY TREND-FILTERED MOMENTUM (the thesis)
----------------------------------------
Time-series (absolute) momentum is the single most robust premium in the
cross-asset literature: an asset above its own long trend tends to keep rising;
below it, to keep falling. Filtering exposure by a 200-day SMA (or a 12-month
absolute-momentum gate vs cash) sidesteps the worst equity drawdowns — you are
in bonds/cash during 2000-02 and 2008 rather than riding them down. The cost is
whipsaw in choppy markets and a lag at turns. The PAYOFF the maximization thesis
cares about is LOWER VARIANCE for a given mean (g ≈ mean − var/2): a shallower
maxDD lets you size larger at the same drawdown budget, which compounds faster.
This sleeve's job is therefore lower-maxDD equity growth, decorrelated from the
intraday edges by both horizon (months vs minutes) and mechanism (trend vs
break/fade).

THREE STACKED COMPONENTS (each a long-only weight contributor)
--------------------------------------------------------------
1. GEM-lite DUAL MOMENTUM (``gem_weight`` of the book):
     RELATIVE: pick the stronger of US (``us_asset``=SPY) vs ex-US
       (``exus_asset``=VXUS) by 12-month total return.
     ABSOLUTE gate: hold that winner only if its own 12m return beats cash
       (``cash_asset``=BIL) OR it is above its 200d SMA (``abs_mom_use_sma``;
       set false to require BOTH). If the gate fails -> bonds (``safe_asset``).
   The canonical Antonacci GEM core trimmed to two risk assets + a safe asset.

2. SECTOR RS ROTATION (``sector_weight`` of the book):
     Rank the 11 SPDR sectors by a BLENDED 3/6/12m total return
     (``blend_weights``), hold the ``top_n`` equal-weight, and gate EACH held
     sector by its OWN 200d SMA — a sector below trend forfeits its slot to the
     safe asset. Cross-sectional relative strength + a per-name absolute filter.

3. VOL-SCALING overlay (applied to the COMBINED book):
     Scale gross exposure inversely to recent realized vol toward ``vol_target``.
     gross = clamp(vol_target / realized_vol, ``min_gross``, ``max_leverage_*``).
     Under the conservative defaults max gross is 1.0 (NO leverage): the overlay
     only ever DE-risks (cuts to cash) when realized vol runs above target.

GROWTH LEVERS (RESEARCH — validate before live; DEFAULTS OFF / conservative)
----------------------------------------------------------------------------
* ``offensive_risk_off``: when ON and the downtrend is STRONG (gated asset below
  its 200d SMA AND 12m momentum negative), route the risk-OFF slot into a -1x
  INVERSE ETF (``inverse_us``=SH for the SPY side, ``inverse_qqq``=PSQ for the
  tech side, ``inverse_smallcap``=RWM) instead of bonds. HONESTY: inverse ETFs
  bleed from daily-reset decay + short-term tax churn; they help only in
  sustained, low-whipsaw downtrends. OFF by default.
* ``leveraged_long``: when ON and in a CONFIRMED strong uptrend (price > 200d SMA
  AND 12m momentum > ``strong_mom_threshold``), tilt ``lev_fraction`` of the
  risk-ON slot into a 2x/3x LONG (``lev_us``=QLD / ``lev_qqq``=TQQQ). HONESTY: 3x
  funds suffer volatility decay and brutal drawdowns; the trend filter mitigates
  but does not remove this. OFF by default.

CONTRACT COMPLIANCE (enforced by the engine)
--------------------------------------------
Every returned weight is a POSITIVE fraction of equity; the total is clamped to
``<= 1`` (remainder is cash). Inverse exposure is a POSITIVE weight on an inverse
ETF — never a short. All lookbacks read point-in-time ADJUSTED daily closes from
the :class:`~backtest.daily.engine.DailyHistory` view, which is sliced to
``<= asof_date`` BEFORE the strategy sees it, so a no-lookahead bug is impossible
by construction.

PURE / DETERMINISTIC: no LLM, no MCP, no network. The signal math below is a set
of free functions operating on plain price frames so each piece (ranking, the
200d gate, weight construction, the toggles) is unit-tested on synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"

# The default variant the integration stage runs uniformly.
DEFAULT_VARIANT = "DEFAULT"
# Variants exposed for uniform sweeping by the integration stage.
VARIANTS = [
    "DEFAULT",
    "OFFENSIVE_OFF_ON",
    "LEVERAGED_ON",
    "BOTH_LEVERS_ON",
    "GEM_ONLY",
    "SECTOR_ONLY",
]


def load_params(
    variant: str = DEFAULT_VARIANT, path: str | Path = DEFAULT_PARAMS_PATH
) -> dict:
    """Load ``defaults`` merged with a named ``variant`` delta from params.yaml."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    params = dict(raw.get("defaults", {}))
    variants = raw.get("variants", {}) or {}
    if variant not in variants:
        raise KeyError(f"unknown variant {variant!r}; have {sorted(variants)}")
    delta = variants[variant] or {}
    params.update(delta)
    params["_variant"] = variant
    return params


# --------------------------------------------------------------------------- #
# Pure signal functions (each unit-tested on synthetic price frames)
# --------------------------------------------------------------------------- #
def total_return(prices: pd.Series, lookback: int) -> float:
    """Total return of an ADJUSTED-close series over the last ``lookback`` steps.

    Uses ``prices[-1] / prices[-1-lookback] - 1`` on a series already clipped to
    ``<= asof_date``. Because the closes are TOTAL-RETURN adjusted, this ratio is
    the realized total return (price + reinvested dividends) over the window.
    Returns ``nan`` when there is not enough history (fewer than ``lookback+1``
    finite points) — the caller treats a NaN momentum as "ineligible".
    """
    s = pd.Series(prices, dtype="float64").dropna()
    if len(s) < lookback + 1 or lookback <= 0:
        return float("nan")
    past = float(s.iloc[-1 - lookback])
    last = float(s.iloc[-1])
    if past <= 0:
        return float("nan")
    return last / past - 1.0


def blended_momentum(
    prices: pd.Series,
    lookbacks: tuple[int, int, int],
    weights: tuple[float, float, float],
) -> float:
    """Weighted average of the 3/6/12m total returns (the sector RS score).

    ``lookbacks`` are the (3m, 6m, 12m) horizons in trading days; ``weights`` are
    their blend weights (need not sum to 1 — only the relative ranking matters).
    A horizon with insufficient history contributes ``nan`` and drops out, with
    the remaining weights renormalized. Returns ``nan`` only if EVERY horizon is
    unavailable (too little history to rank the name at all).
    """
    scores = [total_return(prices, lb) for lb in lookbacks]
    num = 0.0
    den = 0.0
    for sc, w in zip(scores, weights):
        if np.isfinite(sc) and w != 0:
            num += w * sc
            den += w
    if den == 0:
        return float("nan")
    return num / den


def above_sma(prices: pd.Series, window: int) -> bool:
    """True if the latest ADJUSTED close is at/above its ``window``-day SMA.

    The absolute trend gate. ``window`` finite closes are required; with fewer
    (a young ticker) the gate returns ``False`` (untrusted -> treat as below
    trend, the conservative side). Equality counts as above (``>=``).
    """
    s = pd.Series(prices, dtype="float64").dropna()
    if len(s) < window or window <= 0:
        return False
    sma = float(s.iloc[-window:].mean())
    last = float(s.iloc[-1])
    return last >= sma


def rank_sectors(
    panel: pd.DataFrame,
    sectors: list[str],
    lookbacks: tuple[int, int, int],
    weights: tuple[float, float, float],
) -> list[tuple[str, float]]:
    """Rank ``sectors`` by their blended 3/6/12m momentum, best first.

    ``panel`` is a wide ADJUSTED-close frame (columns include the sectors).
    Returns a list of ``(symbol, score)`` sorted DESCENDING by score, dropping
    any sector whose score is ``nan`` (too little history / missing column). Ties
    break by symbol name for determinism.
    """
    scored: list[tuple[str, float]] = []
    for sym in sectors:
        if sym not in panel.columns:
            continue
        score = blended_momentum(panel[sym], lookbacks, weights)
        if np.isfinite(score):
            scored.append((sym, score))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored


def realized_vol(
    prices: pd.Series, window: int, periods_per_year: int = 252
) -> float:
    """Annualized realized vol of daily returns over the last ``window`` days.

    Std (ddof=1) of the most recent ``window`` daily simple returns, annualized
    by ``sqrt(periods_per_year)``. Returns ``nan`` with fewer than 2 usable
    returns. Used by :func:`vol_scalar` to size gross exposure inversely to vol.
    """
    s = pd.Series(prices, dtype="float64").dropna()
    rets = s.pct_change().dropna()
    if window is not None and window > 0:
        rets = rets.iloc[-window:]
    if len(rets) < 2:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def vol_scalar(
    realized: float, vol_target: float, min_gross: float, max_gross: float
) -> float:
    """Gross-exposure multiplier that targets ``vol_target`` realized vol.

    ``gross = clamp(vol_target / realized, min_gross, max_gross)``. A high recent
    vol -> a scalar below 1 (de-risk toward cash); a quiet market -> toward
    ``max_gross`` (1.0 under conservative defaults = no leverage). When realized
    vol is unavailable / non-positive (a brand-new book) we fall back to
    ``min(1.0, max_gross)`` rather than dividing by zero — neutral, never levered
    by an accident of missing data.
    """
    if not np.isfinite(realized) or realized <= 0:
        return min(1.0, max_gross)
    raw = vol_target / realized
    return float(min(max(raw, min_gross), max_gross))


def gem_select(
    panel: pd.DataFrame,
    us_asset: str,
    exus_asset: str,
    cash_asset: str,
    safe_asset: str,
    lookback_12m: int,
    sma_window: int,
    abs_mom_use_sma: bool,
) -> dict:
    """GEM-lite dual-momentum selection -> a one-asset decision dict.

    RELATIVE leg: choose the higher 12m total return of ``us_asset`` vs
    ``exus_asset`` (a NaN momentum loses to a finite one; both NaN -> safe asset).
    ABSOLUTE gate on the winner:
      * 12m return beats cash (``cash_asset``'s 12m total return), OR
      * (``abs_mom_use_sma``) price above its 200d SMA.
      With ``abs_mom_use_sma=False`` BOTH conditions are required (stricter).
    Pass -> hold the winner; fail -> hold ``safe_asset`` (bonds).

    Returns a dict::

        {"asset": <held symbol>, "winner": <relative winner>,
         "risk_on": <bool>, "strong_down": <bool>, "side": "us"|"exus"|None}

    ``risk_on`` is True when the gate passed (we hold equity), False when it
    failed (we hold bonds / the offensive risk-off slot). ``strong_down`` is True
    when the gate failed AND the winner's 12m momentum is negative AND it is below
    its SMA — the "strong downtrend" condition the offensive inverse lever needs.
    ``side`` tags which equity leg won (drives which inverse/leveraged ETF to use).
    """
    mom_us = total_return(panel[us_asset], lookback_12m) if us_asset in panel else float("nan")
    mom_ex = total_return(panel[exus_asset], lookback_12m) if exus_asset in panel else float("nan")
    mom_cash = total_return(panel[cash_asset], lookback_12m) if cash_asset in panel else float("nan")
    if not np.isfinite(mom_cash):
        mom_cash = 0.0  # no cash history -> treat the absolute hurdle as 0%.

    # Relative leg: pick the higher finite momentum; NaN never wins.
    us_ok = np.isfinite(mom_us)
    ex_ok = np.isfinite(mom_ex)
    if not us_ok and not ex_ok:
        return {"asset": safe_asset, "winner": None, "risk_on": False,
                "strong_down": False, "side": None}
    if us_ok and (not ex_ok or mom_us >= mom_ex):
        winner, mom_w, side = us_asset, mom_us, "us"
    else:
        winner, mom_w, side = exus_asset, mom_ex, "exus"

    # Absolute gate on the winner.
    beats_cash = np.isfinite(mom_w) and mom_w > mom_cash
    above = above_sma(panel[winner], sma_window) if winner in panel else False
    if abs_mom_use_sma:
        risk_on = bool(beats_cash or above)
    else:
        risk_on = bool(beats_cash and above)

    strong_down = (not risk_on) and np.isfinite(mom_w) and (mom_w < 0) and (not above)
    asset = winner if risk_on else safe_asset
    return {"asset": asset, "winner": winner, "risk_on": risk_on,
            "strong_down": strong_down, "side": side}


def _add_weight(weights: dict, sym: str, w: float) -> None:
    """Accumulate ``w`` onto ``weights[sym]`` (merging duplicate destinations)."""
    if sym is None or w <= 0:
        return
    weights[sym] = weights.get(sym, 0.0) + float(w)


# --------------------------------------------------------------------------- #
# The strategy (DailyStrategy: target_weights(asof_date, history) -> weights)
# --------------------------------------------------------------------------- #
class MomentumRotationStrategy:
    """Trend-filtered dual-momentum + sector-RS rotation, vol-scaled, long-only.

    Implements the :class:`~backtest.daily.engine.DailyStrategy` protocol. The
    heavy lifting lives in the pure free functions above; this class only reads
    params, pulls the point-in-time price window from ``history``, and assembles
    the three sleeves into one ``{symbol: weight}`` vector clamped to ``<= 1``.
    """

    def __init__(self, params: dict | None = None, variant: str = DEFAULT_VARIANT):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)

        p = self.params
        # universe roles
        self.us_asset = str(p["us_asset"])
        self.exus_asset = str(p["exus_asset"])
        self.cash_asset = str(p["cash_asset"])
        self.safe_asset = str(p["safe_asset"])
        self.sectors = list(p["sectors"])

        # lookbacks + gates
        self.lookback_3m = int(p["lookback_3m"])
        self.lookback_6m = int(p["lookback_6m"])
        self.lookback_12m = int(p["lookback_12m"])
        self.lookbacks = (self.lookback_3m, self.lookback_6m, self.lookback_12m)
        self.blend_weights = tuple(float(w) for w in p["blend_weights"])
        self.sma_window = int(p["sma_window"])

        # sleeve mixing + selection
        self.gem_weight = float(p["gem_weight"])
        self.sector_weight = float(p["sector_weight"])
        self.top_n = int(p["top_n"])
        self.abs_mom_use_sma = bool(p["abs_mom_use_sma"])

        # vol overlay
        self.vol_window = int(p["vol_window"])
        self.vol_target = float(p["vol_target"])
        self.max_leverage_off = float(p["max_leverage_off"])
        self.min_gross = float(p["min_gross"])

        # growth levers (default OFF)
        self.offensive_risk_off = bool(p["offensive_risk_off"])
        self.inverse_us = str(p["inverse_us"])
        self.inverse_qqq = str(p["inverse_qqq"])
        self.inverse_smallcap = str(p["inverse_smallcap"])
        self.leveraged_long = bool(p["leveraged_long"])
        self.lev_us = str(p["lev_us"])
        self.lev_qqq = str(p["lev_qqq"])
        self.lev_fraction = float(p["lev_fraction"])
        self.strong_mom_threshold = float(p["strong_mom_threshold"])

        # The longest window we need before any signal is trustworthy.
        self._min_history = max(self.lookback_12m + 1, self.sma_window)

    # ------------------------------------------------------------ extra symbols
    def extra_symbols(self) -> list[str]:
        """Tickers BEYOND the obvious holdings the engine must also price.

        The universe passed to ``run_daily`` must include every symbol this
        strategy can possibly return a weight on — the GEM legs, cash benchmark,
        safe asset, sectors, AND (when the levers are on) the inverse/leveraged
        ETFs. Returned so a driver can union it into the universe.
        """
        return [
            self.us_asset, self.exus_asset, self.cash_asset, self.safe_asset,
            *self.sectors,
            self.inverse_us, self.inverse_qqq, self.inverse_smallcap,
            self.lev_us, self.lev_qqq,
        ]

    # ------------------------------------------------------------ the interface
    def target_weights(self, asof_date, history) -> dict:
        """Point-in-time target weights for ``asof_date`` (the engine contract).

        Pulls the ADJUSTED-close window (already sliced to ``<= asof_date``),
        builds the GEM-lite core + sector-RS sleeve, applies the optional growth
        levers, then scales the whole book by the vol-target overlay. Returns a
        ``{symbol: fraction}`` dict with every weight >= 0 and the sum <= 1.
        """
        panel = history.prices()
        if panel is None or len(panel) == 0:
            return {}

        # ---- 1. raw sleeve weights (pre vol-scaling), each summing to its sleeve
        weights: dict[str, float] = {}
        self._gem_sleeve(panel, weights)
        self._sector_sleeve(panel, weights)

        if not weights:
            return {}

        # ---- 2. vol-scaling overlay on the COMBINED book ----
        gross = self._gross_scalar(panel, weights)
        scaled = {sym: w * gross for sym, w in weights.items()}

        # ---- 3. final long-only / sum<=1 clamp (defensive; gross<=1 already) ---
        total = sum(scaled.values())
        if total > 1.0:
            scaled = {sym: w / total for sym, w in scaled.items()}
        return {sym: w for sym, w in scaled.items() if w > 0}

    # ------------------------------------------------------------ GEM-lite core
    def _gem_sleeve(self, panel: pd.DataFrame, weights: dict) -> None:
        """Add the GEM-lite dual-momentum sleeve (``gem_weight``) to ``weights``."""
        if self.gem_weight <= 0:
            return
        sel = gem_select(
            panel, self.us_asset, self.exus_asset, self.cash_asset,
            self.safe_asset, self.lookback_12m, self.sma_window,
            self.abs_mom_use_sma,
        )
        w = self.gem_weight
        if sel["risk_on"]:
            asset = sel["asset"]
            # leveraged_long: tilt part of the risk-ON slot into a 2x/3x long when
            # the uptrend is CONFIRMED strong (price>SMA already implied by
            # risk_on, plus 12m momentum above the strong threshold).
            if self.leveraged_long and self._is_strong_up(panel, sel["winner"]):
                lev = self._lev_for_side(sel["side"])
                if lev is not None and lev in panel.columns:
                    _add_weight(weights, lev, w * self.lev_fraction)
                    _add_weight(weights, asset, w * (1.0 - self.lev_fraction))
                    return
            _add_weight(weights, asset, w)
        else:
            # risk-OFF: bonds, or (offensive lever) a -1x inverse ETF in a STRONG
            # downtrend on the winning side.
            if self.offensive_risk_off and sel["strong_down"]:
                inv = self._inverse_for_side(sel["side"])
                if inv is not None and inv in panel.columns:
                    _add_weight(weights, inv, w)
                    return
            _add_weight(weights, self.safe_asset, w)

    # --------------------------------------------------------- sector rotation
    def _sector_sleeve(self, panel: pd.DataFrame, weights: dict) -> None:
        """Add the sector RS rotation sleeve (``sector_weight``) to ``weights``.

        Rank sectors by blended momentum, take the top-N, equal-weight each slot,
        and gate each held sector by its own 200d SMA — a below-trend sector
        forfeits its slot to the safe asset (bonds). With the offensive lever on
        and a strong sector downtrend, that forfeited slot can instead route to an
        inverse ETF; conservative default keeps it in bonds.
        """
        if self.sector_weight <= 0:
            return
        ranked = rank_sectors(panel, self.sectors, self.lookbacks, self.blend_weights)
        if not ranked:
            # No rankable sector (too little history) -> whole sleeve to bonds.
            _add_weight(weights, self.safe_asset, self.sector_weight)
            return
        top = ranked[: self.top_n]
        slot = self.sector_weight / float(self.top_n)  # equal-weight each SLOT
        for sym, score in top:
            if above_sma(panel[sym], self.sma_window):
                # leveraged_long does NOT apply to individual sectors (no 2x/3x
                # sector ETF in the universe) — sectors stay 1x by design.
                _add_weight(weights, sym, slot)
            else:
                # below-trend sector -> safe asset (or inverse if offensive+strong).
                if self.offensive_risk_off and self._sector_strong_down(panel, sym):
                    _add_weight(weights, self.inverse_qqq, slot)
                else:
                    _add_weight(weights, self.safe_asset, slot)

    # ------------------------------------------------------------- vol overlay
    def _gross_scalar(self, panel: pd.DataFrame, weights: dict) -> float:
        """Vol-target gross multiplier for the combined book.

        Realized vol is measured on the WEIGHTED portfolio's synthetic return
        series (each held name's daily returns weighted by its raw pre-scaling
        weight) over ``vol_window`` days — a true book-level vol, not a single
        proxy. Falls back to the US asset's vol if the book series is too short.
        """
        max_gross = self.max_leverage_off
        # Build the weighted portfolio return series from the held names.
        cols = [s for s in weights if s in panel.columns]
        if cols:
            rets = panel[cols].pct_change().dropna(how="all")
            if len(rets) >= 2:
                w = np.array([weights[s] for s in cols], dtype="float64")
                wsum = w.sum()
                if wsum > 0:
                    w = w / wsum
                window = rets.iloc[-self.vol_window:]
                port_ret = window.fillna(0.0).to_numpy() @ w
                if len(port_ret) >= 2:
                    rv = float(np.std(port_ret, ddof=1) * np.sqrt(252))
                    return vol_scalar(rv, self.vol_target, self.min_gross, max_gross)
        # Fallback: vol of the US leg.
        rv = realized_vol(panel.get(self.us_asset, pd.Series(dtype="float64")), self.vol_window)
        return vol_scalar(rv, self.vol_target, self.min_gross, max_gross)

    # --------------------------------------------------------- lever helpers
    def _is_strong_up(self, panel: pd.DataFrame, sym: str | None) -> bool:
        """Confirmed-strong-uptrend test for the leveraged_long lever.

        Requires the name above its 200d SMA AND its 12m total return above
        ``strong_mom_threshold`` — both, so a marginal/early uptrend does not earn
        the 2x/3x tilt (the trend filter that mitigates leverage decay).
        """
        if sym is None or sym not in panel.columns:
            return False
        if not above_sma(panel[sym], self.sma_window):
            return False
        mom = total_return(panel[sym], self.lookback_12m)
        return np.isfinite(mom) and mom > self.strong_mom_threshold

    def _sector_strong_down(self, panel: pd.DataFrame, sym: str) -> bool:
        """A below-trend sector is a STRONG downtrend (negative 12m momentum)."""
        mom = total_return(panel[sym], self.lookback_12m)
        return np.isfinite(mom) and mom < 0 and not above_sma(panel[sym], self.sma_window)

    def _inverse_for_side(self, side: str | None) -> str | None:
        """Map the winning equity side to its -1x inverse ETF."""
        if side == "us":
            return self.inverse_us
        if side == "exus":
            return self.inverse_smallcap  # no -1x ex-US in the universe; small-cap proxy
        return self.inverse_us

    def _lev_for_side(self, side: str | None) -> str | None:
        """Map the winning equity side to its 2x/3x leveraged-long ETF.

        The US leg pairs with QLD (2x S&P-like). There is no leveraged ex-US fund
        in the universe, so an ex-US winner does NOT get a leveraged tilt — it
        stays 1x (returning None keeps the slot at the plain risk-ON asset).
        """
        if side == "us":
            return self.lev_us
        return None
