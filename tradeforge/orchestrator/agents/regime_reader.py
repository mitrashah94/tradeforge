"""orchestrator/agents/regime_reader.py — the daily policy-setter (deterministic).

The regime-reader is the lean roster's daily slow-loop agent (MASTER_PLAN.md §4).
Its LLM definition lives in ``.claude/agents/regime-reader.md``; THIS module is the
deterministic compute that LLM oversees. Per the core principle ("LLM decides
policy; deterministic code decides fast execution"), the *computable* part of the
daily policy is plain, network-free Python so the same DB + config always yield the
same assessment — and the fast loop / risk gate consume a typed event, not a chat.

What it produces each session (MASTER_PLAN.md §1 "regime-scaled exposure"):

  1. REGIME tag for the session — reuses :mod:`backtest.stats.regime` (the P2
     deterministic tagger: trend / chop / vol_shock). One source of truth for
     "what kind of day is this".
  2. A REALIZED-VOL read — today's true range as a multiple of the trailing
     Wilder ATR14 (``tr/atr``), the same volatility feature the tagger keys on,
     surfaced as a number the sizing layer can reason about. IV is a documented
     HOOK ONLY: there is no options feed yet, so ``iv`` is ``None`` (NEVER
     fabricated). Once an IV source exists, the ``iv-rank-skew-read`` skill fills
     this in (see the .md).
  3. ARMED strategies — regime -> which strategies are eligible this session
     (trend arms the continuation/thrust edges; chop arms mean-reversion;
     vol_shock stands down). The fast loop only triggers armed strategies.
  4. An EXPOSURE SCALAR in [0, ~1.25] — the accelerator/brake. It LEANS IN on a
     favorable trend (> 1.0) and CUTS in chop / vol_shock (< 1.0, down to 0 =
     stand down). It modulates conviction sizing by scaling the per-trade
     dollar-risk budget the fast loop computes (see :func:`exposure_scalar_for`
     and the mapping note below). It does NOT touch ``risk/limits.yaml``.

Exposure scalar -> conviction / RI mapping (HOW the fast loop consumes it)
-------------------------------------------------------------------------
The risk gate already turns a setup grade into a risk index and a $-risk budget:

    ri            = risk.sizing.resolve_ri(grade, limits, floor)      # B->5, A->6/7, A+->8
    dollar_risk   = risk.sizing.per_trade_dollar_risk(equity, ri, limits)  # pct * equity
    qty           = vol_target_qty(... dollar_risk ...)               # constant $-risk / stop

The exposure scalar is a MULTIPLIER on that dollar-risk budget, applied AFTER the
conviction tier picks the RI and BEFORE vol-target sizing converts it to a qty:

    effective_dollar_risk = exposure_scalar * dollar_risk

This keeps the conviction tier (edge -> RI -> base %) and the regime accelerator
(regime -> lean in / cut) as two clean, composable dials:

    * scalar 1.0  -> trade the conviction tier's full per-trade % unchanged.
    * scalar > 1.0 (trend) -> press: e.g. 1.20 risks 1.20x the tier's budget,
      i.e. an A setup at RI 6 (1.25%) effectively risks ~1.50% — equivalent to
      shifting RI ~one step up *within the band* without rewriting the live dial.
    * scalar < 1.0 (chop) -> cut: 0.60 risks 0.60x, pulling an RI-6 1.25% down
      toward ~0.75% — a soft de-risk that never widens risk and never edits YAML.
    * scalar 0.0 (vol_shock stand-down) -> no exposure: the fast loop sizes to 0
      and does not enter.

Equivalently, the scalar SHIFTS the effective RI inside the operating band: it is
a continuous in-band lever layered on the discrete conviction tier, so the regime
"accelerates" sizing (§1.A) while the band + catastrophe protections still cap it.
The scalar is intentionally clamped to a modest ceiling so a favorable regime can
lean in but never blow past the band the human set.

Determinism / firewall
-----------------------
NO LLM, NO MCP, NO network on this path. It reads market data + config and emits a
plain event. It WRITES nothing live (the research firewall, §6) — publishing an
event is not a config write. The LLM agent in the .md may *review/override* the
policy, but the numbers here are reproducible from the DB alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.stats.regime import (
    CHOP,
    REGIMES,
    TREND,
    VOL_SHOCK,
    RegimeConfig,
    daily_features,
)
from data.levels import atr14_wilder, daily_ohlc_from_bars
from data.schema import DEFAULT_DB_PATH, connect
from orchestrator.events import Event, EventType

# --------------------------------------------------------------------------- #
# Regime -> armed strategies (MASTER_PLAN.md §1.A edge portfolio by regime)    #
# --------------------------------------------------------------------------- #
# Trend continuation arms the breakout-retest + momentum-thrust edges; chop arms
# the mean-reversion edge (its design regime); a vol_shock stands the book down.
# These names match strategies/registry.yaml.
ARMED_BY_REGIME: dict[str, tuple[str, ...]] = {
    TREND: ("breakout_retest", "momentum_thrust"),
    CHOP: ("level_meanrev",),
    VOL_SHOCK: (),  # stand down: do not arm anything in a shock
}


@dataclass(frozen=True)
class RegimeReaderConfig:
    """Tunables for the daily read (documented defaults; not a live-risk knob).

    ``regime_cfg`` is the shared P2 :class:`RegimeConfig` (the SAME thresholds the
    backtest tagger uses, so "what regime is today" is one source of truth). The
    exposure-scalar knobs map regime -> the per-trade-risk multiplier described in
    the module docstring; they are deliberately conservative (lean in modestly,
    cut hard) and the result is clamped to ``[0, scalar_max]``.
    """

    regime_cfg: RegimeConfig = field(default_factory=RegimeConfig)
    symbol: str = "QQQ"        # the regime is read off the broad-market proxy

    # Exposure scalar by regime (the accelerator/brake). Trend > 1 (press),
    # chop < 1 (cut), vol_shock 0 (stand down).
    trend_scalar: float = 1.20
    chop_scalar: float = 0.60
    vol_shock_scalar: float = 0.0
    scalar_max: float = 1.25   # hard ceiling so a favorable regime never blows past the band


@dataclass(frozen=True)
class RegimeAssessment:
    """The deterministic daily policy for one session.

    Attributes
    ----------
    date
        The session date assessed (an ISO ``YYYY-MM-DD`` string is also exposed
        on the event payload).
    symbol
        The proxy the regime was read from (e.g. ``"QQQ"``).
    regime
        One of ``trend`` / ``chop`` / ``vol_shock`` (from the shared tagger).
    realized_vol
        Today's true range as a multiple of trailing ATR14 (``tr/atr``); ``None``
        when the feature is undefined (no ATR / insufficient history).
    iv
        ALWAYS ``None`` this phase — a documented HOOK. No options feed exists, so
        IV is never fabricated; the ``iv-rank-skew-read`` skill fills it once a
        source exists.
    exposure_scalar
        Per-trade-risk multiplier the fast loop applies to the conviction-tier
        budget (see module docstring). Higher in trend, cut in chop/vol_shock.
    armed
        Strategies eligible to trigger this session (empty in a vol_shock).
    rationale
        A short, human-readable why (for the digest / journal / LLM review).
    """

    date: date
    symbol: str
    regime: str
    realized_vol: Optional[float]
    iv: Optional[float]
    exposure_scalar: float
    armed: tuple[str, ...]
    rationale: str

    def to_event_data(self) -> dict:
        """JSON-serializable payload for the ``REGIME_TAGGED`` event.

        Keys (the consumer contract for the fast loop / risk gate):
        ``date, symbol, regime, realized_vol, iv, exposure_scalar, armed,
        rationale``. ``iv`` is explicitly ``null`` (not omitted) so consumers can
        tell "no IV source" from "forgot to set it".
        """
        return {
            "date": self.date.isoformat(),
            "symbol": self.symbol,
            "regime": self.regime,
            "realized_vol": (
                None if self.realized_vol is None else float(self.realized_vol)
            ),
            "iv": None,  # documented hook; never fabricated this phase
            "exposure_scalar": float(self.exposure_scalar),
            "armed": list(self.armed),
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------- #
# Exposure scalar                                                             #
# --------------------------------------------------------------------------- #
def exposure_scalar_for(regime: str, cfg: RegimeReaderConfig | None = None) -> float:
    """Map a regime to the per-trade-risk exposure multiplier (clamped >= 0).

    trend -> ``trend_scalar`` (lean in, > 1), chop -> ``chop_scalar`` (cut, < 1),
    vol_shock -> ``vol_shock_scalar`` (0 = stand down). The result is clamped to
    ``[0, scalar_max]`` so a favorable regime presses but never exceeds the band
    ceiling the human set. See the module docstring for how the fast loop applies
    it (``effective_dollar_risk = exposure_scalar * conviction_tier_dollar_risk``).
    """
    cfg = cfg or RegimeReaderConfig()
    base = {
        TREND: cfg.trend_scalar,
        CHOP: cfg.chop_scalar,
        VOL_SHOCK: cfg.vol_shock_scalar,
    }.get(regime, cfg.chop_scalar)  # unknown -> conservative (treat as chop)
    return float(min(max(base, 0.0), cfg.scalar_max))


# --------------------------------------------------------------------------- #
# Core assessment                                                             #
# --------------------------------------------------------------------------- #
def _rationale(regime: str, realized_vol: Optional[float], scalar: float,
               armed: tuple[str, ...]) -> str:
    rv = "n/a" if realized_vol is None else f"{realized_vol:.2f}x ATR"
    armed_str = ", ".join(armed) if armed else "none (stand down)"
    if regime == TREND:
        why = "directional regime (far from SMA) — lean in"
    elif regime == VOL_SHOCK:
        why = "volatility shock (wide/gappy range) — cut exposure, do not enter"
    else:
        why = "low-range / mean-reverting regime — trade smaller"
    return (
        f"{regime}: {why}. realized_vol={rv}; exposure_scalar={scalar:.2f}; "
        f"armed={armed_str}. IV unavailable (no options feed)."
    )


def assess_from_daily(
    daily: pd.DataFrame,
    atr_by_date: dict,
    target_date: date,
    *,
    symbol: str = "QQQ",
    cfg: RegimeReaderConfig | None = None,
) -> RegimeAssessment:
    """Build the assessment for ``target_date`` from a daily OHLC frame + ATR map.

    Exposed (DB-free) for tests and for callers that already have the daily frame.
    Computes the per-session feature (``tr/atr`` realized-vol read) and the regime
    tag from the SAME thresholds the backtest tagger uses, then maps regime ->
    armed list + exposure scalar. ``target_date`` must be present in ``daily``.
    """
    cfg = cfg or RegimeReaderConfig()
    target_date = _as_date(target_date)

    feats = daily_features(daily, atr_by_date, cfg.regime_cfg)
    row = feats[feats["session_date"].apply(_as_date) == target_date]
    if len(row) == 0:
        raise KeyError(f"no daily session for {target_date!r} in the frame")

    tr_atr = float(row["tr_atr"].iloc[0]) if pd.notna(row["tr_atr"].iloc[0]) else None
    dist = float(row["dist_sma_atr"].iloc[0]) if pd.notna(row["dist_sma_atr"].iloc[0]) else None

    regime = _classify(tr_atr, dist, cfg.regime_cfg)
    scalar = exposure_scalar_for(regime, cfg)
    armed = ARMED_BY_REGIME.get(regime, ())

    return RegimeAssessment(
        date=target_date,
        symbol=symbol,
        regime=regime,
        realized_vol=tr_atr,
        iv=None,
        exposure_scalar=scalar,
        armed=armed,
        rationale=_rationale(regime, tr_atr, scalar, armed),
    )


def assess(
    target_date: date | str | None = None,
    symbol: str = "QQQ",
    *,
    db_path: str = DEFAULT_DB_PATH,
    cfg: RegimeReaderConfig | None = None,
    con=None,
) -> RegimeAssessment:
    """Assess the daily regime/vol policy for ``symbol`` on ``target_date``.

    Reads the symbol's 5m bars + point-in-time ATR14 from the market DB, rolls
    them into daily OHLC, and delegates to :func:`assess_from_daily`. If
    ``target_date`` is None, the most recent session in the data is used.

    NO LLM / MCP / network: this is the deterministic compute the .md agent
    oversees. Deterministic for a given DB + config.
    """
    cfg = cfg or RegimeReaderConfig()
    symbol = symbol or cfg.symbol

    own_con = con is None
    if own_con:
        con = connect(db_path)
    try:
        bars = con.execute(
            """
            SELECT ts_utc, open, high, low, close, volume
            FROM bars
            WHERE symbol = ? AND timeframe = '5m'
            ORDER BY ts_utc
            """,
            [symbol],
        ).df()
        lv = con.execute(
            "SELECT session_date, atr14 FROM levels WHERE symbol = ? ORDER BY session_date",
            [symbol],
        ).df()
    finally:
        if own_con:
            con.close()

    if len(bars) == 0:
        raise ValueError(f"no bars for symbol {symbol!r} in {db_path!r}")

    asset_class = "crypto" if "/" in symbol else "equity"
    daily = daily_ohlc_from_bars(bars, asset_class)
    if len(daily) == 0:
        raise ValueError(f"no daily sessions derived for {symbol!r}")

    atr_by_date = {
        _as_date(r.session_date): (None if pd.isna(r.atr14) else float(r.atr14))
        for r in lv.itertuples(index=False)
    }

    if target_date is None:
        target_date = _as_date(daily["session_date"].iloc[-1])
    else:
        target_date = _as_date(target_date)

    return assess_from_daily(
        daily, atr_by_date, target_date, symbol=symbol, cfg=cfg
    )


# --------------------------------------------------------------------------- #
# Publish                                                                     #
# --------------------------------------------------------------------------- #
def publish(bus, assessment: RegimeAssessment, *, source: str = "regime_reader") -> Event:
    """Publish a single ``REGIME_TAGGED`` event for ``assessment`` on ``bus``.

    The fast loop / risk gate subscribe to ``REGIME_TAGGED`` and read
    ``data["exposure_scalar"]`` + ``data["armed"]`` to scale conviction sizing
    and gate which strategies may trigger. ``bus`` is duck-typed: any object with
    ``publish(event)`` works (the real :class:`~orchestrator.bus.EventBus` or a
    test fake). Returns the published :class:`~orchestrator.events.Event`.

    Publishing an event is NOT a live-config write — the firewall (§6) is not
    violated. This emits exactly one event.
    """
    event = Event(
        type=EventType.REGIME_TAGGED,
        data=assessment.to_event_data(),
        source=source,
    )
    return bus.publish(event)


# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        return date.fromisoformat(d)
    # pandas Timestamp / numpy datetime64 fallthrough
    return pd.Timestamp(d).date()


def _classify(tr_atr, dist_sma_atr, cfg: RegimeConfig) -> str:
    """Regime from features (vol_shock > trend > chop) — mirrors the P2 tagger."""
    if tr_atr is not None and np.isfinite(tr_atr) and tr_atr >= cfg.vol_shock_tr_mult:
        return VOL_SHOCK
    if (
        dist_sma_atr is not None
        and np.isfinite(dist_sma_atr)
        and dist_sma_atr >= cfg.trend_sma_atr
    ):
        return TREND
    return CHOP


# =========================================================================== #
# PORTFOLIO LAYER — daily multi-sleeve arm/disarm + vol-target exposure scalar #
# =========================================================================== #
# Everything ABOVE this line is the original intraday read (one broad-market
# proxy -> regime tag + armed intraday strategies + a regime accelerator). It is
# untouched and its tests still pass.
#
# Everything BELOW is the ADDITIVE daily portfolio layer for the multi-sleeve
# rotation book that rides the daily-NAV engine (``backtest/daily/engine.py``).
# A daily sleeve answers "what fraction of equity in each symbol tomorrow?"; the
# regime-reader's job for that book is the SLOW, policy half:
#
#   1. ARM / DISARM each sleeve from DAILY history — a long-only sleeve only gets
#      to allocate when its risk-on gate is true. The gate is the classic two-part
#      trend filter used by absolute-momentum / dual-momentum rotations:
#        * a market-trend filter: the broad-market proxy (SPY) above its 200d SMA,
#          AND/OR
#        * an own absolute-momentum filter: the sleeve's own benchmark up over the
#          trailing ~12 months (the absolute-momentum "is the asset itself in an
#          uptrend" test).
#      When the gate is FALSE the sleeve DISARMS and a RISK-OFF sleeve arms in its
#      place (rotate-to-cash/T-bills), so the book is always allocated to
#      *something* — risk-on assets in an uptrend, the defensive sleeve below it.
#   2. A vol-target EXPOSURE SCALAR from RECENT REALIZED VOL vs a target vol
#      (``target_vol / realized_vol``, clamped to the same ``[0, scalar_max]``
#      spirit as the intraday scalar). Calm tape -> scale toward the ceiling;
#      stormy tape -> cut. This is the portfolio-level analogue of the intraday
#      regime accelerator, but driven by a realized-vol *number* rather than a
#      categorical tag — the natural lever for a daily book sized to a vol budget.
#
# HOW A DAILY SLEEVE CONSUMES THIS (the contract for the engine / human MCP)
# -------------------------------------------------------------------------
# The daily engine's strategy returns ``target_weights -> {symbol: fraction}``.
# A sleeve consumes a :class:`PortfolioRegimeAssessment` as a PRE-FILTER + a
# SCALAR on its gross book, BEFORE returning weights:
#
#     pa = assess_portfolio(asof_date)           # this module (slow loop / daily)
#     if not pa.armed_sleeves.get(my_sleeve, False):
#         # disarmed -> hold the risk-off sleeve's weights (or cash) instead
#         return pa.risk_off_weights              # e.g. {"BIL": 1.0}
#     raw = my_alpha_weights(asof_date, history)  # the sleeve's gross long-only book
#     scaled = {s: w * pa.exposure_scalar for s, w in raw.items()}
#     # scalar <= 1.0 only ever DE-levers below 1.0 of book; the > 1.0 headroom is
#     # a documented lever the human/risk gate may allow, never auto-applied to a
#     # long-only book past sum(weights) <= 1 (the engine rejects sum > 1).
#     return scaled
#
# The arm flag is a hard ON/OFF gate (trade the sleeve or rotate it to the
# defensive sleeve); the exposure scalar is a continuous size knob on the armed
# book. Same two-clean-dials decomposition as the intraday path: a categorical
# gate and a continuous accelerator that compose without touching ``limits.yaml``.
#
# DETERMINISM / FIREWALL: identical to the intraday path — reads daily bars +
# config, emits a plain event, writes nothing live, no LLM/MCP/network. The
# numbers are reproducible from the DB alone; the .md agent may review/override.

# Risk-off destination when a sleeve disarms (rotate-to-safety). A short T-bill
# ETF that exists in the daily universe; the sleeve holds it (or cash) instead of
# its risk-on book. Weight 1.0 == fully defensive.
DEFAULT_RISK_OFF_SYMBOL = "BIL"


@dataclass(frozen=True)
class SleeveArmConfig:
    """Tunables for the daily portfolio arm/disarm + vol-target scalar.

    Documented defaults, calibrated against the SPY daily distribution (full
    history): the 200d trend filter is true ~74% of sessions and the 12m absolute
    momentum is positive ~77% of sessions — so the gate spends most of its life
    armed and disarms only in genuine downtrends (the absolute-momentum design).
    The vol target (16%/yr) lands the scalar near 1.0 at SPY's median realized vol
    (~14%), pushes toward the ceiling in calm tape, and cuts to ~0.6 at the 90th
    percentile of realized vol. None of these are live-risk knobs — they are
    policy compute the loop / human-confirmed MCP consumes.
    """

    # -- market-trend filter (the broad-market gate) --
    market_proxy: str = "SPY"      # broad-market proxy for the 200d trend filter
    trend_sma_window: int = 200    # the canonical long-trend SMA

    # -- own absolute-momentum filter --
    abs_mom_lookback: int = 252    # ~12 trading months for the abs-momentum test
    abs_mom_min: float = 0.0       # own benchmark up over the lookback => risk-on

    # How the two filters COMBINE into the arm decision:
    #   "and"  -> arm only if market-trend AND own-momentum (strictest)
    #   "or"   -> arm if EITHER (the default; matches "SPY>200d OR own abs-mom +")
    #   "market" -> market-trend only (ignore own momentum)
    #   "own"    -> own momentum only (ignore the market proxy)
    arm_rule: str = "or"

    # -- vol-target exposure scalar --
    vol_lookback: int = 20         # trailing window for realized vol (~1 month)
    target_vol: float = 0.16       # annualized vol budget for the book
    trading_days: int = 252        # annualization factor
    scalar_max: float = 1.25       # SAME ceiling spirit as the intraday scalar
    scalar_min: float = 0.0        # floor (0 == fully de-levered)

    risk_off_symbol: str = DEFAULT_RISK_OFF_SYMBOL


# Default mapping: sleeve name -> the benchmark whose absolute momentum gates it.
# A daily rotation/mean-reversion sleeve names the symbol that represents "is MY
# kind of asset in an uptrend". Unmapped sleeves fall back to the market proxy.
DEFAULT_SLEEVE_BENCHMARKS: dict[str, str] = {
    "momentum_rotation": "SPY",   # broad-equity momentum sleeve
    "sector_rotation": "SPY",
    "trend_following": "SPY",
}


@dataclass(frozen=True)
class PortfolioRegimeAssessment:
    """The deterministic daily policy for the multi-sleeve rotation book.

    Companion to :class:`RegimeAssessment` (which is the intraday read). Both are
    emitted by the regime-reader; this one drives the DAILY engine's sleeves.

    Attributes
    ----------
    date
        The session date assessed.
    market_proxy
        The broad-market proxy used for the 200d trend filter (e.g. ``"SPY"``).
    market_above_200d
        Whether ``market_proxy`` closed above its ``trend_sma_window`` SMA as of
        ``date`` (the market-trend half of the gate). ``None`` if undefined (not
        enough history).
    realized_vol
        Trailing annualized realized vol of the market proxy (``std(daily ret) *
        sqrt(trading_days)``) over ``vol_lookback`` days; ``None`` if undefined.
    exposure_scalar
        ``target_vol / realized_vol`` clamped to ``[scalar_min, scalar_max]`` —
        the size knob applied to each ARMED sleeve's gross book. Falls back to
        ``1.0`` when realized vol is undefined (neutral, never fabricated extreme).
    armed_sleeves
        ``{sleeve_name: bool}`` — True == risk-on (allocate its book), False ==
        disarmed (rotate to ``risk_off_weights``).
    sleeve_momentum
        ``{sleeve_name: float|None}`` — each sleeve's own trailing abs-momentum
        return over ``abs_mom_lookback`` (diagnostic / rationale; ``None`` if the
        benchmark lacks history).
    risk_off_weights
        The defensive target a DISARMED sleeve holds instead of its book (e.g.
        ``{"BIL": 1.0}``). A ready-to-return ``target_weights`` dict.
    rationale
        Short human-readable why (digest / journal / LLM review).
    """

    date: date
    market_proxy: str
    market_above_200d: Optional[bool]
    realized_vol: Optional[float]
    exposure_scalar: float
    armed_sleeves: Mapping[str, bool]
    sleeve_momentum: Mapping[str, Optional[float]]
    risk_off_weights: Mapping[str, float]
    rationale: str

    def to_event_data(self) -> dict:
        """JSON-serializable payload for the ``REGIME_TAGGED`` portfolio event.

        Keys (the consumer contract for the daily engine / human-confirmed MCP):
        ``date, scope, market_proxy, market_above_200d, realized_vol,
        exposure_scalar, armed_sleeves, sleeve_momentum, risk_off_weights,
        rationale``. ``scope='portfolio'`` lets a consumer distinguish this from
        the intraday :class:`RegimeAssessment` payload on the same event type.
        """
        return {
            "date": self.date.isoformat(),
            "scope": "portfolio",
            "market_proxy": self.market_proxy,
            "market_above_200d": (
                None if self.market_above_200d is None else bool(self.market_above_200d)
            ),
            "realized_vol": (
                None if self.realized_vol is None else float(self.realized_vol)
            ),
            "exposure_scalar": float(self.exposure_scalar),
            "armed_sleeves": {k: bool(v) for k, v in self.armed_sleeves.items()},
            "sleeve_momentum": {
                k: (None if v is None else float(v))
                for k, v in self.sleeve_momentum.items()
            },
            "risk_off_weights": {k: float(v) for k, v in self.risk_off_weights.items()},
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------- #
# Vol-target exposure scalar (the portfolio accelerator/brake)                #
# --------------------------------------------------------------------------- #
def realized_vol_annualized(
    close: pd.Series, lookback: int, trading_days: int = 252
) -> Optional[float]:
    """Trailing annualized realized vol of a daily close series (``None`` if undefined).

    ``std`` of the most recent ``lookback`` simple daily returns, annualized by
    ``sqrt(trading_days)``. Returns ``None`` when there are not enough returns to
    estimate (fewer than 2), so the caller never fabricates a vol number it can't
    compute. Point-in-time safe: ``close`` must already be sliced to ``<= asof``.
    """
    if close is None or len(close) < 2:
        return None
    rets = close.astype("float64").pct_change().dropna()
    if len(rets) == 0:
        return None
    window = rets.iloc[-int(lookback):] if lookback else rets
    if len(window) < 2:
        return None
    sd = float(window.std(ddof=1))
    if not np.isfinite(sd):
        return None
    return sd * float(np.sqrt(trading_days))


def vol_target_scalar(
    realized_vol: Optional[float], cfg: SleeveArmConfig | None = None
) -> float:
    """Map trailing realized vol -> the vol-target exposure scalar (clamped).

    ``scalar = target_vol / realized_vol``, clamped to ``[scalar_min, scalar_max]``
    — calm tape (realized < target) scales UP toward the ceiling, stormy tape
    (realized > target) scales DOWN. Mirrors the intraday ``exposure_scalar_for``
    contract (same ``[0, scalar_max]`` spirit) but is driven by a vol *number*,
    which is what a daily book sized to a vol budget wants.

    ``realized_vol is None`` (undefined) -> a NEUTRAL ``1.0`` (trade the book
    unscaled), never a fabricated extreme. ``realized_vol <= 0`` (degenerate /
    flat) -> the ceiling (vol is below any positive target).
    """
    cfg = cfg or SleeveArmConfig()
    if realized_vol is None or not np.isfinite(realized_vol):
        return 1.0
    if realized_vol <= 0.0:
        return float(cfg.scalar_max)
    raw = cfg.target_vol / realized_vol
    return float(min(max(raw, cfg.scalar_min), cfg.scalar_max))


# --------------------------------------------------------------------------- #
# Per-sleeve arm / disarm                                                     #
# --------------------------------------------------------------------------- #
def _above_sma(close: pd.Series, window: int) -> Optional[bool]:
    """Is the last close >= its trailing ``window``-SMA? ``None`` if undefined."""
    if close is None or len(close) < window:
        return None
    c = close.astype("float64").dropna()
    if len(c) < window:
        return None
    sma = float(c.iloc[-window:].mean())
    last = float(c.iloc[-1])
    if not (np.isfinite(sma) and np.isfinite(last)):
        return None
    return last >= sma


def _abs_momentum(close: pd.Series, lookback: int) -> Optional[float]:
    """Trailing ``lookback``-day total return of a close series (``None`` if undefined)."""
    if close is None or len(close) <= lookback:
        return None
    c = close.astype("float64").dropna()
    if len(c) <= lookback:
        return None
    past = float(c.iloc[-1 - lookback])
    now = float(c.iloc[-1])
    if not (np.isfinite(past) and np.isfinite(now)) or past <= 0:
        return None
    return now / past - 1.0


def _arm_decision(
    market_above: Optional[bool],
    own_mom: Optional[float],
    cfg: SleeveArmConfig,
) -> bool:
    """Combine the market-trend filter + own abs-momentum into an arm flag.

    ``arm_rule`` selects how the two halves compose (see :class:`SleeveArmConfig`).
    A filter whose input is ``None`` (insufficient history) is treated as FALSE
    for that half — undecidable defaults to risk-off, the conservative choice for
    a capital-preservation-first book.
    """
    mkt = bool(market_above) if market_above is not None else False
    own = (own_mom is not None) and (own_mom > cfg.abs_mom_min)
    rule = cfg.arm_rule.lower()
    if rule == "and":
        return mkt and own
    if rule == "market":
        return mkt
    if rule == "own":
        return own
    return mkt or own  # default "or"


def arm_signals(
    panel: pd.DataFrame,
    asof_date: date,
    sleeves: Mapping[str, str] | Sequence[str] | None = None,
    *,
    cfg: SleeveArmConfig | None = None,
) -> dict[str, bool]:
    """Per-sleeve ARM/DISARM flags from a daily ADJUSTED-close panel, point-in-time.

    The standalone arm half of :func:`assess_portfolio_from_panel`, exposed so a
    sleeve (or a test) can ask "am I armed today?" without building the full
    assessment.

    Parameters
    ----------
    panel
        Wide ADJUSTED-close frame (index = dates ascending, columns = symbols) —
        the same panel shape the daily engine uses (``load_daily_bars``).
    asof_date
        The decision date. Only rows ``<= asof_date`` are used (no lookahead).
    sleeves
        ``{sleeve_name: benchmark_symbol}`` mapping, OR a plain iterable of sleeve
        names (each then uses :data:`DEFAULT_SLEEVE_BENCHMARKS`, falling back to
        the market proxy). ``None`` -> :data:`DEFAULT_SLEEVE_BENCHMARKS`.
    cfg
        :class:`SleeveArmConfig` thresholds.

    Returns ``{sleeve_name: armed_bool}``.
    """
    cfg = cfg or SleeveArmConfig()
    bench = _resolve_benchmarks(sleeves, cfg)
    asof_date = _as_date(asof_date)

    visible = panel.loc[[d for d in panel.index if _as_date(d) <= asof_date]]
    proxy_close = visible[cfg.market_proxy] if cfg.market_proxy in visible.columns else None
    market_above = _above_sma(proxy_close, cfg.trend_sma_window)

    out: dict[str, bool] = {}
    for sleeve, benchmark in bench.items():
        bench_close = visible[benchmark] if benchmark in visible.columns else None
        own_mom = _abs_momentum(bench_close, cfg.abs_mom_lookback)
        out[sleeve] = _arm_decision(market_above, own_mom, cfg)
    return out


def _resolve_benchmarks(
    sleeves: Mapping[str, str] | Sequence[str] | None, cfg: SleeveArmConfig
) -> dict[str, str]:
    """Normalize ``sleeves`` -> ``{sleeve: benchmark_symbol}`` (defaults filled)."""
    if sleeves is None:
        return dict(DEFAULT_SLEEVE_BENCHMARKS)
    if isinstance(sleeves, Mapping):
        return {k: (v or cfg.market_proxy) for k, v in sleeves.items()}
    # plain iterable of names
    return {
        name: DEFAULT_SLEEVE_BENCHMARKS.get(name, cfg.market_proxy)
        for name in sleeves
    }


# --------------------------------------------------------------------------- #
# Core portfolio assessment                                                   #
# --------------------------------------------------------------------------- #
def _portfolio_rationale(
    market_above: Optional[bool],
    realized_vol: Optional[float],
    scalar: float,
    armed: Mapping[str, bool],
    cfg: SleeveArmConfig,
) -> str:
    rv = "n/a" if realized_vol is None else f"{realized_vol:.1%}/yr"
    mkt = (
        "n/a" if market_above is None
        else (f"{cfg.market_proxy}>200d" if market_above else f"{cfg.market_proxy}<200d")
    )
    n_armed = sum(1 for v in armed.values() if v)
    armed_str = ", ".join(s for s, v in armed.items() if v) or "none (all risk-off)"
    return (
        f"portfolio: trend {mkt}; realized_vol={rv} vs target {cfg.target_vol:.0%} "
        f"-> exposure_scalar={scalar:.2f}; {n_armed}/{len(armed)} sleeves armed "
        f"({armed_str}); disarmed sleeves rotate to {cfg.risk_off_symbol}."
    )


def assess_portfolio_from_panel(
    panel: pd.DataFrame,
    asof_date: date,
    sleeves: Mapping[str, str] | Sequence[str] | None = None,
    *,
    cfg: SleeveArmConfig | None = None,
) -> PortfolioRegimeAssessment:
    """Build the daily portfolio assessment from a price panel (DB-free).

    Exposed (offline) for tests and for callers that already hold the wide
    ADJUSTED-close panel (e.g. from :func:`backtest.daily.engine.load_daily_bars`
    or :meth:`DailyHistory.prices`). Computes, point-in-time as of ``asof_date``:
    the market 200d-trend flag, trailing realized vol of the proxy + its
    vol-target exposure scalar, and the per-sleeve arm flags + own-momentum reads.

    ``asof_date`` need NOT be an exact index value — the most recent session
    ``<= asof_date`` is used (so a weekend/holiday decision date still resolves).
    """
    cfg = cfg or SleeveArmConfig()
    bench = _resolve_benchmarks(sleeves, cfg)
    asof_date = _as_date(asof_date)

    visible = panel.loc[[d for d in panel.index if _as_date(d) <= asof_date]]
    if len(visible) == 0:
        raise KeyError(f"no daily session <= {asof_date!r} in the panel")
    resolved_date = _as_date(visible.index[-1])

    proxy_close = visible[cfg.market_proxy] if cfg.market_proxy in visible.columns else None
    market_above = _above_sma(proxy_close, cfg.trend_sma_window)
    realized_vol = realized_vol_annualized(
        proxy_close, cfg.vol_lookback, cfg.trading_days
    ) if proxy_close is not None else None
    scalar = vol_target_scalar(realized_vol, cfg)

    armed: dict[str, bool] = {}
    sleeve_mom: dict[str, Optional[float]] = {}
    for sleeve, benchmark in bench.items():
        bench_close = visible[benchmark] if benchmark in visible.columns else None
        own_mom = _abs_momentum(bench_close, cfg.abs_mom_lookback)
        sleeve_mom[sleeve] = own_mom
        armed[sleeve] = _arm_decision(market_above, own_mom, cfg)

    risk_off = {cfg.risk_off_symbol: 1.0}
    return PortfolioRegimeAssessment(
        date=resolved_date,
        market_proxy=cfg.market_proxy,
        market_above_200d=market_above,
        realized_vol=realized_vol,
        exposure_scalar=scalar,
        armed_sleeves=armed,
        sleeve_momentum=sleeve_mom,
        risk_off_weights=risk_off,
        rationale=_portfolio_rationale(market_above, realized_vol, scalar, armed, cfg),
    )


def assess_portfolio(
    asof_date: date | str | None = None,
    sleeves: Mapping[str, str] | Sequence[str] | None = None,
    *,
    universe: Sequence[str] | None = None,
    db_path: str = DEFAULT_DB_PATH,
    cfg: SleeveArmConfig | None = None,
    con=None,
    panel: pd.DataFrame | None = None,
) -> PortfolioRegimeAssessment:
    """Assess the daily multi-sleeve portfolio policy as of ``asof_date``.

    Loads the wide ADJUSTED ``timeframe='1d'`` panel for the market proxy + every
    sleeve benchmark (and any extra ``universe`` symbols) via
    :func:`backtest.daily.engine.load_daily_bars`, then delegates to
    :func:`assess_portfolio_from_panel`. If ``asof_date`` is None the latest
    stored session is used.

    NO LLM / MCP / network: deterministic for a given DB + config. The same slow-
    loop firewall as :func:`assess` — reads data, returns a typed policy, writes
    nothing live.
    """
    from backtest.daily.engine import load_daily_bars  # local import: avoid cycle

    cfg = cfg or SleeveArmConfig()
    bench = _resolve_benchmarks(sleeves, cfg)

    if panel is None:
        needed = {cfg.market_proxy, cfg.risk_off_symbol, *bench.values()}
        if universe:
            needed.update(universe)
        panel = load_daily_bars(sorted(needed), db_path=db_path, con=con)
    if panel is None or len(panel) == 0:
        raise ValueError(f"no daily bars loaded for the portfolio assessment ({db_path!r})")

    if asof_date is None:
        asof_date = _as_date(panel.index[-1])
    return assess_portfolio_from_panel(panel, asof_date, sleeves, cfg=cfg)


def publish_portfolio(
    bus, assessment: PortfolioRegimeAssessment, *, source: str = "regime_reader"
) -> Event:
    """Publish one ``REGIME_TAGGED`` event for a portfolio ``assessment`` on ``bus``.

    Mirror of :func:`publish` for the daily book. The payload carries
    ``scope='portfolio'`` so a consumer can tell it apart from the intraday read.
    The daily engine / human-confirmed MCP read ``armed_sleeves`` (gate) +
    ``exposure_scalar`` (size) + ``risk_off_weights`` (the disarmed fallback).
    Publishing is NOT a live-config write — the firewall is intact. Returns the
    published :class:`~orchestrator.events.Event`.
    """
    event = Event(
        type=EventType.REGIME_TAGGED,
        data=assessment.to_event_data(),
        source=source,
    )
    return bus.publish(event)
