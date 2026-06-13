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
from typing import Optional

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
