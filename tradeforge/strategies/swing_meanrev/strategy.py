"""strategies/swing_meanrev/strategy.py — the SWING MEAN-REVERSION sleeve.

The classic Connors RSI(2) oversold dip-buy, expressed as a daily target-weight
allocator against the :class:`~backtest.daily.engine.DailyStrategy` interface.
LONG-only, MULTI-DAY holds (the daily engine never trades intraday), on broad +
liquid sector index ETFs. Parameters + named variant deltas live in params.yaml.

THE EDGE (and why it is decorrelated, MASTER_PLAN.md §1.A/B)
-----------------------------------------------------------
A momentum / breakout sleeve BUYS strength — a clean break that keeps going. This
sleeve does the opposite: it BUYS short-term WEAKNESS inside a long-term UPTREND.
When an index ETF that is comfortably above its 200d SMA gets briefly oversold
(RSI(2) collapses, the close drops several % below its 5/10d MA), we step in for
the snap-back, then exit on the bounce (RSI(2) recovers) or a time stop. So on
the same name the two sleeves take OPPOSITE sides of the same pullback and are
rarely both right at once — exactly the low correlation the blend wants.

HONESTY: this is a well-known, crowded retail edge. Expect a MODEST, possibly
decayed standalone result. Its job is decorrelation, not raw alpha.

THE RULES (all point-in-time AS OF ``asof_date``'s close)
---------------------------------------------------------
For each ETF in the configured ``universe`` we compute, from the visible window:

  * RSI(rsi_period)  — Wilder RSI on the SHORT period (2 by default). The
    canonical Connors oscillator: it spends most of its life mid-range and only
    dives toward 0 on a sharp multi-bar drop.
  * SMA(ma_window)   — the short MA the close is measured against (5/10d).
  * SMA(sma_gate)    — the 200d trend gate.

ARM (a name ENTERS / stays in the basket) when ALL hold:
  1. close > SMA(sma_gate)                    -> long-term UPTREND (never buy a
                                                 downtrend — the gate is the
                                                 whole risk control of the edge);
  2. RSI(rsi_period) < oversold               -> short-term oversold;
  3. close <= SMA(ma_window) * (1 - pct_below_ma)  -> several % below the MA
                                                 (pct_below_ma=0 -> just "below").

HOLD / EXIT (a held name LEAVES the basket) when ANY fires:
  * RSI(rsi_period) >= exit_rsi               -> the bounce we came for;
  * held for >= max_hold_days trading days    -> time stop (the snap-back failed);
  * close < SMA(sma_gate)                      -> the uptrend gate broke (regime
                                                 changed under us -> step aside).

Names not in the basket are flat (held as cash). The basket is capped at
``max_concurrent`` names; when more than that qualify on an entry day we keep the
MOST oversold (lowest RSI) — size to the strongest signal. The held set is
EQUAL-weighted up to ``weight_cap`` of equity (remainder cash). Because the
engine re-asks ``target_weights`` every rebalance and we recompute the basket
from scratch each call, "hold" is simply "the name still qualifies to stay in"
— there is no hidden cross-call position state to drift out of sync with NAV.

POINT-IN-TIME / NO-LOOKAHEAD: every series is computed from
``history.prices(...)`` which is already sliced to ``<= asof_date`` by the
engine, so a future bar is structurally unreachable. To honor ``max_hold_days``
without carrying mutable state across calls (which the engine's rebalance cadence
would alias against), the time stop is evaluated from the price history itself:
a name is "too old" if it has NOT printed a fresh oversold trigger within the
last ``max_hold_days`` sessions — i.e. the dip that armed it has gone stale.

PURE / DETERMINISTIC: no LLM, no MCP, no network. Just pandas on the panel the
engine hands in.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtest.daily.engine import DailyHistory

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"


def load_params(variant: str = "DEFAULT", path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
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


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI of a close series over ``period`` (the Connors short-period RSI).

    Uses Wilder's smoothing (an EMA with ``alpha = 1/period``, seeded on the first
    ``period`` average gain/loss) — the standard RSI definition, matching what
    Connors' RSI(2) and charting packages compute. Returns a Series aligned to
    ``close``; the first ``period`` values are NaN (insufficient lookback). A flat
    window (no losses) yields 100; an all-down window yields 0.

    Pure / vectorized; no lookahead (each point uses only prior + current closes).
    """
    period = int(period)
    if period < 1:
        raise ValueError(f"rsi period must be >= 1, got {period}")
    delta = close.astype("float64").diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == ewm with alpha = 1/period, min_periods=period so the
    # first `period` rows (no full window) stay NaN rather than seeding on partial
    # data. adjust=False gives the recursive Wilder average exactly.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # All-gain window (avg_loss == 0) -> RSI 100; all-loss (avg_gain == 0) -> 0.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss != 0.0)), 0.0)
    return rsi


class SwingMeanRevStrategy:
    """RSI(2)-oversold dip-buy on index ETFs, gated by a 200d uptrend.

    Implements the daily :class:`~backtest.daily.engine.DailyStrategy` contract:
    ``target_weights(asof_date, history) -> {symbol: fraction}``, long-only,
    summing to ``<= weight_cap``. See the module docstring for the full rule set.
    """

    def __init__(self, params: dict | None = None, variant: str = "DEFAULT"):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)

        self.rsi_period = int(self.params["rsi_period"])
        self.ma_window = int(self.params["ma_window"])
        self.sma_gate = int(self.params["sma_gate"])
        self.oversold = float(self.params["oversold"])
        self.pct_below_ma = float(self.params["pct_below_ma"])
        self.exit_rsi = float(self.params["exit_rsi"])
        self.max_hold_days = int(self.params["max_hold_days"])
        self.max_concurrent = int(self.params["max_concurrent"])
        self.weight_cap = float(self.params.get("weight_cap", 1.0))
        self.universe = list(self.params.get("universe", []))

        # Enough trailing rows to seed the 200d gate (+ slack for the RSI/MA and
        # the max-hold staleness window). Fetch a bit more than sma_gate so the
        # first valid SMA(200) row is real, not partial.
        self._lookback = self.sma_gate + max(self.ma_window, self.max_hold_days) + 5

    # ------------------------------------------------------------ the contract
    def target_weights(
        self, asof_date: date, history: DailyHistory
    ) -> dict[str, float]:
        """Return the equal-weight basket of currently-qualifying oversold names.

        Recomputed from scratch each call off the point-in-time price window, so
        "hold" is implicit: a name stays in the basket iff it still passes the
        ARM/HOLD test today. No cross-call mutable position state (which would
        alias against the engine's rebalance cadence and drift from NAV).
        """
        syms = [s for s in self.universe if s in history.universe]
        if not syms:
            return {}
        px = history.prices(symbols=syms, lookback=self._lookback)
        if len(px) == 0:
            return {}

        qualifying: list[tuple[str, float]] = []  # (symbol, rsi) for ranking
        for sym in syms:
            series = px[sym].dropna()
            # Need a full 200d gate window to trade at all (no partial-trend buys).
            if len(series) < self.sma_gate:
                continue

            close = float(series.iloc[-1])
            sma_gate = float(series.iloc[-self.sma_gate:].mean())
            sma_ma = float(series.iloc[-self.ma_window:].mean())
            rsi = wilder_rsi(series, self.rsi_period)
            rsi_now = float(rsi.iloc[-1])
            if not np.isfinite(rsi_now):
                continue

            # GATE 1: long-term uptrend — never buy below the 200d SMA.
            if close <= sma_gate:
                continue

            # The name STAYS / ENTERS the basket iff it currently ARMS: oversold
            # AND several % below the short MA AND not yet bounced. The bounce
            # exit (rsi >= exit_rsi) and the "below MA" condition are mutually
            # consistent — once RSI recovers past exit_rsi the name simply fails
            # the oversold test and drops out. The explicit exit_rsi check makes
            # the asymmetric band (enter < oversold, exit >= exit_rsi) honest:
            # a name with oversold <= rsi < exit_rsi is in the dead band and we
            # do NOT re-arm it (avoids churn around the threshold).
            armed = (
                rsi_now < self.oversold
                and close <= sma_ma * (1.0 - self.pct_below_ma)
            )
            if not armed:
                continue

            # TIME STOP (stateless): the dip that armed this name must be FRESH —
            # an oversold trigger within the last max_hold_days sessions. If RSI
            # has been stuck oversold for longer than the stop, the snap-back
            # failed; step aside rather than bag-hold a broken dip.
            recent_rsi = rsi.iloc[-self.max_hold_days:]
            if not (recent_rsi < self.oversold).any():
                continue

            qualifying.append((sym, rsi_now))

        if not qualifying:
            return {}

        # Size to the strongest signal: keep the most oversold (lowest RSI) up to
        # the concurrency cap, then equal-weight them within the invested cap.
        qualifying.sort(key=lambda t: t[1])
        chosen = [s for s, _ in qualifying[: self.max_concurrent]]
        if not chosen:
            return {}
        w = self.weight_cap / len(chosen)
        return {s: w for s in chosen}
