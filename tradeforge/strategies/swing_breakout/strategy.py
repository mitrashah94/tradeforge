"""strategies/swing_breakout/strategy.py — the DAILY DONCHIAN-BREAKOUT TREND sleeve.

The classic long-only trend-following breakout, expressed against the bracketed
swing engine's :class:`~backtest.daily.bracket_engine.SwingStrategy` interface
(``entry_score`` required, ``exit_signal`` optional). Each entry is then managed
by the engine's ATR bracket (partial-TP + breakeven + chandelier trail), so the
strategy's only job is to RANK fresh breakouts-in-an-uptrend; the bracket cuts
the losers fast and lets the winners run. Parameters + named variant deltas live
in params.yaml.

THE EDGE (and why it complements the mean-reversion sleeve, MASTER_PLAN §1.A/B)
------------------------------------------------------------------------------
This is the textbook Donchian / turtle breakout: BUY STRENGTH. A name that has
just printed a fresh N-day high WHILE trading above its long trend SMA is in a
confirmed uptrend making new highs — exactly the regime where momentum persists.
We buy that breakout, protect it with an ATR stop, scale a partial into the first
push, ratchet to breakeven, and trail the runner. The asymmetry (small capped
losers, occasional large trailed winners) is the whole expectancy of trend
following — most trades are scratches/small losses, a few runners pay for them
many times over. It is the MIRROR of ``swing_meanrev`` (which buys the dips this
sleeve would never touch), so on the same name they rarely fire together.

HONESTY: trend breakouts have a LOW hit rate (a lot of failed breakouts and
whipsaws) and live or die on the fat right tail of the winners. The bracket
variant (let-it-run trail vs a hard-target cap) is what decides whether that tail
survives. Single-stock breakouts also carry gap risk the index sleeves don't.

THE RULES (all point-in-time AS OF ``asof_date``'s close)
---------------------------------------------------------
``entry_score(symbol, asof_date, history)`` fires (returns a finite score) ONLY
when BOTH hold on ``asof_date``'s ADJUSTED close:

  1. close > the PRIOR ``donchian_n``-day HIGH   -> a FRESH N-day breakout. The
     prior high EXCLUDES today's bar (the highest close over the ``donchian_n``
     sessions ENDING YESTERDAY), so "breakout" means today's close is a genuinely
     new high vs the trailing channel, not merely tying its own bar.
  2. close > SMA(``trend_sma``)                  -> a long-term UPTREND. We only
     buy breakouts that happen ABOVE the trend filter — never catch a falling
     knife breaking out of a downtrend. This gate is the core risk control.

The SCORE (for the engine's cross-sectional ranking when more names break out
than there are free slots) is MOMENTUM STRENGTH = the trailing
``momentum_lookback``-day return (``close / close[-lookback] - 1``). Stronger
trends rank higher and win the scarce slots — "size by conviction" (CLAUDE.md).
``score_mode='above_sma'`` instead ranks by ``close / SMA(trend_sma) - 1`` (how
far above the trend the name is). Either way the score is a monotone momentum
proxy; the engine only uses its ORDER.

OPTIONAL TREND-BREAK EXIT
-------------------------
``exit_signal`` (enabled by ``exit_on_trend_break``) closes the remaining runner
at the close if the name falls back below its trend SMA — the regime that armed
the breakout has broken, so step aside even if the bracket stop has not been hit.
With ``exit_on_trend_break=False`` the position is managed purely by the ATR
bracket (the let-the-trail-decide default).

POINT-IN-TIME / NO-LOOKAHEAD: every series comes from ``history.prices(...)``,
already sliced to ``<= asof_date`` by the engine, so a future bar is structurally
unreachable. The Donchian high explicitly drops today's bar before taking the max
(``iloc[-donchian_n-1:-1]``), so the breakout test compares today's close to a
channel that ends YESTERDAY — no same-bar self-reference.

PURE / DETERMINISTIC: no LLM, no MCP, no network. Just pandas on the close panel
the engine hands in (the engine alone reads O/H/L to resolve the brackets).
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


class SwingBreakoutStrategy:
    """Donchian N-day-high breakout, gated by a trend SMA, ranked by momentum.

    Implements the bracketed-swing :class:`SwingStrategy` contract:
    ``entry_score(symbol, asof_date, history) -> float | None`` (None = no entry
    today; higher = stronger) and an OPTIONAL ``exit_signal`` (trend-break close).
    See the module docstring for the full rule set. Every threshold is read from
    params.yaml so the robustness sweep can probe the plateau.
    """

    def __init__(self, params: dict | None = None, variant: str = "DEFAULT"):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)

        self.donchian_n = int(self.params["donchian_n"])
        self.trend_sma = int(self.params["trend_sma"])
        self.momentum_lookback = int(self.params["momentum_lookback"])
        self.score_mode = str(self.params.get("score_mode", "momentum"))
        self.exit_on_trend_break = bool(self.params.get("exit_on_trend_break", False))

        if self.score_mode not in ("momentum", "above_sma"):
            raise ValueError(
                f"score_mode must be 'momentum' or 'above_sma', got {self.score_mode!r}"
            )

        # Enough trailing rows to seed the trend SMA plus the Donchian channel (the
        # channel ends yesterday, so we need trend_sma rows for the gate AND
        # donchian_n+1 rows for the prior-high; take the max + a little slack).
        self._lookback = max(self.trend_sma, self.donchian_n + 1, self.momentum_lookback + 1) + 5

    # ------------------------------------------------------------ the contract
    def entry_score(
        self, symbol: str, asof_date: date, history: DailyHistory
    ) -> float | None:
        """Score a FRESH N-day breakout-in-uptrend in ``symbol``, else ``None``.

        Returns ``None`` unless today's close is BOTH a new ``donchian_n``-day
        high (vs the channel ending yesterday) AND above the ``trend_sma`` trend
        filter. When it qualifies, returns the momentum strength score the engine
        ranks by (higher = stronger trend). Point-in-time: the price window is
        already sliced to ``<= asof_date`` by the engine.
        """
        px = history.prices(symbols=[symbol], lookback=self._lookback)
        if len(px) == 0:
            return None
        series = px[symbol].dropna()
        # Need a full trend-SMA window AND a full prior-Donchian channel to decide.
        if len(series) < max(self.trend_sma, self.donchian_n + 1):
            return None

        close = float(series.iloc[-1])
        if not np.isfinite(close) or close <= 0:
            return None

        # GATE 1: FRESH N-day breakout. The prior high EXCLUDES today's bar — the
        # max close over the donchian_n sessions ENDING YESTERDAY. Today's close
        # must exceed it to be a genuine new-high breakout (strictly >).
        prior_window = series.iloc[-(self.donchian_n + 1):-1]
        prior_high = float(prior_window.max())
        if not np.isfinite(prior_high) or close <= prior_high:
            return None

        # GATE 2: long-term UPTREND — never buy a breakout below the trend SMA.
        sma = float(series.iloc[-self.trend_sma:].mean())
        if not np.isfinite(sma) or sma <= 0 or close <= sma:
            return None

        # SCORE = momentum strength (monotone; the engine uses only the ORDER).
        if self.score_mode == "above_sma":
            score = close / sma - 1.0
        elif len(series) <= self.momentum_lookback:
            # Not enough history for the trailing-return base yet -> fall back to
            # distance-above-SMA so a fresh listing can still rank.
            score = close / sma - 1.0
        else:  # 'momentum' — trailing-lookback return
            base = float(series.iloc[-(self.momentum_lookback + 1)])
            if not np.isfinite(base) or base <= 0:
                # Fall back to distance-above-SMA when the lookback base is missing.
                score = close / sma - 1.0
            else:
                score = close / base - 1.0

        return float(score) if np.isfinite(score) else None

    # ------------------------------------------------- OPTIONAL discretionary exit
    def exit_signal(
        self, symbol: str, asof_date: date, history: DailyHistory, position=None
    ) -> bool:
        """Close the runner if the name has fallen back BELOW its trend SMA.

        Only active when ``exit_on_trend_break`` is set; otherwise always returns
        False and the position is managed purely by the ATR bracket. Point-in-time
        (the window is sliced to ``<= asof_date``).
        """
        if not self.exit_on_trend_break:
            return False
        px = history.prices(symbols=[symbol], lookback=self._lookback)
        if len(px) == 0:
            return False
        series = px[symbol].dropna()
        if len(series) < self.trend_sma:
            return False
        close = float(series.iloc[-1])
        sma = float(series.iloc[-self.trend_sma:].mean())
        if not (np.isfinite(close) and np.isfinite(sma)):
            return False
        return close < sma
