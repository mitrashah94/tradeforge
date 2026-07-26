"""strategies/level_meanrev/strategy.py — the level-fade MEAN-REVERSION edge.

The chop/range-regime complement to ``breakout_retest``, built as a DELIBERATE
low-correlation counterpart (MASTER_PLAN.md §1.A "edge portfolio", §1.B
"uncorrelated edge-stacking", §5 "build level_meanrev / momentum_thrust as
deliberately low-correlation complements").

WHY THIS IS DECORRELATED FROM breakout_retest (the whole point)
---------------------------------------------------------------
``breakout_retest`` is a CONTINUATION setup: a clean break of PDH (a bar that
CLOSES beyond the level) arms it to BUY the retest and ride the trend. It wins
in TREND/expansion regimes — when a level breaks and price keeps going.

``level_meanrev`` takes the OPPOSITE side of the SAME level event. When price
EXTENDS to a level (tags PDH/PMH from below, or PDL/PML from above) but FAILS to
continue — the bar pokes the level and CLOSES back on the mean-side of it, a
rejection — we FADE it: enter COUNTER-TREND at the extreme, stop just beyond the
level, and target the MEAN (the session midpoint, or a fixed 1-1.5R). It wins in
RANGE/CHOP regimes — exactly the days a continuation break fails and reverses.

Crucially, a *clean break* (a close beyond the level by ``break_buffer_atr*ATR``)
DISARMS the fade on that side: once price genuinely breaks out, that level
belongs to breakout_retest's regime, not ours. So at any given level break the
two strategies react in OPPOSITE directions and are rarely both right at once:

  * Trend day  -> level breaks cleanly -> breakout_retest enters & wins;
                  level_meanrev is disarmed (flat) -> no shared loss/gain.
  * Chop  day  -> level is tagged & rejected -> level_meanrev fades & wins;
                  breakout_retest's retest either never arms or stops out.

Opposite reactions to the same trigger => when one is right the other tends to
be flat or wrong => low / negative PnL correlation by construction. (The
orchestrator computes the actual correlation matrix at the integration stage;
this module's job is to be decorrelated *by design*.)

RULES (per RTH session, using that session's levels from the `levels` table)
----------------------------------------------------------------------------
Setup arming (once per side per session when ``one_per_side``):
  SHORT fade — the bar's HIGH comes within ``tag_atr*ATR`` of an UPPER level
    (PDH/PMH) AND the bar CLOSES back BELOW that level (rejection), AND the bar
    did NOT close beyond the level by ``break_buffer_atr*ATR`` (not a clean
    break). Enter SHORT at the next bar's open.
  LONG fade — symmetric at a LOWER level (PDL/PML): tag from above + close back
    ABOVE the level.

Stop (just beyond the faded extreme):
  short stop = max(bar.high, level) + stop_buffer_atr*ATR
  long  stop = min(bar.low,  level) - stop_buffer_atr*ATR
  risk = abs(signal_close - stop); skip if risk <= 0 or stop wider than
  ``max_stop_atr*ATR`` (degenerate / too-wide fade).

Target (toward the MEAN):
  mean_mode == "fixed_r" (the DEFAULT): target = signal_close -/+ target_r*risk
    — a robust 1-1.5R fade back off the rejected level.
  mean_mode == "session_mid": target = session midpoint = (session_high +
    session_low)/2 measured over the bars seen so far. Skip the entry if the
    implied reward is below ``min_target_r`` (no edge fading into a near mean).

Fill semantics are owned by the engine (Pine parity): the market entry fills at
the NEXT bar's open while stop/target are pinned to the SIGNAL bar's close.

One attempt per side per day (``one_per_side``); all daily state resets at
``on_session_start``. RTH + EOD-flat + one-position-at-a-time are enforced by
the engine.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from backtest.engine.engine import Context, Strategy

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"

# The default variant the integration stage runs uniformly.
DEFAULT_VARIANT = "DEFAULT"
# Variants exposed for uniform sweeping by the integration stage.
VARIANTS = ["DEFAULT", "V0", "V1", "V2"]


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


class LevelMeanRevStrategy(Strategy):
    """Counter-trend FADE of a level tag-and-rejection (chop-regime edge)."""

    def __init__(self, params: dict | None = None, variant: str = DEFAULT_VARIANT):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)

        self.upper_levels = list(self.params["upper_levels"])
        self.lower_levels = list(self.params["lower_levels"])
        self.tag_atr = float(self.params["tag_atr"])
        self.require_rejection = bool(self.params["require_rejection"])
        self.break_buffer_atr = float(self.params["break_buffer_atr"])
        self.stop_buffer_atr = float(self.params["stop_buffer_atr"])
        self.mean_mode = str(self.params["mean_mode"])          # session_mid | fixed_r
        self.target_r = float(self.params["target_r"])
        self.min_target_r = float(self.params["min_target_r"])
        self.max_stop_atr = float(self.params["max_stop_atr"])
        self.warmup_bars = int(self.params["warmup_bars"])
        self.one_per_side = bool(self.params["one_per_side"])
        self.tick = float(self.params["tick"])

        self._reset_daily()

    # ------------------------------------------------------------ daily state
    def _reset_daily(self) -> None:
        self.atr = None
        self.upper_vals: list[float] = []   # resolved upper level prices for the session
        self.lower_vals: list[float] = []   # resolved lower level prices
        self.entered_long = False
        self.entered_short = False
        self.session_high = None
        self.session_low = None

    def on_session_start(self, ctx: Context) -> None:
        self._reset_daily()
        lv = ctx.levels or {}
        self.atr = lv.get("atr14")
        self.upper_vals = [
            float(lv[k]) for k in self.upper_levels if lv.get(k) is not None
        ]
        self.lower_vals = [
            float(lv[k]) for k in self.lower_levels if lv.get(k) is not None
        ]

    # ----------------------------------------------------------------- on_bar
    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar

        # Maintain the running session range (used for the session-midpoint
        # target and to know where "the mean" is).
        self.session_high = bar.high if self.session_high is None else max(self.session_high, bar.high)
        self.session_low = bar.low if self.session_low is None else min(self.session_low, bar.low)

        # Need a usable ATR (sizes tag/break/stop tolerances) and a short warmup
        # so a session range exists before we fade into it.
        if self.atr is None or self.atr <= 0:
            return
        if ctx.bar_index < self.warmup_bars:
            return

        # One position at a time; the engine manages the OCO once we're in.
        if ctx.position is not None:
            return

        tol = self.tag_atr * self.atr
        brk = self.break_buffer_atr * self.atr

        # ---- SHORT fade: tag an upper level and reject it ----
        if not (self.one_per_side and self.entered_short):
            for lvl in self.upper_vals:
                tagged = bar.high >= lvl - tol           # came up to / through the level
                clean_break = bar.close >= lvl + brk     # a real breakout -> NOT a fade
                rejected = bar.close < lvl               # closed back below the level
                if tagged and not clean_break and (rejected or not self.require_rejection):
                    if self._arm_short(ctx, bar, lvl):
                        return

        # ---- LONG fade: tag a lower level and reject it ----
        if not (self.one_per_side and self.entered_long):
            for lvl in self.lower_vals:
                tagged = bar.low <= lvl + tol
                clean_break = bar.close <= lvl - brk
                rejected = bar.close > lvl
                if tagged and not clean_break and (rejected or not self.require_rejection):
                    if self._arm_long(ctx, bar, lvl):
                        return

    # --------------------------------------------------------- entry logic
    def _arm_short(self, ctx: Context, bar, lvl: float) -> bool:
        extreme = max(bar.high, lvl)
        stop = extreme + self.stop_buffer_atr * self.atr
        risk = abs(stop - bar.close)
        if risk <= 0 or risk > self.max_stop_atr * self.atr:
            return False
        target = self._target_short(bar, risk)
        if target is None:
            return False
        ctx.enter_short(stop=stop, target=target)
        self.entered_short = True
        return True

    def _arm_long(self, ctx: Context, bar, lvl: float) -> bool:
        extreme = min(bar.low, lvl)
        stop = extreme - self.stop_buffer_atr * self.atr
        risk = abs(bar.close - stop)
        if risk <= 0 or risk > self.max_stop_atr * self.atr:
            return False
        target = self._target_long(bar, risk)
        if target is None:
            return False
        ctx.enter_long(stop=stop, target=target)
        self.entered_long = True
        return True

    # --------------------------------------------------------- target logic
    def _target_short(self, bar, risk: float) -> float | None:
        if self.mean_mode == "fixed_r":
            return bar.close - self.target_r * risk
        # session_mid: fade back toward the running session midpoint.
        mid = (self.session_high + self.session_low) / 2.0
        if mid >= bar.close:                 # mean is not below us -> no fade edge
            return None
        if (bar.close - mid) < self.min_target_r * risk:
            return None
        return mid

    def _target_long(self, bar, risk: float) -> float | None:
        if self.mean_mode == "fixed_r":
            return bar.close + self.target_r * risk
        mid = (self.session_high + self.session_low) / 2.0
        if mid <= bar.close:                 # mean is not above us -> no fade edge
            return None
        if (mid - bar.close) < self.min_target_r * risk:
            return None
        return mid
