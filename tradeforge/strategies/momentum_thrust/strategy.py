"""strategies/momentum_thrust/strategy.py — the momentum THRUST continuation edge.

The trend/expansion-regime complement to ``level_meanrev`` (and to
``breakout_retest``), built as a DELIBERATE low-correlation counterpart
(MASTER_PLAN.md §1.A "edge portfolio", §1.B "uncorrelated edge-stacking" +
"positive-skew trade structure: trailing runners", §5 "build level_meanrev /
momentum_thrust as deliberately low-correlation complements").

WHY THIS IS DECORRELATED FROM level_meanrev (and breakout_retest)
-----------------------------------------------------------------
``level_meanrev`` FADES extensions: it sells into strength / buys into weakness
and targets the mean. It wins in RANGE/CHOP and is flat-or-wrong on a runaway
trend day. ``momentum_thrust`` does the EXACT OPPOSITE — it FOLLOWS strength:
on a strong directional thrust (consecutive same-direction closes plus an
EXPANSION bar with above-average range and volume) it enters WITH the trend and
RIDES it on a TRAILING stop with NO fixed target, so it captures the big trend
days that the fade misses entirely.

  * Trend day -> a thrust expansion bar prints -> momentum_thrust enters & rides;
                 level_meanrev is fading the same move and gets stopped / stands
                 down -> opposite outcomes.
  * Chop  day -> no qualifying expansion thrust (ranges stay average) ->
                 momentum_thrust is flat; level_meanrev fades the chop and wins.

Follow vs fade — opposite reactions to the same price action — so when one is
right the other is typically flat or wrong => low / negative PnL correlation by
construction. vs breakout_retest it is also decorrelated on EXIT structure:
breakout_retest caps winners at a fixed 2R, momentum_thrust lets them run on a
trail, so the two harvest DIFFERENT parts of the same trend (and momentum_thrust
also fires on intraday thrusts with no PDH/PDL break at all). (The orchestrator
computes the actual correlation matrix at the integration stage; this module's
job is to be decorrelated *by design*.)

RULES (per RTH session)
-----------------------
Thrust detection (signal bar = current bar):
  LONG  — current bar is an UP expansion bar: range >= ``range_mult`` *
    avg_range(``lookback``); volume >= ``vol_mult`` * avg_volume (when
    ``use_volume``); the close sits in the top ``close_loc`` of the bar's range;
    and it is the ``thrust_bars``-th consecutive HIGHER close. Enter LONG next
    open.
  SHORT — symmetric (down expansion bar, consecutive LOWER closes).

Stop (hard protective backstop, enforced by the engine's OCO):
  long  stop = thrust-bar low  - stop_atr*ATR
  short stop = thrust-bar high + stop_atr*ATR
  risk = abs(signal_close - stop); skip if risk <= 0.

Exit — TRAILING ride, NO fixed target (target=None):
  Track the best price since entry; the trail sits ``trail_atr``*ATR behind it
  (a chandelier stop). Each managed bar, if the bar CLOSES through the trail the
  strategy requests a market close (fills next open) — letting winners run while
  the engine's hard stop caps the downside. The trail LATCHES on once the
  best-price excursion reaches ``trail_after_r`` R (0 = immediately) and stays
  armed thereafter, so a runner that spikes then fades still trails out rather
  than waiting for the hard stop. An optional ``time_stop_bars`` closes a stale
  ride. EOD-flat is enforced by the engine.

Fill semantics are owned by the engine (Pine parity): the market entry fills at
the NEXT bar's open while the protective stop is pinned to the SIGNAL bar's
close; strategy-requested closes fill at the next bar's open.

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


class MomentumThrustStrategy(Strategy):
    """With-trend THRUST entry + ATR chandelier trailing exit (trend-regime edge)."""

    def __init__(self, params: dict | None = None, variant: str = DEFAULT_VARIANT):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)

        self.thrust_bars = int(self.params["thrust_bars"])
        self.lookback = int(self.params["lookback"])
        self.range_mult = float(self.params["range_mult"])
        self.vol_mult = float(self.params["vol_mult"])
        self.use_volume = bool(self.params["use_volume"])
        self.close_loc = float(self.params["close_loc"])
        self.stop_atr = float(self.params["stop_atr"])
        self.trail_atr = float(self.params["trail_atr"])
        self.trail_after_r = float(self.params["trail_after_r"])
        self.time_stop_bars = int(self.params["time_stop_bars"])
        self.warmup_bars = int(self.params["warmup_bars"])
        self.one_per_side = bool(self.params["one_per_side"])
        self.tick = float(self.params["tick"])

        self._reset_daily()

    # ------------------------------------------------------------ daily state
    def _reset_daily(self) -> None:
        self.atr = None
        self.entered_long = False
        self.entered_short = False
        # Trail bookkeeping for the open ride.
        self._ride_side = None       # 'long' | 'short' | None
        self._best = None            # best price since entry (high for long, low for short)
        self._entry_ref = None       # signal close at entry (R reference)
        self._risk = None            # per-share risk at entry
        self._bars_in_trade = 0
        self._trail_armed = False    # latches True once trail_after_r profit is reached

    def on_session_start(self, ctx: Context) -> None:
        self._reset_daily()
        lv = ctx.levels or {}
        self.atr = lv.get("atr14")

    # ----------------------------------------------------------------- on_bar
    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar

        # ---- manage an open ride (trailing exit) BEFORE looking for new entry ----
        if ctx.position is not None:
            self._manage_ride(ctx, bar)
            return

        # We were in a ride last bar and now flat -> the engine's hard stop
        # closed us; clear ride state so we don't re-manage a phantom position.
        if self._ride_side is not None:
            self._ride_side = None

        if self.atr is None or self.atr <= 0:
            return
        if ctx.bar_index < self.warmup_bars:
            return

        prev = ctx.prev_bars
        if len(prev) < self.lookback:
            return

        window = prev[-self.lookback:]
        avg_range = sum((b.high - b.low) for b in window) / len(window)
        avg_vol = sum(b.volume for b in window) / len(window)
        if avg_range <= 0:
            return

        bar_range = bar.high - bar.low
        is_expansion = bar_range >= self.range_mult * avg_range
        has_volume = (not self.use_volume) or (avg_vol <= 0) or (bar.volume >= self.vol_mult * avg_vol)
        if not (is_expansion and has_volume):
            return
        close_pos = (bar.close - bar.low) / bar_range if bar_range > 0 else 0.5

        # ---- LONG thrust ----
        up = (
            bar.close > bar.open
            and close_pos >= self.close_loc
            and self._consecutive(prev, bar, "up")
        )
        if up and not (self.one_per_side and self.entered_long):
            self._arm_long(ctx, bar)
            return

        # ---- SHORT thrust ----
        down = (
            bar.close < bar.open
            and close_pos <= (1.0 - self.close_loc)
            and self._consecutive(prev, bar, "down")
        )
        if down and not (self.one_per_side and self.entered_short):
            self._arm_short(ctx, bar)

    # --------------------------------------------------------- thrust helpers
    def _consecutive(self, prev: list, bar, direction: str) -> bool:
        """True if the last ``thrust_bars`` closes (incl. signal bar) are monotone.

        For ``thrust_bars`` = N we need N consecutive higher (up) / lower (down)
        closes ending on the signal bar — i.e. the signal close plus the N-1
        prior closes form a strictly monotone run.
        """
        n = self.thrust_bars
        if n <= 1:
            return True
        if len(prev) < n - 1:
            return False
        closes = [b.close for b in prev[-(n - 1):]] + [bar.close]
        if direction == "up":
            return all(closes[i] > closes[i - 1] for i in range(1, len(closes)))
        return all(closes[i] < closes[i - 1] for i in range(1, len(closes)))

    # --------------------------------------------------------- entry logic
    def _arm_long(self, ctx: Context, bar) -> None:
        stop = bar.low - self.stop_atr * self.atr
        risk = abs(bar.close - stop)
        if risk <= 0:
            return
        # NO fixed target — winners ride the trail.
        ctx.enter_long(stop=stop, target=None)
        self.entered_long = True
        self._ride_side = "long"
        self._best = bar.close
        self._entry_ref = bar.close
        self._risk = risk
        self._bars_in_trade = 0
        self._trail_armed = self.trail_after_r <= 0.0

    def _arm_short(self, ctx: Context, bar) -> None:
        stop = bar.high + self.stop_atr * self.atr
        risk = abs(stop - bar.close)
        if risk <= 0:
            return
        ctx.enter_short(stop=stop, target=None)
        self.entered_short = True
        self._ride_side = "short"
        self._best = bar.close
        self._entry_ref = bar.close
        self._risk = risk
        self._bars_in_trade = 0
        self._trail_armed = self.trail_after_r <= 0.0

    # --------------------------------------------------------- trailing exit
    def _manage_ride(self, ctx: Context, bar) -> None:
        pos = ctx.position
        side = pos.side
        self._bars_in_trade += 1

        # Time stop (stale ride).
        if self.time_stop_bars > 0 and self._bars_in_trade >= self.time_stop_bars:
            ctx.close()
            return

        atr = self.atr or 0.0
        risk = self._risk or 0.0

        if side == "long":
            self._best = bar.high if self._best is None else max(self._best, bar.high)
            # Latch the trail on once profit reaches trail_after_r R (best-price
            # based, so a spike that later fades still trails out — it does not
            # disarm when price dips back below entry).
            if not self._trail_armed and (
                risk <= 0 or (self._best - self._entry_ref) >= self.trail_after_r * risk
            ):
                self._trail_armed = True
            if self._trail_armed:
                trail = self._best - self.trail_atr * atr
                if bar.close <= trail:
                    ctx.close()
        else:  # short
            self._best = bar.low if self._best is None else min(self._best, bar.low)
            if not self._trail_armed and (
                risk <= 0 or (self._entry_ref - self._best) >= self.trail_after_r * risk
            ):
                self._trail_armed = True
            if self._trail_armed:
                trail = self._best + self.trail_atr * atr
                if bar.close >= trail:
                    ctx.close()
