"""strategies/breakout_retest/strategy.py — the PDH/PDL continuation strategy.

The PF-2.24 trend-continuation baseline, ported faithfully from the TradingView
Pine v6 "PDH/PDL Continuation" strategy. Implements the engine
:class:`~backtest.engine.engine.Strategy` interface and reads its parameters
from ``params.yaml`` (with named V0..V4 variant deltas + ``v0_atr_stop``).

V0 (the reproduction gate) = entry_type "retest" + target_mode "fixed_2r" +
PDH/PDL levels only.

PORTED RULES (per RTH session, using that session's levels from `levels`)
-------------------------------------------------------------------------
Break detection (once per day per level/side):
  - the first 5m bar that CLOSES above an UP level (pdh/pmh) sets its broken
    flag (a long-side level);
  - the first that CLOSES below a DOWN level (pdl/pml) sets it (short-side).

bars_since_break counter (per level):
  - increments every bar once broken; it equals 1 ON the break bar itself
    (Pine increments after detection on the same bar), 2 on the next bar, ...

Entry Type B (Retest) — the V0 config:
  A LONG retest fires for an up-level L when
    broken AND bars_since_break in [2,7] AND low <= L AND close > L
    AND not already entered that side today AND not stopped-out that side today
    AND flat.
  A SHORT retest is symmetric on a down-level.

Entry Type A (Break): fires when bars_since_break == 1 (on the break bar).
  Supported for later ablation; V0 does not use it.

Stop (role reversal):
  long stop  = level - stop_buffer_ticks*tick   (or level - atr_stop_k*atr14);
  short stop = level + stop_buffer_ticks*tick    (or level + atr_stop_k*atr14).
  risk = abs(signal_close - stop); skip the entry if risk <= 0.

Target:
  fixed_2r: long target = signal_close + r*risk; short = signal_close - r*risk.
  trailing: no fixed target; the engine manages a partial-at-TP1 + breakeven +
            trailing-runner bracket (V3/V4) via a PartialPlan.

Fill semantics are owned by the engine (Pine parity): the market entry fills at
the NEXT bar's open, while the stop/target are pinned to the SIGNAL bar's close.

One attempt per side per day; no re-entry that side after a stop-out that day.
All daily state resets at on_session_start.

ABLATION COMPONENTS (each gated by a param; MASTER_PLAN §5)
----------------------------------------------------------
  ntz_filter (V1+): block any entry whose breaking level sits INSIDE the
      session's No-Trade Zone [ntz_low, ntz_high] when ntz_valid — the overlap
      of the prior-day and premarket ranges, where price is indecisive.
  use_pmh_pml (V2+): in addition to PDH/PDL, arm break->retest on the premarket
      high/low (pmh up-side, pml down-side) with the identical logic.
  partial_runner + target_mode=trailing (V3+): scale out partial_fraction at
      +partial_tp1_r R, move the stop to breakeven, trail the runner by the
      prior bar's low/high (engine-managed PartialPlan).
  atr_stop_k > 0 (v0_atr_stop): widen the role-reversal stop to level ∓
      atr_stop_k * ATR14 instead of the very tight ∓1 tick — probes whether the
      tight stop is the fragility (it blows avg loss out to ~2.5R under slippage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from backtest.engine.engine import Context, PartialPlan, Strategy

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"

# The two long-side ("up") and short-side ("down") level names, in priority
# order. PDH/PDL are V0; PMH/PML are added by V2+ (use_pmh_pml).
_UP_LEVELS = ("pdh", "pmh")
_DOWN_LEVELS = ("pdl", "pml")


def load_params(variant: str = "V0", path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
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


@dataclass
class _LevelState:
    """Per-level break/retest tracking state, reset each session."""

    name: str               # 'pdh' | 'pdl' | 'pmh' | 'pml'
    side: str               # 'long' (up level) | 'short' (down level)
    price: float | None = None
    broken: bool = False
    bars_since_break: int = 0


class BreakoutRetestStrategy(Strategy):
    """Parameterized PDH/PDL(+PMH/PML) break+retest continuation strategy."""

    def __init__(self, params: dict | None = None, variant: str = "V0"):
        self.params = params if params is not None else load_params(variant)
        self.variant = self.params.get("_variant", variant)

        self.entry_type = self.params["entry_type"]          # break | retest
        self.target_mode = self.params["target_mode"]        # fixed_2r | trailing
        self.window_min = int(self.params["retest_window_min"])
        self.window_max = int(self.params["retest_window_max"])
        self.stop_buffer_ticks = float(self.params["stop_buffer_ticks"])
        self.r_multiple = float(self.params["r_multiple"])
        self.tick = float(self.params["tick"])

        # ---- ablation params (LOGIC implemented; gated by the values) ----
        self.ntz_filter = bool(self.params.get("ntz_filter", False))
        self.use_pmh_pml = bool(self.params.get("use_pmh_pml", False))
        self.partial_runner = bool(self.params.get("partial_runner", False))
        self.partial_tp1_r = float(self.params.get("partial_tp1_r", 1.0))
        self.partial_fraction = float(self.params.get("partial_fraction", 0.5))
        self.trail_mode = str(self.params.get("trail_mode", "prior_bar"))
        # atr_stop_k > 0 switches the role-reversal stop to level ∓ k*ATR14.
        self.atr_stop_k = float(self.params.get("atr_stop_k", 0.0))
        # break_buffer_atr > 0 (V4): a clean break must CLOSE beyond the level by
        # at least break_buffer_atr * ATR14 (filters marginal pokes through).
        self.break_buffer_atr = float(self.params.get("break_buffer_atr", 0.0))

        # Which up/down levels are armed (PDH/PDL always; PMH/PML if enabled).
        self._up_names = list(_UP_LEVELS) if self.use_pmh_pml else ["pdh"]
        self._down_names = list(_DOWN_LEVELS) if self.use_pmh_pml else ["pdl"]

        self._reset_daily()

    # ------------------------------------------------------------ daily state
    def _reset_daily(self) -> None:
        self._levels: dict[str, _LevelState] = {}
        for nm in self._up_names:
            self._levels[nm] = _LevelState(name=nm, side="long")
        for nm in self._down_names:
            self._levels[nm] = _LevelState(name=nm, side="short")
        # One attempt per SIDE per day (not per level) — matches V0 semantics:
        # at most one long and one short attempt, no re-entry after a stop-out.
        self.entered_long = False
        self.entered_short = False
        self.stopped_long = False
        self.stopped_short = False
        self._prev_position_side = None  # to detect stop-outs across bars
        self._ntz = (None, None, False)  # (low, high, valid)
        self._atr14 = None

    def on_session_start(self, ctx: Context) -> None:
        self._reset_daily()
        lv = ctx.levels or {}
        for st in self._levels.values():
            st.price = lv.get(st.name)
        self._ntz = (lv.get("ntz_low"), lv.get("ntz_high"), bool(lv.get("ntz_valid")))
        self._atr14 = lv.get("atr14")

    # ----------------------------------------------------------------- on_bar
    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar

        # Detect a stop-out that happened on THIS or a prior bar so we can block
        # re-entry on that side for the rest of the day. The engine closes a
        # position before on_bar is called, so a position that was open last bar
        # and is now flat (and not via our own entry this bar) was an exit.
        self._track_stopouts(ctx)

        # ---- break detection + counter (per level) ----
        # V4 break_buffer_atr: require the close to clear the level by
        # break_buffer_atr * ATR14 (0 -> any close beyond the level, the V0 rule).
        buf = 0.0
        if self.break_buffer_atr > 0 and self._atr14 is not None and self._atr14 > 0:
            buf = self.break_buffer_atr * self._atr14
        for st in self._levels.values():
            if st.price is None:
                continue
            if not st.broken:
                if st.side == "long" and bar.close > st.price + buf:
                    st.broken = True
                    st.bars_since_break = 0  # becomes 1 after the increment below
                elif st.side == "short" and bar.close < st.price - buf:
                    st.broken = True
                    st.bars_since_break = 0
            if st.broken:
                st.bars_since_break += 1  # 1 on the break bar, 2 next, ...

        # Only one position at a time; if already in a trade, just manage (the
        # engine handles the OCO). Do not arm a new entry while in a position.
        if ctx.position is not None:
            return

        # ---- entries ----
        if self.entry_type == "retest":
            self._try_entry(ctx, bar, mode="retest")
        elif self.entry_type == "break":
            self._try_entry(ctx, bar, mode="break")

    # --------------------------------------------------------- entry logic
    def _try_entry(self, ctx: Context, bar, mode: str) -> None:
        # LONG side: scan up-levels in priority order (pdh before pmh).
        if not self.entered_long and not self.stopped_long:
            for nm in self._up_names:
                st = self._levels[nm]
                if self._qualifies(st, bar, mode):
                    self._arm(ctx, bar, st)
                    return
        # SHORT side: scan down-levels in priority order (pdl before pml).
        if not self.entered_short and not self.stopped_short:
            for nm in self._down_names:
                st = self._levels[nm]
                if self._qualifies(st, bar, mode):
                    self._arm(ctx, bar, st)
                    return

    def _qualifies(self, st: _LevelState, bar, mode: str) -> bool:
        if st.price is None or not st.broken:
            return False
        if mode == "retest":
            if not (self.window_min <= st.bars_since_break <= self.window_max):
                return False
            if st.side == "long":
                return bar.low <= st.price and bar.close > st.price
            return bar.high >= st.price and bar.close < st.price
        # mode == "break": fire on the break bar itself.
        return st.bars_since_break == 1

    def _in_ntz(self, level_price: float) -> bool:
        """True if ``level_price`` sits inside a valid No-Trade Zone band."""
        if not self.ntz_filter:
            return False
        low, high, valid = self._ntz
        if not valid or low is None or high is None:
            return False
        return low <= level_price <= high

    def _stop_for(self, st: _LevelState) -> float:
        """Role-reversal stop: level ∓ (k*ATR14 if atr_stop_k>0 else ticks)."""
        if self.atr_stop_k > 0 and self._atr14 is not None and self._atr14 > 0:
            offset = self.atr_stop_k * self._atr14
        else:
            offset = self.stop_buffer_ticks * self.tick
        return st.price - offset if st.side == "long" else st.price + offset

    def _arm(self, ctx: Context, bar, st: _LevelState) -> None:
        # V1+ NTZ filter: block entries whose breaking level is inside the NTZ.
        if self._in_ntz(st.price):
            # Mark the side as spent so a later qualifying level is still blocked
            # for the same reason this bar; but do NOT permanently consume the
            # day's attempt (other side / later bars may still trade). We simply
            # do not enter and let the next qualifying bar re-evaluate.
            return

        stop = self._stop_for(st)
        risk = abs(bar.close - stop)
        if risk <= 0:
            return

        target = None
        partial = None
        if self.target_mode == "fixed_2r":
            if st.side == "long":
                target = bar.close + self.r_multiple * risk
            else:
                target = bar.close - self.r_multiple * risk
        elif self.target_mode == "trailing" and self.partial_runner:
            # V3/V4: partial at +tp1_r R, breakeven, trail the runner.
            if st.side == "long":
                tp1 = bar.close + self.partial_tp1_r * risk
            else:
                tp1 = bar.close - self.partial_tp1_r * risk
            partial = PartialPlan(
                tp1=tp1,
                tp1_r=self.partial_tp1_r,
                fraction=self.partial_fraction,
                trail_mode=self.trail_mode,
            )
        # else: target_mode=trailing without partial_runner -> pure runner
        #       (no fixed target, engine EOD-flat / stop only). Not used by any
        #       declared variant but kept coherent.

        if st.side == "long":
            ctx.enter_long(stop=stop, target=target, partial=partial)
            self.entered_long = True
        else:
            ctx.enter_short(stop=stop, target=target, partial=partial)
            self.entered_short = True

    # ----------------------------------------------------- stop-out tracking
    def _track_stopouts(self, ctx: Context) -> None:
        """Block re-entry on a side after a stop-out that day.

        We watch position transitions: if we were long last bar and are now flat
        (the engine closed us), the long attempt is spent for the day. Same for
        short. Combined with ``entered_long``/``entered_short`` this gives the
        one-attempt-per-side-per-day-with-no-re-entry-after-stop rule.
        """
        cur_side = ctx.position.side if ctx.position is not None else None
        prev = self._prev_position_side
        if prev == "long" and cur_side != "long":
            self.stopped_long = True
        if prev == "short" and cur_side != "short":
            self.stopped_short = True
        self._prev_position_side = cur_side
