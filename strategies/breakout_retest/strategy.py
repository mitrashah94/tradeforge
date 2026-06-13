"""strategies/breakout_retest/strategy.py — the PDH/PDL continuation strategy.

The PF-2.24 trend-continuation baseline, ported faithfully from the TradingView
Pine v6 "PDH/PDL Continuation" strategy. Implements the engine
:class:`~backtest.engine.engine.Strategy` interface and reads its parameters
from ``params.yaml`` (with named V0..V4 variant deltas).

V0 (the reproduction gate) = entry_type "retest" + target_mode "fixed_2r" +
PDH/PDL levels only.

PORTED RULES (per RTH session, using that session's PDH/PDL from `levels`)
-------------------------------------------------------------------------
Break detection (once per day per side):
  - the first 5m bar that CLOSES above PDH sets ``pdh_broken`` (long side);
  - the first that CLOSES below PDL sets ``pdl_broken`` (short side).

bars_since_break counter:
  - increments every bar once broken; it equals 1 ON the break bar itself
    (Pine increments after detection on the same bar), 2 on the next bar, ...

Entry Type B (Retest) — the V0 config:
  LONG retest fires when
    pdh_broken AND bars_since_pdh_break in [2,7] AND low <= pdh AND close > pdh
    AND not already entered long today AND not stopped-out long today AND flat.
  SHORT retest is symmetric with PDL (high >= pdl AND close < pdl).

Entry Type A (Break): fires when bars_since_break == 1 (on the break bar).
  Supported for later ablation; V0 does not use it.

Stop (role reversal):
  long stop  = pdh - stop_buffer_ticks*tick;
  short stop = pdl + stop_buffer_ticks*tick.
  risk = abs(signal_close - stop); skip the entry if risk <= 0.

Target:
  fixed_2r: long target = signal_close + r*risk; short = signal_close - r*risk.
  trailing: reserved for later ablation (V3/V4) — not exercised in V0.

Fill semantics are owned by the engine (Pine parity): the market entry fills at
the NEXT bar's open, while the stop/target are pinned to the SIGNAL bar's close.

One attempt per side per day; no re-entry that side after a stop-out that day.
All daily state resets at on_session_start.

The ablation hooks (ntz_filter, use_pmh_pml, partial_runner, ...) are read from
params but their LOGIC is intentionally not implemented in Stage 1 — only their
params are reserved so params.yaml is complete for later stages.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from backtest.engine.engine import Context, Strategy

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parent / "params.yaml"


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


class BreakoutRetestStrategy(Strategy):
    """Parameterized PDH/PDL break+retest (or break) continuation strategy."""

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

        # Reserved ablation params (logic not implemented in Stage 1).
        self.ntz_filter = bool(self.params.get("ntz_filter", False))
        self.use_pmh_pml = bool(self.params.get("use_pmh_pml", False))
        self.partial_runner = bool(self.params.get("partial_runner", False))

        self._reset_daily()

    # ------------------------------------------------------------ daily state
    def _reset_daily(self) -> None:
        self.pdh = None
        self.pdl = None
        self.pdh_broken = False
        self.pdl_broken = False
        self.bars_since_pdh_break = 0
        self.bars_since_pdl_break = 0
        self.entered_long = False
        self.entered_short = False
        self.stopped_long = False
        self.stopped_short = False
        self._prev_position_side = None  # to detect stop-outs across bars

    def on_session_start(self, ctx: Context) -> None:
        self._reset_daily()
        lv = ctx.levels or {}
        self.pdh = lv.get("pdh")
        self.pdl = lv.get("pdl")

    # ----------------------------------------------------------------- on_bar
    def on_bar(self, ctx: Context) -> None:
        bar = ctx.bar

        # Detect a stop-out that happened on THIS or a prior bar so we can block
        # re-entry on that side for the rest of the day. The engine closes a
        # position before on_bar is called, so a position that was open last bar
        # and is now flat (and not via our own entry this bar) was an exit.
        self._track_stopouts(ctx)

        # ---- break detection (once per day per side) ----
        if self.pdh is not None and not self.pdh_broken and bar.close > self.pdh:
            self.pdh_broken = True
            self.bars_since_pdh_break = 0  # becomes 1 after the increment below
        if self.pdl is not None and not self.pdl_broken and bar.close < self.pdl:
            self.pdl_broken = True
            self.bars_since_pdl_break = 0

        # ---- bars_since_break counter: 1 on the break bar, 2 next, ... ----
        if self.pdh_broken:
            self.bars_since_pdh_break += 1
        if self.pdl_broken:
            self.bars_since_pdl_break += 1

        # Only one position at a time; if already in a trade, just manage (the
        # engine handles the OCO). Do not arm a new entry while in a position.
        if ctx.position is not None:
            return

        # ---- entries ----
        if self.entry_type == "retest":
            self._try_retest(ctx, bar)
        elif self.entry_type == "break":
            self._try_break(ctx, bar)

    # --------------------------------------------------------- entry logic
    def _try_retest(self, ctx: Context, bar) -> None:
        # LONG retest
        if (
            self.pdh is not None
            and self.pdh_broken
            and self.window_min <= self.bars_since_pdh_break <= self.window_max
            and bar.low <= self.pdh
            and bar.close > self.pdh
            and not self.entered_long
            and not self.stopped_long
        ):
            self._arm_long(ctx, bar)
            return

        # SHORT retest
        if (
            self.pdl is not None
            and self.pdl_broken
            and self.window_min <= self.bars_since_pdl_break <= self.window_max
            and bar.high >= self.pdl
            and bar.close < self.pdl
            and not self.entered_short
            and not self.stopped_short
        ):
            self._arm_short(ctx, bar)

    def _try_break(self, ctx: Context, bar) -> None:
        # Entry Type A: fire on the break bar itself (bars_since_break == 1).
        if (
            self.pdh is not None
            and self.pdh_broken
            and self.bars_since_pdh_break == 1
            and not self.entered_long
            and not self.stopped_long
        ):
            self._arm_long(ctx, bar)
            return
        if (
            self.pdl is not None
            and self.pdl_broken
            and self.bars_since_pdl_break == 1
            and not self.entered_short
            and not self.stopped_short
        ):
            self._arm_short(ctx, bar)

    def _arm_long(self, ctx: Context, bar) -> None:
        stop = self.pdh - self.stop_buffer_ticks * self.tick
        risk = abs(bar.close - stop)
        if risk <= 0:
            return
        target = None
        if self.target_mode == "fixed_2r":
            target = bar.close + self.r_multiple * risk
        ctx.enter_long(stop=stop, target=target)
        self.entered_long = True

    def _arm_short(self, ctx: Context, bar) -> None:
        stop = self.pdl + self.stop_buffer_ticks * self.tick
        risk = abs(bar.close - stop)
        if risk <= 0:
            return
        target = None
        if self.target_mode == "fixed_2r":
            target = bar.close - self.r_multiple * risk
        ctx.enter_short(stop=stop, target=target)
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
