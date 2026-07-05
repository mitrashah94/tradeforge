"""breakout_retest_options: an IV-rank-driven options-expression OVERLAY for the
``breakout_retest`` signal (SPY / QQQ).

It is not an independent edge — it inherits the break-and-retest directional
trigger and decides HOW to express it in listed options: IV rank picks the
structure (low -> long option, mid -> debit vertical, high -> credit spread),
delta picks the strikes, theta gates long-premium bleed, and the RI options
policy gates permission + size. Pure, deterministic, RESEARCH-only — it emits a
decision to inspect on paper; it never places orders (CLAUDE.md P0 #3).
"""

from strategies.breakout_retest_options.overlay import (
    Leg,
    OptionContract,
    OptionsOverlay,
    OverlayDecision,
    UnderlyingSignal,
    choose_structure,
    classify_iv_regime,
    load_params,
    nearest_delta,
)

__all__ = [
    "OptionsOverlay",
    "OverlayDecision",
    "UnderlyingSignal",
    "OptionContract",
    "Leg",
    "classify_iv_regime",
    "choose_structure",
    "nearest_delta",
    "load_params",
]
