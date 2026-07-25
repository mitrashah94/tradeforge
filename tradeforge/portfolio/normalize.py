"""portfolio/normalize.py — turn every sleeve's native signal into Candidates. (#1)

The sleeves speak two different languages:

  * WEIGHT sleeves (``momentum_rotation``, ``swing_meanrev``) answer
    ``target_weights(asof, history) -> {symbol: fraction}`` — a standing
    allocation vector with no per-position stop.
  * SCORE sleeves (``swing_breakout``) answer
    ``entry_score(symbol, asof, history) -> float | None`` — a per-name strength
    that the engine's ATR bracket then manages.

This module's adapters translate BOTH into a uniform ``list[Candidate]`` so the
rest of the pipeline (rank → dedup → resolve → budget) is sleeve-shape-agnostic.
A weight candidate carries ``target_weight`` (and that weight doubles as its rank
score); a score candidate carries ``score`` plus the bracket basis
(``entry_price`` / ``stop`` / ``atr``). PURE — the adapters only read the
point-in-time ``history`` + today's :class:`~portfolio.model.DayBars`.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from portfolio.model import Candidate, DayBars, SleeveSpec


# --------------------------------------------------------------------------- #
# Weight sleeves (rotation, swing_meanrev) → Candidate[target_weight]
# --------------------------------------------------------------------------- #
class WeightSleeveAdapter:
    """Adapt a ``target_weights`` strategy into weight-shaped Candidates.

    Returns a Candidate for EVERY symbol the sleeve targets today (held or not —
    the engine separates held names for reconcile from new names for the open
    competition; unlike a score sleeve, a weight sleeve must surface its full
    standing vector so a dropped name can be closed and a continuing name resized).

    The arm decision is applied HERE (the "normalize" step owns it): a DISARMED
    weight sleeve parks in ``risk_off_weights`` instead of its own book — exactly
    the ``regime_reader`` defensive fallback. ``entry_price`` is today's close and
    ``atr`` is the synthetic-stop-window ATR; the synthetic stop itself
    (``mark - k*ATR``) is applied downstream by the budgeter so the band knob stays
    in one place.
    """

    kind = "weight"

    def __init__(self, spec: SleeveSpec):
        self.spec = spec
        self.strategy = spec.strategy

    def candidates(
        self,
        asof,
        history,
        bars: DayBars,
        *,
        armed: bool = True,
        risk_off_weights: Optional[Mapping[str, float]] = None,
        synthetic_window: int = 14,
    ) -> list:
        if armed:
            raw = self.strategy.target_weights(asof, history) or {}
        else:
            raw = dict(risk_off_weights or {})

        out: list = []
        for sym, w in raw.items():
            try:
                wf = float(w)
            except (TypeError, ValueError):
                continue
            if not (wf > 0):
                continue
            close = bars.close_of(sym)
            atr = bars.atr_of(sym, synthetic_window)
            out.append(
                Candidate(
                    sleeve=self.spec.name,
                    symbol=sym,
                    kind="weight",
                    side="long",
                    grade=self.spec.grade,
                    family=self.spec.resolved_family(),
                    target_weight=wf,
                    score=None,
                    entry_price=close,
                    atr=atr,
                    stop=None,            # synthetic band filled by the budgeter
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Score sleeves (swing_breakout) → Candidate[score + bracket basis]
# --------------------------------------------------------------------------- #
class ScoreSleeveAdapter:
    """Adapt an ``entry_score`` strategy into score-shaped Candidates.

    Mirrors the bracket engine's entry scan: ask ``entry_score`` for every
    universe symbol NOT already held by this sleeve, keep the finite scores, and
    attach the ATR bracket basis (stop = ``close - stop_atr_mult*ATR``, plus the
    TP1 / hard-target prices) from the sleeve's :class:`BracketConfig`. The engine
    ranks on the score and manages the admitted opens with that bracket.

    A symbol with no finite close, no finite bracket-window ATR, a non-positive
    price, or a degenerate stop distance is skipped — the same guards the bracket
    engine applies before sizing.
    """

    kind = "score"

    def __init__(self, spec: SleeveSpec):
        self.spec = spec
        self.strategy = spec.strategy
        self.bracket = spec.bracket     # BracketConfig (atr_window, stop_atr_mult, ...)

    def candidates(
        self,
        asof,
        history,
        bars: DayBars,
        *,
        held_by_sleeve: Optional[Sequence[str]] = None,
        universe: Optional[Sequence[str]] = None,
    ) -> list:
        held = set(held_by_sleeve or [])
        syms = list(universe) if universe is not None else list(history.universe)
        br = self.bracket
        atr_window = int(getattr(br, "atr_window", 14))
        stop_mult = float(getattr(br, "stop_atr_mult", 2.5))
        tp1_R = getattr(br, "tp1_R", None)
        tp1_fraction = float(getattr(br, "tp1_fraction", 0.0) or 0.0)
        hard_target_R = getattr(br, "hard_target_R", None)

        out: list = []
        for sym in syms:
            if sym in held:
                continue
            close = bars.close_of(sym)
            atr = bars.atr_of(sym, atr_window)
            if close is None or atr is None or close <= 0 or atr <= 0:
                continue
            score = self.strategy.entry_score(sym, asof, history)
            if score is None:
                continue
            try:
                sf = float(score)
            except (TypeError, ValueError):
                continue
            if sf != sf:  # NaN
                continue
            stop = close - stop_mult * atr
            rps = close - stop
            if rps <= 0:
                continue
            tp1_price = (
                close + float(tp1_R) * rps
                if (tp1_fraction > 0 and tp1_R is not None) else None
            )
            hard_target = (
                close + float(hard_target_R) * rps
                if hard_target_R is not None else None
            )
            out.append(
                Candidate(
                    sleeve=self.spec.name,
                    symbol=sym,
                    kind="score",
                    side="long",
                    grade=self.spec.grade,
                    family=self.spec.resolved_family(),
                    score=sf,
                    target_weight=None,
                    entry_price=close,
                    stop=stop,
                    atr=atr,
                    tp1_price=tp1_price,
                    hard_target=hard_target,
                )
            )
        return out


def make_adapter(spec: SleeveSpec):
    """Build the right adapter for a :class:`SleeveSpec` (by ``kind``)."""
    if spec.kind == "score":
        return ScoreSleeveAdapter(spec)
    if spec.kind == "weight":
        return WeightSleeveAdapter(spec)
    raise ValueError(f"unknown sleeve kind {spec.kind!r} for {spec.name!r}")
