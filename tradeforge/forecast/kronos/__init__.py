"""forecast/kronos/ — the Kronos OHLCV forecasting overlay (post-cutoff only).

Kronos (MIT, AAAI 2026) is a decoder-only foundation model over OHLCV bars. Here
it (1) filters weak strategy signals, (2) ranks breakout/mean-rev candidates, and
(3) forecasts return / volatility / downside — all in the SLOW loop, writing a
``kronos_forecasts`` table the deterministic engine reads.

THE HONESTY CONSTRAINT (non-negotiable): Kronos was pretrained on history through
an undisclosed cutoff (~2025), so a forecast on PRE-cutoff bars is contaminated by
look-ahead and would silently break the OOS discipline the whole platform rests
on. :mod:`forecast.kronos.leakage` is a HARD GUARD that refuses any forecast whose
inputs predate a conservative assumed cutoff. Kronos therefore accrues a FORWARD
paper track record on the short post-cutoff window; it never "validates" a
strategy through the normal gate on contaminated history.

torch and the model weights are OPTIONAL — the leakage guard, the store, and the
engine's consumption of a seeded table all work without torch; only the live
predictor needs it.
"""

from __future__ import annotations
