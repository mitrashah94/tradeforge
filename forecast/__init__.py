"""forecast/ — ML forecasting overlays for the SLOW (premarket) loop only.

Nothing here runs in the hot path. A forecaster (currently Kronos) runs in the
premarket batch and WRITES a table the deterministic portfolio engine READS as a
plain lookup — the iron rule that no LLM/ML touches money-moving code holds. ML
is an optional dependency (torch et al.): the platform imports and runs fine
without it; the overlay is gated behind a ``use_kronos`` flag.
"""

from __future__ import annotations
