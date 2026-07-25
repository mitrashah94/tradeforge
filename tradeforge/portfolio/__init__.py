"""portfolio/ — the cross-strategy daily PORTFOLIO ENGINE (Phase 1 keystone).

TradeForge's per-sleeve backtesters each run, size, and cap one strategy in
isolation. This package is the layer ABOVE them: a single deterministic allocator
that takes the signals from ALL daily sleeves (momentum rotation, swing
mean-reversion, swing breakout), normalizes them to one shape, ranks them across
sleeves, dedups correlated/duplicate exposure, resolves same-symbol conflicts,
sizes by risk, and enforces every limit at the **book** level off one
``risk/limits.yaml`` row — running the account as ONE risk-budgeted book that
"holds only the best few positions".

It is cross-strategy AND live-bound (the live ``ORDER_INTENT`` seam lives here in
``intents.py``), which is why it is its own top-level package rather than another
module under ``backtest/daily/``. The deterministic decision core
(:class:`~portfolio.engine.PortfolioEngine`) is PURE — no DB, no LLM, no MCP, no
clock — so it runs identically in a backtest (driven by
``backtest/daily/portfolio_backtester.run_portfolio``) and in the live slow loop
(rendered to intents by ``portfolio.intents.allocation_to_intents``).

Iron rules (CLAUDE.md): ``risk/limits.yaml`` is read-only single-source-of-truth;
exits / risk-reducers always execute even when halted; nothing here writes live
config or promotes a sleeve.
"""

from __future__ import annotations
