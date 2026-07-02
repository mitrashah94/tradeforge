"""prop/ — prop-firm EVALUATION + FUNDED-ACCOUNT modeling (the $100k-payout path).

The EV-dominant route to a $100k *payout* on a $1k personal budget is NOT to
100x the $1k (a ~0-EV lottery ticket), but to spend a small eval fee, pass a
funded-account evaluation with a real edge, and harvest payouts on the firm's
much larger capital — where a MODEST, achievable return is worth real dollars,
and the downside is capped at the fee.

This package makes that path quantitative and honest. It encodes a prop firm's
RULES (profit target, trailing/static max drawdown, daily-loss limit, min days,
consistency, profit split), runs a strategy's daily-return stream through the
deterministic account STATE MACHINE, and reports — historically and by
block-bootstrap Monte-Carlo — P(pass), P(bust), time-to-pass, the funded-phase
payout distribution, and the end-to-end EXPECTED VALUE net of the eval fee (so you
can compare the prop path against the moonshot on equal footing).

It reuses the platform's ethos: the firm's trailing-DD and daily-loss rules are
just breakers (the same shape as ``risk/limits.yaml``'s program-abort + daily
halt), and the strategy under test is whatever cleared the validation gate — e.g.
the blended daily book (PF 1.76). PURE / DETERMINISTIC / OFFLINE: no live orders,
no LLM/MCP, seeded RNG for the Monte-Carlo.
"""

from __future__ import annotations
