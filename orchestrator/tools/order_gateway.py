"""Order gateway: paper/live router that sends every order down the identical event path; live orders require hook approval and never bypass the gate (MASTER_PLAN.md §4, CLAUDE.md conventions)."""

# TODO: implement paper/live routing; live path goes review_equity_order -> confirm -> place_equity_order, gated by the PreToolUse hook.
