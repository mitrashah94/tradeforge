"""Broker-vs-ledger reconciliation: runs FIRST on boot for both paper AND live — adopt/cancel orphans, rebuild from the event log, halt on mismatch (MASTER_PLAN.md §4, §7)."""

# TODO: implement reconciliation + orphan recovery; must complete before any new orders are placed.
