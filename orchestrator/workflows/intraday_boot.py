"""Intraday-boot workflow: reconcile with the broker FIRST, recover orphans, rebuild from the event log, then resume the fast loop (MASTER_PLAN.md §4, §7)."""

# TODO: implement boot sequence; reconciliation (orderbook/reconcile.py) must run before any new orders.
