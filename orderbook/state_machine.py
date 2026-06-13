"""Order state machine: STAGED -> APPROVED -> SUBMITTED -> WORKING -> {FILLED | PARTIAL | ...}, the single source of truth for order lifecycle (MASTER_PLAN.md §4)."""

# TODO: implement the deterministic state transitions and persistence to the order ledger.
