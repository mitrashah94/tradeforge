#!/usr/bin/env python3
"""Corpus test for the order-blocking hook. Run after ANY change to
block_order_tools.py or the PreToolUse matcher in .claude/settings.json:

    python3 .claude/hooks/test_block_order_tools.py

Never verify the hook by calling a real order tool — if the hook is broken,
the test itself becomes a live order. Stdlib-only, plain asserts.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RH = "mcp__f2efbdf8-ba97-404e-944a-e5a143eb9758__"

BLOCKED = [
    # real Robinhood order write tools (session of 2026-07-25)
    RH + "place_equity_order", RH + "place_option_order",
    RH + "cancel_equity_order", RH + "cancel_option_order",
    RH + "cancel_option_exercise", RH + "exercise_option",
    RH + "review_equity_order", RH + "review_option_order",
    # modification verbs (rule 1: never place, MODIFY, or cancel)
    "mcp__rh__modify_equity_order", "mcp__alpaca__replace_order",
    "mcp__rh__update_option_order", "mcp__rh__amend_orders",
    "mcp__rh__edit_order", "mcp__rh__change_order",
    "mcp__rh__submit_order", "mcp__rh__execute_orders",
    # suffixed forms (underscore defeats a bare \b boundary)
    "mcp__rh__cancel_order_by_id", "mcp__rh__place_order_v2",
    "mcp__rh__order_cancel_all", "mcp__rh__submit_order_async",
    "mcp__rh__replace_order_by_id", "mcp__rh__cancel_all_orders",
    "mcp__rh__place_order_with_stop", "mcp__rh__orders_update_batch",
    # noun_verb form, gateway, crypto, plural
    "mcp__rh__order_cancel", "mcp__rh__orders_update",
    "order_gateway_live", "mcp__rh__place_crypto_order",
    "mcp__rh__cancel_crypto_order", "mcp__rh__place_orders",
]

ALLOWED = [
    # read tools that must keep working
    RH + "get_equity_orders", RH + "get_option_orders",
    RH + "get_option_quotes", RH + "get_portfolio",
    RH + "get_pnl_trade_history", RH + "get_realized_pnl",
    # write tools that are NOT orders
    RH + "update_scan_config", RH + "update_scan_filters",
    RH + "update_watchlist", RH + "create_scan", RH + "create_watchlist",
    RH + "add_to_watchlist", RH + "remove_from_watchlist",
    "mcp__68fe53a2-d883-4143-9907-65ac995c944a__cancel_trial_auto_renewal",
    # tricky near-misses
    "mcp__x__reorder_watchlist", "mcp__x__update_orderbook_depth_view",
    "mcp__x__get_order_status", "mcp__x__get_ordering_rules",
    "Bash", "Edit", "Read",
]


def main() -> int:
    matcher = json.loads((ROOT / ".claude" / "settings.json").read_text())[
        "hooks"]["PreToolUse"][0]["matcher"]

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "block_order_tools", Path(__file__).with_name("block_order_tools.py"))
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    checks = 0
    for t in BLOCKED:
        # matcher must hit under BOTH anchored and search semantics
        assert re.fullmatch(matcher, t), f"matcher fullmatch MISS: {t}"
        assert re.search(matcher, t), f"matcher search MISS: {t}"
        assert hook.ORDER_TOOL_RE.search(t), f"script regex MISS: {t}"
        checks += 3
    for t in ALLOWED:
        assert not re.fullmatch(matcher, t), f"matcher FALSE POSITIVE: {t}"
        assert not re.search(matcher, t), f"matcher FALSE POSITIVE: {t}"
        assert not hook.ORDER_TOOL_RE.search(t), f"script FALSE POSITIVE: {t}"
        checks += 3

    # matcher and script must agree on every name in the corpus
    for t in BLOCKED + ALLOWED:
        assert bool(re.search(matcher, t)) == bool(hook.ORDER_TOOL_RE.search(t)), \
            f"matcher/script DISAGREE on: {t}"
        checks += 1

    print(f"{checks} checks passed "
          f"({len(BLOCKED)} blocked, {len(ALLOWED)} allowed, corpora agree).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
