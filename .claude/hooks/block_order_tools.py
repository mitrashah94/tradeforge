#!/usr/bin/env python3
"""DayTrading PreToolUse hook: unconditionally deny brokerage order tools.

CLAUDE.md hard rule 1: Claude never places, modifies, cancels, or stages any
order. Unlike tradeforge's gated hook (tradeforge/.claude/hooks/), there is no
evaluator to delegate to — every order write tool is denied, always.
Stdlib-only. Fails closed: any error still emits a deny decision.
"""
import json
import re
import sys

ORDER_TOOL_RE = re.compile(
    r"place_(equity|option|crypto)s?_order"
    r"|cancel_(equity|option)s?_order"
    r"|cancel_option_exercise"
    r"|exercise_option"
    r"|review_(equity|option)s?_order"
    r"|order_gateway_live"
)


def main() -> int:
    tool_name = ""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        tool_name = payload.get("tool_name", "") or ""
    except Exception:
        pass
    # The matcher already selected this call; deny regardless of parse outcome.
    reason = (
        f"BLOCKED: '{tool_name or 'order tool'}' is an order write tool. "
        "DayTrading hard rule 1: Claude never places, modifies, cancels, or "
        "stages orders. Read-only tools (quotes, chains, positions, "
        "portfolio) are fine. The trader clicks submit, not Claude."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
