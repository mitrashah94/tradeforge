#!/usr/bin/env python3
"""DayTrading PreToolUse hook: unconditionally deny brokerage order tools.

CLAUDE.md hard rule 1: Claude never places, modifies, cancels, or stages any
order. Unlike tradeforge's gated hook (tradeforge/.claude/hooks/), there is no
evaluator to delegate to — every order write tool is denied, always.

Patterns are generic verbs, not an enumerated tool list, so new brokerage
tools (e.g. a future cancel_crypto_order) are covered without editing this
file. Keep ORDER_TOOL_RE in sync with the matcher in .claude/settings.json.

Fail-closed contract: this script always emits a deny decision. The settings
command additionally appends a shell fallback that emits a deny if this
script itself fails to execute — exit 1/127 would otherwise be treated by
the hook runner as a non-blocking error and let the tool call proceed.
Stdlib-only.
"""
import json
import re
import sys

# Any write/modify verb attached to "order(s)", in either verb_noun or
# noun_verb form. Read verbs (get/list/fetch/...) are deliberately absent.
# "replace" is Alpaca's order-amendment verb; "modify/update/amend/edit"
# cover CLAUDE.md rule 1's "never place, MODIFY, or cancel".
_ORDER_WRITE_VERBS = (
    "place|replace|cancel|review|modify|update|amend|edit|change"
    "|submit|stage|create|send|execute|open|close|adjust"
)
# (_|\b) instead of \b alone: underscore is a word char, so \b never fires
# before it — plain \b would miss suffixed names like cancel_order_by_id,
# place_order_v2, or order_cancel_all. "orderbook" still doesn't match.
ORDER_TOOL_RE = re.compile(
    rf"(^|_)({_ORDER_WRITE_VERBS})_\w*orders?(_|\b)"
    rf"|(^|_)orders?_({_ORDER_WRITE_VERBS})(_|\b)"
    r"|exercise_options?"
    r"|options?_exercise"
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
