#!/usr/bin/env python3
"""TradeForge PreToolUse hook: block live-order tool calls unless fully gated.
Fails CLOSED (exit 2) for live-order endpoints on any error. Stdlib-only."""
import os, sys, json, re


def _is_live_order_tool(name: str) -> bool:
    return bool(re.search(r"place_(equity|crypto|option)s?_order", name or "")) or "order_gateway_live" in (name or "")


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}
    tool_name = payload.get("tool_name", "")
    # Try to delegate to the rich evaluator; fail closed for live-order tools.
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from orchestrator.hooks import evaluate
        decision = evaluate(payload)
        if decision.allow:
            return 0
        sys.stderr.write(f"[TradeForge hook] BLOCKED live order -> {decision.reason}. Use the paper path.\n")
        return 2
    except Exception as e:
        if _is_live_order_tool(tool_name):
            sys.stderr.write(f"[TradeForge hook] FAIL-CLOSED blocking live order ({e}).\n")
            return 2
        return 0


if __name__ == "__main__":
    sys.exit(main())
