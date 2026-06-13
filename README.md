# TradeForge

Autonomous, self-improving systematic trading platform. **Maximize compounded growth inside a fixed risk band while deterministic code makes catastrophic loss structurally impossible.**

- Design & rationale: [MASTER_PLAN.md](MASTER_PLAN.md)
- Operating contract (mission, resolved decisions, conventions): [CLAUDE.md](CLAUDE.md)

## Quickstart
```bash
bash scripts/bootstrap.sh   # creates .venv, installs deps (pip3), seeds .env, runs tests
```

## Layout
LLM agents decide policy; deterministic Python runs the fast loop and protects capital. `risk/limits.yaml` is the single source of truth for the risk dial, milestone ratchet, and program-abort. A PreToolUse hook (`.claude/`) blocks live orders unless fully gated; paper is the default. See `MASTER_PLAN.md` §4 for the full architecture.
