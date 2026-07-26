#!/usr/bin/env python3
"""common.py -- Braintrust wiring + a zero-dependency offline fallback.

Two ways to run every eval in this directory:

  bt eval evals/eval_decision_quality.py     # real Braintrust run (uploads)
  python3 evals/run_offline.py               # stdlib only, no network, no key

The offline path exists so the harness can be developed, tested and trusted
BEFORE anyone signs up for anything -- and so a failing CI check never depends
on a third-party service being reachable.

MODEL SOURCE (decision + safety evals)
--------------------------------------
Those two suites score what an ASSISTANT REPLIES, so they need model output:
  - live    : ANTHROPIC_API_KEY set and `anthropic` installed -> real call.
  - replay  : otherwise, each case's `recorded_output` is scored instead.
Replay mode tests THE SCORERS, not the model. Every report labels which mode
produced it, because a green replay run says nothing about model quality.
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = "Asymmetric Options Campaign"
MODEL = os.environ.get("EVAL_MODEL", "claude-opus-4-8")

try:                       # pragma: no cover - presence depends on install
    from braintrust import Eval as _BraintrustEval
    HAVE_BRAINTRUST = True
except Exception:
    _BraintrustEval = None
    HAVE_BRAINTRUST = False


def load_cases(name: str) -> List[Dict]:
    """Load evals/datasets/<name>.json -> [{input, expected, metadata...}]."""
    path = os.path.join(HERE, "datasets", name + ".json")
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def model_mode() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return "live"
        except Exception:
            return "replay"
    return "replay"


def call_model(system: str, user: str) -> str:      # pragma: no cover - network
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL, max_tokens=1200, system=system,
        messages=[{"role": "user", "content": user}])
    return "".join(getattr(b, "text", "") for b in msg.content)


def make_task(system: str) -> Callable[[Any], Any]:
    """Task factory. In replay mode returns the case's recorded_output so the
    scorers can be exercised deterministically."""
    mode = model_mode()

    def task(input: Any) -> Any:
        if mode == "live":
            return call_model(system, json.dumps(input, indent=2))
        if not isinstance(input, dict):
            return ""
        rec = input.get("_recorded_output")
        if rec is None:
            return ""
        contract = input.get("_recorded_contract")
        return {"text": rec, "contract": contract} if contract else rec
    return task


# --------------------------------------------------------------------------
# offline Eval shim
# --------------------------------------------------------------------------
def _offline_eval(project: str, data, task, scores, experiment_name=None,
                  metadata=None, **kw) -> Dict:
    cases = data() if callable(data) else data
    rows, totals = [], {}
    for case in cases:
        inp, exp = case.get("input"), case.get("expected")
        out = task(inp)
        per = {}
        for fn in scores:
            name = getattr(fn, "__name__", str(fn))
            try:
                per[name] = float(fn(input=inp, output=out, expected=exp))
            except Exception as e:                    # a broken scorer must be loud
                per[name] = 0.0
                per[name + "__error"] = str(e)[:120]
            if isinstance(per.get(name), float):
                totals.setdefault(name, []).append(per[name])
        rows.append({"name": case.get("name", "case"), "scores": per})
    summary = {k: round(sum(v) / len(v), 4) for k, v in totals.items() if v}
    return {"project": project, "experiment": experiment_name,
            "metadata": metadata or {}, "rows": rows, "summary": summary,
            "n": len(rows)}


def run_eval(name: str, data, task, scores, metadata: Optional[Dict] = None,
             force_offline: bool = False) -> Any:
    """Braintrust when available (and not forced offline), else the shim."""
    meta = dict(metadata or {})
    meta.setdefault("model_mode", model_mode())
    meta.setdefault("model", MODEL)
    if HAVE_BRAINTRUST and not force_offline and os.environ.get("BRAINTRUST_API_KEY"):
        return _BraintrustEval(PROJECT, experiment_name=name, data=data,
                               task=task, scores=scores, metadata=meta)
    return _offline_eval(PROJECT, data, task, scores,
                         experiment_name=name, metadata=meta)


def print_report(res: Dict, gates: tuple = ()) -> bool:
    """Human-readable summary. Returns False if any hard gate failed."""
    if not isinstance(res, dict) or "rows" not in res:
        print("(braintrust run - see the Braintrust UI for results)")
        return True
    print("=" * 72)
    print("{}   n={}   mode={}".format(
        res.get("experiment"), res["n"], res["metadata"].get("model_mode")))
    print("=" * 72)
    failures = []
    for r in res["rows"]:
        bad = {k: v for k, v in r["scores"].items()
               if isinstance(v, float) and v < 1.0}
        flag = "FAIL" if bad else "ok  "
        print("  [{}] {}".format(flag, r["name"]))
        for k, v in sorted(bad.items()):
            print("        {:<28} {:.2f}".format(k, v))
            if k in gates:
                failures.append((r["name"], k))
    print("-" * 72)
    for k, v in sorted(res["summary"].items()):
        mark = "  <-- GATE" if (k in gates and v < 1.0) else ""
        print("  {:<30} {:.3f}{}".format(k, v, mark))
    if failures:
        print("-" * 72)
        print("HARD GATE FAILURES ({}):".format(len(failures)))
        for nm, k in failures:
            print("   {} :: {}".format(nm, k))
    print("=" * 72)
    return not failures
