#!/usr/bin/env python3
"""run_offline.py -- run every eval suite with zero third-party deps.

    python3 evals/run_offline.py

No Braintrust, no network, no API key. Exits non-zero if any HARD GATE fails,
so it works as a pre-flight check before a session or in CI.

Remember what a green run here does and does not mean: in replay mode the
"model output" is each case's recorded reference reply, so a pass proves the
SCORERS behave -- not that a live model would. Set ANTHROPIC_API_KEY (and
install `anthropic`) to score real replies.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_decision_quality        # noqa: E402
import eval_engine_regression       # noqa: E402
import eval_safety                  # noqa: E402
from common import model_mode       # noqa: E402


def main() -> int:
    print("\nAsymmetric campaign eval harness -- OFFLINE mode")
    print("model source: {}\n".format(model_mode()))
    results = {
        "decision_quality": eval_decision_quality.main(force_offline=True),
        "safety": eval_safety.main(force_offline=True),
        "engine": eval_engine_regression.main(force_offline=True),
    }
    print("\n" + "=" * 72)
    for k, ok in results.items():
        print("  {:<20} {}".format(k, "PASS" if ok else "GATE FAILURE"))
    print("=" * 72)
    if model_mode() == "replay":
        print("REPLAY MODE: scorers exercised against recorded reference replies.")
        print("This validates the harness, NOT the model. Set ANTHROPIC_API_KEY")
        print("and install `anthropic` to evaluate live output.")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
