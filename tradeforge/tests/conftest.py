"""Test-suite conftest.

RI_OVERRIDE: simulate an operator risk-dial edit without touching
risk/limits.yaml (which is hand-edited only — no agent may write it).
The suite is required to pass for every risk_index.default in the band
[5, 8] as well as the current setting; sweep it like this:

    for RI in 5 6 7 8; do
        RI_OVERRIDE=$RI .venv/bin/python -m pytest tests -q
    done

The wrapper patches risk.config.load_limits before test modules import it,
so tests and production modules alike see the overridden floor. Reference-
only rows (1-3, 9-10) are out of contract: tests may fail loudly on their
guard assertions there by design.
"""
import os


def pytest_configure(config):
    ri = os.environ.get("RI_OVERRIDE")
    if not ri:
        return
    ri = int(ri)
    import risk.config as rc
    original = rc.load_limits

    def load_limits_with_override(path="risk/limits.yaml"):
        return original(path).model_copy(update={"default_ri": ri})

    rc.load_limits = load_limits_with_override
