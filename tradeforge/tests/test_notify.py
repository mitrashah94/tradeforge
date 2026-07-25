"""tests/test_notify.py — the NOTIFY tool: default-safe file sink + opt-in iMessage.

Deterministic, offline. The default channel is the file/log sink — a real
iMessage is NEVER sent without an explicitly configured recipient. The iMessage
path is guarded so it is only ATTEMPTED when both TRADEFORGE_NOTIFY_TO and the
imessage channel are set; even then we MOCK osascript and never shell out.
"""

from __future__ import annotations

import orchestrator.tools.notify as notify_mod
from orchestrator.tools.notify import notify


# --------------------------------------------------------------------------- #
# default sink = file/log: append + return, no send                           #
# --------------------------------------------------------------------------- #
def test_default_channel_is_file_sink(tmp_path, monkeypatch):
    # No channel / recipient configured anywhere.
    monkeypatch.delenv("TRADEFORGE_NOTIFY_CHANNEL", raising=False)
    monkeypatch.delenv("TRADEFORGE_NOTIFY_TO", raising=False)
    log = tmp_path / "notifications.log"

    out = notify("hello world", log_path=log)

    # The message is RETURNED ...
    assert out == "hello world"
    # ... AND appended to the file sink.
    assert log.exists()
    assert "hello world" in log.read_text(encoding="utf-8")


def test_title_is_prefixed_and_logged(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADEFORGE_NOTIFY_CHANNEL", raising=False)
    log = tmp_path / "n.log"
    out = notify("body", title="EOD 2026-06-13", log_path=log)
    assert out == "[EOD 2026-06-13] body"
    assert "[EOD 2026-06-13] body" in log.read_text(encoding="utf-8")


def test_appends_multiple_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADEFORGE_NOTIFY_CHANNEL", raising=False)
    log = tmp_path / "n.log"
    notify("first", log_path=log)
    notify("second", log_path=log)
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "first" in lines[0] and "second" in lines[1]


# --------------------------------------------------------------------------- #
# imessage requested but NO recipient -> downgrade to file sink, never send    #
# --------------------------------------------------------------------------- #
def test_imessage_without_recipient_does_not_send(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADEFORGE_NOTIFY_TO", raising=False)
    log = tmp_path / "n.log"

    # Guard: if osascript were attempted, fail loudly.
    def _boom(*a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("osascript must NOT be called without a recipient")

    monkeypatch.setattr(notify_mod, "_send_imessage", _boom)

    out = notify("halt!", channel="imessage", log_path=log)

    assert out == "halt!"
    # Fell back to the safe file sink.
    assert "halt!" in log.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# imessage WITH recipient + channel -> attempts osascript (mocked, no shell)   #
# --------------------------------------------------------------------------- #
def test_imessage_with_recipient_attempts_send_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFORGE_NOTIFY_TO", "+15555550123")

    calls = {}

    def _fake_send(text, recipient):  # noqa: ANN001
        calls["text"] = text
        calls["recipient"] = recipient
        return True  # pretend Messages delivered it

    monkeypatch.setattr(notify_mod, "_send_imessage", _fake_send)

    out = notify("digest body", channel="imessage", log_path=tmp_path / "n.log")

    assert out == "digest body"
    assert calls == {"text": "digest body", "recipient": "+15555550123"}
    # On a successful send we do NOT also write the file sink.
    assert not (tmp_path / "n.log").exists()


def test_imessage_send_failure_falls_back_to_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADEFORGE_NOTIFY_TO", "+15555550123")
    log = tmp_path / "n.log"

    monkeypatch.setattr(notify_mod, "_send_imessage", lambda t, r: False)  # noqa: ANN001

    out = notify("important", channel="imessage", log_path=log)

    assert out == "important"
    # Delivery failed -> durable fallback in the file sink.
    assert "important" in log.read_text(encoding="utf-8")


def test_osascript_never_actually_shells_out(monkeypatch, tmp_path):
    """The real _send_imessage must invoke subprocess.run (which we intercept),
    proving the code path is wired but is never allowed to actually shell out in
    the test."""
    monkeypatch.setenv("TRADEFORGE_NOTIFY_TO", "+15555550123")

    import subprocess

    seen = {}

    class _Proc:
        returncode = 0

    def _fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        seen["args"] = args
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    out = notify("ping", channel="imessage", log_path=tmp_path / "n.log")
    assert out == "ping"
    # osascript was the invoked binary, with the recipient + text in argv.
    assert seen["args"][0] == "osascript"
    assert "+15555550123" in seen["args"]
    assert "ping" in seen["args"]
