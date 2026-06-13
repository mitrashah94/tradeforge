"""Notification tool: the NOTIFY sink for alerts, halts, and digests (MASTER_PLAN.md §4).

The journalist (and watchdog/breakers) push human-facing text through one
function — :func:`notify` — which routes to a channel:

- ``log`` / ``file`` (DEFAULT, safe sink): append the message to
  ``journal/notifications.log`` AND return it. Used whenever no recipient is
  configured. Sending nothing outward is the safe default.
- ``imessage`` (macOS, OPT-IN): deliver via AppleScript ``osascript`` to the
  Messages app. Recipient comes from ``TRADEFORGE_NOTIFY_TO``. This is an
  OUTWARD action and is *never* taken unless a recipient is explicitly
  configured — without one the call silently falls back to the file sink.

Channel selection precedence (highest first):
  1. the explicit ``channel=`` argument,
  2. the ``TRADEFORGE_NOTIFY_CHANNEL`` env var,
  3. the default ``file`` sink.

Even when ``imessage`` is selected, a *missing recipient* downgrades to the file
sink — so a misconfigured env can never silently swallow a notification, and an
unconfigured one can never send a real message.

Design notes:
- stdlib only at import time (``os``, ``subprocess``, ``pathlib``); nothing heavy.
- ``subprocess``/``osascript`` is imported lazily inside the iMessage path so the
  module imports cleanly on non-macOS hosts and so tests can monkeypatch it.
- The function ALWAYS returns the rendered text, so callers (and the digest path)
  can both deliver and capture in one call.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# Default file sink lives under journal/ (the journalist's write-only zone).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_PATH = _REPO_ROOT / "journal" / "notifications.log"

# Channel names.
CHANNEL_FILE = "file"
CHANNEL_LOG = "log"  # alias of file
CHANNEL_IMESSAGE = "imessage"

_FILE_CHANNELS = frozenset({CHANNEL_FILE, CHANNEL_LOG})


def _resolve_channel(channel: str | None) -> str:
    """Resolve the effective channel from arg -> env -> default file sink."""
    if channel:
        return channel.strip().lower()
    env = os.environ.get("TRADEFORGE_NOTIFY_CHANNEL")
    if env and env.strip():
        return env.strip().lower()
    return CHANNEL_FILE


def _render(message: str, title: str | None) -> str:
    """Render the human-facing line. A title is prefixed in brackets."""
    msg = "" if message is None else str(message)
    if title:
        return f"[{title}] {msg}"
    return msg


def _append_to_log(text: str, log_path: str | Path | None) -> Path:
    """Append a timestamped line to the file sink, creating it if needed."""
    path = Path(log_path) if log_path else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with path.open("a", encoding="utf-8") as fh:
        # One physical line per notification (newlines in the body are escaped)
        # so the log stays grep-friendly.
        fh.write(f"{ts}\t{text}".replace("\n", "\\n") + "\n")
    return path


def _send_imessage(text: str, recipient: str) -> bool:
    """Deliver ``text`` to ``recipient`` via Messages using osascript.

    Returns True on a clean exit. ``subprocess`` is imported lazily so this
    module never hard-requires it at import time (and tests monkeypatch it).
    Any failure (non-macOS, no Messages, osascript error) returns False so the
    caller can fall back to the file sink rather than raise.
    """
    import subprocess  # lazy — keeps import-time deps to stdlib basics

    # AppleScript: send to a buddy on the iMessage service. The text and
    # recipient are passed as argv (not interpolated into the script body) so a
    # message containing quotes can't break out of the script.
    script = (
        'on run {targetBuddy, targetMessage}\n'
        '    tell application "Messages"\n'
        '        set targetService to 1st account whose service type = iMessage\n'
        '        set theBuddy to participant targetBuddy of targetService\n'
        '        send targetMessage to theBuddy\n'
        '    end tell\n'
        'end run'
    )
    try:
        proc = subprocess.run(
            ["osascript", "-e", script, recipient, text],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 — never let a delivery error escape
        return False


def notify(
    message: str,
    *,
    title: str | None = None,
    channel: str | None = None,
    log_path: str | Path | None = None,
) -> str:
    """Send ``message`` over a channel and return the rendered text.

    Args:
        message: the body text (e.g. a digest, halt alert, fill notice).
        title: optional short prefix rendered as ``[title] message``.
        channel: ``"file"``/``"log"`` (default safe sink) or ``"imessage"``.
            When None, resolved from ``TRADEFORGE_NOTIFY_CHANNEL`` then the file
            default.
        log_path: override the file-sink path (tests point this at tmp).

    Behavior:
        * file/log: append to ``journal/notifications.log`` and return the text.
        * imessage: requires ``TRADEFORGE_NOTIFY_TO``. With NO recipient the call
          downgrades to the file sink (it never sends outward). With a recipient
          it attempts osascript delivery; on ANY failure it ALSO writes the file
          sink as a durable fallback. The text is returned in every case.

    Returns:
        The rendered notification text (always — delivery is best-effort).
    """
    text = _render(message, title)
    effective = _resolve_channel(channel)

    if effective == CHANNEL_IMESSAGE:
        recipient = os.environ.get("TRADEFORGE_NOTIFY_TO", "").strip()
        if not recipient:
            # OPT-IN guard: no recipient -> never send; fall back to file sink.
            _append_to_log(text, log_path)
            return text
        sent = _send_imessage(text, recipient)
        if not sent:
            # Delivery failed: keep a durable record in the file sink.
            _append_to_log(text, log_path)
        return text

    # Default + any unknown channel: the safe file/log sink.
    _append_to_log(text, log_path)
    return text
