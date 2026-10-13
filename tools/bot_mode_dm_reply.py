"""Exactly-once reply delivery for ``message_agent`` runners that outlive their sender.

A ``message_agent`` runner is a background process tracked in the SENDER's in-memory
ProcessRegistry, and its stdout is the completion notification that carries the teammate's
reply. A one-shot sender (``hermes chat -Q``: the cron Bot Chat lane) lingers at most
``terminal.oneshot_completion_wait_seconds`` and then exits, so a reply that lands later had
no consumer: no notification, no result file, no transcript row (charlie-hermes#43). The
runner survives its sender (``start_new_session``); only the pipe that carried its reply dies.

One private record per runner, advanced under a per-record lock by exactly two parties:

- the runner on exit (:func:`run_and_settle`): ``armed`` -> ``finished`` with its outcome, or,
  when the sender already handed off, ``handoff`` -> it appends the outcome itself;
- the sender (:func:`settle`, from interpreter exit or the notify-less poll thread): delivered
  in-process (a consumer drained the completion, or the model read it) -> discard; runner
  gone but undelivered -> it appends the outcome; runner still running at exit -> ``handoff``.

The sender hands off only from interpreter exit, after its last consumer stopped, so the
notification and the transcript row never both happen. The row is the DELIVERY row a
notify-less sender already persisted (``display_kind="process_complete"``), appended between
turns: the runner waits while any live surface holds the sender's Bot Chat, and the next
turn loads it as ordinary history (no past context rewritten, nothing injected mid-loop).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

RECORD_SUFFIX = ".reply.json"
# A live surface (the next one-shot, a Desktop tab) holding the sender's Bot Chat defers the
# runner's append until the turn boundary; past this budget the record is left as evidence.
_OWNER_WAIT_SECONDS = 60 * 60
_OWNER_POLL_SECONDS = 1.0

_armed: dict[str, Any] = {}  # record path -> ProcessSession, for runners this process spawned
_armed_lock = threading.Lock()
_atexit_registered = False


def new_record_path(directory: Path) -> str:
    return str(Path(directory) / f"{uuid.uuid4().hex}{RECORD_SUFFIX}")


@contextlib.contextmanager
def _locked(path: str):
    from hermes_cli.active_sessions import _FileLock

    with _FileLock(Path(path + ".lock")):
        yield


def _read(path: str) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write(path: str, record: dict) -> None:
    from utils import atomic_json_write

    atomic_json_write(path, record, mode=0o600)


def _discard(path: str) -> None:
    for name in (path, path + ".lock"):
        with contextlib.suppress(OSError):
            os.unlink(name)


def arm(path: str, proc: Any, *, db_path: str, session_id: str) -> bool:
    """Sender side, right after spawn: open the record so the reply survives this process.
    Skipped for gateway-watched processes: a watcher delivers those and checkpoint recovery
    re-arms it after a restart, so a runner-side row would duplicate it."""
    if proc is None or proc.watcher_interval > 0 or not db_path or not session_id:
        return False
    global _atexit_registered
    _reap_delivered()
    with _locked(path):
        _write(path, {"state": "armed", "proc_id": proc.id, "command": proc.command,
                      "db_path": str(db_path), "session_id": session_id})
    with _armed_lock:
        _armed[path] = proc
        if not _atexit_registered:
            atexit.register(_settle_all_at_exit)
            _atexit_registered = True
    return True


def settle_when_done(path: str, proc: Any) -> None:
    """Notify-less sender (api_server, poll-only one-shot): settle as soon as the runner exits."""

    def _run() -> None:
        proc._completion_event.wait()
        try:
            settle(path, proc, exiting=False)
        except Exception as exc:
            logger.warning("message_agent: could not settle the reply of %s: %s", proc.id, exc)

    threading.Thread(target=_run, name=f"message-agent-reply-{proc.id}", daemon=True).start()


def settle(path: str, proc: Any, *, exiting: bool) -> None:
    """Sender side. ``exiting``: no consumer of this process will run again."""
    with _locked(path):
        record = _read(path)
        if record is None or record["state"] not in ("armed", "finished"):
            return
        if _delivered_in_process(proc):
            _discard(path)
            return
        if record["state"] == "armed":
            if not proc._completion_event.is_set():
                if exiting:
                    record["state"] = "handoff"
                    _write(path, record)
                return
            # The runner exited without recording (killed, or done before arm): the registry has it.
            record.update(exit_code=proc.exit_code, completion_reason=proc.completion_reason,
                          output=proc.output_buffer)
        _append_row(record)
        _discard(path)


def run_and_settle(path: str, argv: list[str]) -> int:
    """Runner side: run the delivery ``argv`` capturing its combined output (the completion
    text), settle the record, then re-emit the output. Settle first: a sender that already
    exited has closed the pipe this output would ride."""
    proc = subprocess.run(argv, check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    output = proc.stdout or ""
    try:
        _finish(path, proc.returncode, output)
    except Exception as exc:
        logger.warning("message_agent runner could not settle %s: %s", path, exc)
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except OSError:
        # Sender gone: park stdout on devnull so the interpreter's exit flush cannot fail too.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return proc.returncode


def _finish(path: str, exit_code: int, output: str) -> None:
    with _locked(path):
        record = _read(path)
        if record is None:
            return
        if record["state"] == "armed":
            record.update(state="finished", exit_code=exit_code, output=output)
            _write(path, record)
            return
        if record["state"] != "handoff":
            return
        record.update(exit_code=exit_code, output=output)
        if not _wait_for_turn_boundary(Path(record["db_path"]).parent):
            logger.warning("message_agent reply %s kept: the sender's Bot Chat stayed busy", path)
            return
        _append_row(record)
        _discard(path)


def _wait_for_turn_boundary(home: Path) -> bool:
    from tools.bot_live_delivery import find_canonical_owner

    deadline = time.monotonic() + _OWNER_WAIT_SECONDS
    while find_canonical_owner(home) is not None:
        if time.monotonic() >= deadline:
            return False
        time.sleep(_OWNER_POLL_SECONDS)
    return True


def _delivered_in_process(proc: Any) -> bool:
    """Whether this sender already took the completion: read via wait/log/poll, or (notify
    path) drained off the completion queue by a consumer."""
    from tools.process_registry import process_registry

    if process_registry._drain_should_skip(proc.id):
        return True
    if not proc.notify_on_complete or not proc._completion_event.is_set():
        return False
    queue = process_registry.completion_queue
    with queue.mutex:
        return not any(evt.get("type", "completion") == "completion" and evt.get("session_id") == proc.id
                       for evt in queue.queue)


def _reap_delivered() -> None:
    """Drop records of runners already delivered in-process, so a long-lived sender does not
    carry them until exit."""
    with _armed_lock:
        items = list(_armed.items())
    for path, proc in items:
        if _delivered_in_process(proc):
            with _locked(path):
                _discard(path)
            with _armed_lock:
                _armed.pop(path, None)


def _settle_all_at_exit() -> None:
    with _armed_lock:
        items = list(_armed.items())
        _armed.clear()
    for path, proc in items:
        try:
            settle(path, proc, exiting=True)
        except Exception as exc:
            logger.warning("message_agent: could not hand off the reply of %s: %s", proc.id, exc)


def _append_row(record: dict) -> None:
    """The DELIVERY row a push surface persists for the same completion, on the sender's
    current compression tip."""
    from hermes_state_registry import acquire, release_or_close
    from tools.bot_mode_dm import REPLY_COMPLETION_CHARS
    from tools.process_registry import _redact_process_result
    from tools.process_registry_notifications import (
        PROCESS_COMPLETE_DISPLAY_KIND, format_process_notification, process_completion_display_text,
    )

    output = record.get("output") or ""
    evt = {"type": "completion", "session_id": record["proc_id"], "command": record["command"],
           "exit_code": record.get("exit_code"), "completion_reason": record.get("completion_reason", "exited"),
           "output": output[-REPLY_COMPLETION_CHARS:]}
    if len(output) > REPLY_COMPLETION_CHARS:
        evt["output_cut"] = len(output) - REPLY_COMPLETION_CHARS
    _redact_process_result(evt)
    db = acquire(Path(record["db_path"]))
    try:
        session_id = db.resolve_resume_session_id(record["session_id"]) or record["session_id"]
        db.append_message(session_id, "user", content=format_process_notification(evt),
                          display_kind=PROCESS_COMPLETE_DISPLAY_KIND,
                          display_metadata={"display_text": process_completion_display_text([evt])})
    finally:
        release_or_close(db)
