"""Exactly-once reply delivery for ``message_agent`` runners that outlive their sender.

A ``message_agent`` runner is a background process tracked in the SENDER's in-memory
ProcessRegistry, and its stdout is the completion notification that carries the teammate's
reply. A one-shot sender (``hermes chat -Q``: the cron Bot Chat lane) lingers at most
``terminal.oneshot_completion_wait_seconds`` and then exits, so a reply that lands later had
no consumer: no notification, no result file, no transcript row (charlie-hermes#43). The
runner survives its sender (``start_new_session``); only the pipe that carried its reply dies.

One private record per runner, advanced under a per-record lock:

- the runner on exit (:func:`run_and_settle`): ``armed`` -> ``finished`` with its outcome, or,
  when the sender already handed off, ``handoff`` -> ``pending`` with its outcome;
- the sender (:func:`settle`, from interpreter exit or the notify-less poll thread): delivered
  in-process (a consumer drained the completion, or the model read it) -> discard; runner
  gone but undelivered -> ``pending``; runner still running at exit -> ``handoff``.

The sender hands off only from interpreter exit, after its last consumer stopped, so the
notification and the transcript row never both happen. A ``pending`` record holds the outcome
until it lands as the DELIVERY row a notify-less sender already persisted
(``display_kind="process_complete"``), and only at a turn boundary: either a party holding the
outcome takes the conversation's turn lease itself while no live surface holds the session
(:func:`deliver`), or the conversation's next turn appends it under its own lease at admission
(:func:`deliver_pending`) and starts from the reloaded transcript. A refused delivery keeps the
record; no surface's lifetime discards it.

The DM directory is per OS user and shared by every profile, so a record names its owner's
``HERMES_HOME``. :func:`deliver` binds that profile's scope (redaction policy, secrets) whatever
process runs it; admission is already bound to its own conversation's profile. Vault values are
the exception: they live only in the sender's memory, so when the sender held any, a delivery
from another interpreter (the runner, a later turn's admission) appends a generic notice
instead of the reply, logs the record path, and the record stays on disk as ``withheld``. The DM sweeper never
delivers: it only removes records whose parties are gone.
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
# How long a party holding an outcome keeps retrying a refused delivery (a running turn, or a live
# surface holding the session). Past it the record stays ``pending`` for the next turn's admission.
_DELIVERY_RETRY_SECONDS = 60 * 60
_DELIVERY_POLL_SECONDS = 1.0
_DELIVERY_LEASE_TTL_SECONDS = 60.0
# A ``pending`` record nobody could deliver for this long (state.db gone, conversation never resumed)
# is dropped by the DM sweeper. ``handoff`` gets the same horizon: its runner may still be running.
PENDING_MAX_AGE_SECONDS = 7 * 24 * 3600
_LONG_LIVED_STATES = ("handoff", "pending", "withheld")
# Identifies this interpreter in the records it arms. Vault values live only in the memory of the
# process that registered them, so only the sender's own interpreter can scrub them from a reply.
_INTERPRETER = uuid.uuid4().hex

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


@contextlib.contextmanager
def _try_locked(path: str):
    """Non-blocking :func:`_locked`: yields False, holding nothing, when another party holds the
    record. For callers that must never wait on a record (turn admission, the DM sweeper)."""
    with open(path + ".lock", "a+b") as fh:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True  # closing the file releases the lock


@contextlib.contextmanager
def _owner_scope(hermes_home: str):
    """Bind the record owner's profile (home + secret + terminal scope) for a delivery that runs
    outside that profile's turn: a runner, a poll thread, interpreter exit. Same binding as the
    kanban dispatcher's ``_worker_profile_scope``."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    from tui_gateway.launch_profile_policy import launch_secret_scope, launch_terminal_env

    home = Path(hermes_home)
    is_launch_home = home.resolve() == Path(get_process_hermes_home()).resolve()
    # Each reset is registered as soon as its setter returns, so a failure part-way through entry
    # restores what was already bound instead of leaking it into the caller.
    with contextlib.ExitStack() as stack:
        stack.callback(reset_hermes_home_override, set_hermes_home_override(str(home)))
        stack.callback(reset_secret_scope, set_secret_scope(
            launch_secret_scope(home) if is_launch_home else build_profile_secret_scope(home)))
        stack.callback(reset_terminal_scope, install_profile_terminal_scope(
            home, env_overlay=launch_terminal_env() if is_launch_home else None))
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


def arm(path: str, proc: Any, *, db_path: str, session_id: str, label: str = "") -> bool:
    """Sender side, right after spawn: open the record so the reply survives this process.
    Skipped for gateway-watched processes: a watcher delivers those and checkpoint recovery
    re-arms it after a restart, so a runner-side row would duplicate it."""
    if proc is None or proc.watcher_interval > 0 or not db_path or not session_id:
        return False
    from hermes_constants import get_hermes_home

    global _atexit_registered
    _reap_delivered()
    record = {"state": "armed", "proc_id": proc.id, "command": proc.command, "label": label,
              "db_path": str(db_path), "session_id": session_id, "hermes_home": str(get_hermes_home()),
              "armed_by": _INTERPRETER}
    with _locked(path):
        _write(path, _note_vault(record))
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
            if settle(path, proc, exiting=False):
                _deliver_until(path, _DELIVERY_RETRY_SECONDS)
        except Exception as exc:
            logger.warning("message_agent: could not settle the reply of %s: %s", proc.id, exc)

    threading.Thread(target=_run, name=f"message-agent-reply-{proc.id}", daemon=True).start()


def settle(path: str, proc: Any, *, exiting: bool) -> bool:
    """Sender side. ``exiting``: no consumer of this process will run again. Returns whether the
    record is now ``pending`` (the caller owns a delivery attempt)."""
    with _locked(path):
        record = _read(path)
        if record is None or record["state"] not in ("armed", "finished"):
            return False
        if _delivered_in_process(proc):
            _discard(path)
            _forget(path)
            return False
        _note_vault(record)
        if record["state"] == "armed":
            if not proc._completion_event.is_set():
                if exiting:
                    record["state"] = "handoff"
                    _write(path, record)
                return False
            # The runner exited without recording (killed, or done before arm): the registry has it.
            record.update(exit_code=proc.exit_code, completion_reason=proc.completion_reason,
                          output=proc.output_buffer)
        record["state"] = "pending"
        _write(path, record)
    _forget(path)
    return True


def run_and_settle(path: str, argv: list[str]) -> int:
    """Runner side: run the delivery ``argv`` capturing its combined output (the completion
    text), settle the record, then re-emit the output. Settle first: a sender that already
    exited has closed the pipe this output would ride."""
    proc = subprocess.run(argv, check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    output = proc.stdout or ""
    try:
        if _finish(path, proc.returncode, output):
            _deliver_until(path, _DELIVERY_RETRY_SECONDS)
    except Exception as exc:
        logger.warning("message_agent runner could not settle %s: %s", path, exc)
    try:
        sys.stdout.write(output)
        sys.stdout.flush()
    except OSError:
        # Sender gone: park stdout on devnull so the interpreter's exit flush cannot fail too.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return proc.returncode


def _finish(path: str, exit_code: int, output: str) -> bool:
    """Record the outcome before anything can wait; returns whether it is now ``pending``."""
    with _locked(path):
        record = _read(path)
        if record is None or record["state"] not in ("armed", "handoff"):
            return False
        pending = record["state"] == "handoff"
        record.update(state="pending" if pending else "finished", exit_code=exit_code, output=output)
        _write(path, record)
        return pending


def _deliver_until(path: str, budget: float) -> None:
    deadline = time.monotonic() + budget
    while not deliver(path):
        if time.monotonic() >= deadline:
            logger.info("message_agent reply %s stays pending for the conversation's next turn", path)
            return
        time.sleep(_DELIVERY_POLL_SECONDS)


def deliver(path: str) -> bool:
    """Append a ``pending`` outcome at a turn boundary taken here: this party holds the
    conversation's turn lease (no turn is running, and a turn starting now waits and then reloads
    the transcript) and no live surface holds the session (its next turn would start from the
    history it already has; :func:`deliver_pending` hands the row to that turn instead). Runs in
    the record owner's profile scope, never the caller's. Returns whether the record is settled;
    False keeps it for a retry."""
    with _locked(path):
        record = _read(path)
        if record is None or record["state"] != "pending":
            return True
        if not record.get("hermes_home") or not Path(record["db_path"]).is_file():
            return False
        with _owner_scope(record["hermes_home"]):
            delivered = _deliver_locked(path, record)
        if delivered:
            _delivered(path, record)
        return delivered


def _deliver_locked(path: str, record: dict) -> bool:
    from hermes_state_registry import acquire, release_or_close

    db = acquire(Path(record["db_path"]))
    try:
        session_id = db.resolve_resume_session_id(record["session_id"]) or record["session_id"]
        holder = f"pid={os.getpid()}:message-agent-reply:{uuid.uuid4().hex[:8]}"
        if not db.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=_DELIVERY_LEASE_TTL_SECONDS,
                                                 patience_s=0.5):
            return False
        try:
            if _surface_holds(Path(record["db_path"]).parent, db, session_id):
                return False
            _append_row(db, session_id, path, record, turn_lease_holder=holder)
        finally:
            db.release_session_turn_lease(session_id, holder)
    finally:
        release_or_close(db)
    return True


def deliver_pending(db: Any, session_id: str, turn_lease_holder: str) -> int:
    """Turn admission, with ``turn_lease_holder`` owning the conversation's lease: append every
    ``pending`` reply addressed to this conversation before the turn loads history. Returns the
    number appended; the caller reloads the transcript when it is non-zero. Never waits: a record
    another party holds stays ``pending`` for its holder or the next turn."""
    from tools.bot_mode_dm import _dm_dir

    directory = _dm_dir(create=False)
    if directory is None or not getattr(db, "db_path", None):
        return 0
    db_path = Path(db.db_path).resolve()
    tip = db.resolve_resume_session_id(session_id) or session_id
    delivered = 0
    for record_path in sorted(directory.glob(f"*{RECORD_SUFFIX}")):
        path = str(record_path)
        try:
            if not _addressed_to(_read(path), db, db_path, tip):
                continue
            with _try_locked(path) as locked:
                if not locked or not _addressed_to(record := _read(path), db, db_path, tip):
                    continue
                _append_row(db, tip, path, record, turn_lease_holder=turn_lease_holder)
                _delivered(path, record)
                delivered += 1
        except Exception as exc:
            logger.warning("message_agent: could not deliver pending reply %s: %s", path, exc)
    return delivered


def sweep(directory: Path, cutoff: float) -> int:
    """DM-cache sweep for reply records; returns files removed. Never delivers: the sweep runs in
    whatever profile writes a DM or runs housekeeping, not the record owner's. ``handoff`` and
    ``pending`` records are removed only past :data:`PENDING_MAX_AGE_SECONDS`; any other record
    older than ``cutoff`` lost both parties before settling. State and age are judged under the
    record lock, so a party advancing the record meanwhile keeps it; a held record is skipped."""
    removed = 0
    for record_path in directory.glob(f"*{RECORD_SUFFIX}"):
        path = str(record_path)
        try:
            with _try_locked(path) as locked:
                if not locked or (record := _read(path)) is None:
                    continue
                long_lived = record.get("state") in _LONG_LIVED_STATES
                if record_path.stat().st_mtime >= (time.time() - PENDING_MAX_AGE_SECONDS if long_lived else cutoff):
                    continue
                if record.get("state") == "pending":
                    logger.warning("message_agent: dropping reply %s, undelivered for 7 days", path)
                _discard(path)
            removed += 1
        except Exception as exc:  # the sweep runs inside message_agent's DM write: never fail it
            logger.debug("message_agent: reply record sweep skipped %s: %s", path, exc)
    for lock_path in directory.glob(f"*{RECORD_SUFFIX}.lock"):
        with contextlib.suppress(OSError):
            if not Path(str(lock_path)[:-len(".lock")]).exists() and lock_path.stat().st_mtime < cutoff:
                lock_path.unlink()
                removed += 1
    return removed


def _addressed_to(record: Optional[dict], db: Any, db_path: Path, tip: str) -> bool:
    """Whether ``record`` is a ``pending`` reply for this database's conversation ``tip``."""
    return (bool(record) and record.get("state") == "pending" and Path(record["db_path"]).resolve() == db_path
            and (db.resolve_resume_session_id(record["session_id"]) or record["session_id"]) == tip)


def _surface_holds(home: Path, db: Any, tip: str) -> bool:
    """Whether a live surface (a one-shot, a Desktop tab) holds this conversation's session.
    An unreadable registry counts as held (the delivery is retried, never forced)."""
    from hermes_cli.active_sessions import active_session_registry_snapshot

    try:
        entries = active_session_registry_snapshot(registry_home=home)
    except Exception as exc:
        logger.debug("message_agent: active-session registry unreadable: %s", exc)
        return True
    for entry in entries:
        held = entry.get("session_id")
        if held and (held == tip or db.resolve_resume_session_id(held) == tip):
            return True
    return False


def _note_vault(record: dict) -> dict:
    """Sender side: flag the record (once set, for good) when this process holds vault values for
    the owner's profile. Only a flag: the values themselves never leave this process's memory."""
    from agent.redact import has_registered_vault_values

    if record.get("hermes_home") and has_registered_vault_values(record["hermes_home"]):
        record["vault_guarded"] = True
    return record


def _withheld(record: dict) -> bool:
    """Whether the reply may carry the sender's vault values and this interpreter cannot scrub them."""
    return bool(record.get("vault_guarded")) and record.get("armed_by") != _INTERPRETER


def _delivered(path: str, record: dict) -> None:
    """Called under the record lock once its row is appended. A withheld reply's record stays on
    disk (``withheld``, never delivered again) for an operator until the 7-day sweep."""
    if _withheld(record):
        record["state"] = "withheld"
        _write(path, record)
    else:
        _discard(path)


def _forget(path: str) -> None:
    with _armed_lock:
        _armed.pop(path, None)


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
            # One attempt: exit must not wait. A refused delivery stays pending for the next turn.
            if settle(path, proc, exiting=True):
                deliver(path)
        except Exception as exc:
            logger.warning("message_agent: could not hand off the reply of %s: %s", proc.id, exc)


def _append_row(db: Any, session_id: str, path: str, record: dict, *, turn_lease_holder: str) -> None:
    """The DELIVERY row a push surface persists for the same completion, written under the
    conversation's turn lease (the append transaction re-checks ``turn_lease_holder``). A
    :func:`_withheld` reply and its command are replaced by a generic notice and the teammate's
    label. The record path goes only to the log: a later turn that knows it could ``read_file`` the
    raw reply without the vault values to mask it, and ``display_metadata`` is not model-proof
    (``hermes sessions export`` emits it)."""
    from tools.bot_mode_dm import REPLY_COMPLETION_CHARS
    from tools.process_registry import _redact_process_result
    from tools.process_registry_notifications import (
        PROCESS_COMPLETE_DISPLAY_KIND, format_process_notification, process_completion_display_text,
    )

    output, command = record.get("output") or "", record["command"]
    withheld = _withheld(record)
    if withheld:  # the command is the ``--reply-record <path>`` wrapper, so it names the record too
        label = record.get("label") or "the teammate"
        output = (f"The reply from {label} arrived after the sending session exited. It is withheld from "
                  "this transcript because that session held vault secrets that only it could redact. "
                  "Ask the operator for it.")
        command = f"message_agent {label}"
    evt = {"type": "completion", "session_id": record["proc_id"], "command": command,
           "exit_code": record.get("exit_code"), "completion_reason": record.get("completion_reason", "exited"),
           "output": output[-REPLY_COMPLETION_CHARS:]}
    if len(output) > REPLY_COMPLETION_CHARS:
        evt["output_cut"] = len(output) - REPLY_COMPLETION_CHARS
    _redact_process_result(evt)
    db.append_message(session_id, "user", content=format_process_notification(evt),
                      display_kind=PROCESS_COMPLETE_DISPLAY_KIND,
                      display_metadata={"display_text": process_completion_display_text([evt])},
                      turn_lease_holder=turn_lease_holder)
    if withheld:
        logger.warning("message_agent: withheld the vault-guarded reply of %s in session %s; "
                       "the raw reply is in %s (kept for 7 days)", record["proc_id"], session_id, path)
