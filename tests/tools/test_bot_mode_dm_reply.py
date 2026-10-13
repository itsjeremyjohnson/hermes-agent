"""A ``message_agent`` reply reaches its sender exactly once, even when the sender exits first.

Regression for Robben-Media/charlie-hermes#43: a one-shot Bot Chat sender (``hermes chat -Q``)
lingers at most ``terminal.oneshot_completion_wait_seconds`` for its runners' completion
notifications, then exits; a teammate reply landing later was lost (no notification, no row).

The sender here is a real process running the real ``message_agent`` dispatch, runner and
one-shot linger code against a temp ``HERMES_HOME``. The only stand-in is the teammate's
``hermes`` binary, which prints its reply once the test opens a gate file. Competing turns and
surfaces hold the real SessionDB turn lease and the real active-session registry slot.
"""

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

REPO = Path(__file__).resolve().parents[2]
REPLY = "REPLY_SENTINEL_43"

_SENDER = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from types import SimpleNamespace
    from hermes_cli.quiet_single_query import continue_quiet_notify_completions
    from hermes_state import SessionDB
    from tools import bot_mode_dm, bot_relay
    from tools.process_registry import process_registry

    mode, home, fake_hermes = sys.argv[2:5]
    if len(sys.argv) > 5:  # a vault fill this turn: only this process can scrub the value
        from agent.redact import register_vault_redaction_value
        from hermes_logging import setup_logging
        setup_logging(hermes_home=Path(home))  # the model-readable logs/ a CLI session writes
        if sys.argv[5]:
            register_vault_redaction_value(sys.argv[5])
    bot_relay._hermes_cli = lambda: fake_hermes
    db = SessionDB(db_path=Path(home) / "state.db")
    db.create_session("bot-chat", source="cli")
    db.set_session_title("bot-chat", "Bot Chat")
    agent = SimpleNamespace(_session_db=db, session_id="bot-chat", _session_title_hint="Bot Chat",
                            _bot_mode_protocol=True, _bot_mode_manager=True)
    ack = json.loads(bot_mode_dm.message_agent_tool(target="researcher", message="status?", agent=agent))
    assert ack["status"] == "queued", ack
    delivered = []
    if mode == "finished_undrained":
        # The finalize linger: waits for the runner, never drains its notification.
        process_registry.wait_for_pending_completions(None, timeout=60)
    else:
        budget = 60 if mode == "linger_delivers" else 0.5
        continue_quiet_notify_completions("bot-chat", delivered.append, owns_event=lambda evt: True,
                                          linger_budget=budget)
    print(json.dumps(delivered))
""")


def _setup(tmp_path, reply=REPLY):
    home = tmp_path / "home"
    profile = home / "profiles" / "researcher"
    profile.mkdir(parents=True)
    (profile / "profile.yaml").write_text("description: teammate\nui_meta:\n  hermes-bots:\n    shape: cloud\n",
                                          encoding="utf-8")
    gate = tmp_path / "gate"
    fake_hermes = tmp_path / "bin" / "hermes"
    fake_hermes.parent.mkdir()
    fake_hermes.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        f"while not os.path.exists({str(gate)!r}):\n"
        "    time.sleep(0.05)\n"
        f"print({reply!r})\n", encoding="utf-8")
    fake_hermes.chmod(0o755)
    sender = tmp_path / "sender.py"
    sender.write_text(_SENDER, encoding="utf-8")
    temp = tmp_path / "tmp"
    temp.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "TMPDIR": str(temp)}
    return home, gate, fake_hermes, sender, env


def _messages(home):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        return db.get_messages("bot-chat")
    finally:
        db.close()


def _reply_rows(home):
    return [m for m in _messages(home) if REPLY in str(m.get("content"))]


def _send_and_exit(tmp_path):
    """The sender dispatches, lingers briefly and exits while the teammate is still replying."""
    home, gate, fake_hermes, sender, env = _setup(tmp_path)
    run = subprocess.run([sys.executable, str(sender), str(REPO), "exits_before_reply", str(home), str(fake_hermes)],
                         env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return home, gate


def _wait_for_reply_row(home, timeout=30):
    deadline = time.monotonic() + timeout
    while not _reply_rows(home) and time.monotonic() < deadline:
        time.sleep(0.2)
    return _messages(home)


@pytest.mark.parametrize("mode, expect_notification", [
    ("linger_delivers", True),      # sender alive: the notification delivers, no row
    ("exits_before_reply", False),  # sender gone before the reply: the runner writes the row
    ("finished_undrained", False),  # reply landed, sender exited without draining it: sender writes the row
])
def test_reply_is_delivered_exactly_once_whether_or_not_the_sender_outlives_the_runner(
        tmp_path, mode, expect_notification):
    home, gate, fake_hermes, sender, env = _setup(tmp_path)
    if mode != "exits_before_reply":
        gate.touch()

    run = subprocess.run([sys.executable, str(sender), str(REPO), mode, str(home), str(fake_hermes)],
                         env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    notifications = json.loads(run.stdout.strip().splitlines()[-1])

    gate.touch()  # the teammate replies now, after the sender process is gone
    deadline = time.monotonic() + 30
    rows = _reply_rows(home)
    while not expect_notification and not rows and time.monotonic() < deadline:
        time.sleep(0.2)
        rows = _reply_rows(home)
    if expect_notification:
        time.sleep(1.0)  # a duplicate row would come from the runner right after its exit
        rows = _reply_rows(home)

    assert [REPLY in text for text in notifications] == ([True] if expect_notification else [])
    assert len(rows) == (0 if expect_notification else 1)
    if rows:
        assert rows[0]["role"] == "user"
        assert rows[0]["display_kind"] == "process_complete"


def test_reply_waits_out_a_running_turn_and_lands_after_it(tmp_path):
    from hermes_state import SessionDB

    home, gate = _send_and_exit(tmp_path)
    db = SessionDB(db_path=home / "state.db")
    holder = f"pid={os.getpid()}:turn=competing:platform=cli"
    try:
        assert db.try_acquire_session_turn_lease("bot-chat", holder)
        db.append_message("bot-chat", "user", content="next request", turn_lease_holder=holder)
        gate.touch()  # the teammate replies while that turn is running
        time.sleep(4.0)
        assert _reply_rows(home) == []
        db.append_message("bot-chat", "assistant", content="answer", turn_lease_holder=holder)
    finally:
        db.release_session_turn_lease("bot-chat", holder)
        db.close()

    tail = [(m["role"], m["content"] if REPLY not in str(m["content"]) else REPLY)
            for m in _wait_for_reply_row(home)]
    assert tail[-3:] == [("user", "next request"), ("assistant", "answer"), ("user", REPLY)]


def _admit(db, session_id):
    """A real durable turn admission of ``session_id``; returns the history the turn starts from."""
    from agent.turn_facade_lease import admit_durable_turn_lease

    agent = SimpleNamespace(_session_db=db, session_id=session_id, _persist_disabled=False,
                            _interrupt_requested=False, _interrupt_message=None,
                            _execution_thread_id=None, _session_turn_lease_refresh_interval=60.0)
    agent._emit_status = agent._emit_warning = agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = threading.Lock
    admission = admit_durable_turn_lease(
        agent, session_id=session_id, relay_turn_id=f"{session_id}:t:abcd",
        task_context={"session_id": session_id, "task_id": "t", "platform": "desktop"},
        conversation_history=[])
    try:
        return admission.conversation_history
    finally:
        admission.lease.release()


def _kill_runners(tmp_path):
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = proc.info["cmdline"] or []
        if "--reply-record" in cmdline and any(str(tmp_path) in part for part in cmdline):
            proc.send_signal(signal.SIGKILL)
            proc.wait(10)


# Real SIGKILL of this test's own runner: it was reparented away when its sender exited, so the
# guard sees a PID outside the test subtree. _kill_runners matches only argv under tmp_path.
@pytest.mark.live_system_guard_bypass
def test_reply_held_by_an_idle_surface_survives_its_runner_and_reaches_that_surfaces_next_turn(
        tmp_path, monkeypatch):
    from hermes_cli.active_sessions import release_active_session, try_acquire_active_session
    from hermes_state import SessionDB

    home, gate = _send_and_exit(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))  # the DM directory the runner used
    surface, refusal = try_acquire_active_session(session_id="bot-chat", surface="desktop", config={},
                                                  registry_home=home)
    assert refusal is None
    db = SessionDB(db_path=home / "state.db")
    try:
        gate.touch()  # the reply lands while an idle Desktop tab holds the Bot Chat
        time.sleep(4.0)
        assert _reply_rows(home) == []
        _kill_runners(tmp_path)  # the runner's retry budget is gone; only the record remains

        history = _admit(db, "bot-chat")
    finally:
        db.close()
        release_active_session(surface)

    assert [(m["role"], REPLY in str(m["content"])) for m in history] == [("user", True)]
    assert len(_reply_rows(home)) == 1


# ---- Reply records driven in-process: the sender's real record calls, no teammate process ----

def _record(db_path, session_id, *, output=None, proc_id="proc_test"):
    """A record the sender armed in the CURRENT profile scope. With ``output`` the runner finished
    and the sender settled it undelivered (``pending``); without, the sender exited while the
    runner was still running (``handoff``)."""
    from tools.bot_mode_dm import _dm_dir
    from tools.bot_mode_dm_reply import arm, new_record_path, settle
    from tools.process_registry import ProcessSession

    proc = ProcessSession(id=proc_id, command="message_agent runner")
    path = new_record_path(_dm_dir())
    assert arm(path, proc, db_path=str(db_path), session_id=session_id)
    if output is not None:
        proc.output_buffer, proc.exit_code = output, 0
        proc._completion_event.set()
    assert settle(path, proc, exiting=output is None) is (output is not None)
    return path


def _profile(home, *, redact):
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(f"security:\n  redact_secrets: {str(redact).lower()}\n", encoding="utf-8")
    return home


@contextlib.contextmanager
def _in_profile(home):
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(str(home))
    secret_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def test_a_reply_lands_under_its_owners_redaction_whichever_profile_runs_the_delivery(tmp_path, monkeypatch):
    """The DM directory is shared by every profile of the OS user: profile A's DM-write sweep, and a
    delivery running in A's scope (a poll thread, a runner, interpreter exit), must not decide how
    profile B's reply is redacted."""
    from agent.redact import redact_sensitive_text, register_vault_redaction_value
    from hermes_state import SessionDB
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_mode_dm_reply import deliver

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    credential, vault = "ghp_Rv7Qk2Lm9Xc4Tz8Wn1Bp6Hs3Jd5Fg0Ya2Ue", "vault-opaque-value-5517"
    home_a, home_b = _profile(tmp_path / "a", redact=False), _profile(tmp_path / "b", redact=True)
    db = SessionDB(db_path=home_b / "state.db")
    try:
        db.create_session("chat-b", source="cli")
        with _in_profile(home_b):
            register_vault_redaction_value(vault)
            via_runner = _record(db.db_path, "chat-b", output=f"first {credential} {vault}", proc_id="proc_1")
            _record(db.db_path, "chat-b", output=f"second {credential} {vault}", proc_id="proc_2")
        with _in_profile(home_a):
            cleanup_bot_dm_cache()  # A writes a DM: the sweep sees B's pending records
            assert deliver(via_runner)  # a delivery that runs in A's scope
        with _in_profile(home_b):
            history = _admit(db, "chat-b")  # B's next turn takes the other one
        with _in_profile(home_a):
            a_still_unredacted = redact_sensitive_text(credential)
        rows = [m["content"] for m in db.get_messages("chat-b")]
    finally:
        db.close()

    assert [(credential in row, vault in row, "ghp_Rv...a2Ue" in row) for row in rows] == [(False, False, True)] * 2
    seen = "\n".join(m["content"] for m in history)  # consecutive user rows merge on reload
    assert ("first" in seen, "second" in seen) == (True, True)
    assert a_still_unredacted == credential


def _waits_on_lock(lock_path):
    """Whether some process is blocked acquiring the flock on ``lock_path`` (Linux /proc/locks)."""
    inode = f":{os.stat(lock_path).st_ino} "
    return any("->" in line and inode in line for line in Path("/proc/locks").read_text().splitlines())


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs Linux /proc/locks")
def test_a_runner_outcome_recorded_while_the_sweeper_judges_an_old_handoff_still_reaches_the_conversation(
        tmp_path, monkeypatch):
    """A runner can outlive the 24 h DM sweep horizon. Its outcome, recorded while a sweep is
    deciding whether its old ``handoff`` record is stale, must still reach the next turn."""
    from hermes_cli.active_sessions import _FileLock
    from hermes_state import SessionDB
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_mode_dm_reply import _finish

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("chat", source="cli")
        path = _record(db.db_path, "chat")
        two_days_ago = time.time() - 2 * 24 * 3600
        os.utime(path, (two_days_ago, two_days_ago))

        holder = _FileLock(Path(path + ".lock"))  # the runner is mid-update of its record
        holder.__enter__()
        try:
            sweeper = threading.Thread(target=cleanup_bot_dm_cache)
            sweeper.start()
            deadline = time.monotonic() + 10
            while sweeper.is_alive() and not _waits_on_lock(path + ".lock") and time.monotonic() < deadline:
                time.sleep(0.01)
            finisher = threading.Thread(target=_finish, args=(path, 0, REPLY))  # the runner's real exit step
            finisher.start()
        finally:
            holder.__exit__(None, None, None)
        sweeper.join(10)
        finisher.join(10)
        cleanup_bot_dm_cache()  # the next housekeeping pass

        history = _admit(db, "chat")
    finally:
        db.close()

    assert [REPLY in m["content"] for m in history] == [True]


def test_turn_admission_never_waits_on_a_reply_record_another_party_holds(tmp_path, monkeypatch):
    """A runner holding a record lock (another conversation's, or this one's) must not stall a
    turn before its supervision starts; a held record stays pending for a later turn."""
    from hermes_cli.active_sessions import _FileLock
    from hermes_state import SessionDB

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("chat-n", source="cli")
        db.create_session("chat-m", source="cli")
        own = _record(db.db_path, "chat-n", output="reply for n", proc_id="proc_n")
        other = _record(db.db_path, "chat-m", output="reply for m", proc_id="proc_m")

        admitted, result = threading.Event(), {}

        def admit_n():
            result["history"] = _admit(db, "chat-n")
            admitted.set()

        held = [_FileLock(Path(own + ".lock")), _FileLock(Path(other + ".lock"))]
        for lock in held:
            lock.__enter__()
        try:
            threading.Thread(target=admit_n, daemon=True).start()
            admitted_while_held = admitted.wait(10)
        finally:
            for lock in held:
                lock.__exit__(None, None, None)
        admitted.wait(30)

        later_n, later_m = _admit(db, "chat-n"), _admit(db, "chat-m")
    finally:
        db.close()

    assert admitted_while_held
    assert result["history"] == []
    assert [m["content"].count("reply for n") > 0 for m in later_n] == [True]
    assert [m["content"].count("reply for m") > 0 for m in later_m] == [True]


VAULT = "opaque-vault-value-4391"


def _stored(home):
    """Every byte of every transcript row, display metadata included."""
    import sqlite3

    with contextlib.closing(sqlite3.connect(home / "state.db")) as conn:
        rows = conn.execute("SELECT * FROM messages").fetchall()
    return "\n".join(v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for row in rows for v in row)


def _records(tmp_path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "tmp").glob("hermes-dm-*/*.reply.json")]


def _files_holding(tmp_path, home, *needles):
    """Files under the DM temp dir or the profile's logs/ (both readable by a later model's file
    tools) that hold any of ``needles``."""
    return sorted(str(f) for root in (tmp_path / "tmp", home / "logs") if root.exists() for f in root.rglob("*")
                  if f.is_file() and any(n.encode() in f.read_bytes() for n in needles))


def _wait_for_runner_delivery(tmp_path, home, timeout=30):
    """Until the runner appended its row and discarded the record."""
    deadline = time.monotonic() + timeout
    while (not _messages(home) or _records(tmp_path)) and time.monotonic() < deadline:
        time.sleep(0.1)


def _assert_discarded(tmp_path, home, secret):
    """A guarded reply lands as the generic notice and leaves no byte of itself anywhere a later
    model can read: transcript columns, the DM temp dir, the profile's logs."""
    assert secret not in _stored(home)
    assert (_records(tmp_path), _files_holding(tmp_path, home, secret, REPLY)) == ([], [])
    [row] = _messages(home)
    assert (REPLY in row["content"], "was discarded" in row["content"]) == (False, True)


@pytest.mark.live_system_guard_bypass  # the "admission" case SIGKILLs this test's own reparented runner
@pytest.mark.parametrize("mode", ["linger_delivers", "runner", "admission"])
def test_a_reply_never_publishes_the_senders_vault_values_from_another_process(tmp_path, monkeypatch, caplog, mode):
    """Vault values live only in the memory of the sender that registered them. Its own consumer
    masks them (``linger_delivers``); a delivery by the runner or by a later turn's admission runs
    in another interpreter that cannot, so the reply is discarded: no transcript column, record,
    temp file or log line keeps a byte of it or names where it was."""
    from hermes_cli.active_sessions import release_active_session, try_acquire_active_session
    from hermes_state import SessionDB

    home, gate, fake_hermes, sender, env = _setup(tmp_path, reply=f"{REPLY} {VAULT}")
    if mode == "linger_delivers":
        gate.touch()
    run = subprocess.run([sys.executable, str(sender), str(REPO),
                          "linger_delivers" if mode == "linger_delivers" else "exits_before_reply",
                          str(home), str(fake_hermes), VAULT], env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    notifications = json.loads(run.stdout.strip().splitlines()[-1])

    surface = None
    if mode == "admission":  # an idle surface holds the chat, so only the next turn's admission delivers
        surface, refusal = try_acquire_active_session(session_id="bot-chat", surface="desktop", config={},
                                                      registry_home=home)
        assert refusal is None
    try:
        gate.touch()
        if mode == "runner":
            _wait_for_runner_delivery(tmp_path, home)
        elif mode == "admission":
            deadline = time.monotonic() + 30
            while [r["state"] for r in _records(tmp_path)] != ["pending"] and time.monotonic() < deadline:
                time.sleep(0.1)
            _kill_runners(tmp_path)
            pending_holds = _files_holding(tmp_path, home, VAULT, REPLY)
            record_dir = str(next((tmp_path / "tmp").glob("hermes-dm-*")))
            monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
            db = SessionDB(db_path=home / "state.db")
            try:
                with caplog.at_level(logging.WARNING, logger="tools.bot_mode_dm_reply"):
                    _admit(db, "bot-chat")
            finally:
                db.close()
    finally:
        if surface is not None:
            release_active_session(surface)

    assert VAULT not in json.dumps(notifications) + _stored(home)
    if mode == "linger_delivers":
        assert [("«redacted-vault-secret»" in n, REPLY in n) for n in notifications] == [(True, True)]
        assert (_messages(home), _records(tmp_path)) == ([], [])
        return
    _assert_discarded(tmp_path, home, VAULT)
    if mode == "admission":
        assert pending_holds == []  # nor while it waited for the turn
        assert ("proc_" in caplog.text, record_dir in caplog.text, VAULT in caplog.text) == (True, False, False)


@pytest.mark.skipif(not Path("/proc/locks").exists(), reason="needs Linux /proc/locks")
def test_a_vault_value_registered_after_the_senders_last_flag_read_still_keeps_the_reply_out(tmp_path):
    """A vault fill on another thread can register right after the exiting sender read the record's
    flag and before it hands the record off. The sender holds the value before it exits, so the
    runner must not publish the reply it later receives."""
    value = "opaque-late-vault-value-2208"
    home, gate, fake_hermes, sender, env = _setup(tmp_path, reply=f"{REPLY} {value}")
    sender.write_text(_SENDER + textwrap.dedent(f"""
        import os, threading, time
        from agent.redact import redact_sensitive_text
        from tools import bot_mode_dm_reply as reply

        registered = threading.Event()

        def fill():
            register_vault_redaction_value({value!r})
            registered.set()

        def fill_blocked_on_a_record():
            inodes = [f":{{os.stat(p).st_ino}} " for p in bot_mode_dm._dm_dir().glob("*.reply.json.lock")]
            return any("->" in line and any(i in line for i in inodes)
                       for line in Path("/proc/locks").read_text().splitlines())

        note_vault, settle = reply._note_vault, reply.settle

        def note_vault_then_fill(record):  # the exiting sender's final flag read
            result = note_vault(record)
            threading.Thread(target=fill, daemon=True).start()
            deadline = time.monotonic() + 10
            while not (registered.is_set() or fill_blocked_on_a_record()):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            return result

        def settle_then_hold_the_value(*args, **kwargs):
            result = settle(*args, **kwargs)
            assert registered.wait(10)
            assert {value!r} not in redact_sensitive_text({value!r})  # the sender masks it from here on
            return result

        reply._note_vault, reply.settle = note_vault_then_fill, settle_then_hold_the_value
    """), encoding="utf-8")

    run = subprocess.run([sys.executable, str(sender), str(REPO), "exits_before_reply", str(home), str(fake_hermes),
                          ""], env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    gate.touch()
    _wait_for_runner_delivery(tmp_path, home)

    _assert_discarded(tmp_path, home, value)


def test_a_vault_fill_in_flight_when_the_reply_module_first_loads_still_keeps_the_reply_out(tmp_path):
    """A fill on another thread starts registering before anything imported the reply module, so it
    read no fence. Dispatch then loads the module and arms a record while that registration is in
    flight; the fill completes at the exiting sender's final flag read. The sender holds the value
    before it exits, so the runner must not publish the reply."""
    value = "opaque-first-import-vault-5813"
    home, gate, fake_hermes, sender, env = _setup(tmp_path, reply=f"{REPLY} {value}")
    dispatch = 'ack = json.loads(bot_mode_dm.message_agent_tool(target="researcher", message="status?", agent=agent))\n'
    assert dispatch in _SENDER
    sender.write_text(_SENDER.replace(dispatch, textwrap.dedent(f"""
        import threading, time
        from agent import redact
        assert "tools.bot_mode_dm_reply" not in sys.modules
        paused, resume, registered = threading.Event(), threading.Event(), threading.Event()

        class PauseBeforeInsert:  # the fill stops after reading the fence list, before inserting
            def __init__(self, lock):
                self.lock = lock
            def __enter__(self):
                if threading.current_thread() is fill:
                    paused.set()
                    assert resume.wait(30)
                return self.lock.__enter__()
            def __exit__(self, *exc):
                return self.lock.__exit__(*exc)

        def fill_value():
            redact.register_vault_redaction_value({value!r})
            registered.set()

        def resume_fill():
            resume.set()
            assert registered.wait(10)
            assert {value!r} not in redact.redact_sensitive_text({value!r})  # the sender masks it from here on
            print("sender holds the value", file=sys.stderr, flush=True)  # at exit an assert failure keeps exit 0

        redact._VAULT_REDACTION_LOCK = PauseBeforeInsert(redact._VAULT_REDACTION_LOCK)
        fill = threading.Thread(target=fill_value, daemon=True)
        fill.start()
        assert paused.wait(10)
        acks = []
        dispatcher = threading.Thread(target=lambda: acks.append(bot_mode_dm.message_agent_tool(
            target="researcher", message="status?", agent=agent)), daemon=True)
        dispatcher.start()
        dispatcher.join(3)
        if dispatcher.is_alive():  # loading the reply module waits for the fill
            resume_fill()
        else:  # armed with the fill in flight: it completes at the sender's final flag read
            from tools import bot_mode_dm_reply as reply
            note_vault = reply._note_vault

            def note_vault_then_fill(record):
                result = note_vault(record)
                resume_fill()
                return result

            reply._note_vault = note_vault_then_fill
        dispatcher.join(30)
        ack = json.loads(acks[0])
    """)), encoding="utf-8")

    run = subprocess.run([sys.executable, str(sender), str(REPO), "exits_before_reply", str(home), str(fake_hermes)],
                         env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "sender holds the value" in run.stderr
    gate.touch()
    _wait_for_runner_delivery(tmp_path, home)

    _assert_discarded(tmp_path, home, value)


def test_a_failure_while_binding_the_owners_scope_leaves_the_callers_scope_intact(tmp_path, monkeypatch):
    from agent import secret_scope
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    from tools.bot_mode_dm_reply import _owner_scope

    home_a, home_b = _profile(tmp_path / "a", redact=False), _profile(tmp_path / "b", redact=False)
    (home_a / ".env").write_text("OWNER_KEY=A\n", encoding="utf-8")
    (home_b / ".env").write_text("OWNER_KEY=B\n", encoding="utf-8")
    with _owner_scope(str(home_a)):
        monkeypatch.setattr(secret_scope, "build_profile_secret_scope",
                            lambda home: (_ for _ in ()).throw(OSError("unreadable profile")))
        with pytest.raises(OSError):
            with _owner_scope(str(home_b)):
                pass
        after = (get_hermes_home(), get_secret("OWNER_KEY"))
    assert after == (home_a, "A")
