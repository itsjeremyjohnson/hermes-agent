"""A ``message_agent`` reply reaches its sender exactly once, even when the sender exits first.

Regression for Robben-Media/charlie-hermes#43: a one-shot Bot Chat sender (``hermes chat -Q``)
lingers at most ``terminal.oneshot_completion_wait_seconds`` for its runners' completion
notifications, then exits; a teammate reply landing later was lost (no notification, no row).

The sender here is a real process running the real ``message_agent`` dispatch, runner and
one-shot linger code against a temp ``HERMES_HOME``. The only stand-in is the teammate's
``hermes`` binary, which prints its reply once the test opens a gate file. Competing turns and
surfaces hold the real SessionDB turn lease and the real active-session registry slot.
"""

import json
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


def _setup(tmp_path):
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
        f"print({REPLY!r})\n", encoding="utf-8")
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
    from agent.turn_facade_lease import admit_durable_turn_lease
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

        agent = SimpleNamespace(_session_db=db, session_id="bot-chat", _persist_disabled=False,
                                _interrupt_requested=False, _interrupt_message=None,
                                _execution_thread_id=None, _session_turn_lease_refresh_interval=60.0)
        agent._emit_status = agent._emit_warning = agent._touch_activity = lambda *a, **k: None
        agent._liveness_activity_lock = threading.Lock
        admission = admit_durable_turn_lease(
            agent, session_id="bot-chat", relay_turn_id="bot-chat:t:abcd",
            task_context={"session_id": "bot-chat", "task_id": "t", "platform": "desktop"},
            conversation_history=[])
        try:
            history = admission.conversation_history
        finally:
            admission.lease.release()
    finally:
        db.close()
        release_active_session(surface)

    assert [(m["role"], REPLY in str(m["content"])) for m in history] == [("user", True)]
    assert len(_reply_rows(home)) == 1
