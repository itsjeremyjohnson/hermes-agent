"""A ``message_agent`` reply reaches its sender exactly once, even when the sender exits first.

Regression for Robben-Media/charlie-hermes#43: a one-shot Bot Chat sender (``hermes chat -Q``)
lingers at most ``terminal.oneshot_completion_wait_seconds`` for its runners' completion
notifications, then exits; a teammate reply landing later was lost (no notification, no row).

The sender here is a real process running the real ``message_agent`` dispatch, runner and
one-shot linger code against a temp ``HERMES_HOME``. The only stand-in is the teammate's
``hermes`` binary, which prints its reply once the test opens a gate file.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

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


def _reply_rows(home):
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        return [m for m in db.get_messages("bot-chat") if REPLY in str(m.get("content"))]
    finally:
        db.close()


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
