"""Inline manager batches keep peer and Desktop-relay identity at the real boundary."""
import json
import shlex
from pathlib import Path

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from agent.turn_author import bot_author_id, local_origin
from hermes_state import SessionDB
from tests.tools.test_bot_mode_dm import _FakeAgent, _managed_home, _runner_author, _runner_parts
from tools import bot_mode_dm, bot_mode_probe, bot_relay


def test_inline_batch_peer_and_desktop_relay_keep_route_identity(tmp_path, monkeypatch):
    """One batch crosses the real peer command and Desktop envelope paths.

    Only the external process launch is fake. Author ownership, the queued
    envelope, indexed receipts, and the canonical Bot Chat signature are the
    production objects.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = _managed_home(tmp_path, peers=("spark",))
    bot_relay.write_remote_roster(home, [{
        "profile": "scribe",
        "handle": "scribe",
        "connection_id": "desktop-mac",
        "connection_label": "Desktop",
        "title": "Scribe",
        "online": True,
    }])
    db = SessionDB(home / "state.db")
    db.create_session("manager-batch", source="cli")
    db.set_session_title("manager-batch", "Bot Chat")
    manager = _FakeAgent(home)
    manager._session_db = db
    manager.session_id = "manager-batch"
    manager._session_title_hint = None
    manager._bot_mode_manager = True
    calls = []

    def fake_terminal(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return json.dumps({"output": "Background process started", "session_id": f"proc-{len(calls)}"})

    monkeypatch.setattr("tools.terminal_tool.terminal_tool", fake_terminal)
    monkeypatch.setattr(bot_relay, "_hermes_cli", lambda: "hermes")
    bot_mode_probe._reset_cache_for_tests()
    try:
        refused = json.loads(INLINE_TOOL_EXECUTORS["message_agent"](
            _FakeAgent(home, title="ordinary chat"),
            {"assignments": [{"target": "spark", "message": "no"}]},
            InlineToolContext(effective_task_id="refused"),
        ))
        assert "error" in refused and not calls

        result = json.loads(INLINE_TOOL_EXECUTORS["message_agent"](manager, {"assignments": [
            {"target": "spark", "message": "peer payload"},
            {"target": "scribe", "message": "relay payload"},
        ]}, InlineToolContext(effective_task_id="batch-routes")))
        assert result["status"] == "sent" and result["sent"] == 2
        assert [entry["index"] for entry in result["results"]] == [0, 1]
        assert [entry["target"] for entry in result["results"]] == ["spark", "scribe"]
        assert len(calls) == 2
        assert all(call["task_id"] == "batch-routes" and call["notify_on_complete"] is True for call in calls)

        peer_mode, peer_file, peer_argv = _runner_parts(calls[0]["command"])
        assert peer_mode == "stdin"
        assert peer_argv == ["hermes", "-p", "default", "peer", "dm", "spark"]
        assert _runner_author(calls[0]["command"]) == {
            "id": bot_author_id("default", local_origin()), "name": "hermes", "is_bot": True,
        }
        peer_text = Path(peer_file).read_text(encoding="utf-8")
        assert peer_text.startswith("Message from 🤖 hermes (@hermes): peer payload")
        assert result["results"][0]["result"]["delivery_id"] == bot_mode_dm._dm_delivery_id(peer_file)
        assert result["results"][0]["result"]["status"] == "queued"

        waiter = shlex.split(calls[1]["command"])
        assert waiter[1:3] == [str(Path(bot_mode_dm.__file__).resolve()), "--wait-reply"]
        envelope_id = result["results"][1]["result"]["delivery_id"]
        assert envelope_id and envelope_id in waiter[3]
        envelope_path = bot_relay.relay_root(home) / bot_relay.OUTBOX_DIR / f"{envelope_id}.json"
        envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        assert envelope["id"] == envelope_id
        assert envelope["from_profile"] == "default"
        assert envelope["from_handle"] == "hermes"
        assert envelope["target_connection"] == "desktop-mac"
        assert envelope["target_profile"] == "scribe"
        assert envelope["target_handle"] == "scribe"
        assert envelope["message"].startswith("Message from 🤖 hermes (@hermes): relay payload")
        assert "peer payload" not in envelope["message"]
        assert result["results"][1]["result"]["status"] == "queued"
        assert result["results"][1]["result"]["to"].startswith("@scribe on ")
    finally:
        db.close()
        bot_mode_probe._reset_cache_for_tests()
