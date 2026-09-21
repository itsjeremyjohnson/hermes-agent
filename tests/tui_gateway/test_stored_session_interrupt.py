"""Stop a live profile conversation by its durable identity without resuming it."""
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def live_profiles(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from tui_gateway import server
    from tui_gateway.turn_marker import record_turn_start, read_turn_marker

    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_tts_stream_stop", lambda: None)
    calls = []
    key = "same-opaque-stored-key"
    for name in ("default", "alpha", "beta"):
        profile_home = home if name == "default" else home / "profiles" / name
        profile_home.mkdir(parents=True, exist_ok=True)
        ready = threading.Event()
        ready.set()
        server._sessions[name + "-runtime"] = {
            "session_key": key,
            "profile_home": None if name == "default" else str(profile_home),
            "agent": SimpleNamespace(session_id=key, hard_interrupt=lambda n=name: calls.append(n)),
            "agent_ready": ready,
            "history_lock": threading.Lock(),
            "running": True,
            "queued_prompt": "synthetic queued work",
            "_active_turn_marker_key": key,
        }
        record_turn_start(profile_home, key, "synthetic accepted work")

    def forbidden(*args, **kwargs):
        pytest.fail("cancellation must not create/resume/build an agent or schedule continuation")

    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", forbidden)
    monkeypatch.setitem(server._methods, "session.resume", forbidden)
    monkeypatch.setitem(server._methods, "session.create", forbidden)
    yield server, calls, key, home, read_turn_marker
    server._sessions.clear()


@pytest.mark.parametrize("profile", ["default", "alpha", "beta"])
@pytest.mark.parametrize("identity", ["stored", "runtime"])
def test_stored_interrupt_selects_exact_profile_and_retires_only_its_marker(live_profiles, monkeypatch, profile, identity):
    server, calls, key, home, read_marker = live_profiles
    if identity == "stored":
        monkeypatch.setattr(server, "_start_agent_build", lambda *a, **kw: pytest.fail("stored interrupt must not build an agent"))
    params = {"profile": profile}
    params["stored_session_id" if identity == "stored" else "session_id"] = (
        key if identity == "stored" else profile + "-runtime")
    response = server.handle_request({
        "jsonrpc": "2.0", "id": "stop", "method": "session.interrupt", "params": params,
    })
    assert response["result"]["status"] == "interrupted"
    assert calls == [profile]
    for name in ("default", "alpha", "beta"):
        session = server._sessions[name + "-runtime"]
        profile_home = home if name == "default" else home / "profiles" / name
        if name == profile:
            assert session["_turn_cancel_requested"] is True
            assert session["queued_prompt"] is None
            assert read_marker(profile_home, key) is None
        else:
            assert "_turn_cancel_requested" not in session
            assert session["queued_prompt"] == "synthetic queued work"
            assert read_marker(profile_home, key) is not None


@pytest.mark.parametrize("params,code", [
    ({"stored_session_id": "same-opaque-stored-key"}, 4002),
    ({"stored_session_id": "same-opaque-stored-key", "profile": " "}, 4002),
    ({"stored_session_id": "same-opaque-stored-key", "profile": "alpha", "session_id": "beta-runtime"}, 4002),
    *[({"stored_session_id": "same-opaque-stored-key", "profile": "alpha", "session_id": value}, 4002)
      for value in ("", None, False, 0)],
    ({"stored_session_id": "missing", "profile": "alpha"}, 4001),
    ({"stored_session_id": "same-opaque-stored-key ", "profile": "alpha"}, 4001),
    ({"stored_session_id": 123, "profile": "alpha"}, 4002),
])
def test_ambiguous_or_missing_stored_interrupt_never_touches_live_work(live_profiles, params, code):
    server, calls, key, home, read_marker = live_profiles
    before = set(server._sessions)
    response = server.handle_request({
        "jsonrpc": "2.0", "id": "stop", "method": "session.interrupt", "params": params,
    })
    assert response["error"]["code"] == code
    assert calls == []
    assert set(server._sessions) == before
    for name in ("default", "alpha", "beta"):
        assert "_turn_cancel_requested" not in server._sessions[name + "-runtime"]
        profile_home = home if name == "default" else home / "profiles" / name
        assert read_marker(profile_home, key) is not None


def test_stored_interrupt_binds_selected_profile_for_hooks(live_profiles, monkeypatch):
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home
    from hermes_cli import plugins

    server, calls, key, home, _ = live_profiles
    seen = []
    for name in ("alpha", "beta"):
        (home / "profiles" / name / ".env").write_text(f"INTERRUPT_TEST_KEY={name}\n")
    monkeypatch.setattr(plugins, "invoke_hook", lambda *a, **kw: seen.append(
        (get_hermes_home(), get_secret("INTERRUPT_TEST_KEY"))))
    for name in ("alpha", "beta", "alpha"):
        server._sessions[name + "-runtime"]["running"] = True
        response = server.handle_request({
            "jsonrpc": "2.0", "id": name, "method": "session.interrupt",
            "params": {"stored_session_id": key, "profile": name},
        })
        assert response["result"]["status"] == "interrupted"
        assert get_hermes_home() == home
    assert seen == [(home / "profiles" / name, name) for name in ("alpha", "beta", "alpha")]
