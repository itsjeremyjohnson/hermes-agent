import json
import sys
from types import SimpleNamespace

import pytest

from tools import bot_mode_dm


def test_delivery_uses_refusal_code_before_human_wording(tmp_path, capsys):
    for code, message, busy in (
        ("SESSION_NOT_OWNED", "Ce chat est occupé.", True),
        ("SESSION_COORDINATION_UNAVAILABLE", "Cannot verify whether session already has a live owner", False),
        ("SESSION_NOT_OWNED_EXTRA", "Different failure", False),
    ):
        dm = tmp_path / "message.txt"
        dm.write_text("isolated probe", encoding="utf-8")
        child = tmp_path / "child.py"
        child.write_text(f"import sys\nprint({f'hermes-refusal-reason: {code}'!r}, file=sys.stderr)\nprint({message!r}, file=sys.stderr)\nraise SystemExit(1)\n", encoding="utf-8")
        assert bot_mode_dm._run_delivery([sys.executable, str(child)], str(dm), stdin_file=False) == 1
        output = capsys.readouterr()
        assert (json.loads(output.out)["reason"] == "target_busy") if busy else not output.out
        assert not dm.exists()


def test_one_shot_cli_preserves_refusal_reason(monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli import active_sessions
    refusal = active_sessions.ActiveSessionRefusal("Ce chat est occupé.", reason=active_sessions.SESSION_NOT_OWNED)
    monkeypatch.setattr(active_sessions, "try_acquire_active_session", lambda **kwargs: (None, refusal))
    cli = SimpleNamespace(_active_session_lease=None, session_id="isolated", config={})
    assert not HermesCLI._claim_active_session(cli, stderr=True)
    assert "hermes-refusal-reason: SESSION_NOT_OWNED\n" in capsys.readouterr().err


def test_one_shot_cli_refused_by_an_owned_session_submits_no_turn(tmp_path, monkeypatch, capsys):
    """Cron requeues a ``SESSION_NOT_OWNED`` child only because the refusal precedes any turn work.
    Moving the lease claim after credentials, agent init or the turn breaks that and fails here."""
    import cli as cli_module
    from cli import HermesCLI
    from hermes_cli.active_sessions import try_acquire_active_session
    from hermes_cli.cli_single_query import _run_single_query_mode
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    owner, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None
    turns = []
    monkeypatch.setattr(cli_module, "_route_single_query_images", lambda cli, query, *a: query)
    monkeypatch.setattr(cli_module, "_configure_quiet_agent", lambda agent: None)
    monkeypatch.setattr(cli_module, "_run_quiet_single_query", lambda cli, query, **kw: turns.append(query))
    child = SimpleNamespace(_active_session_lease=None, session_id="chat", config={}, agent=None,
                            _active_agent_route_signature=None, _release_active_session=lambda: None,
                            _ensure_runtime_credentials=lambda: True, _init_agent=lambda **kw: True,
                            _resolve_turn_agent_config=lambda q: {"signature": "s", "model": "m", "runtime": {}})
    child._claim_active_session = HermesCLI._claim_active_session.__get__(child)
    try:
        with pytest.raises(SystemExit) as exited:
            _run_single_query_mode(child, "NEW alert", None, quiet=True, oneshot=True)
        assert exited.value.code == 1
        assert "hermes-refusal-reason: SESSION_NOT_OWNED\n" in capsys.readouterr().err
        assert turns == []
    finally:
        owner.release()
