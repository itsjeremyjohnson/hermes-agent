"""A recovered user-only transcript must not look settled while auto-continue builds."""
import threading

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.turn_marker import record_turn_start


@pytest.fixture()
def pending_recovery(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('recover-me', source='desktop')
    db.append_message('recover-me', 'user', 'Unfinished original group task')
    record_turn_start(tmp_path, 'recover-me', 'Unfinished original group task')
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    submissions = []
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    monkeypatch.setattr(server, '_get_db', lambda: db)
    monkeypatch.setattr(server, '_profile_home', lambda _profile: None)
    monkeypatch.setattr(server, '_load_cfg', lambda: {})
    monkeypatch.setattr(server, '_enable_gateway_prompts', lambda: None)
    monkeypatch.setattr(server, '_schedule_agent_build', lambda *a, **k: None)
    monkeypatch.setattr(server, '_schedule_session_cap_enforcement', lambda: None)
    monkeypatch.setattr(server, '_default_session_cwd', lambda: str(tmp_path))
    monkeypatch.setattr(server, '_start_agent_build', lambda *a: None)
    monkeypatch.setattr(server, '_ensure_active_session_slot', lambda *a: None)
    monkeypatch.setattr(server, '_emit', lambda *a: None)

    def wait_for_build(*a, **k):
        entered.set()
        assert release.wait(3), 'test did not release recovery build'
        return None

    def submit(_rid, _sid, session, text, **kwargs):
        submissions.append((text, kwargs))
        session['running'] = False
        session.pop('_auto_continue_scheduled', None)
        finished.set()

    monkeypatch.setattr(server, '_wait_agent', wait_for_build)
    monkeypatch.setattr(server, '_run_prompt_submit', submit)
    known = set(server._sessions)
    try:
        yield entered, release, finished, submissions
    finally:
        release.set()
        if entered.is_set():
            assert finished.wait(3)
        for sid in set(server._sessions) - known:
            server._sessions.pop(sid, None)
        db.close()


def resume():
    response = server.handle_request({'id': 'read', 'method': 'session.resume',
                                      'params': {'session_id': 'recover-me', 'source': 'desktop'}})
    assert 'error' not in response
    return response['result']


@pytest.mark.parametrize('warm', [False, True])
def test_cold_and_warm_resume_keep_recovery_pending_busy(pending_recovery, warm):
    entered, release, finished, submissions = pending_recovery
    cold = resume()
    assert entered.wait(3)
    assert cold.get('auto_continue', {}).get('attempt') == 1
    snapshot = resume() if warm else cold
    # Exactly the production edge: count grew only because the pre-crash user
    # persisted. The renderer's !running + count>before must not end this turn.
    assert [entry['role'] for entry in snapshot['messages']] == ['user']
    assert snapshot['message_count'] == 1
    assert snapshot['running'] is True
    assert snapshot['status'] == 'starting'
    assert submissions == []
    release.set()
    assert finished.wait(3)
    settled = resume()
    assert settled['running'] is False
    assert settled['status'] == 'idle'
    assert len(submissions) == 1

