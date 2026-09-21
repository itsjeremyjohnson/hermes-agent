"""Manager opt-in and batch admission through the real inline tool boundary."""
import json
import os
from pathlib import Path
import shlex
import sys
import time

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tests.tools.test_bot_mode_dm import _FakeAgent, _managed_home
from tools import bot_mode_dm, bot_mode_probe


@pytest.mark.parametrize("notify", [True, False])
def test_manager_config_schema_protocol_and_batch_containment(tmp_path, monkeypatch, notify):
    from agent.agent_init import _apply_agent_section
    from agent.system_prompt import _bot_mode_parts
    from hermes_cli.config import load_config_readonly

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = _managed_home(tmp_path, teammates=('researcher', 'coder'))
    monkeypatch.setenv('HERMES_HOME', str(home))
    bot_mode_probe._reset_cache_for_tests()
    agent = _FakeAgent(home, title='Bot Chat')
    agent.run_budget_seconds = None
    config = home / 'config.yaml'
    config.write_text('agent:\n  environment_probe: false\n')
    _apply_agent_section(agent, load_config_readonly())
    assert agent._bot_mode_manager is False
    assert bot_mode_dm.ensure_message_agent_tool(agent)
    ordinary_schema = json.dumps(agent.tools)
    ordinary_protocol = _bot_mode_parts(agent)
    fingerprint = bot_mode_probe.capability_fingerprint(home)
    config.write_text('agent:\n  environment_probe: false\n  bot_mode_manager: true\n')
    # Existing objects retain their session snapshot; no config reload in dispatch.
    assert bot_mode_dm.ensure_message_agent_tool(agent)
    assert json.dumps(agent.tools) == ordinary_schema
    manager = _FakeAgent(home, title='Bot Chat')
    manager.run_budget_seconds = None
    _apply_agent_section(manager, load_config_readonly())
    assert manager._bot_mode_manager is True
    assert bot_mode_dm.ensure_message_agent_tool(manager)
    assert 'assignments' in manager.tools[0]['function']['parameters']['properties']
    assert 'assignments' not in agent.tools[0]['function']['parameters']['properties']
    manager_protocol = _bot_mode_parts(manager)
    assert 'assignments=' in manager_protocol[0]
    assert 'assignments=' not in ordinary_protocol[0]
    assert _bot_mode_parts(manager) == manager_protocol
    assert bot_mode_probe.capability_fingerprint(home) != fingerprint

    launched = []
    def spawn(command, **kwargs):
        launched.append((command, kwargs))
        return json.dumps({'session_id': f'proc-{len(launched)}', 'notify_on_complete': notify})
    monkeypatch.setattr('tools.terminal_tool.terminal_tool', spawn)
    call = INLINE_TOOL_EXECUTORS['message_agent']
    ctx = InlineToolContext(effective_task_id='manager-test')
    assignment = {'target': 'researcher', 'message': 'Read only'}
    group_agent = _FakeAgent(home, title='Group: Team')
    group_agent._bot_mode_manager = True
    for candidate in (agent, group_agent):
        assert 'error' in json.loads(call(candidate, {'assignments': [assignment]}, ctx))
    for args in ({'assignments': None}, {'assignments': None, 'target': 'researcher', 'message': 'no'},
                 {'assignments': [assignment], 'target': '', 'message': ''},
                 {'assignments': []}, {'assignments': [assignment] * 9},
                 {'assignments': [assignment, {'target': 'coder'}]},
                 {'assignments': [assignment], 'target': 'coder', 'message': 'mixed'}):
        assert 'error' in json.loads(call(manager, args, ctx))
    assert not launched
    result = json.loads(call(manager, {'assignments': [assignment, {'target': 'missing', 'message': 'no'}]}, ctx))
    assert result['status'] == 'partial' and result['sent'] == 1
    assert result['results'][0]['result']['status'] == 'queued'
    assert 'error' in result['results'][1]['result']
    receipt = result['results'][0]['result']
    assert receipt['reply_delivery'] == ('notification' if notify else 'poll')
    assert 'before ending' in receipt['detail'].lower()
    if not notify:
        assert receipt['process_id'] in receipt['detail']
    assert len(launched) == 1
    limit = json.loads(call(manager, {'assignments': [assignment] * 8}, ctx))
    assert limit['status'] == 'sent' and limit['sent'] == 8
    assert [entry['index'] for entry in limit['results']] == list(range(8))
    assert len(launched) == 9
    # Captured launches never ran their cleanup-owning runner.
    for command, _ in launched:
        Path(shlex.split(command)[shlex.split(command).index("query-file") + 1]).unlink(missing_ok=True)
    bot_mode_probe._reset_cache_for_tests()


@pytest.mark.linux_only
def test_batch_workers_overlap_through_real_delivery_runners(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = _managed_home(tmp_path, teammates=('researcher', 'coder'))
    monkeypatch.setenv('HERMES_HOME', str(home))
    bot_mode_probe._reset_cache_for_tests()
    agent = _FakeAgent(home, title='Bot Chat')
    agent._bot_mode_manager = True
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    worker = bin_dir / 'hermes'
    worker.write_text(f'''#!{sys.executable}
import os, sys, time
from pathlib import Path
root = Path(os.environ['BATCH_TEST_DIR'])
name = sys.argv[sys.argv.index('-p') + 1]
query = Path(sys.argv[sys.argv.index('--query-file') + 1]).read_text()
(root / (name + '.started')).write_text(query)
deadline = time.monotonic() + 20
while not (root / 'release').exists():
    if time.monotonic() > deadline: raise SystemExit(2)
    time.sleep(0.02)
print(name + ' completed')
''')
    worker.chmod(0o755)
    monkeypatch.setattr("tools.bot_relay._hermes_cli", lambda: str(worker))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SHELL', '/bin/bash')
    # Recreate the installed CLI/import environment inside this isolated shell home.
    (tmp_path / '.bash_profile').write_text(
        f'export PATH={shlex.quote(str(bin_dir))}:$PATH\n'
        f'export PYTHONPATH={shlex.quote(str(Path.cwd()))}\n'
    )
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('BATCH_TEST_DIR', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(Path.cwd()))
    from tools.process_registry import ProcessRegistry
    registry = ProcessRegistry()
    monkeypatch.setattr('tools.process_registry.process_registry', registry)
    monkeypatch.setenv('HERMES_SESSION_KEY', 'manager-overlap')
    processes = []
    try:
        result = json.loads(INLINE_TOOL_EXECUTORS['message_agent'](agent, {'assignments': [
            {'target': 'researcher', 'message': 'research payload'},
            {'target': 'coder', 'message': 'code payload'}
        ]}, InlineToolContext(effective_task_id='overlap-test')))
        assert result['sent'] == 2, json.dumps(result, indent=2)
        processes = [registry.get(entry['result']['process_id']) for entry in result['results']]
        deadline = time.monotonic() + 15
        while not all((tmp_path / f'{name}.started').exists() for name in ('researcher', 'coder')):
            assert time.monotonic() < deadline, [p.output_buffer for p in processes]
            time.sleep(0.02)
        assert all(p is not None and p.notify_on_complete and not p.exited for p in processes)
        assert 'research payload' in (tmp_path / 'researcher.started').read_text()
        assert 'code payload' in (tmp_path / 'coder.started').read_text()
        (tmp_path / 'release').touch()
        deadline = time.monotonic() + 10
        while registry.completion_queue.qsize() < 2:
            assert time.monotonic() < deadline, 'Both workers must produce completion notifications'
            time.sleep(0.02)
        assert registry.drain_notifications('other-manager') == []
        notifications = registry.drain_notifications('manager-overlap')
        assert len(notifications) == 2
        assert {event['session_id'] for event, _ in notifications} == {p.id for p in processes}
        text = '\n'.join(text for _, text in notifications)
        assert 'researcher completed' in text and 'coder completed' in text
        assert registry.drain_notifications('manager-overlap') == []
    finally:
        (tmp_path / 'release').touch()
        for process in processes:
            if not process.exited:
                registry.kill_process(process.id)
        bot_mode_probe._reset_cache_for_tests()


def test_batch_containment_and_failure_continuation(tmp_path, monkeypatch, caplog):
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = _managed_home(tmp_path, teammates=('researcher', 'coder'))
    monkeypatch.setenv('HERMES_HOME', str(home))
    bot_mode_probe._reset_cache_for_tests()
    db = SessionDB(home / 'state.db')
    manager = _FakeAgent(home)
    manager._session_db = db
    manager._bot_mode_manager = True
    db.create_session(manager.session_id, source='cli')
    call = INLINE_TOOL_EXECUTORS['message_agent']
    ctx = InlineToolContext(effective_task_id='failure-test')
    valid = {'target': 'researcher', 'message': 'review'}
    launched = []

    def spawn(command, **kwargs):
        launched.append(command)
        # Fail the middle admission; the last must still be attempted exactly once.
        if len(launched) == 2:
            return json.dumps({'error': 'test startup failure'})
        return json.dumps({'session_id': f'proc-{len(launched)}'})

    monkeypatch.setattr('tools.terminal_tool.terminal_tool', spawn)
    try:
        for title in ('Group: Team', 'ordinary chat'):
            db.set_session_title(manager.session_id, title)
            assert 'error' in json.loads(call(manager, {'assignments': [valid]}, ctx))
        db.set_session_title(manager.session_id, 'Bot Chat')
        for bad in (None, 'wrong', {}, {'target': 1, 'message': 'x'},
                    {'target': 'coder', 'message': ' '},
                    {'target': 'coder', 'message': 'x' * (bot_mode_dm.MESSAGE_MAX_CHARS + 1)},
                    {**valid, 'extra': True}):
            assert 'error' in json.loads(call(manager, {'assignments': [valid, bad]}, ctx))
        assert not launched
        assignments = [valid, {'target': 'coder', 'message': 'review'}, valid]
        result = json.loads(call(manager, {'assignments': assignments}, ctx))
        assert result['status'] == 'partial' and result['sent'] == 2
        assert [entry['index'] for entry in result['results']] == [0, 1, 2]
        assert [entry['target'] for entry in result['results']] == ['researcher', 'coder', 'researcher']
        assert 'failed to start' in result['results'][1]['result']['error']
        assert len(launched) == 3

        # An adapter can fail after side effects; preserve ambiguity and continue,
        # without retrying either the uncertain or successful entry.
        attempted = []
        real_delivery = bot_mode_dm.message_agent_tool
        def uncertain_delivery(**kwargs):
            if 'assignments' in kwargs:
                return real_delivery(**kwargs)
            target = kwargs['target']
            attempted.append(target)
            if len(attempted) == 1:
                raise RuntimeError('private assignment content')
            return json.dumps({'status': 'queued'})
        monkeypatch.setattr(bot_mode_dm, 'message_agent_tool', uncertain_delivery)
        result = json.loads(call(manager, {'assignments': assignments[:2]}, ctx))
        assert attempted == ['researcher', 'coder']
        assert 'Batch acknowledgement unavailable at index 0 (RuntimeError)' in caplog.text
        assert 'private assignment content' not in caplog.text
        assert result['results'][0]['result']['status'] == 'unknown'
        assert result['results'][1]['result']['status'] == 'queued'
    finally:
        for command in launched:
            Path(shlex.split(command)[shlex.split(command).index("query-file") + 1]).unlink(missing_ok=True)
        db.close()
        bot_mode_probe._reset_cache_for_tests()
