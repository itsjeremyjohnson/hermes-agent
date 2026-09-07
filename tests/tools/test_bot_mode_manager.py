"""Manager opt-in and batch admission through the real inline tool boundary."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tests.tools.test_bot_mode_dm import _FakeAgent, _managed_home
from tools import bot_mode_dm, bot_mode_probe


def test_manager_config_schema_protocol_and_batch_containment(tmp_path, monkeypatch):
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
        return json.dumps({'session_id': f'proc-{len(launched)}'})
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
    assert result['results'][0]['result']['status'] == 'sent'
    assert 'error' in result['results'][1]['result']
    assert 'before ending' in result['results'][0]['result']['detail']
    assert len(launched) == 1
    # Captured launches never ran their cleanup-owning runner.
    for command, _ in launched:
        Path(shlex.split(command)[4]).unlink(missing_ok=True)
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
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('BATCH_TEST_DIR', str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(Path.cwd()))
    processes = []
    def spawn(command, **kwargs):
        assert kwargs['background'] and kwargs['notify_on_complete'] and kwargs['_host_local']
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(process)
        return json.dumps({'session_id': f'proc-{process.pid}'})
    monkeypatch.setattr('tools.terminal_tool.terminal_tool', spawn)
    try:
        result = json.loads(INLINE_TOOL_EXECUTORS['message_agent'](agent, {'assignments': [
            {'target': 'researcher', 'message': 'research payload'},
            {'target': 'coder', 'message': 'code payload'}
        ]}, InlineToolContext(effective_task_id='overlap-test')))
        assert result['sent'] == 2
        deadline = time.monotonic() + 15
        while not all((tmp_path / f'{name}.started').exists() for name in ('researcher', 'coder')):
            assert time.monotonic() < deadline, 'Both workers must start before release'
            time.sleep(0.02)
        assert all(p.poll() is None for p in processes)
        assert 'research payload' in (tmp_path / 'researcher.started').read_text()
        assert 'code payload' in (tmp_path / 'coder.started').read_text()
        (tmp_path / 'release').touch()
        for process in processes:
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0, error
            assert 'completed' in output
    finally:
        (tmp_path / 'release').touch()
        for process in processes:
            if process.poll() is None:
                process.communicate(timeout=10)
        bot_mode_probe._reset_cache_for_tests()
