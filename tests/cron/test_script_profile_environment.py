"""Actual script children must use their own multiplex profile environment."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import json
from unittest.mock import Mock

import pytest

from agent import secret_scope as scopes
from cron.scheduler_script import _run_job_script
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('CLICKUP_API_KEY', 'synthetic-default')
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-provider')
    prior = scopes.is_multiplex_active()
    script = "import json,os\nprint(json.dumps({k:os.getenv(k) for k in ['CLICKUP_API_KEY','PROFILE_ONLY','OPENAI_API_KEY','HERMES_HOME']}))\n"
    homes = []
    for name in ('one', 'two'):
        home = tmp_path/name
        (home/'scripts').mkdir(parents=True)
        (home/'scripts/check.py').write_text(script)
        homes.append(home)
    yield homes
    scopes.set_multiplex_active(prior)


def run(home, scope):
    home_token = set_hermes_home_override(str(home))
    secret_token = scopes.set_secret_scope(scope)
    try:
        return _run_job_script('check.py', script_timeout_seconds=5)
    finally:
        scopes.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def owned_execution(home):
    from cron import executions
    token = set_hermes_home_override(str(home))
    try:
        attempt = executions.create_execution('worker-binding-job', source='manual')
        assert executions.mark_execution_running(attempt['id'])
        return attempt['id']
    finally:
        reset_hermes_home_override(token)


def tree_state(root):
    return {
        str(path.relative_to(root)): (
            path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob('*')
        if path.is_file() and not path.name.endswith(('-shm', '-wal'))
    }


@pytest.mark.parametrize('multiplex,scope,key,only', [
    (True, {'CLICKUP_API_KEY':'synthetic-own','OPENAI_API_KEY':'synthetic-provider-own'}, 'synthetic-own', None),
    (True, {}, None, None),
    (True, {'PROFILE_ONLY':'synthetic-scope-only'}, None, 'synthetic-scope-only'),
    (True, None, None, None),
    (False, {'CLICKUP_API_KEY':'synthetic-own'}, 'synthetic-default', None),
])
def test_own_scope_missing_scope_and_legacy_contract(profiles, multiplex, scope, key, only):
    scopes.set_multiplex_active(multiplex)
    success, output = run(profiles[0], scope)
    if multiplex and scope is None:
        assert not success and 'scope' in output.lower()
        return
    assert success, output
    result = json.loads(output)
    assert result == {'CLICKUP_API_KEY':key, 'PROFILE_ONLY':only,
                      'OPENAI_API_KEY':None, 'HERMES_HOME':str(profiles[0])}


def test_simultaneous_script_children_keep_distinct_profile_credentials(profiles):
    scopes.set_multiplex_active(True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(run, home, {'CLICKUP_API_KEY':'synthetic-'+home.name}) for home in profiles]
        for home, future in zip(profiles, calls):
            success, output = future.result(timeout=10)
            assert success, output
            row = json.loads(output)
            assert row['CLICKUP_API_KEY'] == 'synthetic-'+home.name
            assert row['HERMES_HOME'] == str(home)


def test_native_dotenv_globals_do_not_override_deployment_environment(profiles, monkeypatch):
    home = profiles[0]
    (home/'.env').write_text('PATH=/profile-only-bin\nTERMINAL_HOME_MODE=profile\nCLICKUP_API_KEY=synthetic-own\n')
    (home/'home').mkdir()
    (home/'scripts/check.py').write_text(
        "import json,os\nprint(json.dumps({k:os.getenv(k) for k in ['PATH','HOME','CLICKUP_API_KEY']}))\n")
    monkeypatch.setenv('PATH', '/usr/bin:/bin')
    monkeypatch.setenv('TERMINAL_HOME_MODE', 'real')
    monkeypatch.setattr('hermes_cli.env_loader.get_secret_source_values', lambda _: {})
    scopes.set_multiplex_active(True)
    # Use the actual native dotenv loader: it can include global names even
    # though get_secret resolves those names exclusively from the process.
    scope = scopes.build_profile_secret_scope(home)
    success, output = run(home, scope)
    assert success, output
    row = json.loads(output)
    assert '/profile-only-bin' not in row['PATH'].split(':')
    assert '/usr/bin' in row['PATH'].split(':')
    assert row['HOME'] == str(home.parent)
    assert row['CLICKUP_API_KEY'] == 'synthetic-own'


def test_multiplex_script_receives_owned_worker_execution(profiles, monkeypatch):
    home = profiles[0]
    (home/'scripts/check.py').write_text(
        "import json,os\nprint(json.dumps({'worker':os.getenv('_HERMES_CRON_EXTERNAL_WORKER')}))\n")
    execution_id = owned_execution(home)
    monkeypatch.setenv('_HERMES_CRON_EXTERNAL_WORKER', execution_id)
    scopes.set_multiplex_active(True)
    success, output = run(home, {'PROFILE_ONLY': 'synthetic-scope-only'})
    assert success, output
    assert json.loads(output) == {'worker': execution_id}


def test_invalid_worker_marker_fails_before_popen_without_writes(profiles, monkeypatch):
    home = profiles[0]
    execution_id = owned_execution(home)
    marker = 'a' * 31
    monkeypatch.setenv('_HERMES_CRON_EXTERNAL_WORKER', marker)
    popen = Mock(side_effect=AssertionError('must not execute'))
    monkeypatch.setattr('subprocess.Popen', popen)
    before = tree_state(home)
    scopes.set_multiplex_active(True)
    success, output = run(home, {'PROFILE_ONLY': 'synthetic-scope-only'})
    assert not success
    assert 'cron worker execution binding invalid' in output
    popen.assert_not_called()
    assert tree_state(home) == before


def test_foreign_worker_execution_fails_before_popen_without_writes(profiles, monkeypatch):
    home = profiles[0]
    execution_id = owned_execution(home)
    foreign = ('f' * 32) if execution_id != 'f' * 32 else ('e' * 32)
    monkeypatch.setenv('_HERMES_CRON_EXTERNAL_WORKER', foreign)
    popen = Mock(side_effect=AssertionError('must not execute'))
    monkeypatch.setattr('subprocess.Popen', popen)
    before = tree_state(home)
    scopes.set_multiplex_active(True)
    success, output = run(home, {'PROFILE_ONLY': 'synthetic-scope-only'})
    assert not success
    assert 'cron worker execution binding invalid' in output
    popen.assert_not_called()
    assert tree_state(home) == before
