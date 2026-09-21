"""Finite budgets survive cadence edits; based on the independent migration bite probe."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cron import jobs, executions, scheduler


@pytest.mark.parametrize('edit_kind', ['none', 'schedule_only', 'schedule_and_budget'])
def test_schedule_edit_keeps_repeat_budget(tmp_path, monkeypatch, edit_kind):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(scheduler, '_hermes_home', tmp_path)
    monkeypatch.setattr(executions, 'EXECUTIONS_FILE', tmp_path / 'cron/executions.db')
    for name in ('_launch_external_cron_worker', '_maybe_run_worktree_maintenance', '_sweep_mcp_orphans'):
        monkeypatch.setattr(scheduler, name, lambda *args: False)
    clock = [datetime.now(timezone.utc)]
    monkeypatch.setattr(jobs, '_hermes_now', lambda: clock[0])
    script = tmp_path / 'scripts/effect.py'
    script.parent.mkdir()
    effect = tmp_path / 'effect'
    script.write_text(f"from pathlib import Path\nwith Path({str(effect)!r}).open('a') as f: f.write('once\\n')\nprint('done')\n")
    with jobs.use_cron_store(tmp_path):
        created = jobs.create_job(None, 'every 6h', script=str(script), no_agent=True,
                                  deliver='local', repeat=1,
                                  schedule_anchor_ms=int(clock[0].timestamp()*1000)-1000)
        # Use the corrected non-force direct-manual admission, independently of Hound's force path.
        claim = jobs.claim_job_for_fire(created['id'], manual_run=True, return_job=True)
        assert claim
        if edit_kind != 'none':
            updates = {'schedule': {'kind': 'interval', 'minutes': 60,
                                    'anchor_ms': created['schedule']['anchor_ms']}}
            if edit_kind == 'schedule_and_budget':
                updates['repeat'] = 2
            edited = jobs.update_job(created['id'], updates)
            assert edited['repeat']['times'] == (2 if edit_kind == 'schedule_and_budget' else 1)
            assert edited['fire_claim']['by'] == claim['fire_claim']['by']
        assert scheduler.run_one_job(claim)
        completed = jobs.get_job(created['id'])
        if completed['next_run_at']:
            clock[0] = datetime.fromisoformat(completed['next_run_at']) + timedelta(seconds=1)
        tick = scheduler.tick(sync=True, verbose=False)
        final = jobs.get_job(created['id'])
        observed = {'edit_kind': edit_kind, 'after_manual': {k: completed[k] for k in
                    ('repeat', 'state', 'enabled', 'next_run_at')},
                    'after_tick': {k: final[k] for k in ('repeat', 'state', 'enabled', 'next_run_at')},
                    'tick': tick, 'effects': effect.read_text().splitlines(),
                    'statuses': [x['status'] for x in executions.list_executions(job_id=created['id'])]}
        expected = 2 if edit_kind == 'schedule_and_budget' else 1
        assert len(observed['effects']) == expected, json.dumps(observed)
        assert final['repeat']['completed'] == expected
