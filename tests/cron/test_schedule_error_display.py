"""Scheduling refusal stays visible without overwriting actual execution history."""
from pathlib import Path


def test_native_job_readback_and_existing_displays_preserve_error_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    from cron import jobs
    from hermes_cli.cron import _job_rows
    from tools.cronjob_job_args import _format_job
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job('fixture', 'every 6h', paused=True)
        jobs.update_job(job['id'], {
            'last_status': 'error', 'last_error': 'Previous execution failed',
            'schedule_error': {'at': '2026-09-08T12:00:00+00:00',
                               'detail': 'Unsupported local time; https://fixture:synthetic-password@example.com/'}})
        stored = jobs.get_job(job['id'])
        tool = _format_job(stored)
        rows = dict(_job_rows(stored))
        assert tool['last_error'] == 'Previous execution failed'
        assert 'Unsupported local time' in tool['schedule_error']
        assert tool['schedule_error'] == rows['Schedule error']
        assert 'synthetic-password' not in tool['schedule_error']
        for value in (None, {}, 'malformed'):
            jobs.update_job(job['id'], {'schedule_error': value})
            stored = jobs.get_job(job['id'])
            assert 'schedule_error' not in _format_job(stored)
            assert 'Schedule error' not in dict(_job_rows(stored))
