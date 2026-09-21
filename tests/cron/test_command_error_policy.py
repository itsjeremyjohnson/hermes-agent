"""Actual command/store/SQLite outcomes for the opt-in source retry schedule."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cron import error_policy, executions, interval_schedule, jobs, scheduler


@pytest.fixture
def runner(tmp_path, monkeypatch):
    for name in ("HOME", "USERPROFILE", "HERMES_HOME"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    clock = [datetime(2026, 9, 8, 12, 17, tzinfo=timezone.utc)]
    for module in (jobs, executions, scheduler):
        monkeypatch.setattr(module, "_hermes_now", lambda: clock[0])
    (tmp_path / "config.yaml").write_text("{}\n")
    script = tmp_path / "scripts/retry.py"
    script.parent.mkdir()
    effect = tmp_path / "effects"
    native_script = scheduler._run_job_script_with_claim_heartbeat

    def advance_clock(*args, **kwargs):
        result = native_script(*args, **kwargs)
        clock[0] += timedelta(seconds=7)
        return result

    monkeypatch.setattr(scheduler, "_run_job_script_with_claim_heartbeat", advance_clock)

    def create(*, calendar=False, selected=True, repeat=None):
        job = jobs.create_job(
            None, "0 8 * * *" if calendar else "every 6h", script=str(script), no_agent=True,
            repeat=repeat, deliver="local", schedule_timezone="America/Chicago" if calendar else None,
            schedule_anchor_ms=None if calendar else int(datetime(2026, 9, 8, tzinfo=timezone.utc).timestamp()*1000),
            )
        if selected:
            job = jobs.update_job(job["id"], {"schedule":{**job["schedule"], "error_policy":error_policy.POLICY, "auto_disable_profile":"default"}})
        return job

    def run(job, error=None, *, manual=False):
        script.write_text(
            f"from pathlib import Path\nwith Path({str(effect)!r}).open('a') as f: f.write('ran\\n')\n"
            + ("import time; time.sleep(10)\n" if error == "timeout" else
               f"raise SystemExit({error!r})\n" if error else "print('complete')\n"))
        current = jobs.get_job(job["id"])
        jobs.update_job(job["id"], {"script_timeout_seconds": 1})
        if not manual:
            jobs.update_job(job["id"], {"next_run_at": clock[0].isoformat()})
        claimed = jobs.claim_job_for_fire(job["id"], force=manual, return_job=True)
        assert claimed
        started = clock[0]
        assert scheduler.run_one_job(claimed)
        final = jobs.get_job(job["id"])
        assert final["last_status"] == ("error" if error else "ok")
        return final, started, clock[0]

    with jobs.use_cron_store(tmp_path):
        yield create, run, clock, effect


def test_native_retry_sequence_manual_ownership_and_disable(runner):
    create, run, clock, effect = runner
    job = create()
    for count, delay in ((1,30), (2,60), (3,300)):
        if count == 2:
            jobs.update_job(job["id"], {"script_max_output_bytes":1024})
        final, started, end = run(job, "timeout")
        assert final["failure_streak"] == count
        assert final["last_error_classification"] == {"kind":"reason", "reason":"timeout"}
        assert datetime.fromisoformat(final["next_run_at"]) == end + timedelta(seconds=delay)
        # No early due admission; the future retry survives normal native scans.
        assert jobs.get_due_jobs() == []
        clock[0] = datetime.fromisoformat(final["next_run_at"])
    final, started, end = run(job)
    assert final["failure_streak"] == 0
    assert final["next_run_at"] == interval_schedule.next_run(final["schedule"], end)
    for count in range(1,10):
        final, started, end = run(job, "HTTP status 503 is incidental stderr")
        assert final["enabled"]
        assert final["last_error_classification"] == {"kind":"permanent"}
        expected = interval_schedule.next_run(final["schedule"], end, started.isoformat() if count == 1 else None)
        assert datetime.fromisoformat(final["next_run_at"]) == max(
            datetime.fromisoformat(expected), end + timedelta(seconds=(30,60,300,900,3600)[min(count,5)-1]))
        clock[0] = end + timedelta(seconds=1)
    # Manual tenth error is out of band and cannot consume or disable its regular slot.
    slot = final["next_run_at"]
    final, _, _ = run(job, "HTTP status 503", manual=True)
    assert final["enabled"] and final["next_run_at"] == slot
    final, _, end = run(job, "timeout")
    assert not final["enabled"] and final["next_run_at"] is None
    assert final["auto_disabled"]["reason"] == "consecutive-failures"
    assert final["auto_disabled"]["at"] == end.isoformat()
    assert final["auto_disabled"]["consecutive_errors"] == 11
    assert final["auto_disabled"]["notification_id"]
    assert jobs.get_due_jobs() == []


def test_calendar_bound_default_behavior_validation_and_finite_budget(runner):
    create, run, clock, _ = runner
    calendar = create(calendar=True)
    final, _, end = run(calendar, "timeout")
    assert datetime.fromisoformat(final["next_run_at"]) == end + timedelta(seconds=30)
    assert jobs.get_due_jobs() == []
    clock[0] = datetime.fromisoformat(final["next_run_at"])
    assert calendar["id"] in {row["id"] for row in jobs.get_due_jobs()}
    # A 30-second transient retry beyond the natural occurrence falls back to the same floor.
    clock[0] = datetime(2026,9,9,12,59,45,tzinfo=timezone.utc)
    jobs.update_job(calendar["id"], {"failure_streak":0})
    final, _, end = run(calendar, "timeout")
    assert datetime.fromisoformat(final["next_run_at"]) == end + timedelta(seconds=30)
    jobs.update_job(calendar["id"], {"failure_streak":9})
    slot = final["next_run_at"]
    final, _, _ = run(calendar, "HTTP status 503", manual=True)
    assert final["enabled"] and final["next_run_at"] == slot
    native = create(selected=False)
    final, _, end = run(native, "HTTP status 503")
    assert datetime.fromisoformat(final["next_run_at"]) == end + timedelta(hours=6)
    finite = create(repeat=1)
    final, _, _ = run(finite, "HTTP status 503")
    assert final["state"] == "completed" and final["repeat"]["completed"] == 1
    assert not final["enabled"] and final["next_run_at"] is None
    original = jobs.get_job(native["id"])
    for policy in (None, False, {}, "typo"):
        with pytest.raises(ValueError):
            jobs.update_job(native["id"], {"schedule":{**original["schedule"], "error_policy":policy}})
        assert jobs.get_job(native["id"]) == original
    with pytest.raises(ValueError):
        jobs.update_job(calendar["id"], {"no_agent":False})
    # A short interval's one-hour backoff must survive native stale-error recovery.
    short = create()
    short = jobs.update_job(short["id"], {"schedule":{**short["schedule"], "minutes":1}, "failure_streak":4})
    final, _, end = run(short, "permanent synthetic failure")
    expected_retry = end + timedelta(hours=1)
    assert datetime.fromisoformat(final["next_run_at"]) == expected_retry
    clock[0] = end + timedelta(minutes=5)
    assert short["id"] not in {row["id"] for row in jobs.get_due_jobs()}
    assert datetime.fromisoformat(jobs.get_job(short["id"])["next_run_at"]) == expected_retry
    # A regular in-flight operator edit owns cadence and enablement at the tenth failure.
    edited = create()
    jobs.update_job(edited["id"], {"failure_streak":9, "next_run_at":clock[0].isoformat()})
    claimed = jobs.claim_job_for_fire(edited["id"], return_job=True)
    changed = jobs.update_job(edited["id"], {"schedule":{**edited["schedule"], "minutes":60}})
    assert jobs.mark_job_run(edited["id"], False, "synthetic exit", expected_fire_owner=claimed["fire_claim"]["by"])
    final = jobs.get_job(edited["id"])
    assert final["enabled"] and final["next_run_at"] == changed["next_run_at"]
    for message in ("context limit 512 exceeded", "pid 511 killed", "exited with 503 lines"):
        assert not error_policy.transient(message)
    for message in ("HTTP status 503", "503", "network disconnected", "status 429", "gateway timeout"):
        assert error_policy.transient(message)
