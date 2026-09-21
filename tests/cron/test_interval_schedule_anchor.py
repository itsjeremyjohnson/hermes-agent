"""Anchored intervals preserve elapsed cadence and durable manual schedule ownership."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from cron import jobs, scheduler, executions


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))


def test_anchor_store_validation_recurrence_and_manual_slot(tmp_path, monkeypatch):
    now = [datetime(2026, 9, 8, 12, 0, 0, 372000, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    anchor = int(now[0].timestamp() * 1000)
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True,
                              schedule_anchor_ms=anchor)
        slot = job["next_run_at"]
        assert datetime.fromisoformat(slot) == now[0] + timedelta(hours=6)
        original = (tmp_path / "cron/jobs.json").read_bytes()
        for invalid in (True, -1, 1.5, "1", None, {}, 2**63):
            with pytest.raises(ValueError):
                jobs.update_job(job["id"], {"schedule": {"kind": "interval", "minutes": 360, "anchor_ms": invalid}})
            assert (tmp_path / "cron/jobs.json").read_bytes() == original
        with pytest.raises(ValueError):
            jobs.create_job(None, "0 8 * * *", script="synthetic.py", no_agent=True, schedule_anchor_ms=anchor)
        for invalid in (True, -1, 1.5, "1", {}, 2**63):
            with pytest.raises(ValueError):
                jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True, schedule_anchor_ms=invalid)
            assert (tmp_path / "cron/jobs.json").read_bytes() == original
        for invalid_minutes in ("360", True, None, float("nan"), float("inf"), 0, -1):
            with pytest.raises(ValueError):
                jobs.update_job(job["id"], {"schedule": {"kind": "interval", "minutes": invalid_minutes, "anchor_ms": anchor}})
            assert (tmp_path / "cron/jobs.json").read_bytes() == original
        # The original slot survives repeated run-now requests and pre-run advancement.
        jobs.trigger_job(job["id"])
        now[0] += timedelta(seconds=10)
        jobs.trigger_job(job["id"])
        jobs.advance_next_run(job["id"])
        claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
        assert claimed["next_run_at"] == slot
        started = now[0].isoformat()
        now[0] += timedelta(seconds=45)
        owner = claimed["fire_claim"]["by"]
        assert not jobs.mark_job_run(job["id"], True, expected_fire_owner="stale", execution_started_at=started)
        assert jobs.mark_job_run(job["id"], True, expected_fire_owner=owner, execution_started_at=started)
        final = jobs.get_job(job["id"])
        assert final["next_run_at"] == slot
        assert final["last_started_at"] == started
        assert final["last_run_at"] == now[0].isoformat()
        assert "manual_next_run_at" not in final
        # Force alone preserves the same future slot, including across an interpreter restart.
        claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
        code = (
            "import sys,json; from cron import jobs; from datetime import datetime\n"
            "jobs._hermes_now=lambda: datetime.fromisoformat(sys.argv[3])\n"
            "with jobs.use_cron_store(sys.argv[1]):\n"
            " j=jobs.get_job(sys.argv[2]); jobs.mark_job_run(j['id'],True,expected_fire_owner=j['fire_claim']['by'],execution_started_at=sys.argv[3]); print(json.dumps(jobs.get_job(j['id'])))\n")
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path), job["id"], now[0].isoformat()],
                               capture_output=True, text=True, timeout=5, env=dict(os.environ))
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout)["next_run_at"] == slot
        # Normal completion ignores runtime; native error cadence is explicitly unchanged.
        now[0] += timedelta(hours=6)
        started = now[0].isoformat()
        now[0] += timedelta(seconds=45)
        jobs.mark_job_run(job["id"], True, execution_started_at=started)
        assert datetime.fromisoformat(jobs.get_job(job["id"])["next_run_at"]) == datetime.fromisoformat(started)+timedelta(hours=6)
        jobs.mark_job_run(job["id"], False, "synthetic error", execution_started_at=started)
        assert datetime.fromisoformat(jobs.get_job(job["id"])["next_run_at"]) == now[0]+timedelta(hours=6)
        jobs.mark_job_run(job["id"], True, status="skipped", execution_started_at=started)
        # Opted-in skips now reset the prior error cadence to the strict anchor.
        from cron import interval_schedule
        assert jobs.get_job(job["id"])["next_run_at"] == interval_schedule.next_run(job["schedule"], now[0])
        jobs.trigger_job(job["id"])
        edited = jobs.update_job(job["id"], {"schedule": {"kind": "interval", "minutes": 60}})
        assert "manual_next_run_at" not in edited
        assert "manual_run_at" not in edited
        # Removing an anchor also clears manual intent when no future slot existed to save.
        jobs.update_job(job["id"], {"schedule": {"kind": "interval", "minutes": 360, "anchor_ms": anchor}})
        jobs.update_job(job["id"], {"next_run_at": now[0].isoformat()})
        jobs.trigger_job(job["id"])
        edited = jobs.update_job(job["id"], {"schedule": {"kind": "interval", "minutes": 60}})
        assert "manual_run_at" not in edited
        # A direct persisted malformed anchor is rejected before due dispatch or direct execution.
        raw = jobs.load_jobs(); raw[0]["schedule"]["anchor_ms"] = False
        raw[0]["next_run_at"] = (now[0] - timedelta(seconds=1)).isoformat(); jobs.save_jobs(raw)
        assert jobs.get_due_jobs() == []
        with pytest.raises(ValueError):
            scheduler.run_one_job(raw[0])


def test_anchor_epoch_boundaries_and_legacy_defaults(tmp_path, monkeypatch):
    from cron import interval_schedule
    chicago = ZoneInfo("America/Chicago")
    cases = [datetime(2026, 3, 7, 22, 0, 0, 372000, tzinfo=chicago),
             datetime(2026, 10, 31, 22, 0, 0, 372000, tzinfo=chicago)]
    for start in cases:
        anchor = int(start.timestamp()*1000)
        schedule = {"kind": "interval", "minutes": 360, "anchor_ms": anchor}
        now = datetime.fromtimestamp(start.timestamp()+45, chicago)
        monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
        nxt = datetime.fromisoformat(jobs.compute_next_run(schedule, start.isoformat()))
        assert nxt.timestamp() - start.timestamp() == 21600
        assert nxt.microsecond == 372000
        # Strictly later lattice at an exact boundary; overdue/future-corrupt starts fall back.
        now = datetime.fromtimestamp(start.timestamp()+43200, chicago)
        for last in (start.isoformat(), (now+timedelta(days=1)).isoformat(), None):
            result = datetime.fromisoformat(jobs.compute_next_run(schedule, last))
            assert result.timestamp() == start.timestamp()+64800
        now = datetime.fromtimestamp(start.timestamp()-1, chicago)
        assert datetime.fromisoformat(jobs.compute_next_run(schedule)).timestamp() == start.timestamp()
        # Omitted anchor keeps existing completion-based, wall-clock default behavior.
        legacy = {"kind": "interval", "minutes": 360}
        assert jobs.compute_next_run(legacy, start.isoformat()) == (start+timedelta(hours=6)).isoformat()
        assert interval_schedule.validate(legacy) is None
    with jobs.use_cron_store(tmp_path):
        plain = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True)
        assert "anchor_ms" not in plain["schedule"]
        jobs.mark_job_run(plain["id"], True)
        assert "last_started_at" not in jobs.get_job(plain["id"])


def test_native_script_completion_uses_durable_start(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    script = tmp_path / "scripts/synthetic.py"
    script.parent.mkdir(parents=True)
    script.write_text("import time\ntime.sleep(2.1)\nprint('interval synthetic result')\n")
    with jobs.use_cron_store(tmp_path):
        start_before = datetime.now(timezone.utc)
        job = jobs.create_job(None, "every 6h", script=str(script), no_agent=True,
                              deliver="local", schedule_anchor_ms=int(start_before.timestamp()*1000)-1000)
        claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
        assert scheduler.run_one_job(claimed)
        final = jobs.get_job(job["id"])
        row = executions.get_execution(claimed["execution_id"])
        assert row["status"] == "completed", row.get("error")
        assert final["last_started_at"] == row["started_at"]
        start = datetime.fromisoformat(row["started_at"])
        end = datetime.fromisoformat(final["last_run_at"])
        nxt = datetime.fromisoformat(final["next_run_at"])
        assert end.timestamp() - start.timestamp() >= 2
        assert abs(nxt.timestamp() - start.timestamp() - 21600) < .001
        assert nxt.timestamp() < end.timestamp() + 21600
        # Native script execution through a forced manual dispatch keeps that future slot.
        slot = final["next_run_at"]
        jobs.trigger_job(job["id"])
        manual = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
        assert scheduler.run_one_job(manual)
        assert jobs.get_job(job["id"])["next_run_at"] == slot
        # Finite completion retains the separate start audit before retirement.
        jobs.update_job(job["id"], {"repeat": 1})
        jobs.mark_job_run(job["id"], True, execution_started_at=row["started_at"])
        final = jobs.get_job(job["id"])
        assert final["state"] == "completed"
        assert final["last_started_at"] == row["started_at"]


def test_pause_resume_discards_pending_manual_slot(tmp_path,monkeypatch):
    now=[datetime(2026,9,8,12,16,tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs,'_hermes_now',lambda:now[0])
    with jobs.use_cron_store(tmp_path):
        j=jobs.create_job(None,'every 6h',script='synthetic.py',no_agent=True,schedule_anchor_ms=int(datetime(2026,9,8,tzinfo=timezone.utc).timestamp()*1000))
        jobs.mark_job_run(j['id'],True,execution_started_at='2026-09-08T12:15:00+00:00')
        assert jobs.get_job(j['id'])['next_run_at']=='2026-09-08T18:15:00+00:00'
        now[0]=datetime(2026,9,8,13,tzinfo=timezone.utc)
        jobs.trigger_job(j['id'])
        jobs.pause_job(j['id'])
        resumed=jobs.resume_job(j['id'])
        assert resumed['next_run_at']=='2026-09-08T18:00:00+00:00'
        now[0]=datetime(2026,9,8,18,tzinfo=timezone.utc)
        due=jobs.get_due_jobs()
        assert len(due)==1 and due[0]['_scheduled_instant'] is not None
        jobs.advance_next_runs([j['id']])
        claimed=jobs.claim_job_for_fire(j['id'],return_job=True)
        started=now[0].isoformat()
        now[0]+=timedelta(minutes=1)
        jobs.mark_job_run(j['id'],True,expected_fire_owner=claimed['fire_claim']['by'],execution_started_at=started)
        final=jobs.get_job(j['id'])
        assert final['next_run_at']=='2026-09-09T00:00:00+00:00',final



@pytest.fixture
def outcome_runner(tmp_path, monkeypatch):
    """Native claims, script children and SQLite; only clock and external handoff are controlled."""
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    clock = [datetime(2026, 9, 8, 12, 17, 0, 125000, tzinfo=timezone.utc)]
    for module in (jobs, executions, scheduler):
        monkeypatch.setattr(module, "_hermes_now", lambda: clock[0])
    (tmp_path / "config.yaml").write_text("{}\n")
    script = tmp_path / "scripts/outcome.py"
    script.parent.mkdir()
    effect = tmp_path / "effects"
    duration = [timedelta(minutes=2)]
    real_script = scheduler._run_job_script_with_claim_heartbeat

    def timed_script(*args, **kwargs):
        result = real_script(*args, **kwargs)
        clock[0] += duration[0]
        return result

    monkeypatch.setattr(scheduler, "_run_job_script_with_claim_heartbeat", timed_script)

    def no_agent_allowed(*args, **kwargs):
        raise AssertionError("native wake gate must return before agent construction")

    monkeypatch.setattr(scheduler, "_construct_cron_agent", no_agent_allowed)
    anchor = datetime(2026, 9, 8, microsecond=372000, tzinfo=timezone.utc)

    def create(anchored=True, cooldown=True):
        return jobs.create_job(
            None, "every 6h", script=str(script), no_agent=True, deliver="local",
            schedule_anchor_ms=int(anchor.timestamp() * 1000) if anchored else None,
            failure_alert_cooldown_seconds=21600 if cooldown else None)

    def run(job_id, mode="ok", *, long=False, manual=False):
        current = jobs.get_job(job_id)
        if current.get("last_run_at"):
            # A late actual start stays distinct from the anchor and completion phases.
            clock[0] = datetime.fromisoformat(current["next_run_at"]) + timedelta(minutes=17)
        duration[0] = timedelta(hours=7) if long else timedelta(minutes=2)
        script.write_text(
            f"from pathlib import Path\nwith Path({str(effect)!r}).open('a') as f: f.write({mode!r} + '\\n')\n"
            + {"error": "raise SystemExit('synthetic execution failure')\n",
               "skip": 'print(\'{"wakeAgent": false}\')\n',
               "wake_command": 'print(\'{"wakeAgent": false}\')\n',
               "empty": "",
               "silent": "print('[SILENT]')\n",
               "ok": "print('synthetic success')\n"}[mode])
        jobs.update_job(job_id, {"no_agent": mode != "skip"})
        claim = jobs.claim_job_for_fire(job_id, return_job=True, manual_run=manual)
        assert claim
        assert scheduler.run_one_job(claim)
        final = jobs.get_job(job_id)
        row = executions.get_execution(claim["execution_id"])
        assert row["status"] == ("failed" if mode == "error" else "completed"), row
        assert datetime.fromisoformat(row["finished_at"]) == clock[0]
        assert final["fire_claim"] is None
        assert final["last_run_at"] == clock[0].isoformat()
        if "anchor_ms" in final["schedule"]:
            assert final["last_started_at"] == row["started_at"]
        assert effect.read_text().splitlines()[-1] == mode
        return final, row

    with jobs.use_cron_store(tmp_path):
        yield create, run, clock, effect


def assert_outcome_cadence(final, row, *, reanchor=False, legacy=False):
    start = datetime.fromisoformat(row["started_at"])
    end = datetime.fromisoformat(final["last_run_at"])
    period = timedelta(hours=6)
    if legacy:
        expected = end + period
    else:
        anchor = datetime.fromtimestamp(final["schedule"]["anchor_ms"] / 1000, timezone.utc)
        lattice = anchor + ((end - anchor) // period + 1) * period
        expected = lattice if reanchor or start + period <= end else start + period
        assert lattice != start + period
        assert lattice != end + period
    assert start < end
    assert datetime.fromisoformat(final["next_run_at"]) == expected
    assert expected > end


@pytest.mark.parametrize("mode", ["ok", "skip"])
@pytest.mark.parametrize("prior_error", [False, True])
def test_native_outcome_cadence_and_recovery(outcome_runner, mode, prior_error):
    create, run, _, effect = outcome_runner
    job = create()
    if prior_error:
        failed, row = run(job["id"], "error")
        assert failed["failure_streak"] == 1
        assert_outcome_cadence(failed, row, legacy=True)
    final, row = run(job["id"], mode)
    assert final["last_status"] == ("skipped" if mode == "skip" else "ok")
    assert final["failure_streak"] == 0
    assert_outcome_cadence(final, row, reanchor=prior_error)
    # The first healthy/skip completion clears errors exactly once; later success uses S+P.
    clean, row = run(job["id"])
    assert_outcome_cadence(clean, row)
    assert len(effect.read_text().splitlines()) == (3 if prior_error else 2)


@pytest.mark.parametrize("mode", ["ok", "skip"])
def test_native_long_outcome_returns_strict_future_anchor(outcome_runner, mode):
    create, run, _, _ = outcome_runner
    final, row = run(create()["id"], mode, long=True)
    assert_outcome_cadence(final, row, reanchor=True)


@pytest.mark.parametrize("mode", ["wake_command", "empty", "silent"])
def test_command_silence_remains_success_and_resets_cooldown(outcome_runner, mode):
    from cron import failure_alerts
    create, run, _, _ = outcome_runner
    job = create()
    failed, _ = run(job["id"], "error")
    with failure_alerts.failure_alert_request(job["id"], cooldown_seconds=21600) as audit:
        audit["outcome"] = "failed"
    stamp = jobs.get_job(job["id"])["failure_alert_state"]["requested_at"]
    final, row = run(job["id"], mode)
    assert final["last_status"] == "ok"
    assert final["failure_streak"] == 0
    assert_outcome_cadence(final, row, reanchor=True)
    assert final["failure_alert_state"]["requested_at"] == stamp
    assert final["failure_alert_state"]["cooldown_active"] is False
    assert final["failure_alert_state"]["outcome"] == "reset_success"


def test_native_skip_preserves_requested_cooldown(outcome_runner):
    from cron import failure_alerts
    create, run, _, _ = outcome_runner
    job = create()
    run(job["id"], "error")
    with failure_alerts.failure_alert_request(job["id"], cooldown_seconds=21600) as audit:
        audit["outcome"] = "failed"
    before = jobs.get_job(job["id"])["failure_alert_state"]
    final, row = run(job["id"], "skip")
    assert final["failure_alert_state"]["requested_at"] == before["requested_at"]
    assert final["failure_alert_state"]["cooldown_active"] is True
    assert final["failure_alert_state"]["outcome"] == "suppressed_skipped"
    assert_outcome_cadence(final, row, reanchor=True)


@pytest.mark.parametrize("anchored,cooldown", [(True, False), (False, False), (False, True)])
def test_explicit_false_success_skip_boundary_retains_native_receipt(outcome_runner, anchored, cooldown):
    create, run, clock, _ = outcome_runner
    job = create(anchored=anchored, cooldown=cooldown)
    run(job["id"], "error")
    # Existing completion API can receive a false-success skip (e.g. drift precheck).
    # Classification is supplied here; real failed command above establishes durable errors.
    clock[0] += timedelta(minutes=17)
    claim = jobs.claim_job_for_fire(job["id"], return_job=True)
    execution_id = executions.create_execution(job["id"], source="direct")["id"]
    assert executions.mark_execution_running(execution_id)
    clock[0] += timedelta(minutes=2)
    d = scheduler._RunDelivery(job=claim, success=False, error="synthetic precheck skip", skipped=True)
    before = jobs.get_job(job["id"])
    assert not jobs.mark_job_run(job["id"], False, status="skipped", expected_fire_owner="stale")
    assert jobs.get_job(job["id"]) == before
    assert scheduler._finish_completed_run(d, claim["fire_claim"]["by"], execution_id)
    final = jobs.get_job(job["id"])
    row = executions.get_execution(execution_id)
    assert row["status"] == "failed"
    assert final["last_status"] == "skipped"
    assert final["last_error"] == "synthetic precheck skip"
    assert final["failure_streak"] == (0 if anchored or cooldown else 2)
    assert_outcome_cadence(final, row, reanchor=anchored, legacy=not anchored)
    clean, row = run(job["id"])
    assert_outcome_cadence(clean, row, legacy=not anchored)


def test_delivery_failure_does_not_become_execution_error(outcome_runner, monkeypatch):
    create, run, _, _ = outcome_runner
    job = create()
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *args, **kwargs: "synthetic transport failure")
    final, row = run(job["id"])
    assert final["last_status"] == "delivery_failed"
    assert final["last_delivery_error"] == "synthetic transport failure"
    assert final["failure_streak"] == 0
    assert_outcome_cadence(final, row)
    clean, row = run(job["id"])
    assert_outcome_cadence(clean, row)


@pytest.mark.parametrize("prior_counter", [-1, "bad"])
def test_success_repairs_prior_counter_without_losing_execution(outcome_runner, prior_counter):
    create, run, _, effect = outcome_runner
    job = create()
    jobs.update_job(job["id"], {"failure_streak": prior_counter})
    final, row = run(job["id"])
    assert final["failure_streak"] == 0
    assert final["last_status"] == "ok"
    assert effect.read_text().splitlines() == ["ok"]
    assert_outcome_cadence(final, row)
