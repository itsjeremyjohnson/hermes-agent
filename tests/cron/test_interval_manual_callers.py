"""Anchored manual admission keeps cadence without gaining force permissions."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from cron import executions, jobs, scheduler
from gateway import session_context
from tools import cronjob_tools
from tools.registry import registry


@pytest.fixture
def local_job(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr(scheduler, "_maybe_run_worktree_maintenance", lambda: None)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: None)
    monkeypatch.setattr(session_context, "async_delivery_supported", lambda: False)
    monkeypatch.setattr(cronjob_tools, "_notify_provider_jobs_changed_safe", lambda: None)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    script = tmp_path / "scripts/marker.py"
    script.parent.mkdir()
    marker = tmp_path / "marker"
    script.write_text(
        f"from pathlib import Path\nwith Path({str(marker)!r}).open('a') as f: f.write('effect\\n')\n"
        "print('synthetic')\n")

    def create(slot, anchored=True):
        job = jobs.create_job(
            None, "every 6h", script=str(script), no_agent=True, deliver="local",
            schedule_anchor_ms=int(now.timestamp() * 1000) - 1000 if anchored else None)
        stored = jobs.load_jobs()
        stored[-1]["next_run_at"] = slot
        jobs.save_jobs(stored)
        return job

    with jobs.use_cron_store(tmp_path):
        yield now, marker, create


def run_manual(job_id, route):
    if route == "registry":
        result = json.loads(registry.dispatch("cronjob_manage", {"action": "run", "job_id": job_id}))
        return result["job"]["executed"], result["job"]["execution_success"]
    # Both native sync and background dispatch use this same atomic admission owner.
    claimed, error = cronjob_tools._claim_for_manual_run(job_id, "background run")
    if error is not None:
        return error["claimed"], error["success"]
    result = cronjob_tools._run_claimed_job(claimed)
    return result["claimed"], result["success"]


@pytest.mark.parametrize("route", ["registry", "background_admission"])
@pytest.mark.parametrize("slot_kind", ["future", "missing", "overdue", "legacy"])
def test_manual_callers_preserve_only_anchored_cadence(local_job, route, slot_kind):
    now, marker, create = local_job
    slot = None if slot_kind == "missing" else (
        now + timedelta(seconds=-2 if slot_kind == "overdue" else 3600)).isoformat()
    anchored = slot_kind != "legacy"
    job = create(slot, anchored=anchored)
    for _ in range(2):
        prior = jobs.get_job(job["id"])["next_run_at"]
        assert run_manual(job["id"], route) == (True, True)
        final = jobs.get_job(job["id"])
        assert "manual_next_run_at" not in final
        if anchored:
            assert final["next_run_at"] == slot
        else:
            assert final["next_run_at"] == jobs.compute_next_run(final["schedule"], now.isoformat())
        row = executions.list_executions(job_id=job["id"], limit=1)[0]
        assert row["status"] == "completed"
        assert row["scheduled_instant"] == (None if anchored else prior)
    assert marker.read_text().splitlines() == ["effect", "effect"]


@pytest.mark.parametrize("route", ["registry", "background_admission"])
@pytest.mark.parametrize("gate", ["paused", "live_claim"])
def test_manual_intent_does_not_override_admission(local_job, route, gate):
    now, marker, create = local_job
    job = create((now + timedelta(hours=1)).isoformat())
    if gate == "paused":
        jobs.pause_job(job["id"])
    else:
        assert jobs.claim_job_for_fire(job["id"], return_job=True)
    before = jobs.get_job(job["id"])
    assert run_manual(job["id"], route) == (False, False)
    assert jobs.get_job(job["id"]) == before
    assert not marker.exists()
    assert executions.list_executions(job_id=job["id"]) == []


def test_ticker_manual_identity_survives_restored_completed_slot(local_job):
    now, marker, create = local_job
    slot = (now - timedelta(seconds=2)).isoformat()
    job = create(slot)
    prior = jobs.claim_job_for_fire(job["id"], return_job=True)
    assert scheduler.run_one_job(prior)
    prior_row = executions.get_execution(prior["execution_id"])
    assert prior_row["status"] == "completed"
    assert prior_row["scheduled_instant"] == slot

    def restore_slot():
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = slot
        jobs.save_jobs(stored)

    restore_slot()
    # A completed ordinary occurrence must remain protected against duplicate execution.
    assert jobs.claim_job_for_fire(job["id"], return_job=True) is False
    restore_slot()
    jobs.trigger_job(job["id"])
    assert scheduler.tick(verbose=False, sync=True) == 1
    assert marker.read_text().splitlines() == ["effect", "effect"]
    final = jobs.get_job(job["id"])
    assert final["next_run_at"] == slot
    assert "manual_next_run_at" not in final
    rows = executions.list_executions(job_id=job["id"])
    assert len(rows) == 2
    manual = next(row for row in rows if row["id"] != prior["execution_id"])
    assert manual["status"] == "completed"
    assert manual["scheduled_instant"] is None
