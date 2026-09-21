"""Manual interval ownership survives elapsed slots and concurrent operator edits."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from cron import executions, jobs, scheduler


@pytest.fixture(autouse=True)
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)


@pytest.mark.parametrize("slot_kind", ["future", "overdue", "missing", "crossed"])
def test_manual_slot_round_trip_then_regular_execution(tmp_path, monkeypatch, slot_kind):
    start = datetime.now(timezone.utc)
    clock = [start]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    script = tmp_path / "scripts/synthetic.py"
    script.parent.mkdir()
    effects = tmp_path / "effects"
    script.write_text(f"from pathlib import Path\nwith Path({str(effects)!r}).open('a') as f: f.write('effect\\n')\nprint('synthetic')\n")
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(None, "every 6h", script=str(script), no_agent=True, deliver="local",
                              schedule_anchor_ms=int(start.timestamp() * 1000) - 1000)
        slot = {"future": start + timedelta(hours=6), "overdue": start - timedelta(seconds=2),
                "missing": None, "crossed": start + timedelta(seconds=2)}[slot_kind]
        slot = slot.isoformat() if slot else None
        stored = jobs.load_jobs()
        stored[0]["next_run_at"] = slot
        jobs.save_jobs(stored)  # Existing missing state: update_job deliberately repairs missing slots.
        jobs.trigger_job(job["id"])
        clock[0] += timedelta(seconds=1 if slot_kind == "crossed" else 3)
        jobs.trigger_job(job["id"])
        jobs.advance_next_runs([job["id"]])
        claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
        assert claimed["next_run_at"] == slot
        assert "manual_next_run_at" in claimed
        assert claimed["manual_next_run_at"] == slot
        if slot_kind == "crossed":
            # Admission precedes the slot; the controlled completion clock crosses it.
            assert clock[0] < datetime.fromisoformat(slot)
            clock[0] += timedelta(seconds=3)
        # A fresh interpreter reads the nullable ownership and completes the actual script.
        code = (
            "import json,sys;from pathlib import Path;from datetime import datetime\n"
            "from cron import jobs,scheduler,executions\n"
            "jobs._hermes_now=lambda:datetime.fromisoformat(sys.argv[3])\n"
            "scheduler._hermes_home=Path(sys.argv[1])\n"
            "scheduler._launch_external_cron_worker=lambda job:False\n"
            "with jobs.use_cron_store(sys.argv[1]):\n"
            " j=jobs.get_job(sys.argv[2]);ownership=['manual_next_run_at' in j,j.get('manual_next_run_at')]\n"
            " assert scheduler.run_one_job(j)\n"
            " print(json.dumps({'ownership':ownership,'execution_id':j['execution_id']}))\n")
        child = subprocess.run([sys.executable, "-c", code, str(tmp_path), job["id"], clock[0].isoformat()],
                               env=dict(os.environ), capture_output=True, text=True, timeout=8)
        assert child.returncode == 0, child.stderr
        result = json.loads(child.stdout)
        assert result["ownership"] == [True, slot]
        completed = jobs.get_job(job["id"])
        assert completed["next_run_at"] == slot
        assert completed["state"] == "scheduled"
        assert "manual_next_run_at" not in completed
        assert "manual_run_at" not in completed
        row = executions.get_execution(result["execution_id"])
        assert row["status"] == "completed", row.get("error")
        assert effects.read_text().splitlines() == ["effect"]
        # Missing prior slot is repaired on the next scan, not fabricated during manual completion.
        if slot is None:
            assert jobs.get_due_jobs() == []
            slot = jobs.get_job(job["id"])["next_run_at"]
            assert slot is not None
        clock[0] = max(clock[0], datetime.fromisoformat(slot)) + timedelta(seconds=1)
        due = jobs.get_due_jobs()
        assert len(due) == 1 and due[0]["_scheduled_instant"] is not None
        jobs.advance_next_runs([job["id"]])
        regular = jobs.claim_job_for_fire(job["id"], return_job=True)
        assert "manual_next_run_at" not in regular
        assert scheduler.run_one_job(regular)
        assert effects.read_text().splitlines() == ["effect", "effect"]
        final = jobs.get_job(job["id"])
        assert final["next_run_at"] != slot
        assert "manual_next_run_at" not in final


@pytest.mark.parametrize("operator_edit", ["pause", "schedule", "remove_anchor"])
def test_operator_edit_owns_cadence_while_manual_script_finishes(tmp_path, monkeypatch, operator_edit):
    start = datetime.now(timezone.utc)
    clock = [start]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0])
    ready, release = tmp_path / "ready", tmp_path / "release"
    script = tmp_path / "scripts/waiting.py"
    script.parent.mkdir()
    script.write_text(
        "from pathlib import Path\nimport time\n"
        f"Path({str(ready)!r}).touch()\ndeadline=time.monotonic()+8\n"
        f"while not Path({str(release)!r}).exists():\n"
        " if time.monotonic()>deadline: raise TimeoutError('test release missing')\n time.sleep(.02)\n"
        "print('completed once after operator edit')\n")
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(None, "every 6h", script=str(script), no_agent=True, deliver="local",
                              schedule_anchor_ms=int(start.timestamp() * 1000) - 1000)
        jobs.trigger_job(job["id"])
        claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
        results = []
        def run():
            with jobs.use_cron_store(tmp_path):
                results.append(scheduler.run_one_job(claimed))
        thread = threading.Thread(target=run)
        thread.start()
        try:
            deadline = time.monotonic() + 6
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert ready.exists(), "Native script did not reach the synchronization point"
            clock[0] += timedelta(minutes=5)
            if operator_edit == "pause":
                edited = jobs.pause_job(job["id"])
            else:
                schedule = {"kind": "interval", "minutes": 60}
                if operator_edit == "schedule":
                    schedule["anchor_ms"] = int(start.timestamp() * 1000) - 1000
                edited = jobs.update_job(job["id"], {"schedule": schedule})
            expected = {key: edited[key] for key in ("schedule", "next_run_at", "enabled", "state")}
            assert edited["fire_claim"]["by"] == claimed["fire_claim"]["by"]
            assert "manual_next_run_at" not in edited
        finally:
            release.touch()
            thread.join(10)
        assert not thread.is_alive()
        assert results == [True]
        final = jobs.get_job(job["id"])
        assert {key: final[key] for key in expected} == expected
        assert final["last_status"] == "ok"
        assert final["fire_claim"] is None
        row = executions.get_execution(claimed["execution_id"])
        assert row["status"] == "completed", row.get("error")
