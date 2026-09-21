"""Opt-in failure alerts use durable per-job request cooldowns, independently of incidents."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import contextvars
import json
import os
import subprocess
import sys

from cron import jobs, scheduler, incidents, executions, failure_alerts


def test_durable_request_policy_is_atomic_scoped_and_owner_fenced(tmp_path, monkeypatch):
    now = [datetime(2026, 9, 8, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    home = tmp_path / "profile-a"
    other = tmp_path / "profile-b"
    with jobs.use_cron_store(home):
        job = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True,
                              paused=True, failure_alert_cooldown_seconds=21600)
        assert job["failure_alert_cooldown_seconds"] == 21600
        original = (home / "cron/jobs.json").read_bytes()
        for invalid in (0, -1, True, 1.5, "21600", [], {}, 2**63):
            try:
                jobs.update_job(job["id"], {"failure_alert_cooldown_seconds": invalid})
            except ValueError:
                pass
            else:
                raise AssertionError("invalid policy was accepted")
            assert (home / "cron/jobs.json").read_bytes() == original
            try:
                jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True,
                                paused=True, failure_alert_cooldown_seconds=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid create policy was accepted")
            assert (home / "cron/jobs.json").read_bytes() == original

        def request():
            with failure_alerts.failure_alert_request(job["id"], cooldown_seconds=21600) as audit:
                result = audit["outcome"]
                if result == "requested":
                    # Delivery failure still consumes the request cooldown.
                    audit["outcome"] = "failed"
                return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(contextvars.copy_context().run, request) for _ in range(2)]
            assert sorted(f.result(timeout=5) for f in futures) == ["requested", "suppressed_cooldown"]
        stamp = jobs.get_job(job["id"])["failure_alert_state"]["requested_at"]

        # A fresh interpreter reads the persisted request rather than in-process cache state.
        code = (
            "import json,sys; from datetime import datetime; from cron import jobs, failure_alerts\n"
            "jobs._hermes_now=lambda: datetime.fromisoformat(sys.argv[3])\n"
            "with jobs.use_cron_store(sys.argv[1]):\n"
            " with failure_alerts.failure_alert_request(sys.argv[2],cooldown_seconds=21600) as a:\n"
            "  print(json.dumps(a))\n"
        )
        child = subprocess.run([sys.executable, "-c", code, str(home), job["id"], now[0].isoformat()],
                               capture_output=True, text=True, timeout=5, env=dict(os.environ))
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout)["outcome"] == "suppressed_cooldown"

        with jobs.use_cron_store(other):
            jobs.save_jobs([job])  # Same opaque job ID in a separate profile.
            with failure_alerts.failure_alert_request(job["id"], cooldown_seconds=21600) as audit:
                assert audit["outcome"] == "requested"
                audit["outcome"] = "delivered"

        # Exact boundary permits a new request; a future timestamp also cannot suppress.
        now[0] += timedelta(hours=6)
        assert request() == "requested"
        now[0] -= timedelta(seconds=1)
        assert request() == "requested"
        assert jobs.get_job(job["id"])["failure_alert_state"]["requested_at"] != stamp

        before = (home / "cron/jobs.json").read_bytes()
        jobs.update_job(job["id"], {"fire_claim": {"by": "new-owner"}})
        fenced = (home / "cron/jobs.json").read_bytes()
        for stale_owner in (None, "old-owner"):
            with failure_alerts.failure_alert_request(job["id"], cooldown_seconds=21600,
                                            expected_fire_owner=stale_owner) as audit:
                assert audit is None
            assert (home / "cron/jobs.json").read_bytes() == fenced
        assert fenced != before
        now[0] += timedelta(hours=6)
        with failure_alerts.failure_alert_request(job["id"], cooldown_seconds=21600,
                                        expected_fire_owner="new-owner") as audit:
            assert audit["outcome"] == "requested"
            audit["outcome"] = "delivered"
        jobs.update_job(job["id"], {"fire_claim": None})
        jobs.mark_job_run(job["id"], True, status="skipped")
        assert request() == "suppressed_cooldown"
        jobs.mark_job_run(job["id"], True, delivery_error="synthetic transport failure")
        assert request() == "suppressed_cooldown"
        jobs.mark_job_run(job["id"], True)
        assert jobs.get_job(job["id"])["failure_alert_state"]["outcome"] == "reset_success"
        assert request() == "requested"
        plain = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True, paused=True)
        jobs.mark_job_run(plain["id"], False, status="skipped")
        assert jobs.get_job(plain["id"])["failure_streak"] == 1
        assert "failure_alert_state" not in jobs.get_job(plain["id"])
        # Source cooldowns use elapsed epoch time, including DST transitions and folds.
        chicago = ZoneInfo("America/Chicago")
        cases = [
            (datetime(2026, 3, 7, 22, tzinfo=chicago), datetime(2026, 3, 8, 4, tzinfo=chicago),
             "suppressed_cooldown"),  # Six wall hours, five elapsed.
            (datetime(2026, 3, 7, 22, tzinfo=chicago), datetime(2026, 3, 8, 5, tzinfo=chicago),
             "requested"),  # Exact six elapsed hours.
            (datetime(2026, 10, 31, 22, tzinfo=chicago), datetime(2026, 11, 1, 3, tzinfo=chicago),
             "requested"),  # Five wall hours, six elapsed.
            (datetime(2026, 11, 1, 1, 30, tzinfo=chicago, fold=1),
             datetime(2026, 11, 1, 1, 45, tzinfo=chicago, fold=0), "requested"),  # Future instant.
            (datetime(2026, 11, 1, 1, 45, tzinfo=chicago, fold=0),
             datetime(2026, 11, 1, 1, 30, tzinfo=chicago, fold=1), "suppressed_cooldown"),
        ]
        for last, current, expected in cases:
            now[0] = current
            jobs.update_job(job["id"], {"failure_alert_state": {
                "requested_at": last.isoformat(), "cooldown_active": True, "outcome": "failed"}})
            assert request() == expected
        cleared = jobs.update_job(job["id"], {"failure_alert_cooldown_seconds": None})
        assert cleared.get("failure_alert_cooldown_seconds") is None


def test_scheduler_normal_crash_skip_and_ack_share_request_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    now = [datetime(2026, 9, 8, 12, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(scheduler, "_is_interrupted", lambda *_: False)
    deliveries = []
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True,
                              paused=True, deliver="local", failure_deliver="bot-chat:default",
                              failure_alert_cooldown_seconds=21600)

        def deliver(jb, content, **kwargs):
            state = jobs.get_job(jb["id"])["failure_alert_state"]
            assert state["outcome"] == "requested"  # Durable BEFORE transport.
            assert state["requested_at"] == now[0].isoformat()
            assert kwargs["for_failure"] is True
            deliveries.append(content)
            return "synthetic transport failure"

        monkeypatch.setattr(scheduler, "_deliver_result", deliver)

        def normal(error, success=False, skipped=False):
            live = jobs.get_job(job["id"])
            if skipped:
                live["_failure_alert_skipped"] = True
            execution = executions.create_execution(job["id"], source="direct")
            d = scheduler._RunDelivery(job=live, success=success, error=error)
            scheduler._save_compose_deliver(
                d, scheduler._FireOwnership(live, None), "synthetic result", "synthetic output",
                adapters=None, loop=None, verbose=False, execution_token=None)
            assert scheduler._finish_completed_run(d, None, execution["id"])
            if not success:
                assert executions.get_execution(execution["id"])["status"] == "failed"
                assert jobs.get_job(job["id"])["last_error"] == error
            return d

        first = normal("first failure")
        assert first.delivery_attempted and first.delivery_error
        assert jobs.get_job(job["id"])["failure_alert_state"]["outcome"] == "failed"
        second = normal("different failure signature")
        assert not second.delivery_attempted
        error, outcome = scheduler._deliver_crash_failure(
            jobs.get_job(job["id"]), "third failure signature", adapters=None, loop=None)
        assert error is None and outcome == "suppressed"
        assert jobs.get_job(job["id"])["failure_alert_state"]["outcome"] == "suppressed_cooldown"
        assert len(deliveries) == 1

        now[0] += timedelta(hours=6)
        error, outcome = scheduler._deliver_crash_failure(
            jobs.get_job(job["id"]), "crash after boundary", adapters=None, loop=None)
        assert error and outcome == "failed" and len(deliveries) == 2
        now[0] += timedelta(hours=6)
        skipped = normal("synthetic drift", skipped=True)
        assert not skipped.delivery_attempted
        assert jobs.get_job(job["id"])["failure_alert_state"]["outcome"] == "suppressed_skipped"
        configured = normal(scheduler.BLOCKED_CONFIG_MARKER + " synthetic malformed config")
        assert configured.delivery_attempted  # includeSkipped=false does not mute config errors.

        now[0] += timedelta(hours=6)
        incident_id, _ = incidents.upsert_incident(job["id"], "acknowledged failure")
        incidents.ack_incident(incident_id)
        assert not normal("acknowledged failure").delivery_attempted
        assert jobs.get_job(job["id"])["failure_alert_state"]["outcome"] == "suppressed_acked"

        # An explicit no-change gate is skipped; SILENT alone is not a skip classifier.
        from cron import monitor
        monitored = dict(job, no_agent=False, monitor_url="https://synthetic.invalid")
        monkeypatch.setattr(monitor, "check_monitor", lambda _: SimpleNamespace(ok=True, changed=False))
        scheduler._apply_monitor_gate(monitored, job["id"], "synthetic", None)
        assert monitored["_failure_alert_skipped"] is True
        d = scheduler._RunDelivery(job=monitored, success=True, error=None)
        scheduler._save_compose_deliver(
            d, scheduler._FireOwnership(monitored, None), scheduler.SILENT_MARKER, "synthetic output",
            adapters=None, loop=None, verbose=False, execution_token=None)
        assert d.skipped is True

        # Omitted policy leaves existing every-failure behavior and no new audit state.
        plain = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True, paused=True)
        monkeypatch.setattr(scheduler, "_deliver_result", lambda *a, **k: deliveries.append("legacy"))
        for _ in range(2):
            scheduler._deliver_crash_failure(plain, "legacy failure", adapters=None, loop=None)
        assert deliveries[-2:] == ["legacy", "legacy"]
        assert "failure_alert_state" not in jobs.get_job(plain["id"])
