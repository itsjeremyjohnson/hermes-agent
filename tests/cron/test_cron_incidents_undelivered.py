"""A failure ping that never arrived is reported once on recovery, never closed silently (JOH-125).

A cron failure's notice could fail (send error, or a deferred Bot Chat turn ending ``ambiguous``)
and the next green run then resolved the incident with ``alerted_at`` null: the operator was never
told and the ledger showed nothing wrong.
"""
import pytest

from cron import bot_chat_delivery as queue
from cron import executions, incidents, jobs, scheduler
from cron import scheduler_delivery as delivery


def _run(job_id, *, success, error=None):
    live = jobs.get_job(job_id)
    execution = executions.create_execution(job_id, source="direct")
    d = scheduler._RunDelivery(job=live, success=success, error=error)
    scheduler._save_compose_deliver(
        d, scheduler._FireOwnership(live, None), "report", "output",
        adapters=None, loop=None, verbose=False, execution_token=None)
    assert scheduler._finish_completed_run(d, None, execution["id"])


@pytest.mark.parametrize("first_send_error", ["synthetic transport failure", None])
def test_recovery_reports_a_failure_alert_that_never_arrived(tmp_path, monkeypatch, first_send_error):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "_is_interrupted", lambda *_: False)
    sends = []

    def deliver(job, content, **kwargs):
        sends.append((str(job.get("execution_id") or ""), content, kwargs["for_failure"]))
        return first_send_error if len(sends) == 1 else None

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True, paused=True,
                              name="Blog Portfolio", deliver="local", failure_deliver="bot-chat:default")
        _run(job["id"], success=False, error="provider 503")
        (incident,) = incidents.list_incidents()
        assert bool(incident["undelivered_at"]) is bool(first_send_error)

        _run(job["id"], success=True)
        _run(job["id"], success=True)

    incident = incidents.get_incident(incident["id"])
    assert incident["state"] == "resolved"
    digests = [s for s in sends[1:] if s[2]]
    if first_send_error:
        # One fresh failure-lane notice under its own delivery id, never a replay of the lost one.
        (digest,) = digests
        assert digest[0].endswith(":undelivered-digest")
        assert "never delivered" in digest[1] and "provider 503" in digest[1]
        assert incident["alerted_at"] and incident["undelivered_at"] is None
    else:
        assert digests == []


@pytest.mark.parametrize("error, delivered", [("SESSION_NOT_OWNED", False), (None, True)])
def test_drain_records_whether_a_deferred_failure_notice_arrived(tmp_path, monkeypatch, error, delivered):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    incident_id, _ = incidents.upsert_incident("job", "provider 503")
    queue.defer("c" * 64, {"id": "job", "_failure_incident_ids": [incident_id]}, "failed", "", tmp_path,
                for_failure=True)
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda *a, **kw: error)

    queue.drain()

    incident = incidents.get_incident(incident_id)
    assert queue.read_pending("c" * 64)["status"] == ("settled" if delivered else "ambiguous")
    assert bool(incident["alerted_at"]) is delivered
    assert bool(incident["undelivered_at"]) is not delivered
    assert incident["state"] == ("alerted" if delivered else "detected")
