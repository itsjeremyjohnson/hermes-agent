"""Selected agent completion through the native scheduler, store and SQLite ledger."""
from concurrent.futures import CancelledError
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

import pytest

from cron import delivery_queue, error_policy, executions, jobs, scheduler


@pytest.fixture
def runner(tmp_path, monkeypatch):
    for name in ("HOME", "USERPROFILE", "HERMES_HOME"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    clock = [datetime(2026, 9, 8, 12, 17, tzinfo=timezone.utc)]
    for module in (jobs, executions, scheduler):
        monkeypatch.setattr(module, "_hermes_now", lambda: clock[0])
    (tmp_path / "config.yaml").write_text("{}\n")
    action = {"result": {"completed":True, "final_response":"complete"}}
    class Agent:
        provider = "resolved-synthetic"
        model = "controlled"
        def run_conversation(self, prompt, task_id):
            self.provider = "fallback-synthetic"
            clock[0] += timedelta(seconds=7)
            if action.get("edit"):
                action["edit"]()
            outcome = action["result"]
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return outcome(self)
            return outcome
        def close(self):
            pass
        def interrupt(self, reason, **kwargs):
            action["released"].set()
    agent = Agent()
    def setup(*args):
        if "pre_error" in action:
            raise action["pre_error"]
        return scheduler._CronAgentSetup(model="controlled", runtime={"provider":"resolved-synthetic"})
    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", setup)
    monkeypatch.setattr(scheduler, "_construct_cron_agent", lambda *a, **k: agent)
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda job: None)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 0)

    def create(selected=True, repeat=None):
        return jobs.create_job("Synthetic controlled cron", "0 8 * * *", model="controlled", deliver="local",
            repeat=repeat, schedule_timezone="America/Chicago",
            **({"schedule_error_policy":error_policy.AGENT_POLICY, "schedule_auto_disable_profile":"owner"} if selected else {}))
    def run(job, outcome, *, manual=False):
        action["result"] = outcome
        if not manual:
            jobs.update_job(job["id"], {"next_run_at":clock[0].isoformat()})
        claimed = jobs.claim_job_for_fire(job["id"], force=manual, return_job=True)
        assert claimed
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            with pytest.raises(type(outcome)):
                scheduler.run_one_job(claimed)
        else:
            assert scheduler.run_one_job(claimed)
        final = jobs.get_job(job["id"])
        ledger = executions.latest_execution(job["id"])
        assert ledger["status"] in {"completed", "failed"}
        assert (ledger["status"] == "completed") == (final["last_status"] == "ok")
        return final
    with jobs.use_cron_store(tmp_path):
        yield create, run, clock, action, tmp_path


def test_agent_metadata_controls_retry_without_text_overrides(runner, monkeypatch):
    create, run, clock, action, home = runner
    for reason, retry in [("upstream_rate_limit",True),("overloaded",True),("timeout",True),
                          ("auth",False),("billing",False),("context_overflow",False),
                          ("ssl_cert_verification",False),("format_error",False),("unknown",False)]:
        final = run(create(), {"failed":True, "failure_reason":reason, "failure_retryable":True,
                               "error":"HTTP status 503 and timeout"})
        metadata = final["last_error_classification"]
        assert metadata["provider"] == "fallback-synthetic" and metadata["execution_started"]
        assert error_policy.transient(final["last_error"], metadata) == retry
        expected = clock[0]+timedelta(seconds=30) if retry else jobs._parse_aware(jobs.compute_next_run(final["schedule"], clock[0].isoformat()))
        assert jobs._parse_aware(final["next_run_at"]) == expected
        assert final["id"] not in {j["id"] for j in jobs.get_due_jobs()}
    for outcome in ({"failed":True,"error":"timeout"}, TimeoutError("unlabelled"), ConnectionError("unlabelled")):
        final = run(create(), outcome)
        assert error_policy.transient(final["last_error"], final["last_error_classification"])
        assert jobs._parse_aware(final["next_run_at"]) == clock[0]+timedelta(seconds=30)
    for outcome in ("timeout", InterruptedError("timeout"), CancelledError("timeout"), {"completed":True,"final_response":""},
                    {"failed":True,"interrupted":True,"error":"timeout"}, KeyboardInterrupt("timeout")):
        final = run(create(), outcome)
        assert final["last_error_classification"]["kind"] == "permanent"
        assert not error_policy.transient(final["last_error"], final["last_error_classification"])
    class ProviderError(Exception):
        status_code = 429
        def __init__(self, message):
            super().__init__(message)
            self.body = {"error":{"message":message}}
    for message, reason in (("Rate limit exceeded", "rate_limit"),
                            ("Your credit balance is too low", "billing")):
        final = run(create(), ProviderError(message))
        assert final["last_error_classification"]["reason"] == reason
        assert error_policy.transient(final["last_error"], final["last_error_classification"]) == (reason == "rate_limit")
    # Drive the real watchdog expiry/interrupt branch; the worker is released by
    # its real cancellation call, not by replacing the watchdog with TimeoutError.
    entered = threading.Event()
    action["released"] = threading.Event()
    def stalled(agent):
        entered.set()
        assert action["released"].wait(5)
        return {"completed":False,"interrupted":True}
    def expire(**kwargs):
        assert entered.wait(5)
        return True
    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 1)
        patch.setattr(scheduler, "_inactivity_watchdog_loop", expire)
        native_wait = scheduler.concurrent.futures.wait
        patch.setattr(scheduler.concurrent.futures, "wait", lambda fs, timeout: native_wait(fs, timeout=0.01))
        final = run(create(), stalled)
    assert action["released"].is_set()
    assert final["last_error_classification"]["reason"] == "timeout"
    assert jobs._parse_aware(final["next_run_at"]) == clock[0]+timedelta(seconds=30)
    # A control signal after a transient result wins over the earlier reason.
    for control in (KeyboardInterrupt, InterruptedError, CancelledError):
        with monkeypatch.context() as patch:
            def stop_output(*args, **kwargs):
                raise control("synthetic stop after provider result")
            patch.setattr(scheduler, "save_job_output", stop_output)
            late = create()
            jobs.update_job(late["id"], {"next_run_at":clock[0].isoformat()})
            claimed = jobs.claim_job_for_fire(late["id"], return_job=True)
            action["result"] = {"failed":True,"failure_reason":"rate_limit","error":"throttle"}
            if issubclass(control, Exception):
                assert scheduler.run_one_job(claimed) is False
            else:
                with pytest.raises(control):
                    scheduler.run_one_job(claimed)
            final = jobs.get_job(late["id"])
            assert final["last_error_classification"]["reason"] == "interrupted"
            assert not error_policy.transient(final["last_error"], final["last_error_classification"])
            assert jobs._parse_aware(final["next_run_at"]) > clock[0]+timedelta(seconds=30)
    # Exercise actual filesystem failure after successful or failed agent work.
    # Both must record the outer failure without granting a provider retry.
    for outcome in ({"completed":True,"final_response":"workflow finished"},
                    {"failed":True,"failure_reason":"rate_limit","error":"throttle"}):
        late = create()
        jobs.update_job(late["id"], {"next_run_at":clock[0].isoformat()})
        claimed = jobs.claim_job_for_fire(late["id"], return_job=True)
        target = jobs._job_output_dir(late["id"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stale file blocks output directory")
        action["result"] = outcome
        assert scheduler.run_one_job(claimed) is False
        final = jobs.get_job(late["id"])
        assert executions.latest_execution(late["id"])["status"] == "failed"
        assert final["last_error_classification"]["reason"] == "orchestration_failed"
        assert not error_policy.transient(final["last_error"], final["last_error_classification"])
        assert jobs._parse_aware(final["next_run_at"]) > clock[0]+timedelta(seconds=30)
    conflict = 'Session "synthetic" changed while starting work. Retry.' 
    action["pre_error"] = RuntimeError(conflict)
    final = run(create(), {})
    assert final["last_error_classification"]["execution_started"] is False
    assert jobs._parse_aware(final["next_run_at"]) == clock[0]+timedelta(seconds=30)
    del action["pre_error"]
    final = run(create(), RuntimeError(conflict))
    assert final["last_error_classification"]["execution_started"] is True
    assert not error_policy.transient(final["last_error"], final["last_error_classification"])


def test_agent_ownership_disable_queue_and_native_defaults(runner, monkeypatch):
    create, run, clock, action, home = runner
    outcome = {"failed":True,"failure_reason":"rate_limit","error":"synthetic throttle"}
    job = create()
    for delay in (30,60,300):
        final = run(job, outcome)
        assert jobs._parse_aware(final["next_run_at"]) == clock[0]+timedelta(seconds=delay)
    jobs.update_job(job["id"], {"failure_streak":9})
    slot = final["next_run_at"]
    final = run(job, outcome, manual=True)
    assert final["enabled"] and final["next_run_at"] == slot
    final = run(job, outcome)
    assert not final["enabled"] and final["next_run_at"] is None
    notification = delivery_queue.get_status(final["auto_disabled"]["notification_id"])
    assert notification["status"] == "pending" and 'bot-chat:owner' in notification["job_json"]
    recovering = create()
    final = run(recovering, outcome)
    assert final["failure_streak"] == 1
    final = run(recovering, {"completed":True,"final_response":"finished"})
    assert final["failure_streak"] == 0 and final["last_error_classification"] is None
    assert "error_next_run_at" not in final
    assert jobs._parse_aware(final["next_run_at"]) == jobs._parse_aware(
        jobs.compute_next_run(final["schedule"], clock[0].isoformat()))
    edited = create()
    jobs.update_job(edited["id"], {"failure_streak":9})
    action["edit"] = lambda: jobs.update_job(edited["id"], {"schedule":{**edited["schedule"],"expr":"0 9 * * *"}})
    final = run(edited, outcome)
    assert final["enabled"] and final["schedule"]["expr"] == "0 9 * * *"
    del action["edit"]
    final = run(create(selected=False), outcome)
    assert "last_error_classification" not in final
    assert final["next_run_at"] == jobs.compute_next_run(final["schedule"], clock[0].isoformat())
    final = run(create(repeat=1), outcome)
    assert final["state"] == "completed" and final["repeat"]["completed"] == 1
    for changes in ({"no_agent":True}, {"script":"unselected.py"},
                    {"schedule":{**edited["schedule"], "kind":"interval","minutes":60,"anchor_ms":0}}):
        with pytest.raises(ValueError):
            jobs.update_job(edited["id"], changes)

    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    for profile in ("alpha", "beta"):
        source_home = home / "profiles" / profile
        source_home.mkdir(parents=True)
        (source_home / "config.yaml").write_text("{}\n")
        token = set_hermes_home_override(source_home)
        try:
            with jobs.use_cron_store(source_home):
                isolated = create()
                jobs.update_job(isolated["id"], {"failure_streak":9})
                final = run(isolated, outcome)
                assert not final["enabled"] and not final["auto_disable_pending"]
                event_id = final["auto_disabled"]["notification_id"]
                assert delivery_queue.get_status(event_id)["status"] == "pending"
                assert final["last_error_classification"]["provider"] == "fallback-synthetic"
        finally:
            reset_hermes_home_override(token)
        assert jobs.get_job(isolated["id"]) is None
        assert delivery_queue.get_status(event_id) is None
