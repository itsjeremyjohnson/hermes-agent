"""Persist a disable event before owner wake and never replay an uncertain delivery."""
from datetime import datetime, timezone
import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys

import pytest

from cron import delivery_queue, error_policy, jobs, scheduler


@pytest.fixture
def private_store(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", None)
    clock = datetime(2026, 9, 9, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock)
    owner_home = home / "profiles/owner"
    owner_home.mkdir(parents=True)
    (owner_home / ".env").write_text("", encoding="utf-8")
    with jobs.use_cron_store(home):
        yield home, clock


def create_command(clock, owner="owner"):
    job = jobs.create_job(
        None, "every 6h", name="synthetic command", script="synthetic.py", no_agent=True,
        schedule_anchor_ms=int(clock.timestamp() * 1000))
    jobs.update_job(job["id"], {"schedule":{**job["schedule"], "error_policy":error_policy.POLICY,
                                         "auto_disable_profile":owner}, "failure_streak":9})
    return job


def child(home, code):
    result = subprocess.run([sys.executable, "-c", code], env=dict(os.environ),
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_disable_commit_queue_crash_and_fresh_owner_drain(private_store, monkeypatch):
    home, clock = private_store
    job = create_command(clock)
    enqueue = delivery_queue.enqueue

    def crash_after_queue(*args, **kwargs):
        enqueue(*args, **kwargs)
        raise RuntimeError("synthetic crash after queue commit before outbox acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(delivery_queue, "enqueue", crash_after_queue)
        assert jobs.mark_job_run(job["id"], False, "synthetic exit", error_classification={"kind":"permanent"})
    disabled = jobs.get_job(job["id"])
    assert not disabled["enabled"] and disabled["next_run_at"] is None
    event = disabled["auto_disable_pending"][0]
    assert event["id"] == disabled["auto_disabled"]["notification_id"]
    queued = delivery_queue.get_status(event["id"])
    assert queued["status"] == "pending"
    assert json.loads(queued["job_json"])["deliver"] == "bot-chat:owner"
    assert "synthetic exit" not in queued["content"]
    assert child(home, 'from cron import error_policy; import json; print(json.dumps(error_policy.enqueue_pending_notifications()))') == 1
    assert jobs.get_job(job["id"])["auto_disable_pending"] == []
    sent = []

    def send(queued_job, content, **kwargs):
        from cron.scheduler_delivery import _resolve_bot_chat_target
        target = _resolve_bot_chat_target(queued_job, "owner")
        assert target["chat_id"] == "owner"
        sent.append((queued_job, content))
        return None

    monkeypatch.setattr(scheduler, "_deliver_result", send)
    assert scheduler.drain_delivery_queue(None, None) == 1
    assert len(sent) == 1
    assert delivery_queue.get_status(event["id"])["status"] == "delivered"
    # Reappearing pre-ack state is an idempotent replay, including after queue retention.
    monkeypatch.setattr(delivery_queue, "MAX_TERMINAL_DELIVERIES", 0)
    delivery_queue.recover_abandoned()
    jobs.update_job(job["id"], {"auto_disable_pending":[event]})
    assert scheduler.drain_delivery_queue(None, None) == 0
    assert len(sent) == 1
    assert delivery_queue.get_status(event["id"])["status"] == "delivered"
    assert jobs.get_job(job["id"])["auto_disable_pending"] == []
    # Another disable at the same clock instant is a different transition.
    jobs.update_job(job["id"], {"enabled":True, "state":"scheduled", "paused_at":None, "failure_streak":9})
    assert jobs.mark_job_run(job["id"], False, "synthetic exit")
    again = jobs.get_job(job["id"])["auto_disabled"]["notification_id"]
    assert again != event["id"]
    assert scheduler.drain_delivery_queue(None, None) == 1
    assert len(sent) == 2


def test_unknown_owner_wake_is_not_retried_and_route_is_explicit(private_store, monkeypatch):
    home, clock = private_store
    native = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True)
    assert jobs.mark_job_run(native["id"], False, "synthetic error")
    assert not delivery_queue._path().exists()
    before = (home / "cron/jobs.json").read_bytes()
    for invalid in (None, "", "default,owner", "../owner", "OWNER"):
        with pytest.raises(ValueError):
            jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True,
                            schedule_anchor_ms=int(clock.timestamp()*1000),
                            schedule_error_policy=error_policy.POLICY, schedule_auto_disable_profile=invalid)
        assert (home / "cron/jobs.json").read_bytes() == before
    job = jobs.create_job(None, "every 6h", script="synthetic.py", no_agent=True,
                          schedule_anchor_ms=int(clock.timestamp()*1000),
                          schedule_error_policy=error_policy.POLICY, schedule_auto_disable_profile="default")
    jobs.update_job(job["id"], {"failure_streak":9})
    # Leave the durable outbox present, like a producer dying before queue insertion.
    with monkeypatch.context() as patch:
        patch.setattr(delivery_queue, "enqueue", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("queue unavailable")))
        assert jobs.mark_job_run(job["id"], False, "synthetic exit")
    event = jobs.get_job(job["id"])["auto_disable_pending"][0]
    assert child(home, 'from cron import error_policy,delivery_queue; import json; error_policy.enqueue_pending_notifications(); print(json.dumps(delivery_queue.claim_next()["execution_id"]))') == event["id"]
    # The real child PID is now dead. Claim recovery must terminalize unknown, never send again.
    assert delivery_queue.recover_abandoned() == 1
    assert delivery_queue.get_status(event["id"])["status"] == "unknown"
    jobs.update_job(job["id"], {"auto_disable_pending":[event]})
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *args, **kwargs: pytest.fail("uncertain owner wake must never be replayed"))
    assert scheduler.drain_delivery_queue(None, None) == 0
    assert jobs.get_job(job["id"])["auto_disable_pending"] == []


@pytest.mark.parametrize("profile", ["alpha", "beta"])
def test_shutdown_disable_queue_uses_source_profile_and_restores_context(private_store, monkeypatch, profile):
    from hermes_constants import (
        get_hermes_home, get_hermes_home_override,
        set_hermes_home_override, reset_hermes_home_override)

    home, clock = private_store
    source_home = home / "profiles" / profile
    source_home.mkdir()
    with jobs.use_cron_store(source_home):
        job = create_command(clock)
        jobs.update_job(job["id"], {"next_run_at":clock.isoformat()})
        claimed = jobs.claim_job_for_fire(job["id"], return_job=True)
        assert claimed
    token = object()
    owner = claimed["fire_claim"]["by"]
    scheduler._remember_inflight_home(source_home)
    inflight = scheduler._inflight_key(job["id"], source_home)
    monkeypatch.setattr(scheduler, "_running_fire_owners", {inflight:{token:(owner,source_home)}})
    monkeypatch.setattr(scheduler, "_restart_safe_waiter_job_ids", set())
    monkeypatch.setattr(scheduler, "_running_job_ids", {inflight})
    monkeypatch.setattr(scheduler, "_interrupted_job_ids", set())
    original_override = get_hermes_home_override()
    original_env = os.environ["HERMES_HOME"]
    assert scheduler.mark_running_jobs_interrupted(
        "Synthetic shutdown", only_owners={(job["id"],owner)}) == [job["id"]]
    with jobs.use_cron_store(source_home):
        disabled = jobs.get_job(job["id"])
        assert disabled["auto_disabled"]["consecutive_errors"] == 10
        assert not disabled["enabled"] and disabled["auto_disable_pending"] == []
    assert (source_home / "cron/deliveries.db").exists()
    assert not (home / "cron/deliveries.db").exists()
    assert not (home / "profiles/owner/cron/deliveries.db").exists()
    assert get_hermes_home() == home
    assert get_hermes_home_override() == original_override
    assert os.environ["HERMES_HOME"] == original_env
    scope = set_hermes_home_override(source_home)
    try:
        queued = delivery_queue.get_status(disabled["auto_disabled"]["notification_id"])
        assert json.loads(queued["job_json"])["deliver"] == "bot-chat:owner"
    finally:
        reset_hermes_home_override(scope)
    # Failure during insertion also restores the previous context and keeps the source outbox.
    with jobs.use_cron_store(source_home):
        jobs.update_job(job["id"], {"enabled":True, "state":"scheduled", "paused_at":None, "failure_streak":9})
        with monkeypatch.context() as patch:
            def fail_enqueue(*args, **kwargs):
                assert get_hermes_home() == source_home
                raise RuntimeError("synthetic queue failure")
            patch.setattr(delivery_queue, "enqueue", fail_enqueue)
            assert jobs.mark_job_run(job["id"], False, "synthetic error")
        assert len(jobs.get_job(job["id"])["auto_disable_pending"]) == 1
    assert get_hermes_home_override() == original_override
    assert os.environ["HERMES_HOME"] == original_env


def test_failed_notice_tombstone_and_changed_outbox_cannot_replay_or_acknowledge(private_store, monkeypatch):
    from copy import deepcopy

    home, clock = private_store
    job = create_command(clock)
    enqueue = delivery_queue.enqueue

    def leave_pending(*args, **kwargs):
        enqueue(*args, **kwargs)
        raise RuntimeError("synthetic producer failure after queue commit")

    with monkeypatch.context() as patch:
        patch.setattr(delivery_queue, "enqueue", leave_pending)
        assert jobs.mark_job_run(job["id"], False, "synthetic error")
    original = jobs.get_job(job["id"])["auto_disable_pending"][0]
    # Both destination and content are immutable for a queued event ID.
    for field in ("route", "content"):
        changed = deepcopy(original)
        if field == "route":
            changed["job"]["deliver"] = "bot-chat:default"
        else:
            changed["content"] += " altered"
        jobs.update_job(job["id"], {"auto_disable_pending":[changed]})
        assert error_policy.enqueue_pending_notifications() == 0
        assert jobs.get_job(job["id"])["auto_disable_pending"] == [changed]
        queued = delivery_queue.get_status(original["id"])
        assert json.loads(queued["job_json"]) == original["job"]
        assert queued["content"] == original["content"]
    # A writer replacing the outbox after enqueue is not acknowledged as the old snapshot.
    replacement = deepcopy(original)
    replacement["content"] += " replacement"
    jobs.update_job(job["id"], {"auto_disable_pending":[original]})

    def replace_after_enqueue(*args, **kwargs):
        result = enqueue(*args, **kwargs)
        jobs.update_job(job["id"], {"auto_disable_pending":[replacement]})
        return result

    with monkeypatch.context() as patch:
        patch.setattr(delivery_queue, "enqueue", replace_after_enqueue)
        assert error_policy.enqueue_pending_notifications() == 0
    assert jobs.get_job(job["id"])["auto_disable_pending"] == [replacement]
    jobs.update_job(job["id"], {"auto_disable_pending":[original]})
    assert error_policy.enqueue_pending_notifications() == 1
    sent = []

    def fail_send(*args, **kwargs):
        sent.append(args)
        return "synthetic owner transport failed"

    monkeypatch.setattr(scheduler, "_deliver_result", fail_send)
    monkeypatch.setattr(delivery_queue, "MAX_TERMINAL_DELIVERIES", 0)
    assert scheduler.drain_delivery_queue(None, None) == 1
    assert len(sent) == 1
    assert delivery_queue.get_status(original["id"])["status"] == "failed"
    with sqlite3.connect(delivery_queue._path()) as connection:
        assert connection.execute("SELECT terminal_status FROM delivery_tombstones WHERE execution_id=?",
                                  (original["id"],)).fetchone() == ("failed",)
        assert connection.execute("SELECT COUNT(*) FROM deliveries WHERE execution_id=?",
                                  (original["id"],)).fetchone() == (0,)
    jobs.update_job(job["id"], {"auto_disable_pending":[original]})
    assert scheduler.drain_delivery_queue(None, None) == 0
    assert len(sent) == 1
    assert jobs.get_job(job["id"])["auto_disable_pending"] == []
