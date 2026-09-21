"""Busy fire fence is exclusive ownership, not a stolen claim.

PrimeVPS/Blog delivered cards then native last_status became "Interrupted by
shutdown" because heartbeat_fire_claim treated a fire-fence timeout as
ownership loss. A live same-job OS fence prevents claim theft, so fence-busy
must not age into heartbeat grace. Stolen owners still return False.
"""
import threading
import time

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json does not touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _claimed_job():
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="fence-heartbeat")
    assert claim_job_for_fire(job["id"]) is True
    stored = get_job(job["id"])
    assert stored is not None
    return stored


def test_busy_fence_raises_typed_busy_not_false(temp_home, monkeypatch):
    """A live delivery lock must raise FireClaimFenceBusy, not look stolen."""
    import cron.jobs as jobs

    job = _claimed_job()
    owner = job["fire_claim"]["by"]
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.08)
    held = threading.Event()
    release = threading.Event()
    result = {}

    def hold_delivery_lock():
        with jobs.fire_claim_fence(job["id"], expected_owner=owner) as owns:
            result["owns"] = owns
            held.set()
            release.wait(5)

    thread = threading.Thread(target=hold_delivery_lock)
    thread.start()
    assert held.wait(timeout=2)
    assert result["owns"] is True

    with pytest.raises(jobs.FireClaimFenceBusy, match="fire fence busy"):
        jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)

    assert jobs.get_job(job["id"])["fire_claim"]["by"] == owner
    release.set()
    thread.join(timeout=2)
    assert thread.is_alive() is False
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner) is True


def test_stolen_claim_returns_false_when_fence_is_free(temp_home):
    import cron.jobs as jobs

    job = _claimed_job()
    original = job["fire_claim"]["by"]
    assert jobs.heartbeat_fire_claim(job["id"], expected_owner="replacement-owner") is False
    assert jobs.get_job(job["id"])["fire_claim"]["by"] == original


def test_delivery_hold_longer_than_grace_does_not_set_lost(temp_home, monkeypatch):
    """Hold the real delivery fence longer than grace, then steal after release.

    Production constants are heartbeat 60s and grace 180s (3x). Prime delivery
    held the fence ~151s. This test uses the same lock and scheduler loop with
    scaled intervals so the hold exceeds grace.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    job = _claimed_job()
    owner = job["fire_claim"]["by"]
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.08)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.22)
    seen = {}

    def run(lost_ownership):
        with jobs.fire_claim_fence(job["id"], expected_owner=owner) as owns:
            seen["owns"] = owns
            time.sleep(0.45)
            seen["lost_during_hold"] = lost_ownership.is_set()
            seen["owner_during_hold"] = jobs.get_job(job["id"])["fire_claim"]["by"]
        seen["refresh_after_release"] = jobs.heartbeat_fire_claim(
            job["id"], expected_owner=owner)
        records = jobs.load_jobs()
        claim = dict(records[0]["fire_claim"])
        records[0]["fire_claim"] = {"at": claim["at"], "by": "thief"}
        jobs.save_jobs(records)
        seen["lost_after_steal"] = lost_ownership.wait(timeout=2)
        return True

    assert scheduler._run_with_fire_claim_heartbeat(job, run) is True
    assert seen["owns"] is True
    assert seen["lost_during_hold"] is False
    assert seen["owner_during_hold"] == owner
    assert seen["refresh_after_release"] is True
    assert seen["lost_after_steal"] is True
    assert jobs.get_job(job["id"])["fire_claim"]["by"] == "thief"
