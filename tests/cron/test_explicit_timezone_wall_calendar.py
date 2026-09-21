"""Explicit cron zones preserve the earliest real wall-calendar occurrence."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from cron import jobs as J
from cron.schedule_timezone import next_explicit_cron_run


@pytest.mark.parametrize("expr, base, expected", [
    ("0 8 * * 1-5", "2026-03-08T06:00:00+00:00", "2026-03-09T08:00:00-05:00"),
    ("0 8 * * 1-5", "2026-11-01T05:00:00+00:00", "2026-11-02T08:00:00-06:00"),
    ("0 7 2 * *", "2026-03-08T06:00:00+00:00", "2026-04-02T07:00:00-05:00"),
    ("0 7 2 * *", "2026-11-01T05:00:00+00:00", "2026-11-02T07:00:00-06:00"),
    ("0 9-17 * * 1-5", "2026-11-01T05:00:00+00:00", "2026-11-02T09:00:00-06:00"),
])
def test_next_occurrence_preserves_earliest_wall_time(expr, base, expected):
    actual = next_explicit_cron_run(expr, datetime.fromisoformat(base), ZoneInfo("America/Chicago"))
    assert actual.isoformat() == expected
    assert actual.timestamp() > datetime.fromisoformat(base).timestamp()
    assert actual.astimezone(timezone.utc).astimezone(actual.tzinfo) == actual


@pytest.mark.parametrize("expr, base, reason", [
    ("30 2 * * *", "2026-03-08T06:00:00+00:00", "nonexistent"),
    ("30 1 * * *", "2026-11-01T05:00:00+00:00", "ambiguous"),
])
def test_unresolved_transition_policy_fails_visibly(expr, base, reason):
    with pytest.raises(ValueError, match=reason):
        next_explicit_cron_run(expr, datetime.fromisoformat(base), ZoneInfo("America/Chicago"))
    with pytest.raises(ValueError, match=reason):
        J.compute_next_run({"kind": "cron", "expr": expr, "timezone": "America/Chicago"}, base)


@pytest.mark.parametrize("explicit_first", [False, True])
def test_cadence_cache_separates_explicit_policy_from_profile_default(monkeypatch, explicit_first):
    base = datetime(2026, 3, 8, tzinfo=ZoneInfo("America/Chicago"))
    monkeypatch.setattr(J, "_hermes_now", lambda: base)
    legacy = {"kind": "cron", "expr": "0 8 * * 1-5"}
    explicit = dict(legacy, timezone="America/Chicago")
    ordered = [(legacy, 3600), (explicit, 86400)]
    if explicit_first:
        ordered.reverse()
    J._cron_cadence_cache.clear()
    try:
        for schedule, expected in ordered + ordered:
            assert J._schedule_cadence_seconds(schedule) == expected
    finally:
        J._cron_cadence_cache.clear()


@pytest.mark.parametrize("base, first, second, cadence", [
    ("2026-03-07T06:00:00+00:00", "2026-03-07T08:00:00-06:00", "2026-03-08T08:00:00-05:00", 23 * 3600),
    ("2026-10-31T05:00:00+00:00", "2026-10-31T08:00:00-05:00", "2026-11-01T08:00:00-06:00", 25 * 3600),
])
def test_real_store_due_advance_and_cadence_across_dst(tmp_path, monkeypatch, base, first, second, cadence):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    clock = [datetime.fromisoformat(base).astimezone(ZoneInfo("Asia/Tokyo"))]
    monkeypatch.setattr(J, "_hermes_now", lambda: clock[0])
    J._cron_cadence_cache.clear()
    try:
        with J.use_cron_store(tmp_path):
            job = J.create_job("fixture", "0 8 * * *", schedule_timezone="America/Chicago")
            assert J.get_job(job["id"])["next_run_at"] == first
            assert J._schedule_cadence_seconds(job["schedule"]) == cadence
            assert J.get_due_jobs() == []
            clock[0] = datetime.fromisoformat(first).astimezone(ZoneInfo("Asia/Tokyo"))
            assert [row["id"] for row in J.get_due_jobs()] == [job["id"]]
            assert J._cron_next_run_matches_expr(job["schedule"], clock[0])
            J.advance_next_run(job["id"])
            assert J.get_job(job["id"])["next_run_at"] == second
            assert J.get_due_jobs() == []
    finally:
        J._cron_cadence_cache.clear()


def test_refused_next_occurrence_does_not_abort_healthy_tick(tmp_path, monkeypatch):
    from cron import executions as E, scheduler as S

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(S, "_hermes_home", tmp_path)
    monkeypatch.setattr(E, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    for name in ("_maybe_reap_dead_owners", "_maybe_run_worktree_maintenance", "_sweep_mcp_orphans"):
        monkeypatch.setattr(S, name, lambda: None)
    monkeypatch.setattr(S, "_should_yield_tick_to_fresh_gateway", lambda: None)
    monkeypatch.setattr(S, "_launch_external_cron_worker", lambda job: False)
    script = tmp_path / "scripts" / "synthetic.py"
    script.parent.mkdir()
    script.write_text("print('synthetic healthy execution')\n")
    now = [datetime(2026, 3, 7, tzinfo=timezone.utc)]
    monkeypatch.setattr(J, "_hermes_now", lambda: now[0])
    with J.use_cron_store(tmp_path):
        gap = J.create_job(None, "30 2 * * *", script=str(script), no_agent=True,
                           deliver="local", schedule_timezone="America/Chicago")
        healthy = J.create_job(None, "every 6h", script=str(script), no_agent=True, deliver="local")
        now[0] = datetime(2026, 3, 7, 8, 30, tzinfo=timezone.utc)
        J.trigger_job(healthy["id"])
        assert {j["id"] for j in J.get_due_jobs()} == {gap["id"], healthy["id"]}
        S.tick(verbose=False)
        assert J.get_job(healthy["id"])["last_status"] == "ok", J.get_job(healthy["id"])["last_error"]
        refused = J.get_job(gap["id"])
        assert refused["enabled"] is True and refused["state"] == "error"
        assert refused["next_run_at"] is None and refused["last_run_at"] is None
        assert not refused.get("fire_claim")
        assert refused["repeat"]["completed"] == 0
        assert "nonexistent" in refused["schedule_error"]["detail"]
        assert J.claim_job_for_fire(gap["id"], return_job=True) is False
        assert not J.get_job(gap["id"]).get("fire_claim")
        listed = next(j for j in J.list_jobs(include_disabled=True) if j["id"] == gap["id"])
        assert listed["schedule_error"] == J.get_job(gap["id"])["schedule_error"]


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("paused", [False, True])
def test_completion_preserves_actual_outcome_when_next_becomes_unsupported(tmp_path, monkeypatch, success, paused):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    now = [datetime(2026, 3, 7, tzinfo=timezone.utc)]
    monkeypatch.setattr(J, "_hermes_now", lambda: now[0])
    with J.use_cron_store(tmp_path):
        job = J.create_job("fixture", "30 2 * * *", schedule_timezone="America/Chicago")
        # The claim has a valid future occurrence; the run crosses that occurrence
        # before finishing, so only completion discovers the following day's gap.
        now[0] = datetime(2026, 3, 7, 8, 29, tzinfo=timezone.utc)
        claimed = J.claim_job_for_fire(job["id"], return_job=True)
        owner = claimed["fire_claim"]["by"]
        now[0] = datetime(2026, 3, 7, 8, 31, tzinfo=timezone.utc)
        assert not J.mark_job_run(job["id"], success, expected_fire_owner="stale-owner")
        assert J.get_job(job["id"])["fire_claim"]["by"] == owner
        if paused:
            J.pause_job(job["id"])
        actual_error = None if success else "synthetic execution failure"
        assert J.mark_job_run(job["id"], success, error=actual_error, expected_fire_owner=owner)
        final = J.get_job(job["id"])
        assert final["last_run_at"] == now[0].isoformat()
        assert final["last_status"] == ("ok" if success else "error")
        assert final["last_error"] == actual_error
        assert final["repeat"]["completed"] == 1 and final["fire_claim"] is None
        assert final["enabled"] is (not paused)
        assert final["state"] == ("paused" if paused else "error")
        assert final["next_run_at"] is None and "nonexistent" in final["schedule_error"]["detail"]
        now[0] = datetime(2026, 3, 8, 10, tzinfo=timezone.utc)
        assert J.get_due_jobs() == []
        if paused:
            assert J.get_job(job["id"])["state"] == "paused"
            J.resume_job(job["id"])
        recovered = J.get_job(job["id"])
        assert recovered["next_run_at"] == "2026-03-09T02:30:00-05:00"
        assert recovered["state"] == "scheduled" and not recovered.get("schedule_error")
        assert recovered["last_status"] == final["last_status"]
        assert recovered["last_error"] == actual_error and recovered["repeat"]["completed"] == 1
