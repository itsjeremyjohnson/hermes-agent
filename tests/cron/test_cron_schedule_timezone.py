"""Explicit cron zones preserve UTC cadence without changing profile defaults."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from cron import jobs as J


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(J, "_hermes_now", lambda: datetime(2026, 9, 6, 18, 0, tzinfo=ZoneInfo("America/Chicago")))
    J._cron_cadence_cache.clear()
    with J.use_cron_store(tmp_path):
        yield
    J._cron_cadence_cache.clear()


@pytest.mark.parametrize("instant, legacy_result", [
    ("2026-03-01T00:00:00+00:00", "2026-03-01T23:00:00-06:00"),
    ("2026-03-08T00:00:00+00:00", "2026-03-08T22:00:00-05:00"),
    ("2026-11-01T00:00:00+00:00", "2026-11-02T00:00:00-06:00"),
])
def test_explicit_utc_next_run_ignores_chicago_dst(instant, legacy_result):
    schedule = {"kind": "cron", "expr": "0 23 * * 0", "timezone": "UTC"}
    actual = datetime.fromisoformat(J.compute_next_run(schedule, instant))
    assert actual.weekday() == 6
    assert (actual.hour, actual.minute, actual.utcoffset().total_seconds()) == (23, 0, 0)
    # Preserve the installed croniter's existing profile-default DST behavior;
    # this opt-in change does not repair its separate transition-hour quirk.
    assert J.compute_next_run({"kind": "cron", "expr": schedule["expr"]}, instant) == legacy_result


def test_actual_store_due_tick_keeps_utc_occurrence_and_advances_once():
    job = J.create_job("fixture", "0 23 * * 0", schedule_timezone="UTC", paused=True)
    assert job["next_run_at"] is None and job["enabled"] is False
    assert "UTC" in J.get_job(job["id"])["schedule_display"]
    rows = J.load_jobs()
    rows[0].update(enabled=True, state="scheduled", paused_at=None, next_run_at="2026-09-06T23:00:00+00:00")
    J.save_jobs(rows)
    due = J.get_due_jobs()
    assert [j["id"] for j in due] == [job["id"]]
    assert J.get_job(job["id"])["next_run_at"] == "2026-09-06T23:00:00+00:00"
    J.advance_next_run(job["id"])
    assert J.get_job(job["id"])["next_run_at"] == "2026-09-13T23:00:00+00:00"
    assert J.get_due_jobs() == []


@pytest.mark.parametrize("zone", ["Not/A_Zone", "", None, False, {}])
def test_invalid_explicit_zone_never_falls_back_or_blocks_healthy_sibling(zone):
    schedule = {"kind": "cron", "expr": "0 18 * * 0", "timezone": zone}
    with pytest.raises(ValueError, match="timezone"):
        J.compute_next_run(schedule)
    bad = {"id": "bad", "name": "bad", "prompt": "x", "schedule": schedule,
           "enabled": True, "next_run_at": "2026-09-06T18:00:00-05:00"}
    good = dict(bad, id="good", name="good", schedule={"kind": "cron", "expr": "0 18 * * 0"})
    J.save_jobs([bad, good])
    assert [j["id"] for j in J.get_due_jobs()] == ["good"]
    assert J.get_job("bad")["state"] == "error"
    assert not J.is_job_runnable(bad)
    with pytest.raises(ValueError, match="timezone"):
        J.update_job("good", {"schedule": schedule})
    assert "timezone" not in J.get_job("good")["schedule"]


def test_cadence_cache_separates_effective_zones_and_noncron_is_unchanged(monkeypatch):
    with monkeypatch.context() as clock:
        clock.setattr(J, "_hermes_now", lambda: datetime(2026, 9, 30, 20, tzinfo=ZoneInfo("UTC")))
        # UTC still awaits October1; Tokyo already passed it. Their next
        # monthly gaps are31/30days, even though the expressions are identical.
        for zone, days in (("UTC", 31), ("Asia/Tokyo", 30), ("UTC", 31)):
            assert J._schedule_cadence_seconds({"kind": "cron", "expr": "0 0 1 * *", "timezone": zone}) == days * 86400
    for schedule in ({"kind": "interval", "minutes": 30},
                     {"kind": "once", "run_at": "2026-09-06T23:01:00+00:00"}):
        assert J.compute_next_run(schedule) == J.compute_next_run(dict(schedule, timezone="Not/A_Zone"))


def test_paused_schedule_update_validates_zone_and_displays_it():
    job = J.create_job("fixture", "0 23 * * 0", paused=True)
    updated = J.update_job(job["id"], {"schedule": {"kind": "cron", "expr": "0 23 * * 0", "timezone": "UTC"}})
    assert updated["enabled"] is False and updated["next_run_at"] is None
    assert "UTC" in J.get_job(job["id"])["schedule_display"]


def test_manual_run_keeps_exact_instant_identity_with_explicit_zone():
    job = J.create_job("fixture", "0 1 * * *", schedule_timezone="UTC", paused=True)
    rows = J.load_jobs()
    instant = "2026-09-06T18:00:00-05:00"
    rows[0].update(enabled=True, state="scheduled", paused_at=None, next_run_at=instant, manual_run_at=instant)
    J.save_jobs(rows)
    due = J.get_due_jobs()
    assert [j["id"] for j in due] == [job["id"]]
    assert due[0]["_scheduled_instant"] is None
    assert J.get_job(job["id"])["next_run_at"] == instant
