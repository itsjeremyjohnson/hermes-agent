"""Opt-in elapsed interval timing and durable manual occurrence ownership.

The anchor is an integer UTC epoch millisecond, never a wall-clock phase. A normal
completed run uses its durable execution start; missed starts use the future lattice.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def validate(schedule):
    """Validate only opted-in schedules; absent anchors preserve legacy semantics."""
    if not isinstance(schedule, dict) or "anchor_ms" not in schedule:
        return None
    anchor = schedule["anchor_ms"]
    if schedule.get("kind") != "interval" or type(anchor) is not int or not 0 <= anchor < 253402300800000:
        raise ValueError("schedule.anchor_ms must be a nonnegative epoch millisecond within datetime range, for intervals only.")
    try:
        raw = schedule.get("minutes")
        if type(raw) not in (int, float):
            raise ValueError
        period = Decimal(str(raw)) * 60000
        if not period.is_finite() or period <= 0 or period != period.to_integral_value():
            raise ValueError
        period = int(period)
        if anchor + period >= 253402300800000:
            raise ValueError
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("Anchored interval minutes must yield a positive whole-millisecond period within datetime range.") from None
    return anchor, period


def _milliseconds(instant):
    delta = instant.astimezone(timezone.utc) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def next_run(schedule, now, started_at=None):
    """Start plus period when future, otherwise strictly later anchor lattice."""
    from cron import jobs
    anchor, period = validate(schedule)
    current = _milliseconds(now)
    start = jobs._parse_aware(started_at) if started_at else None
    start_ms = _milliseconds(start) if start is not None else None
    if start_ms is not None and anchor <= start_ms <= current and start_ms + period > current:
        following = start_ms + period
    else:
        following = anchor if anchor > current else anchor + ((current - anchor) // period + 1) * period
    return (_EPOCH + timedelta(milliseconds=following)).astimezone(now.tzinfo).isoformat()


def _manual_policy(schedule):
    from cron import error_policy
    return validate(schedule) is not None or error_policy.selected(schedule)


def remember_manual_slot(job):
    """Key presence owns the prior slot; JSON null preserves an absent occurrence."""
    if not _manual_policy(job.get("schedule")):
        return
    if "manual_next_run_at" not in job:
        job["manual_next_run_at"] = job.get("next_run_at")


def preserved_slot(job):
    """Return ownership separately from the nullable slot, without a clock cutoff."""
    owned = _manual_policy(job.get("schedule")) and "manual_next_run_at" in job
    return owned, job.get("manual_next_run_at") if owned else None


def record_start(job, now, started_at):
    """Keep valid actual start audit even when a finite run retires the job."""
    from cron import jobs
    if validate(job.get("schedule")) is not None and started_at:
        parsed = jobs._parse_aware(started_at)
        if parsed is not None and parsed.timestamp() <= jobs._parse_aware(now).timestamp():
            job["last_started_at"] = started_at


def completion_next(job, now, started_at, success, *, skipped=False, previous_failure_streak=0):
    """Consume manual ownership once, then apply the explicitly selected outcome policy."""
    from cron import jobs
    schedule = job["schedule"]
    current = jobs._parse_aware(now)
    owned, slot = preserved_slot(job)
    job.pop("manual_next_run_at", None)
    if owned:
        return slot
    from cron import error_policy
    if error_policy.validate(job) and job.get("last_status") == "error":
        return error_policy.failure_next(job, now, started_at, previous_failure_streak)
    if validate(schedule) is None:
        return jobs.compute_next_run(schedule, now)
    if success or skipped:
        # Healthy completion already repairs the counter; stale malformed state
        # must not turn an executed success into a failed completion receipt.
        try:
            had_errors = int(previous_failure_streak or 0) > 0
        except (TypeError, ValueError, OverflowError):
            had_errors = False
        return next_run(schedule, current, None if had_errors else started_at)
    legacy = {key: value for key, value in schedule.items() if key != "anchor_ms"}
    return jobs.compute_next_run(legacy, now)


def preserve_manual_update(job, updated, updates):
    """Capture a manual slot atomically; normal lifecycle edits discard old intent."""
    manual = "manual_run_at" in updates and updates.get("manual_run_at") == updates.get("next_run_at")
    from cron import error_policy
    old_schedule = job.get("schedule") or {}
    new_schedule = updated.get("schedule") or {}
    anchored = _manual_policy(old_schedule) or _manual_policy(new_schedule)
    edited = "schedule" in updates
    lifecycle = not manual and (
        "paused_at" in updates or updates.get("state") == "paused" or updates.get("enabled") is False)
    if (anchored or "manual_next_run_at" in updated) and (edited or lifecycle):
        claim = updated.get("fire_claim")
        if ("manual_next_run_at" in job or error_policy.selected(old_schedule)) and isinstance(claim, dict):
            # Keep execution identity/outcome ownership while transferring cadence
            # to this operator edit, including a later edit back to the old value.
            updated["fire_claim"] = {**claim, "schedule_edited": True}
        if updated.get("manual_run_at") == updated.get("next_run_at"):
            from cron import jobs as cron_jobs
            updated["next_run_at"] = cron_jobs.compute_next_run(updated.get("schedule") or {})
        for key in ("manual_next_run_at", "manual_run_at", "manual_run_prompt", "error_next_run_at"):
            updated.pop(key, None)
        return
    if not manual:
        return
    original = dict(job)
    remember_manual_slot(original)
    if "manual_next_run_at" in original:
        updated["manual_next_run_at"] = original["manual_next_run_at"]
    else:
        updated.pop("manual_next_run_at", None)
