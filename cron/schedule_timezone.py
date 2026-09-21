"""Explicit cron zones use local calendar fields, not croniter's DST adjustment.

Gap and fold recurrence policy is deliberately unresolved: reject selected local
times that do not identify one real instant, rather than silently choosing one.
Schedules without an explicit zone continue through the existing jobs path.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


class UnsupportedCronLocalTime(ValueError):
    """The next explicit-zone occurrence needs an unspecified gap/fold policy."""


def record_refusal(job: dict, error: UnsupportedCronLocalTime, at: str) -> None:
    """Keep scheduling failure separate from the outcome of an actual execution."""
    import logging

    job.update(next_run_at=None, schedule_error={"at": at, "detail": str(error)})
    if job.get("state") != "paused":
        job["state"] = "error"
    logging.getLogger(__name__).error("Cron job %r schedule error: %s", job.get("id"), error)


def clear_refusal(job: dict) -> None:
    """A newly computed occurrence resolves only this scheduling diagnostic."""
    if job.pop("schedule_error", None) is not None and job.get("state") == "error":
        job["state"] = "scheduled"


def schedule_error_message(job: dict) -> str | None:
    """Render only a valid diagnostic, applying the standard credential redactor."""
    from agent.redact import redact_sensitive_text

    error = job.get("schedule_error")
    detail = error.get("detail") if isinstance(error, dict) else None
    if not isinstance(detail, str) or not detail.strip():
        return None
    return redact_sensitive_text(detail.strip(), force=True, redact_url_credentials=True)


def next_explicit_cron_run(expr: str, base_time: datetime, zone: ZoneInfo) -> datetime:
    """Return the next unique local occurrence strictly after an aware base."""
    from croniter import croniter

    if base_time.tzinfo is None or base_time.utcoffset() is None:
        raise ValueError("Explicit cron timezone requires an aware base instant.")
    local_base = base_time.astimezone(zone)
    wall = croniter(expr, local_base.replace(tzinfo=None)).get_next(datetime)
    candidate = wall.replace(tzinfo=zone, fold=0)
    round_trip = candidate.astimezone(timezone.utc).astimezone(zone)
    if round_trip.replace(tzinfo=None) != wall:
        raise UnsupportedCronLocalTime(
            f"Explicit cron timezone selected nonexistent local time {wall.isoformat()} in {zone}."
        )
    if candidate.utcoffset() != wall.replace(tzinfo=zone, fold=1).utcoffset():
        raise UnsupportedCronLocalTime(
            f"Explicit cron timezone selected ambiguous local time {wall.isoformat()} in {zone}."
        )
    if candidate.timestamp() <= base_time.timestamp():
        raise ValueError("Explicit cron timezone did not produce a future instant.")
    return candidate
