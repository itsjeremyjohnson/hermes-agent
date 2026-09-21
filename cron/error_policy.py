"""Opt-in recurring retry timing matched to OpenClaw v2026.9.1.

The classifier regexes below come from the equality-verified public retry-hint.ts.
Agent outcomes retain classification and runner admission before display flattening.
"""
from datetime import timedelta
import re
import json
import logging
import uuid

logger = logging.getLogger(__name__)

POLICY = "openclaw-v2026.9.1-command"
AGENT_POLICY = "openclaw-v2026.9.1-agent"
_BACKOFF = (30, 60, 300, 900, 3600)
_SERVER_ERROR_PATTERN = re.compile('\\b(?:https?|status(?:[ _]code)?|response(?:[ _]code)?|http(?:[ _]status)?)\\b[\\s:=#"\']{0,4}5\\d{2}\\b|\\b5\\d{2}\\b[\\s:)\\].,-]*(?:internal server error|server error|bad gateway|service unavailable|gateway time-?out)\\b|\\binternal server error\\b|\\bbad gateway\\b|\\bservice unavailable\\b|\\bgateway time-?out\\b|\\b5xx\\b|^\\s*5\\d{2}\\s*$', re.IGNORECASE)
_RATE_LIMIT_PATTERN = re.compile('\\b(?:https?(?:\\/\\d(?:\\.\\d)?)?|status(?:[ _-]?code)?|response(?:[ _-]?code)?|http(?:[ _-]?status)?)\\b[\\s:=#"\'(]{0,6}429\\b|\\b(?:provider\\s+)?api[ _-]?error\\b[\\s:=#"\'(]{0,6}429\\b|\\b(?:requested\\s+)?url\\s+returned\\s+error\\b[\\s:=#"\'(]{0,6}429\\b|\\b429\\b[\\s:)\\].,-]*(?:rate[_ -]?limit(?:ed|ing)?(?:[_ -](?:error|exceeded|reached))?|too many requests|resource has been exhausted|quota(?:\\s+(?:exceeded|exhausted|depleted|reached))?)\\b|\\brate[_ -]?limit(?:ed|ing)?(?:[_ -](?:error|exceeded|reached))?\\b|\\btoo many requests\\b|\\bresource has been exhausted\\b|\\btokens per day\\b|^\\s*429\\s*$', re.IGNORECASE)
_SESSION_LIFECYCLE_CLAIM_ERROR_PATTERN = re.compile('^(?:(?:CronSessionLifecycleClaimError|Error): )?Session "[^"\\n]+" (?:changed|was deleted) while starting work\\. Retry\\.$', re.IGNORECASE)
_OVERLOADED = re.compile('^\\s*529(?:\\s*$|[\\s:)\\].,-]*(?:api\\b.*\\bbusy\\b|(?:please\\s+)?try\\s+again\\b))|\\b(?:https?(?:\\/\\d(?:\\.\\d)?)?|status(?:[ _-]?code)?|response(?:[ _-]?code)?|http(?:[ _-]?status)?|(?:provider\\s+)?api[ _-]?error|(?:requested\\s+)?url\\s+returned\\s+error)\\b[\\s:=#"\'(]{0,6}529\\b|\\boverloaded(?:_error)?\\b|high demand|temporar(?:ily|y) overloaded|capacity exceeded', re.IGNORECASE)
_NETWORK = re.compile('(network|fetch failed|socket|econnreset|econnrefused|eai_again|enetdown|ehostunreach|ehostdown|enetreset|enetunreach|epipe)', re.IGNORECASE)
_TIMEOUT = re.compile('(timeout|timed out|stalled before execution start|etimedout)', re.IGNORECASE)


def selected(schedule):
    return isinstance(schedule, dict) and "error_policy" in schedule


def agent_selected(job):
    return (job.get("schedule") or {}).get("error_policy") == AGENT_POLICY


def validate(job):
    schedule = job.get("schedule") or {}
    if not selected(schedule):
        return False
    if schedule["error_policy"] not in (POLICY, AGENT_POLICY):
        raise ValueError("Unknown schedule.error_policy.")
    if agent_selected(job):
        if (job.get("no_agent") or job.get("script") or schedule.get("kind") != "cron"
                or not schedule.get("timezone")):
            raise ValueError("Agent error policy requires an agent-only explicit-zone cron.")
    elif (job.get("no_agent") is not True or not job.get("script")
            or schedule.get("kind") not in {"interval", "cron"}
            or (schedule["kind"] == "interval" and "anchor_ms" not in schedule)
            or (schedule["kind"] == "cron" and not schedule.get("timezone"))):
        raise ValueError("Command error policy requires a no-agent script and an anchored interval or explicit-zone cron.")
    profile = schedule.get("auto_disable_profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("Error policy requires an explicit schedule.auto_disable_profile.")
    from hermes_cli.profiles import validate_profile_name
    validate_profile_name(profile)
    return True


def transient(error, classification=None):
    metadata = classification if isinstance(classification, dict) else {}
    if metadata.get("kind") == "permanent":
        return False
    if not isinstance(error, str) or not error:
        return False
    if metadata.get("session_lifecycle_conflict") or _SESSION_LIFECYCLE_CLAIM_ERROR_PATTERN.search(error):
        return metadata.get("execution_started", True) is not True
    if metadata.get("kind") == "reason":
        return metadata.get("reason") in {"rate_limit", "overloaded", "network", "timeout", "server_error"}
    return any(pattern.search(error) for pattern in (
        _RATE_LIMIT_PATTERN, _OVERLOADED, _NETWORK, _TIMEOUT, _SERVER_ERROR_PATTERN))


def owns_retry(job):
    return (validate(job) and job.get("error_next_run_at") is not None
            and job.get("error_next_run_at") == job.get("next_run_at"))


def disable_after_failures(job, now):
    """Persist the disable fact; notification delivery requires the owning workflow."""
    if (validate(job) and job.get("enabled", True) and job.get("last_status") == "error"
            and job.get("failure_streak", 0) >= 10):
        job.update(enabled=False, state="paused", next_run_at=None, paused_at=now,
                   paused_reason="Auto-disabled after consecutive run failures.")
        job.pop("error_next_run_at", None)
        notification_id = "cron-auto-disabled:" + uuid.uuid4().hex
        job["auto_disabled"] = {"reason": "consecutive-failures", "at": now,
                                "consecutive_errors": job["failure_streak"],
                                "notification_id": notification_id}
        name = " ".join(str(job.get("name") or job["id"]).split())[:120]
        notification = {
            "id": notification_id,
            "job": {"id": job["id"], "name": name,
                    "deliver": "bot-chat:" + job["schedule"]["auto_disable_profile"]},
            "content": (f'Automation "{name}" was auto-disabled after '
                        f'{job["failure_streak"]} consecutive run failures.\n'
                        f'Inspect job {job["id"]} and fix the cause before resuming it.')}
        job.setdefault("auto_disable_pending", []).append(notification)
        return True
    return False


def failure_next(job, now, started_at, previous_failure_streak):
    from cron import jobs, interval_schedule
    current = jobs._parse_aware(now)
    count = job["failure_streak"]
    retryable = count <= 3 and transient(job.get("last_error"), job.get("last_error_classification"))
    schedule = job["schedule"]
    if schedule["kind"] == "interval":
        normal = interval_schedule.next_run(
            schedule, current, None if retryable or previous_failure_streak > 0 else started_at)
    else:
        normal = jobs.compute_next_run(schedule, now)
    if normal is None:
        return None
    backoff = current + timedelta(seconds=_BACKOFF[min(max(count, 1), 5) - 1])
    natural = jobs._parse_aware(normal)
    return (backoff if retryable and backoff < natural else max(natural, backoff)).isoformat()


def enqueue_pending_notifications(job_id=None):
    """Transfer persisted disable events to the existing profile-local durable queue.

    A crash after queue insertion leaves the same event for an idempotent next pass.
    Terminal/unknown queue tombstones prevent replay; queue acceptance is not delivery.
    """
    from cron import delivery_queue, jobs
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    if not jobs._current_cron_store().jobs_file.exists():
        return 0
    try:
        with jobs._jobs_lock():
            snapshots = jobs._peek_jobs_unlocked()
        if snapshots is None:
            raise ValueError("Cannot read pending disable notifications from corrupt job store")
    except Exception:
        logger.exception("Cannot read pending auto-disable notifications")
        return 0
    transferred = 0
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        if job_id is not None and snapshot.get("id") != job_id:
            continue
        for notification in list(snapshot.get("auto_disable_pending") or []):
            try:
                # Shutdown can scope only the jobs store while the launch home
                # remains another profile. The queue belongs to the source store;
                # its explicit Bot Chat recipient is a separate identity.
                token = set_hermes_home_override(jobs._current_cron_store().cron_dir.parent)
                try:
                    queued = delivery_queue.enqueue(
                        notification["id"], notification["job"], notification["content"])
                finally:
                    reset_hermes_home_override(token)
                # Pending rows retain immutable first-writer payloads. A reused ID
                # with altered routing/content must not acknowledge a different event.
                if queued["status"] in {"pending", "delivering"} and (
                        json.loads(queued["job_json"]) != notification["job"]
                        or queued["content"] != notification["content"]):
                    raise ValueError("Disable notification identity has a different queued payload")

                def acknowledge(records, _index, current):
                    pending = current.get("auto_disable_pending") or []
                    if notification not in pending:
                        return False
                    current["auto_disable_pending"] = [item for item in pending if item != notification]
                    jobs.save_jobs(records)
                    return True

                if jobs._with_job(snapshot["id"], acknowledge, False):
                    transferred += 1
            except Exception:
                # The disable already committed. Keep its event for the next
                # gateway queue pass rather than turning completion into a replay.
                logger.exception("Cannot enqueue auto-disable notification for job %s", snapshot["id"])
    return transferred
