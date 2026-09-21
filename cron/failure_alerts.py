"""Opt-in per-job failure notification cooldowns and durable request audit.

Job-store primitives own locking and persistence; scheduler delivery owns transport routing.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

from cron import jobs

logger = logging.getLogger("cron.scheduler")


@contextlib.contextmanager
def failure_alert_request(
    job_id: str, *, cooldown_seconds: int, expected_fire_owner: Optional[str] = None,
    suppression: Optional[str] = None,
):
    """Claim a durable notification request and hold the job fence through its transport.

    The yielded audit has outcome ``requested`` or a suppression reason; None means the
    job/fence/owner is unavailable. The sender replaces a requested outcome with its delivery
    result before leaving the context. Failed transport still consumes cooldown. This state
    is per job, independent of incident signatures and their operator acknowledgement gate.
    """
    cooldown = jobs._normalize_script_limit(cooldown_seconds, "failure_alert_cooldown_seconds")
    if cooldown is None:
        raise ValueError("failure_alert_cooldown_seconds must be configured.")

    def owns(job):
        claim = job.get("fire_claim")
        if claim is None:
            return expected_fire_owner is None
        if not isinstance(claim, dict):
            return False
        actual = str(claim.get("by") or "")
        return bool(actual) and actual == expected_fire_owner

    def request(records, _i, job):
        if not owns(job):
            return None
        now = jobs._hermes_now()
        state = job.get("failure_alert_state")
        audit = dict(state) if isinstance(state, dict) else {}
        raw_last = audit.get("requested_at")
        last = jobs._parse_aware(raw_last) if isinstance(raw_last, str) else None
        # datetime subtraction/comparison in one ZoneInfo uses wall time, which is
        # wrong across DST and fold boundaries. Cooldown is elapsed epoch time.
        elapsed = now.timestamp() - last.timestamp() if last is not None else None
        in_cooldown = (
            audit.get("cooldown_active", True) and elapsed is not None
            and 0 <= elapsed < cooldown)
        audit.update(checked_at=now.isoformat(), outcome=(
            suppression or ("suppressed_cooldown" if in_cooldown else "requested")))
        if audit["outcome"] == "requested":
            audit.update(requested_at=now.isoformat(), cooldown_active=True)
        job["failure_alert_state"] = dict(audit)
        jobs.save_jobs(records)
        return audit

    with jobs._fire_job_lock(job_id) as acquired:
        audit = jobs._with_job(job_id, request) if acquired else None
        requested_at = audit.get("requested_at") if audit and audit["outcome"] == "requested" else None
        try:
            yield audit
        except BaseException:
            if requested_at is not None:
                audit["outcome"] = "failed"
            raise
        finally:
            if requested_at is not None:
                def finish(records, _i, job):
                    state = job.get("failure_alert_state")
                    if not owns(job) or not isinstance(state, dict) or state.get("requested_at") != requested_at:
                        return
                    state["outcome"] = audit["outcome"]
                    jobs.save_jobs(records)
                jobs._with_job(job_id, finish)


def _deliver_cooldown_failure(
    job: dict, content: str, *, adapters, loop, suppression: Optional[str] = None,
) -> tuple[Optional[str], str, bool]:
    """Opt-in notification policy shared by normal and crash failures; never changes run status."""
    from cron import scheduler as _sched

    claim = job.get("fire_claim")
    owner = (str(claim.get("by") or "") or None) if isinstance(claim, dict) else None
    attempted = False
    try:
        with failure_alert_request(
            job["id"], cooldown_seconds=job["failure_alert_cooldown_seconds"],
            expected_fire_owner=owner, suppression=suppression,
        ) as audit:
            if audit is None:
                return None, "suppressed", False
            if audit["outcome"] != "requested":
                # Keep the existing monitoring vocabulary; the durable audit retains why.
                outcome = "suppressed_acked" if audit["outcome"] == "suppressed_acked" else "suppressed"
                return None, outcome, False
            attempted = True
            try:
                error = _sched._deliver_result(job, content, adapters=adapters, loop=loop, for_failure=True)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
            unresolved = (
                not error
                and _sched._normalize_deliver_value(_sched._delivery_lane_value(job, for_failure=True)) == "origin"
                and not _sched._resolve_delivery_targets(job, for_failure=True))
            outcome = _sched._classify_delivery_outcome(
                delivery_error=error, should_deliver=True, unresolved_origin=unresolved,
                normalized_deliver=_sched._normalize_deliver_value(_sched._delivery_lane_value(job, for_failure=True)),
                incident_acked=False, success=False)
            audit["outcome"] = outcome
            return error, outcome, True
    except Exception as exc:
        # A failed durable request cannot safely authorize a send. Preserve a visible failure.
        logger.error("Job '%s': failure-alert state unavailable: %s", job.get("id"), exc)
        return "Failure-alert state unavailable", "failed", attempted
