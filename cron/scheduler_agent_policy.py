"""Selected cron outcome metadata, captured before scheduler display flattening.

Inner-loop retryability is not whole-workflow retry permission. Known reasons
remain authoritative; the native provider classifier is reused for typed errors.
"""
from concurrent.futures import CancelledError

from cron.error_policy import agent_selected, _SESSION_LIFECYCLE_CLAIM_ERROR_PATTERN

_REASON_NAMES = {
    "upstream_rate_limit": "rate_limit",
    "ssl_cert_verification": "tls_certificate",
    "format_error": "format",
}
_KEY = "_agent_error_classification"


def begin(job):
    if agent_selected(job):
        job[_KEY] = {"execution_started": False}


def provider(job, value):
    if agent_selected(job) and isinstance(value, str) and value:
        job[_KEY]["provider"] = value


def terminal(job, reason):
    if agent_selected(job):
        job.setdefault(_KEY, {"execution_started": False}).update(kind="permanent", reason=reason)


def run_agent(agent, prompt, job, task_id):
    # The worker thread entering the selected runner is the admission boundary;
    # claiming a schedule or constructing its agent is not runner admission.
    if agent_selected(job):
        job[_KEY]["execution_started"] = True
        provider(job, getattr(agent, "provider", None))
    return agent.run_conversation(prompt, task_id=task_id)


def exception(job, error, agent=None, *, orchestration=False):
    if not agent_selected(job):
        return
    metadata = job.setdefault(_KEY, {"execution_started": False})
    provider(job, getattr(agent, "provider", None))
    # A later explicit stop replaces an earlier provider failure. Likewise,
    # saving/delivering an already-run outcome is not a provider attempt:
    # local I/O failure must not infer permission to replay the workflow.
    if not isinstance(error, Exception) or isinstance(error, (InterruptedError, CancelledError)):
        terminal(job, "interrupted")
        return
    if orchestration:
        terminal(job, "orchestration_failed")
        return
    # Preserve result metadata when the scheduler formats its RuntimeError.
    if "kind" in metadata:
        return
    if _SESSION_LIFECYCLE_CLAIM_ERROR_PATTERN.search(str(error)):
        metadata["session_lifecycle_conflict"] = True
    from agent.error_classifier import classify_api_error
    classified = classify_api_error(error, provider=metadata.get("provider", ""),
                                    model=getattr(agent, "model", "") or "")
    reason = classified.reason.value
    # An unclassified exception is absent evidence, unlike an explicit unknown
    # reason authored by the agent. Only the former may reach cron text inference.
    if reason != "unknown":
        metadata.update(kind="reason", reason=_REASON_NAMES.get(reason, reason))


def result(job, outcome, agent):
    if not agent_selected(job):
        return
    provider(job, getattr(agent, "provider", None))
    if outcome.get("failed") is not True and outcome.get("completed") is not False:
        return
    metadata = job[_KEY]
    message = str(outcome.get("error") or outcome.get("final_response") or "agent reported failure")
    if _SESSION_LIFECYCLE_CLAIM_ERROR_PATTERN.search(message):
        metadata["session_lifecycle_conflict"] = True
    if outcome.get("interrupted") is True:
        terminal(job, "interrupted")
    elif isinstance(outcome.get("failure_reason"), str) and outcome["failure_reason"]:
        reason = outcome["failure_reason"]
        metadata.update(kind="reason", reason=_REASON_NAMES.get(reason, reason))
    else:
        exception(job, RuntimeError(message), agent)
