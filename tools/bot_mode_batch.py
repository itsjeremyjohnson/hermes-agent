"""Opt-in manager batching over the existing Bot Chat delivery path."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_ASSIGNMENTS = 8


def configure_manager_schema(schema: dict) -> None:
    function = schema["function"]
    function["description"] = (
        "Delegate to existing teammates in their canonical Bot Chats. For independent "
        "specialist work, provide assignments (up to eight target/message objects) in "
        "one call; for one message, provide target and message instead. Never mix the "
        "two forms. Compose self-contained assignments within the user's authorized "
        "scope, with distinct outputs and evidence to review. Answer small questions "
        "directly; serialize actual dependencies and conflicting writes. Each delivery "
        "starts in the background and returns its own acknowledgement or error, not "
        "the worker's result. Dispatch all ready work before ending your turn; do not "
        "wait or poll unless a receipt returns reply_delivery=poll; then follow its wait instruction "
        "after dispatching ready work. Review automatic worker returns. Partial failures do not undo "
        "successful sends: never resend the whole batch. Use the live roster and "
        "connection-qualified targets where names are ambiguous."
    )
    parameters = function["parameters"]
    parameters.pop("required")
    parameters["properties"]["assignments"] = {
        "type": "array",
        "description": "Independent assignments to dispatch without waiting for worker completion.",
        "minItems": 1,
        "maxItems": MAX_ASSIGNMENTS,
        "items": {
            "type": "object",
            "properties": {
                "target": dict(parameters["properties"]["target"]),
                "message": dict(parameters["properties"]["message"]),
            },
            "required": ["target", "message"],
            "additionalProperties": False,
        },
    }


def dispatch_batch(assignments: Any, *, mixed_form: bool,
                   task_id: str | None, agent: Any) -> str:
    from tools.bot_mode_dm import MESSAGE_MAX_CHARS, message_agent_authorized, message_agent_tool

    if not message_agent_authorized(agent) or getattr(agent, "_bot_mode_manager", False) is not True:
        return json.dumps({"error": "Batch dispatch requires an enabled manager in a managed Bot Chat."})
    if mixed_form:
        return json.dumps({"error": "Use assignments or target/message, not both. No assignments dispatched."})
    if not isinstance(assignments, list) or not 1 <= len(assignments) <= MAX_ASSIGNMENTS:
        return json.dumps({"error": f"Provide 1–{MAX_ASSIGNMENTS} assignments. No assignments dispatched."})
    # Reject malformed payloads before any side effects. Routing errors remain
    # per-recipient outcomes, so one unavailable specialist cannot block the others.
    for index, assignment in enumerate(assignments):
        invalid = {"error": f"Invalid assignment at index {index}. No assignments dispatched."}
        if not isinstance(assignment, dict) or set(assignment) != {"target", "message"}:
            return json.dumps(invalid)
        if any(not isinstance(assignment[key], str) or not assignment[key].strip()
               for key in ("target", "message")):
            return json.dumps(invalid)
        if len(assignment["message"].strip()) > MESSAGE_MAX_CHARS:
            return json.dumps(invalid)

    results = []
    for index, assignment in enumerate(assignments):
        # The existing handler returns after spawning, not after the worker's
        # final. Sequential admissions preserve profile context without adding
        # a second scheduler or bypassing per-recipient turn locks.
        try:
            result = json.loads(message_agent_tool(**assignment, task_id=task_id, agent=agent))
            if not isinstance(result, dict):
                raise ValueError("Delivery returned a non-object acknowledgement")
        except Exception as exc:
            # Earlier entries may already be running. Keep their receipts and
            # mark this entry ambiguous; never automatically resend it.
            # Exception text can contain private payloads from an adapter.
            logger.warning("Batch acknowledgement unavailable at index %s (%s)", index, type(exc).__name__)
            result = {"status": "unknown", "error": "Dispatch acknowledgement unavailable; check recipient before retrying."}
        results.append({"index": index, "target": assignment["target"], "result": result})
    sent = sum(entry["result"].get("status") == "queued" for entry in results)
    if sent == len(results):
        status = "sent"
    elif sent:
        status = "partial"
    elif any(entry["result"].get("status") in {"unknown", "ambiguous"} for entry in results):
        status = "unknown"
    else:
        status = "failed"
    return json.dumps({
        "status": status,
        "sent": sent,
        "results": results,
        "detail": "Dispatch complete, not worker completion. Follow each receipt's reply instructions. Entries with notification_error will not wake you; inspect their outcome without resending. For reply_delivery=poll, follow the wait instruction. Do not resend successful or ambiguous entries.",
    })
