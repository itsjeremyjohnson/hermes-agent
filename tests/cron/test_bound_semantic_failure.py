"""Bound semantic failures stay on the normal route and fail the real ledgers.

The agent output is the exact prepared card. Failure accounting is a validated
metadata declaration, not a text prefix and not a failure-lane rewrite.
"""
from cron import delivery_receipts, executions, incidents, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig

CARD = "Bound failure card\nReason: preparation-failed"


def _run(monkeypatch, *, content, metadata, quiet=False, transport_error=None,
         failure_deliver="telegram:failure-lane"):
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    real_load = scheduler.load_config

    def load_config():
        loaded = real_load()
        cron_cfg = loaded.setdefault("cron", {})
        cron_cfg["wrap_response"] = False
        return loaded

    monkeypatch.setattr(scheduler, "load_config", load_config)
    observed = []
    sent = []
    real_deliver = scheduler._deliver_result

    def deliver(job, content, adapters=None, loop=None, *, for_failure=False):
        observed.append({"content": content, "for_failure": for_failure})
        return real_deliver(
            job, content, adapters=adapters, loop=loop, for_failure=for_failure)

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)

    async def send(platform, pconfig, chat_id, text, **kwargs):
        sent.append((str(chat_id), text))
        if transport_error:
            return {"success": False, "error": transport_error}
        return {"success": True, "message_id": "receipt"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)

    def run(job, **kwargs):
        delivery_receipts.prepare(job, content, metadata=metadata, quiet=quiet)
        return (True, "retained raw output", content, None)

    monkeypatch.setattr(scheduler, "run_job", run)
    job = jobs.create_job(
        prompt="fixture only",
        schedule="every 1h",
        deliver="telegram:fixture",
        failure_deliver=failure_deliver,
        name="bound semantic",
    )
    incidents.upsert_incident(job["id"], "earlier outage")
    assert scheduler.run_one_job(job) is True
    saved = jobs.get_job(job["id"])
    execution = executions.get_execution(job["execution_id"])
    history = delivery_receipts.history(job["id"])
    open_incidents = [item for item in incidents.list_incidents() if item["job_id"] == job["id"]]
    return observed, sent, saved, execution, history, open_incidents


def test_bound_semantic_failure_uses_normal_route_and_failed_ledgers(monkeypatch):
    observed, sent, saved, execution, history, open_incidents = _run(
        monkeypatch,
        content=CARD,
        metadata={"kind": "example-failure", "execution_error": "preparation-failed"},
    )
    assert observed == [{"content": CARD, "for_failure": False}]
    assert sent == [("fixture", CARD)]
    assert saved["last_status"] == "error"
    assert saved["last_error"] == "preparation-failed"
    assert saved["failure_streak"] == 1
    assert not saved.get("last_delivery_error")
    assert execution["status"] == "failed"
    assert execution["error"] == "preparation-failed"
    assert execution["delivery_outcome"] == "delivered"
    receipt = history["records"][0]
    assert receipt["outcome"] == "delivered"
    assert receipt["content_sha256"] == delivery_receipts.digest(CARD)
    assert open_incidents[0]["state"] == "detected"


def test_same_card_without_declaration_stays_successful(monkeypatch):
    observed, sent, saved, execution, history, open_incidents = _run(
        monkeypatch,
        content=CARD,
        metadata={"kind": "example-result"},
    )
    assert observed == [{"content": CARD, "for_failure": False}]
    assert sent == [("fixture", CARD)]
    assert saved["last_status"] == "ok"
    assert saved["failure_streak"] == 0
    assert execution["status"] == "completed"
    assert history["records"][0]["outcome"] == "delivered"
    assert open_incidents[0]["state"] == "resolved"


def test_quiet_binding_stays_successful_and_unsent(monkeypatch):
    observed, sent, saved, execution, history, _open = _run(
        monkeypatch,
        content="NO_REPLY",
        metadata={"quiet": True},
        quiet=True,
    )
    assert observed == []
    assert sent == []
    assert saved["last_status"] == "ok"
    assert execution["status"] == "completed"
    assert history["records"][0]["outcome"] == "not_requested"


def test_malformed_execution_error_is_not_inferred_from_the_card(monkeypatch):
    _observed, _sent, saved, execution, history, _open = _run(
        monkeypatch,
        content=CARD,
        metadata={"execution_error": "preparation failed"},
    )
    assert saved["last_status"] == "ok"
    assert execution["status"] == "completed"
    assert history["records"][0]["outcome"] == "delivered"


def test_transport_failure_remains_a_delivery_failure(monkeypatch):
    observed, sent, saved, execution, history, _open = _run(
        monkeypatch,
        content=CARD,
        metadata={"execution_error": "preparation-failed"},
        transport_error="synthetic transport error",
    )
    assert observed == [{"content": CARD, "for_failure": False}]
    assert sent == [("fixture", CARD)]
    assert saved["last_status"] == "error"
    assert saved["last_error"] == "preparation-failed"
    assert "synthetic transport error" in (saved.get("last_delivery_error") or "")
    assert execution["status"] == "failed"
    assert execution["delivery_outcome"] == "failed"
    assert history["records"][0]["outcome"] == "failed"
