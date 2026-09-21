"""A transport's ambiguous outcome must never establish durable delivery."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode, expected", [
    ("inflight_timeout", "unknown"), ("standalone_no_receipt", "unverified"),
    ("standalone_bare_success", "unverified"), ("standalone_confirmed", "delivered"),
    ("live_bare_success", "unverified"), ("live_confirmed", "delivered"),
])
def test_actual_transport_ambiguity_does_not_become_confirmed_delivery(tmp_path, monkeypatch, mode, expected):
    from cron import delivery_receipts as receipts, executions, scheduler_delivery as delivery
    from cron import delivery_queue, scheduler
    import gateway.config
    import agent.async_utils
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr(gateway.config, "load_gateway_config", lambda: None)
    record = executions.create_execution("receipt-confirmation", source="builtin")
    executions.mark_execution_running(record["id"])
    job = {"id": record["job_id"], "execution_id": record["id"], "deliver": "telegram:123"}
    receipts.prepare(job, "synthetic card", metadata={})
    target = SimpleNamespace(job=job, config=None, target_adapters={}, platform="telegram",
        platform_name="telegram", chat_id="123", thread_id=None, loop=None,
        where="telegram:123", is_relay=False, origin_user_id=None, mirror_text="synthetic card",
        mirror_this_target=False, live_adapter_ready=mode.startswith("live_") or mode == "inflight_timeout")
    monkeypatch.setattr(delivery, "_prepare_target_delivery", lambda *args, **kwargs: target)
    monkeypatch.setattr(delivery, "_live_route_metadata", lambda t: (None, {}, {}))
    monkeypatch.setattr(delivery, "_seed_live_delivery_sessions", lambda *args: None)
    monkeypatch.setattr(delivery, "_maybe_mirror_cron_delivery", lambda *args, **kwargs: None)
    class InFlight:
        def result(self, timeout):
            if mode == "live_bare_success":
                return {"success": True}
            if mode == "live_confirmed":
                return {"success": True, "message_id": "synthetic-message"}
            raise TimeoutError("synthetic confirmation timeout")

        def cancel(self):
            return False

    def scheduled(coroutine, loop):
        coroutine.close()
        return InFlight()

    monkeypatch.setattr(agent.async_utils, "safe_schedule_threadsafe", scheduled)
    standalone_result = ({"success": True} if mode == "standalone_bare_success" else
                         {"success": True, "message_id": "synthetic-message"}
                         if mode == "standalone_confirmed" else None)
    monkeypatch.setattr(delivery, "_standalone_send", lambda *args: (standalone_result, None))
    assert delivery._deliver_result(job, "synthetic card") is None
    assert receipts.history(job["id"])["records"][0]["outcome"] == expected


def test_actual_recipient_uses_the_validated_bound_target(tmp_path, monkeypatch):
    from cron import delivery_receipts as receipts, executions, scheduler_delivery as delivery
    from cron import delivery_queue, scheduler
    import gateway.config
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(scheduler, "load_config", lambda: {"cron": {"wrap_response": False}})
    monkeypatch.setattr(gateway.config, "load_gateway_config", lambda: None)
    record = executions.create_execution("receipt-route", source="builtin")
    executions.mark_execution_running(record["id"])
    job = {"id": record["job_id"], "execution_id": record["id"], "deliver": "bot-chat"}
    resolved = []

    def current_route(job, **kwargs):
        target = "wren" if len(resolved) < 2 else "rm-infra"
        resolved.append(target)
        return [{"platform": delivery.BOT_CHAT_PLATFORM, "chat_id": target, "thread_id": None}]

    monkeypatch.setattr(delivery, "_resolve_delivery_targets", current_route)
    receipts.prepare(job, "synthetic card", metadata={})
    recipients = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p: recipients.append(p))
    assert delivery._deliver_result(job, "synthetic card") is None
    assert recipients == ["wren"]
