"""Actual native ledger/queue/delivery hook; recipient transport is synthetic."""
from pathlib import Path
import sqlite3
import json

import pytest

from cron import delivery_receipts as R, delivery_queue as Q, executions as E
from cron import scheduler_delivery as D, scheduler as S


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for key in ("HOME", "USERPROFILE", "HERMES_HOME"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(Q, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    monkeypatch.setattr(E, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    monkeypatch.setattr(S, "load_config", lambda: {"cron": {"wrap_response": False}})
    import gateway.config
    monkeypatch.setattr(gateway.config, "load_gateway_config", lambda: None)


def job():
    execution = E.create_execution("synthetic-job", source="builtin")
    E.mark_execution_running(execution["id"])
    return {"id": "synthetic-job", "execution_id": execution["id"], "deliver": "bot-chat"}


def test_actual_deferred_queue_does_not_become_delivered_until_drain(monkeypatch):
    item = job()
    original_start = E.get_execution(item["execution_id"])["started_at"]
    R.prepare(item, "bound card", metadata={"version": 1})
    monkeypatch.setattr(Q, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", item["execution_id"])
    assert D._deliver_result(item, "bound card") is None
    assert Q.get_status(item["execution_id"])["status"] == "pending"
    assert R.history(item["id"])["records"][0]["outcome"] == "prepared"
    E.finish_execution(item["execution_id"], success=True)
    assert R.history(item["id"])["records"][0]["outcome"] == "prepared"
    # Normal execution retention is independent of still-pending delivery.
    monkeypatch.setattr(E, "MAX_TERMINAL_EXECUTIONS", 1)
    later = job(); E.finish_execution(later["execution_id"], success=True)
    assert E.get_execution(item["execution_id"]) is None
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER")
    seen = []
    monkeypatch.setattr(D, "_deliver_to_bot_chat", lambda j, c, p, **kwargs: seen.append(c))
    assert Q.drain(lambda j, c, f: D._deliver_result(j, c, for_failure=f)) == 1
    assert seen == ["bound card"]
    queued = Q.get_status(item["execution_id"])
    assert queued["status"] == "delivered" and queued["content"] == "" and queued["job_json"] == "{}"
    receipt = R.history(item["id"])["records"][0]
    assert receipt["outcome"] == "delivered" and receipt["content_sha256"] == R.digest("bound card")
    assert receipt["started_at"] == original_start
    assert json.loads(receipt["metadata_json"]) == {"version": 1}
    with pytest.raises(ValueError, match="no replay"):
        D._deliver_result(item, "bound card")
    assert seen == ["bound card"]


def test_immutable_binding_and_distinct_outcomes(monkeypatch):
    seen = []
    monkeypatch.setattr(D, "_deliver_to_bot_chat", lambda j, c, p, **kwargs: seen.append(c) or "synthetic failed send")
    item = job()
    R.prepare(item, "one", metadata={})
    for changed in (dict(item, deliver="local"), dict(item, id="foreign-job")):
        with pytest.raises(ValueError):
            D._deliver_result(changed, "one")
    with pytest.raises(ValueError):
        D._deliver_result(item, "two")
    with pytest.raises(ValueError, match="immutable"):
        R.prepare(item, "one", metadata={"changed": True})
    assert seen == []
    # Late run failure owns a different summary; do not block the native alert
    # or claim that sending it delivered the prepared normal card.
    assert D._deliver_result(item, "failure summary", for_failure=True) == "synthetic failed send"
    assert seen == ["failure summary"]
    assert R._lookup(item["execution_id"])["outcome"] == "prepared"
    assert D._deliver_result(item, "one") == "synthetic failed send"
    assert R._lookup(item["execution_id"])["outcome"] == "failed"
    for verify, raises, expected in [(False, False, "unverified"), (False, True, "unknown"), (True, False, "delivered")]:
        item = job()
        R.prepare(item, "synthetic", metadata={})
        @R.track_delivery
        def send(j, c, **kwargs):
            if raises:
                raise RuntimeError("synthetic interrupted transport")
            R.note_verification(verify)
        if raises:
            with pytest.raises(RuntimeError):
                send(item, "synthetic")
        else:
            assert send(item, "synthetic") is None
        assert R._lookup(item["execution_id"])["outcome"] == expected


def test_history_retention_unknown_owners_and_unbound_defaults(tmp_path, monkeypatch):
    assert R.history("synthetic-job")["status"] == "unavailable"
    assert not Q._path().exists()
    calls = []
    monkeypatch.setattr(D, "_deliver_to_bot_chat", lambda j, c, p, **kwargs: calls.append(c))
    assert D._deliver_result({"id": "unbound", "deliver": "bot-chat"}, "plain") is None
    assert calls == ["plain"] and not Q._path().exists()
    item = job()
    R.prepare(item, "lost", metadata={})
    with sqlite3.connect(Q._path()) as conn:
        conn.execute("UPDATE delivery_receipts SET outcome='sending',owner_pid=999999999,owner_started_at=1")
    assert R.history(item["id"])["records"][0]["outcome"] == "unknown"
    assert R._lookup(item["execution_id"])["outcome"] == "sending"  # read-only history
    with pytest.raises(ValueError):
        D._deliver_result(item, "lost")
    monkeypatch.setattr(R, "MAX_RECEIPTS", 1)
    quiet = job(); R.prepare(quiet, "NO_REPLY", metadata={}, quiet=True)
    assert R._lookup(quiet["execution_id"])["outcome"] == "not_requested"
    with pytest.raises(ValueError):
        D._deliver_result(quiet, "NO_REPLY")
    next_quiet = job(); R.prepare(next_quiet, "NO_REPLY", metadata={}, quiet=True)
    with pytest.raises(ValueError, match="retired"):
        R.prepare(quiet, "NO_REPLY", metadata={}, quiet=True)
    old = job(); R.prepare(old, "old", metadata={}); D._deliver_result(old, "old")
    new = job(); R.prepare(new, "new", metadata={}); D._deliver_result(new, "new")
    assert R.history(new["id"])["status"] == "truncated"
    for action in (lambda: R.prepare(old, "old", metadata={}), lambda: D._deliver_result(old, "old")):
        with pytest.raises(ValueError, match="retired"):
            action()
