"""Only never-started cron delivery may wait for a CLI owner's release."""
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from cron import bot_chat_delivery as queue
from cron import scheduler_delivery as delivery
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB


@pytest.mark.parametrize("error", [None, subprocess.TimeoutExpired("hermes", 1)])
def test_cli_owner_deferral_and_attempt_fence(tmp_path, monkeypatch, error):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None and lease is not None
    run = Mock(side_effect=error, return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/bin/hermes")
    job = {"id": "job", "execution_id": "execution"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "output", "")
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        queue.drain()
        run.assert_not_called()
        lease.release()
        queue.drain()
        assert run.call_count == 1
        expected = "ambiguous" if error else "settled"
        assert queue.read_pending(key)["status"] == expected
        queue.drain()
        delivery._deliver_to_bot_chat(job, "output", "")
        # The failed turn is never retried; a timeout additionally drains ONE short
        # degraded-delivery marker (its own turn), and its timeout queues nothing more.
        assert run.call_count == (2 if error else 1)
        markers = [r for _, r in queue._records(queue._root()) if r.get("degraded")]
        assert len(markers) == (1 if error else 0)
    finally:
        lease.release()
        db.close()


def test_delivery_exception_retains_attempt_and_continues_siblings(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    blocked_parent = tmp_path / "blocked"
    blocked_home = blocked_parent / "recipient"
    blocked_home.mkdir(parents=True)
    for home in (blocked_home, tmp_path):
        db = SessionDB(db_path=home / "state.db")
        db.create_session(session_id="chat", source="cli")
        db.set_session_title("chat", "Bot Chat")
        db.close()
    queue.defer("b" * 64, {"id": "bad"}, "bad output", "", blocked_home)
    queue.defer("a" * 64, {"id": "good"}, "good output", "", tmp_path)
    calls = []
    original_is_dir = Path.is_dir
    armed = False

    def resolve_cli(name, *args, **kwargs):
        # Delivery resolves the CLI (the running install's ``hermes_cli``) right after discovery.
        nonlocal armed
        armed = True
        return object()

    def is_dir(self):
        if armed and self == blocked_home:
            raise PermissionError("target traversal denied after discovery")
        return original_is_dir(self)

    def run(argv, env, report_path, timeout):
        calls.append(env["HERMES_HOME"])
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(importlib.util, "find_spec", resolve_cli)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    monkeypatch.setattr(Path, "is_dir", is_dir)
    queue.drain()
    assert queue.read_pending("b" * 64)["status"] == "ambiguous"
    assert "PermissionError" in queue.read_pending("b" * 64)["error"]
    assert queue.read_pending("a" * 64)["status"] == "settled"
    queue.drain()
    assert calls == [str(tmp_path)]


def test_pending_queue_uses_admission_order_and_keeps_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "job"}
    queue.defer("f" * 64, job, "older", "", tmp_path)
    queue.defer("a" * 64, job, "newer", "", tmp_path)
    seen = []

    def interrupted(job, content, profile, **kwargs):
        seen.append(content)
        raise KeyboardInterrupt

    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        queue.drain()
    assert seen == ["older"]
    assert queue.read_pending("f" * 64)["status"] == "claimed"
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    queue.drain()
    queue.drain()
    assert seen == ["older", "newer"]


def test_policy_change_settles_diagnostic_without_waiting_for_cli_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None
    run = Mock()
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    job = {"id": "failure", "execution_id": "run"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
        key = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]
        assert queue.read_pending(key)["for_failure"] is True
        with pytest.raises(ValueError, match="different payload"):
            queue.defer(key, job, "diagnostic", "", tmp_path, for_failure=False)
        (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
        queue.drain()
        assert queue.read_pending(key)["status"] == "suppressed"
        queue.drain()
        run.assert_not_called()
    finally:
        lease.release()
        db.close()


def test_live_receipt_outcome_survives_later_suppression(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="tui")
    db.set_session_title("chat", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="chat", surface="desktop", config={}, registry_home=tmp_path,
        metadata={"bot_live_delivery_consumer": True, "live_session_id": "live"})
    assert refusal is None
    job = {"id": "failure", "execution_id": "run"}
    try:
        assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
        (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
        assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
        assert not job.get("_notification_all_targets_suppressed")
    finally:
        lease.release()
        db.close()

@pytest.mark.skipif(not hasattr(__import__("os"), "geteuid") or __import__("os").geteuid() == 0,
                    reason="needs POSIX file permissions for an unreadable receipt")
def test_unreadable_deferred_receipt_does_not_block_siblings(tmp_path, monkeypatch, caplog):
    """One permission-denied receipt beside healthy queued work must degrade to a logged skip,
    not abort the drain (same class as the live-owner mailbox wedge, #109820)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    queue.defer("a" * 64, {"id": "job"}, "healthy", "", tmp_path)
    bad = queue._root() / f"{'e' * 64}.json"
    bad.write_text('{"status": "queued"}', encoding="utf-8")
    bad.chmod(0)
    seen = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    with caplog.at_level("ERROR", logger=queue.logger.name):
        queue.drain()
        queue.drain()  # every scheduler tick drains; the same bad receipt must not re-log
    assert seen == ["healthy"]
    assert [r for r in caplog.records
            if "Unreadable deferred Bot Chat receipt" in r.message and "Permission denied" in r.message] and \
        sum("Unreadable deferred Bot Chat receipt" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("payload", ["42", '"oops"', "[1, 2, 3]"])
def test_non_dict_deferred_receipt_is_skipped_by_the_drain_and_fails_exact_id_reads_closed(
        tmp_path, monkeypatch, caplog, payload):
    """A receipt that parses but is not a JSON object is a bad file like any other: the drain
    and new admissions skip it (warned once, preserved as evidence) and an exact-id read of it
    never licenses an overwrite."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    queue.defer("a" * 64, {"id": "job"}, "healthy", "", tmp_path)
    bad = queue._root() / f"{'e' * 64}.json"
    bad.write_text(payload, encoding="utf-8")
    seen = []
    monkeypatch.setattr(delivery, "_deliver_to_bot_chat", lambda j, c, p, **kw: seen.append(c))
    with caplog.at_level("ERROR", logger=queue.logger.name):
        queue.drain()
        queue.drain()
        later = queue.defer("f" * 64, {"id": "job"}, "later", "", tmp_path)
    assert seen == ["healthy"] and later["status"] == "queued"
    assert sum("Unreadable deferred Bot Chat receipt" in r.message for r in caplog.records) == 1
    with pytest.raises(ValueError):
        queue.defer("e" * 64, {"id": "job"}, "same id", "", tmp_path)
    assert bad.read_text(encoding="utf-8") == payload


def _bot_chat_home(home):
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="chat", source="cli")
    db.set_session_title("chat", "Bot Chat")
    return db


def _sibling_takes_session_then_child_runs(home, stderr, turns):
    """The ``hermes chat -Q`` child boundary: the first launch loses the lease to a sibling that
    claimed it after the parent's probe and exits with ``stderr``; later launches record the
    query they were handed in ``turns``. That counts child submissions, not persisted session turns."""
    from hermes_cli.active_sessions import format_refusal_stderr
    sibling = []

    def run(argv, env, report_path, timeout):
        if not sibling:
            lease, _ = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=home)
            _, refusal = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=home,
                                                    metadata={"live_session_id": "child"})
            sibling.append(lease)
            text = stderr.replace("{refusal}", format_refusal_stderr(refusal))
            return subprocess.CompletedProcess(argv, 1, "", text)
        turns.append(Path(argv[argv.index("--query-file") + 1]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(argv, 0, "", "")
    return run, sibling


# The marker sits behind an update warning and ahead of more stderr than the persisted
# diagnostic tail keeps, as in the Oct 5 rm-hosting reader refusal.
_REFUSAL = "Update available: run `hermes update`\n{refusal}\n" + "x" * 600
_NOT_A_REFUSAL = [pytest.param("Traceback: provider exploded", id="generic-exit"),
                  pytest.param("hermes-refusal-reason: SESSION_NOT_OWNED_EXTRA\nunknown", id="unknown-marker"),
                  pytest.param("hermes-refusal-reason: SESSION_COORDINATION_UNAVAILABLE\ncannot tell",
                               id="coordination-unavailable")]


@pytest.mark.parametrize("for_failure", [False, True])
def test_child_session_not_owned_after_probe_queues_then_settles_once(tmp_path, monkeypatch, for_failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _bot_chat_home(tmp_path)
    turns = []
    run, sibling = _sibling_takes_session_then_child_runs(tmp_path, _REFUSAL, turns)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    job = {"id": "reader", "execution_id": "execution"}
    try:
        outcome = delivery._deliver_to_bot_chat(job, "NEW alert", "", for_failure=for_failure)
        assert sibling, "the probe saw no owner, so the child was launched"
        receipt = job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]
        key = receipt["delivery_id"]
        assert receipt["status"] == "queued"
        assert outcome == f"bot-chat:(own) queued (receipt {key}): completion unverified; do not resend"
        record = queue.read_pending(key)
        assert (record["status"], record["content"], record["home"], bool(record.get("for_failure"))) == (
            "queued", "NEW alert", str(tmp_path.resolve()), for_failure)
        # The same execution re-admitted while the sibling holds the session stays on its original key.
        assert delivery._deliver_to_bot_chat(job, "NEW alert", "", for_failure=for_failure) == outcome
        queue.drain()
        assert turns == []
        sibling[0].release()
        queue.drain()
        assert queue.read_pending(key)["status"] == "settled"
        queue.drain()
        assert len(turns) == 1 and "NEW alert" in turns[0]
    finally:
        for lease in sibling:
            lease.release()
        db.close()


@pytest.mark.parametrize("stderr", _NOT_A_REFUSAL)
def test_untyped_child_failure_after_probe_is_not_queued(tmp_path, monkeypatch, stderr):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _bot_chat_home(tmp_path)
    turns = []
    run, sibling = _sibling_takes_session_then_child_runs(tmp_path, stderr, turns)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    job = {"id": "reader", "execution_id": "execution"}
    try:
        outcome = delivery._deliver_to_bot_chat(job, "NEW alert", "")
        assert outcome.startswith("Hermes could not deliver this result to Bot Chat")
        assert "_bot_chat_delivery_receipts" not in job
        sibling[0].release()
        queue.drain()
        assert queue._records(queue._root()) == [] and turns == []
    finally:
        for lease in sibling:
            lease.release()
        db.close()


@pytest.mark.parametrize("stderr, requeued", [
    pytest.param(_REFUSAL, True, id="session-not-owned"),
    *(pytest.param(p.values[0], False, id=p.id) for p in _NOT_A_REFUSAL)])
def test_child_refusal_during_claimed_drain_requeues_then_settles_once(tmp_path, monkeypatch, stderr, requeued):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = _bot_chat_home(tmp_path)
    turns = []
    run, sibling = _sibling_takes_session_then_child_runs(tmp_path, stderr, turns)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    key = "c" * 64
    queue.defer(key, {"id": "reader", "execution_id": "execution"}, "NEW alert", "", tmp_path.resolve())
    try:
        queue.drain()
        assert sibling, "the drain claimed the record and launched its child"
        record = queue.read_pending(key)
        assert record["status"] == ("queued" if requeued else "ambiguous")
        queue.drain()
        assert turns == []
        sibling[0].release()
        queue.drain()
        queue.drain()
        assert queue.read_pending(key)["status"] == ("settled" if requeued else "ambiguous")
        assert len(turns) == (1 if requeued else 0)
    finally:
        for lease in sibling:
            lease.release()
        db.close()


def test_suppression_after_a_refused_drain_releases_its_claim_stands(tmp_path, monkeypatch):
    """A failure route ``bot-chat,bot-chat:default`` admits the same key twice. If the second
    alias suppresses the record after a refused drain re-queued it, the drain must not overwrite
    that suppression with its own ``ambiguous`` disposition or record a failed incident delivery."""
    from cron import incidents
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron/executions.db")
    incident_id, _ = incidents.upsert_incident("reader", "provider 503")
    db = _bot_chat_home(tmp_path)
    turns = []
    run, sibling = _sibling_takes_session_then_child_runs(tmp_path, _REFUSAL, turns)
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)
    job = {"id": "reader", "execution_id": "execution", "_failure_incident_ids": [incident_id]}
    owner, _ = try_acquire_active_session(session_id="chat", surface="cli", config={}, registry_home=tmp_path)
    delivery._deliver_to_bot_chat(dict(job), "NEW alert", "", for_failure=True)
    owner.release()
    key = queue._records(queue._root())[0][1]["id"]
    release_claim = queue.release_claim

    def release_then_alias_suppresses(released_key):
        record = release_claim(released_key)
        (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
        alias = dict(job)
        assert delivery._deliver_to_bot_chat(alias, "NEW alert", "default", for_failure=True) is None
        assert alias["_bot_chat_delivery_receipts"]["bot-chat:default"] == {"status": "suppressed", "delivery_id": key}
        return record

    monkeypatch.setattr(queue, "release_claim", release_then_alias_suppresses)
    try:
        queue.drain()
        assert sibling, "the drain claimed the record and its child was refused"
        assert queue.read_pending(key)["status"] == "suppressed"
        incident = incidents.get_incident(incident_id)
        assert (incident["state"], incident["alerted_at"], incident["undelivered_at"]) == ("detected", None, None)
        sibling[0].release()
        queue.drain()
        queue.drain()
        assert queue.read_pending(key)["status"] == "suppressed" and turns == []
    finally:
        for lease in sibling:
            lease.release()
        db.close()
