"""Canonical manager -Q must review real completion events before exiting."""

from queue import Queue
from types import SimpleNamespace

import pytest
import yaml

import cli
from hermes_cli.cli_process_notifications import CLIProcessNotificationsMixin
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump({"ui_meta": {"hermes-bots": {}}}))
    registry = ProcessRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr(registry, "_oneshot_completion_wait_seconds", lambda: 1.0)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    class TestCLI(CLIProcessNotificationsMixin):
        session_id = "manager"
        conversation_history = []
        _pending_input = Queue()
        _session_db = SimpleNamespace(resolve_resume_session_id=lambda key: "manager-next" if key == "manager" else key)

    instance = TestCLI()
    instance.agent = SimpleNamespace(_bot_mode_manager=True, session_id="manager",
                                     _session_title_hint="Bot Chat")

    def process(name):
        session = ProcessSession(id="proc_" + name, command=name, task_id="host-local-default",
                                 session_key="ephemeral-turn-id", parent_session_id="manager", notify_on_complete=True,
                                 cwd=str(tmp_path), output_buffer=name + " actual result")
        registry._running[session.id] = session
        return session

    return instance, registry, process


def test_fast_review_precedes_slow_completion_and_keeps_history_after_compression(runtime, capsys):
    instance, registry, process = runtime
    fast, slow = process("fast"), process("slow")
    history = []
    calls = []

    def run(*, user_message, conversation_history):
        nonlocal history
        assert conversation_history == history
        calls.append(user_message)
        if len(calls) == 1:
            registry._finish_exited(fast, 0)
            response = "DISPATCHED"
        elif len(calls) == 2:
            assert "fast actual result" in user_message
            assert not slow.exited
            registry._finish_exited(slow, 0)
            instance.agent.session_id = "manager-next"
            response = "FAST REVIEWED"
        else:
            assert len(calls) == 3
            assert "slow actual result" in user_message
            response = "BOTH REVIEWED"
        history = history + [{"role": "user", "content": user_message},
                             {"role": "assistant", "content": response}]
        return {"final_response": response, "messages": history}

    instance.agent.run_conversation = run
    with pytest.raises(SystemExit) as exit_info:
        cli._run_quiet_single_query(instance, "delegate two checks")
    assert exit_info.value.code == 0
    assert len(calls) == 3
    assert instance.session_id == "manager-next"
    assert registry.completion_queue.empty()
    output = capsys.readouterr().out
    assert output.index("DISPATCHED") < output.index("FAST REVIEWED") < output.index("BOTH REVIEWED")


@pytest.mark.parametrize("manager,title", [(False, "Bot Chat"), (True, "ordinary chat")])
def test_other_quiet_callers_still_run_one_turn(runtime, manager, title):
    instance, registry, process = runtime
    instance.agent._bot_mode_manager = manager
    instance.agent._session_title_hint = title
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        registry._finish_exited(process("background"), 0)
        return {"final_response": "one answer"}

    instance.agent.run_conversation = run
    with pytest.raises(SystemExit):
        cli._run_quiet_single_query(instance, "hello")
    assert len(calls) == 1
    assert not registry.completion_queue.empty()


def test_finished_process_waits_for_its_notification_publication(runtime):
    instance, registry, process = runtime
    late = process("late")
    calls = []

    def publish(timeout):
        registry.completion_queue.put({"type": "completion", "session_id": late.id,
                                       "session_key": "manager", "command": "late",
                                       "exit_code": 0, "output": "late actual result"})
        late._completion_event.set()
        return True

    late._completion_event.wait = publish

    def run(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            late.mark_exited(0)
            registry._running.pop(late.id)
            registry._finished[late.id] = late
        else:
            assert "late actual result" in kwargs["user_message"]
        return {"final_response": "reviewed"}

    instance.agent.run_conversation = run
    with pytest.raises(SystemExit):
        cli._run_quiet_single_query(instance, "go")
    assert len(calls) == 2


def test_foreign_process_parent_is_not_claimed_even_when_turn_key_matches(runtime):
    instance, registry, process = runtime
    foreign = process("foreign")
    foreign.parent_session_id = "unrelated-conversation"
    foreign.session_key = "manager"
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        registry._finish_exited(foreign, 0)
        return {"final_response": "no own work"}

    instance.agent.run_conversation = run
    with pytest.raises(SystemExit):
        cli._run_quiet_single_query(instance, "go")
    assert len(calls) == 1
    assert registry.completion_queue.get_nowait()["session_id"] == foreign.id


def test_incomplete_worker_wait_has_one_shared_deadline_and_nonzero_exit(runtime, monkeypatch):
    instance, registry, process = runtime
    process("unfinished")
    monkeypatch.setattr(registry, "_oneshot_completion_wait_seconds", lambda: 0.01)
    instance.agent.run_conversation = lambda **kwargs: {"final_response": "dispatched"}
    with pytest.raises(SystemExit) as exited:
        cli._run_quiet_single_query(instance, "go")
    assert exited.value.code == 1
    waits = []
    monkeypatch.setattr(registry, "wait_for_pending_completions",
                        lambda task, **kwargs: waits.append(kwargs) or {})
    cli._wait_for_oneshot_background_completions(instance)
    assert waits == [{"timeout": 0.0}]


def test_interrupt_prevents_manager_review(runtime, monkeypatch):
    instance, registry, process = runtime
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *args, **kwargs: None)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        registry._finish_exited(process("completed"), 0)
        return {"final_response": "dispatched"}

    instance.agent.run_conversation = run
    with pytest.raises(SystemExit) as exited:
        cli._run_quiet_single_query(instance, "go")
    assert exited.value.code == 130
    assert len(calls) == 1
    assert not registry.completion_queue.empty()
