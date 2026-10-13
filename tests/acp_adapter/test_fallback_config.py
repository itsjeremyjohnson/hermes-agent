"""ACP must use the owning profile's saved fallback, including edits between turns."""

from types import SimpleNamespace
import time

import pytest
import yaml

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class CapturingAgent:
    def __init__(self, **kwargs):
        self.model = kwargs.get("model", "primary")
        self._fallback_chain = kwargs.get("fallback_model") or []
        self._fallback_activated = False
        self._rate_limited_until = 0

    def run_conversation(self, **kwargs):
        return {"chain": self._fallback_chain}


@pytest.fixture
def acp_runtime(monkeypatch):
    monkeypatch.setattr("run_agent.AIAgent", CapturingAgent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kwargs: {
        "provider": "custom", "base_url": "https://example.invalid/v1", "api_key": "test-key",
    })
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda *args: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **kwargs: None)
    manager = SessionManager()
    manager._get_db = lambda: None
    return manager


def write_config(home, model):
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "primary", "provider": "custom"},
        "fallback_providers": [{"provider": "custom:cliproxyapi", "model": model}],
    }))


def run_turn(server, agent):
    return server._run_agent_turn(
        state=SimpleNamespace(agent=agent, cwd=".", history=[]), session_id="fallback-test",
        user_text="status", user_content="status", conn=None, loop=None,
        approval_cb=None, edit_approval_requester=None,
    )["chain"]


def test_new_acp_agent_receives_saved_fallback(acp_runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_config(tmp_path, "gpt-6.1-sol")
    agent = acp_runtime._make_agent(session_id="new", cwd=".")
    assert agent._fallback_chain == [{"provider": "custom:cliproxyapi", "model": "gpt-6.1-sol"}]


def test_cached_acp_agent_refreshes_profile_fallback_and_keeps_it_on_bad_yaml(acp_runtime, monkeypatch, tmp_path):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, model in zip(homes, ["gpt-6.1-sol", "gpt-6-luna"]):
        write_config(home, model)
    server = HermesACPAgent(session_manager=acp_runtime)
    agent = CapturingAgent()
    for home, model in [(homes[0], "gpt-6.1-sol"), (homes[1], "gpt-6-luna"), (homes[0], "gpt-6.1-sol")]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert run_turn(server, agent) == [{"provider": "custom:cliproxyapi", "model": model}]
    (homes[0] / "config.yaml").write_text("fallback_providers: [")
    assert run_turn(server, agent) == [{"provider": "custom:cliproxyapi", "model": "gpt-6.1-sol"}]
    (homes[0] / "config.yaml").write_text("{}")
    assert run_turn(server, agent) == []


def test_acp_refresh_preserves_activated_fallback_during_cooldown(acp_runtime, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    write_config(tmp_path, "gpt-6.1-sol")
    agent = CapturingAgent(fallback_model=[{"provider": "custom:cliproxyapi", "model": "gpt-6-luna"}])
    agent._fallback_activated = True
    agent._rate_limited_until = time.monotonic() + 60
    server = HermesACPAgent(session_manager=acp_runtime)
    assert run_turn(server, agent) == [{"provider": "custom:cliproxyapi", "model": "gpt-6-luna"}]
    agent._rate_limited_until = 0
    assert run_turn(server, agent) == [{"provider": "custom:cliproxyapi", "model": "gpt-6.1-sol"}]
