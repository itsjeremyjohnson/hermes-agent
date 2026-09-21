"""Manager memory must not be opened or exposed to excluded background contexts."""

import json
from types import SimpleNamespace

import pytest
import yaml

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _init_memory
from plugins.memory import load_memory_provider


def _configure(tmp_path, monkeypatch, excluded_contexts):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "memory": {"provider": "holographic", "memory_enabled": False,
                   "user_profile_enabled": False},
        "plugins": {"hermes-memory-store": {
            "excluded_contexts": excluded_contexts, "auto_extract": False,
            "hrr_dim": 64,
        }},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    return config


@pytest.mark.parametrize("context", [{"platform": "cron", "agent_context": "primary"},
                                    {"platform": "cli", "agent_context": "flush"}])
def test_excluded_context_never_opens_store_or_exposes_tools(tmp_path, monkeypatch, context):
    _configure(tmp_path, monkeypatch, ["cron", "flush"])
    provider = load_memory_provider("holographic", register_skills=False)
    assert provider is not None
    provider._config["auto_extract"] = True
    provider.initialize("background", **context)
    try:
        assert not (tmp_path / "memory_store.db").exists()
        assert provider.get_tool_schemas() == []
        assert provider.system_prompt_block() == ""
        assert provider.prefetch("manager") == ""
        result = json.loads(provider.handle_tool_call("fact_store", {
            "action": "add", "content": "must not be stored"}))
        assert "excluded" in result["error"].lower()
        feedback = json.loads(provider.handle_tool_call("fact_feedback", {
            "action": "helpful", "fact_id": 1}))
        assert "excluded" in feedback["error"].lower()
        provider.on_memory_write("add", "memory", "must not be mirrored")
        provider.on_session_end([{"role": "user", "content": "I prefer manager memory"}])
        assert not (tmp_path / "memory_store.db").exists()
    finally:
        provider.shutdown()


def test_real_agent_memory_init_excludes_cron_and_delegate_but_keeps_manager(tmp_path, monkeypatch):
    config = _configure(tmp_path, monkeypatch, "cron, flush")

    def create(platform, skip_memory=False):
        agent = SimpleNamespace(
            session_id=platform, enabled_toolsets=["memory"], disabled_toolsets=[],
            tools=[], valid_tool_names=set(), _session_db=None,
            _emit_warning=lambda *a: None, _emit_status=lambda *a: None,
            **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
        )
        _init_memory(agent, config, skip_memory, platform)
        return agent

    cron = create("cron")
    worker = create("cli", skip_memory=True)
    try:
        assert "fact_store" not in cron.valid_tool_names
        assert "fact_store" not in worker.valid_tool_names
        assert not (tmp_path / "memory_store.db").exists()
        manager = create("cli")
        try:
            assert "fact_store" in manager.valid_tool_names
            provider = manager._memory_manager.providers[0]
            assert json.loads(provider.handle_tool_call("fact_store", {
                "action": "add", "content": "manager-approved durable fact"}))["status"] == "added"
            provider.on_session_end([{"role": "user", "content": "I prefer unreviewed extraction"}])
            facts = json.loads(provider.handle_tool_call("fact_store", {"action": "list"}))["facts"]
            assert [fact["content"] for fact in facts] == ["manager-approved durable fact"]
        finally:
            manager._memory_manager.shutdown_all()
    finally:
        cron._memory_manager.shutdown_all()


def test_default_preserves_cron_and_exclusion_reinitialization_closes_store(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, [])
    provider = load_memory_provider("holographic", register_skills=False)
    provider.initialize("existing-cron", platform="cron")
    assert (tmp_path / "memory_store.db").exists()
    assert provider.get_tool_schemas()
    provider._config["excluded_contexts"] = ["cron"]
    provider.initialize("excluded-cron", platform="cron")
    assert provider._store is None
    assert not provider.get_tool_schemas()
    provider.initialize("manager", platform="cli")
    try:
        assert provider._store is not None
        assert provider.get_tool_schemas()
    finally:
        provider.shutdown()


@pytest.mark.parametrize("invalid", [True, False, 0, {}, [True]])
def test_invalid_exclusions_cannot_retain_previously_open_manager_store(tmp_path, monkeypatch, invalid):
    _configure(tmp_path, monkeypatch, [])
    provider = load_memory_provider("holographic", register_skills=False)
    provider.initialize("manager", platform="cli")
    provider._config["excluded_contexts"] = invalid
    with pytest.raises(ValueError, match="excluded_contexts"):
        provider.initialize("cron", platform="cron")
    assert provider._store is None
    assert not provider.get_tool_schemas()
    assert "excluded" in json.loads(provider.handle_tool_call("fact_store", {
        "action": "add", "content": "must not write"}))["error"].lower()
