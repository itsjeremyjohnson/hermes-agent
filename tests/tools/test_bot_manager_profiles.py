"""Manager protocol and schemas remain scoped to the owning profile."""
import json
from pathlib import Path
from types import SimpleNamespace

from agent.agent_init import _apply_agent_section
from agent.system_prompt import _bot_mode_parts
from gateway.run import _profile_runtime_scope
from hermes_cli.config import load_config_readonly
from hermes_state import SessionDB
from tests.tools.test_bot_mode_dm import _managed_home
from tools import bot_mode_probe
from tools.mcp_tool_agent import _reinject_authorized_dynamic_tools


def test_manager_profile_state_isolated_across_multiplex_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = _managed_home(tmp_path, teammates=("ordinary",))
    ordinary = home / "profiles" / "ordinary"
    for profile, manager in ((home, True), (ordinary, False)):
        (profile / "config.yaml").write_text(
            f"agent:\n  bot_mode_manager: {str(manager).lower()}\n  environment_probe: false\n"
        )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    bot_mode_probe._reset_cache_for_tests()
    observed = []
    try:
        for profile, expected in ((home, True), (ordinary, False), (home, True)):
            db = SessionDB(profile / "state.db")
            try:
                db.create_session("canonical", source="cli")
                db.set_session_title("canonical", "Bot Chat")
                with _profile_runtime_scope(profile, prepared_secret_scope={}):
                    config = load_config_readonly()
                    agent = SimpleNamespace(run_budget_seconds=None, _session_db=db,
                                            session_id="canonical", tools=[], valid_tool_names=set())
                    _apply_agent_section(agent, config)
                    protocol = "\n".join(_bot_mode_parts(agent))
                    _reinject_authorized_dynamic_tools(agent, agent.tools, agent.valid_tool_names)
                    assert agent._bot_mode_manager is expected
                    assert ("assignments=" in protocol) is expected
                    assert ("assignments" in agent.tools[0]["function"]["parameters"]["properties"]) is expected
                    assert "message_agent" in agent.valid_tool_names
                    observed.append((protocol, bot_mode_probe.epoch_line(profile), json.dumps(agent.tools)))
            finally:
                db.close()
        assert observed[0] == observed[2]
        assert observed[0] != observed[1]
    finally:
        bot_mode_probe._reset_cache_for_tests()


def test_protocol_refresh_through_home_alias_updates_same_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = _managed_home(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    bot_mode_probe._reset_cache_for_tests()
    try:
        old = bot_mode_probe.get_bot_mode_protocol_section(home, manager=True)
        metadata = home / "profiles" / "researcher" / "profile.yaml"
        metadata.write_text(metadata.read_text().replace("teammate for tests", "changed researcher role"))
        refreshed = bot_mode_probe.get_bot_mode_protocol_section(alias, force_refresh=True, manager=True)
        assert "changed researcher role" in refreshed and refreshed != old
        assert bot_mode_probe.get_bot_mode_protocol_section(home, manager=True) == refreshed
    finally:
        bot_mode_probe._reset_cache_for_tests()
