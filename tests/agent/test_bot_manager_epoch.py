"""Manager snapshots stay consistent across compaction and persisted continuation."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from agent.agent_init import _apply_agent_section
from agent.conversation_compression import _rebuild_system_prompt_at_boundary
from agent.conversation_loop import _persist_system_prompt, _restore_or_build_system_prompt
from agent.system_prompt import _bot_mode_parts
from hermes_cli.config import load_config_readonly
from hermes_state import SessionDB
from tests.tools.test_bot_mode_dm import _managed_home
from tools import bot_mode_probe
from tools.bot_mode_dm import ensure_message_agent_tool


@pytest.mark.parametrize('before', [False, True])
def test_manager_epoch_uses_agent_snapshot_across_persisted_restore(tmp_path, monkeypatch, before):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = _managed_home(tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    bot_mode_probe._reset_cache_for_tests()
    db = SessionDB(home / 'state.db')
    sid = 'canonical-bot-chat'
    db.create_session(sid, source='cli')
    db.set_session_title(sid, 'Bot Chat')
    history = [{'role': 'user', 'content': 'existing history'}]

    def configure(mode):
        (home / 'config.yaml').write_text(
            f'agent:\n  bot_mode_manager: {str(mode).lower()}\n  environment_probe: false\n'
        )

    def new_agent():
        agent = SimpleNamespace(run_budget_seconds=None, _session_db=db, session_id=sid,
                                tools=[], valid_tool_names=set(), _cached_system_prompt=None)
        _apply_agent_section(agent, load_config_readonly())
        agent.build_count = 0

        def build(_):
            agent.build_count += 1
            return '\n'.join(_bot_mode_parts(agent))

        agent._build_system_prompt = build
        agent._invalidate_system_prompt = lambda: setattr(agent, '_cached_system_prompt', None)
        assert ensure_message_agent_tool(agent)
        return agent

    try:
        configure(before)
        old = new_agent()
        old._cached_system_prompt = old._build_system_prompt('')
        original = old._cached_system_prompt
        configure(not before)
        # Exclude unrelated MCP discovery; retain real prompt, epoch, DB and restore paths.
        monkeypatch.setattr('agent.conversation_compression._refresh_agent_tool_definitions', lambda _: None)
        rebuilt = _rebuild_system_prompt_at_boundary(old, '')
        assert rebuilt == original
        _persist_system_prompt(old, 'persist failed: %s %s')
        fresh = new_agent()
        _restore_or_build_system_prompt(fresh, '', history)
        assert fresh.build_count == 1
        assert ('assignments=' in fresh._cached_system_prompt) is (not before)
        assert ('assignments' in fresh.tools[0]['function']['parameters']['properties']) is (not before)
        assert db.get_session(sid)['system_prompt'] == fresh._cached_system_prompt
        continued = new_agent()
        # Disk changing after construction must not change the intended schema/protocol.
        configure(before)
        _restore_or_build_system_prompt(continued, '', history)
        assert continued.build_count == 0
        assert continued._cached_system_prompt == fresh._cached_system_prompt
        assert history == [{'role': 'user', 'content': 'existing history'}]
        assert db.get_session_title(sid) == 'Bot Chat'
    finally:
        db.close()
        bot_mode_probe._reset_cache_for_tests()
