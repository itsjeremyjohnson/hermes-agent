"""Batch receipts distinguish uncertainty and missing reply notifications."""
import json

import pytest

from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext
from tests.tools.test_bot_mode_dm import _FakeAgent, _managed_home
from tools import bot_mode_dm, bot_mode_probe


@pytest.mark.parametrize('outcomes, expected', [
    ([{'status': 'ambiguous'}, {'status': 'unknown'}], 'unknown'),
    ([{'status': 'ambiguous'}, {'error': 'not admitted'}], 'unknown'),
    ([{'error': 'not admitted'}, {'error': 'not admitted'}], 'failed'),
    ([{'status': 'queued', 'notification_error': 'waiter unavailable'}], 'sent'),
])
def test_batch_preserves_admission_uncertainty_and_reply_obligations(tmp_path, monkeypatch, outcomes, expected):
    home = _managed_home(tmp_path)
    manager = _FakeAgent(home)
    manager._bot_mode_manager = True
    delivered = []
    def deliver(**kwargs):
        delivered.append(kwargs)
        return json.dumps(outcomes[len(delivered) - 1])
    monkeypatch.setattr(bot_mode_dm, 'message_agent_tool', deliver)
    from tools.bot_mode_batch import dispatch_batch
    assignments = [{'target': 'researcher', 'message': 'read only'}] * len(outcomes)
    result = json.loads(dispatch_batch(assignments, mixed_form=False, task_id='receipts', agent=manager))
    assert len(delivered) == len(outcomes)
    assert result['status'] == expected
    assert [entry['result'] for entry in result['results']] == outcomes
    if any('notification_error' in outcome for outcome in outcomes):
        assert 'will not wake you' in result['detail']
        assert 'without resending' in result['detail']


@pytest.mark.parametrize('manager_mode', [False, True])
@pytest.mark.parametrize('waiter', [{'process_id': 'waiter'}, {'error': 'waiter unavailable'}])
def test_live_owner_receipt_keeps_manager_dispatch_guidance(tmp_path, monkeypatch, manager_mode, waiter):
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home)
    agent._bot_mode_manager = manager_mode
    monkeypatch.setattr(bot_mode_dm, '_admit_live_dm', lambda *args, **kwargs: {
        'status': 'queued', 'delivery_id': 'accepted-once',
    })
    guidance = ('Dispatch remaining ready independent assignments before ending your turn.'
                if manager_mode else 'Finish your turn now.')
    monkeypatch.setattr(bot_mode_dm, '_spawn_delivery',
                        lambda *args, **kwargs: json.dumps({**waiter, 'detail': guidance}))
    bot_mode_probe._reset_cache_for_tests()
    try:
        result = json.loads(INLINE_TOOL_EXECUTORS['message_agent'](agent, {
            'target': 'researcher', 'message': 'read only',
        }, InlineToolContext(effective_task_id='live-owner')))
        assert result['status'] == 'queued' and result['delivery_id'] == 'accepted-once'
        if 'error' in waiter:
            assert result['notification_error'] == waiter['error']
            assert 'will not wake you' in result['detail']
        else:
            assert guidance in result['detail']
        if manager_mode:
            assert 'finish your turn' not in result['detail'].lower()
    finally:
        bot_mode_probe._reset_cache_for_tests()
