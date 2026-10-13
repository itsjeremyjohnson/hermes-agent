"""Bot-root context survives restarts and standalone cron sends without widening access."""
from unittest.mock import MagicMock

import pytest

from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tests.gateway.test_google_chat import (
    GoogleChatAdapter, _gc_mod, _FakeAiohttpResponse, _FakeAiohttpSession,
    _install_fake_aiohttp, _install_fake_google_auth_transport,
)
from tests.gateway.test_google_chat_source_parity import _parity_config, _SPACE, _USERS


def adapter(home):
    token = set_hermes_home_override(home)
    try:
        instance = GoogleChatAdapter(_parity_config(groups={_SPACE: {
            "users": list(_USERS), "allow_unmentioned_bot_threads": True,
        }}))
    finally:
        reset_hermes_home_override(token)
    instance._chat_api = MagicMock()
    instance._new_authed_http = MagicMock()
    return instance


def incoming(thread="old", sender=_USERS[0]):
    return {"name": f"{_SPACE}/messages/reply", "text": "This task is done",
            "thread": {"name": f"{_SPACE}/threads/{thread}"},
            "sender": {"name": sender, "type": "HUMAN"}}


def response(thread):
    return {"name": f"{_SPACE}/messages/{thread}",
            "thread": {"name": f"{_SPACE}/threads/{thread}"},
            "createTime": "2026-09-29T14:00:00Z"}


@pytest.mark.asyncio
async def test_original_roots_survive_restart_profile_isolation_and_preserve_sender_gate(tmp_path):
    home_a, home_b = tmp_path / "A", tmp_path / "B"
    first = adapter(home_a)
    api = first._chat_api.spaces().messages().create
    api.return_value.execute.side_effect = [response("old"), response("new"), response("old")]
    old_text = "Standup September 28\nTask from https://example.com/source-28"
    await first._create_message(_SPACE, {"text": old_text})
    await first._create_message(_SPACE, {"text": "Standup September 29\nhttps://example.com/source-29"})
    await first._create_message(_SPACE, {"text": "A later bot reply", "thread": {"name": f"{_SPACE}/threads/old"}})
    space = {"name": _SPACE, "spaceType": "SPACE"}
    restarted = adapter(home_a)
    assert restarted._should_accept_inbound(incoming(), space)
    event = await restarted._build_message_event(incoming(), {"space": space})
    assert event.reply_to_text == old_text
    assert event.reply_to_message_id == f"{_SPACE}/messages/old"
    assert event.reply_to_is_own_message is True
    assert event.source.thread_id == f"{_SPACE}/threads/old"
    assert not restarted._should_accept_inbound(incoming("untracked"), space)
    assert not restarted._should_accept_inbound(incoming(sender="users/outsider"), space)
    assert not restarted._should_accept_inbound(incoming(), {"name": "spaces/Other", "spaceType": "SPACE"})
    assert not adapter(home_b)._should_accept_inbound(incoming(), space)
    assert adapter(home_a)._should_accept_inbound(incoming(), space)
    restarted._groups[_SPACE]["allow_unmentioned_bot_threads"] = False
    assert not restarted._should_accept_inbound(incoming(), space)


@pytest.mark.asyncio
async def test_standalone_post_is_seen_by_already_running_gateway(tmp_path, monkeypatch):
    running = adapter(tmp_path / "A")
    _install_fake_google_auth_transport(monkeypatch)
    creds = MagicMock(token="test-token")
    monkeypatch.setattr(_gc_mod, "_load_sa_credentials_from", lambda value: creds)
    session = _FakeAiohttpSession([_FakeAiohttpResponse(200, response("old"))])
    _install_fake_aiohttp(monkeypatch, session)
    original = "September 28: task and source https://example.com/original"
    token = set_hermes_home_override(tmp_path / "A")
    try:
        result = await _gc_mod._standalone_send(_parity_config(), _SPACE, original)
    finally:
        reset_hermes_home_override(token)
    assert result["success"] is True
    assert result["thread_id"] == f"{_SPACE}/threads/old"
    assert result["context_recorded"] is True
    space = {"name": _SPACE, "spaceType": "SPACE"}
    assert running._should_accept_inbound(incoming(), space)
    event = await running._build_message_event(incoming(), {"space": space})
    assert event.reply_to_text == original
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_accepted_standalone_post_keeps_receipt_when_context_write_fails(tmp_path, monkeypatch):
    _install_fake_google_auth_transport(monkeypatch)
    monkeypatch.setattr(_gc_mod, "_load_sa_credentials_from", lambda value: MagicMock(token="test-token"))
    store = MagicMock()
    store.record.side_effect = OSError("disk full")
    monkeypatch.setattr(_gc_mod, "profile_thread_context_store", lambda: store)
    session = _FakeAiohttpSession([_FakeAiohttpResponse(200, response("old"))])
    _install_fake_aiohttp(monkeypatch, session)
    result = await _gc_mod._standalone_send(_parity_config(), _SPACE, "original")
    assert result == {"success": True, "message_id": f"{_SPACE}/messages/old",
                      "thread_id": f"{_SPACE}/threads/old", "context_recorded": False,
                      "context_error": "disk full"}
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_typing_marker_is_untrusted_until_final_root_patch(tmp_path):
    running = adapter(tmp_path)
    create = running._chat_api.spaces().messages().create
    create.return_value.execute.return_value = response("old")
    patch = running._chat_api.spaces().messages().patch
    patch.return_value.execute.return_value = response("old")
    space = {"name": _SPACE, "spaceType": "SPACE"}
    await running.send_typing(_SPACE)
    assert not running._should_accept_inbound(incoming(), space)
    result = await running.send(_SPACE, "Final original post with source https://example.com")
    assert result.success is True
    assert running._should_accept_inbound(incoming(), space)
    event = await running._build_message_event(incoming(), {"space": space})
    assert event.reply_to_text == "Final original post with source https://example.com"


@pytest.mark.asyncio
async def test_explicit_reply_never_establishes_trusted_root(tmp_path):
    running = adapter(tmp_path)
    running._chat_api.spaces().messages().create.return_value.execute.return_value = response("old")
    await running._create_message(_SPACE, {
        "text": "Bot reply in unrelated thread", "thread": {"name": f"{_SPACE}/threads/old"},
    })
    assert not running._should_accept_inbound(incoming(), {"name": _SPACE, "spaceType": "SPACE"})


@pytest.mark.asyncio
async def test_related_summary_uses_exact_thread_and_persists_root_pointer(tmp_path, monkeypatch):
    from plugins.platforms.google_chat.thread_context import profile_thread_context_store
    _install_fake_google_auth_transport(monkeypatch)
    monkeypatch.setattr(_gc_mod, "_load_sa_credentials_from", lambda value: MagicMock(token="test-token"))
    accepted_reply = response("old")
    accepted_reply["name"] = f"{_SPACE}/messages/summary"
    session = _FakeAiohttpSession([_FakeAiohttpResponse(200, response("old")),
                                  _FakeAiohttpResponse(200, accepted_reply)])
    _install_fake_aiohttp(monkeypatch, session)
    token = set_hermes_home_override(tmp_path)
    try:
        root = await _gc_mod._standalone_send(_parity_config(), _SPACE, "Standup done | September 28")
        summary = "Meeting September 28\nTask from https://example.com/source"
        reply = await _gc_mod._standalone_send(_parity_config(), _SPACE, summary, thread_id=root["thread_id"])
        store = profile_thread_context_store()
        assert not store.record_related_message(_SPACE, root["thread_id"], "wrong-root", reply["message_id"], summary)
        assert not store.record_related_message(_SPACE, f"{_SPACE}/threads/unknown", root["message_id"], reply["message_id"], summary)
        assert store.record_related_message(_SPACE, root["thread_id"], root["message_id"], reply["message_id"], summary)
        assert store.record_related_message(_SPACE, root["thread_id"], root["message_id"], reply["message_id"], summary)
    finally:
        reset_hermes_home_override(token)
    assert "params" not in session.calls[0][1]
    assert session.calls[1][1]["params"] == {"messageReplyOption": "REPLY_MESSAGE_OR_FAIL"}
    assert session.calls[1][1]["json"]["thread"]["name"] == f"{_SPACE}/threads/old"
    restarted = adapter(tmp_path)
    event = await restarted._build_message_event(incoming(), {"space": {"name": _SPACE, "spaceType": "SPACE"}})
    assert event.reply_to_message_id == root["message_id"]
    assert event.reply_to_text == "Standup done | September 28\n\nMeeting September 28\nTask from https://example.com/source"
