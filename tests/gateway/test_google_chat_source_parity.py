"""Google Chat source-parity: app-url HTTP auth, group/DM own-policy, mention, threads.

Intake tests leave GOOGLE_CHAT_ALLOWED_USERS unset. Group users stay group-only.
DMs use pairing on the exact users/{id} principal. Continuity tests authorize the
event email without treating a different resource id as the same user.
"""

from __future__ import annotations

import asyncio
import json

import yaml
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from tests.gateway.test_google_chat import (
    GoogleChatAdapter,
    _GC,
    _base_config,
    _gc_mod,
    _make_chat_envelope,
    _make_pubsub_message,
)

_SPACE = "spaces/AAQABxzAoFQ"
_USERS = (
    "users/114569381988673522973",
    "users/116002164628005659760",
    "users/103908867453150333526",
)
_BOT = "users/111683293596317960871"
_PRINCIPAL = "101024956389977120814"
_CHAT_ISSUER = "chat@system.gserviceaccount.com"
_ADDON = "service-123@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
_AUDIENCE = "https://charlie.robbenmedia.com/googlechat"


def _parity_extra(**overrides):
    extra = {
        "http_events_url": _AUDIENCE,
        "http_events_audience": _AUDIENCE,
        "http_events_app_principal": _PRINCIPAL,
        "http_events_service_account_email": _CHAT_ISSUER,
        "bot_user": _BOT,
        "dm_policy": "pairing",
        "group_policy": "allowlist",
        "require_mention": True,
        "groups": {_SPACE: {"enabled": True, "users": list(_USERS)}},
    }
    extra.update(overrides)
    return extra


def _parity_config(reply_to_mode="all", **overrides):
    cfg = _base_config(**_parity_extra(**overrides))
    cfg.reply_to_mode = reply_to_mode
    return cfg


def _mention(bot=_BOT):
    return [{"type": "USER_MENTION", "userMention": {"user": {"name": bot, "type": "BOT"}}}]


def _group_envelope(sender=_USERS[0], space=_SPACE, text="ping", annotations=None, sender_type="HUMAN"):
    env = _make_chat_envelope(
        text=text,
        sender_email="alice@example.com",
        sender_type=sender_type,
        thread_name=f"{space}/threads/T1",
    )
    payload = env["chat"]["messagePayload"]
    payload["space"] = {"name": space, "spaceType": "SPACE"}
    payload["message"]["space"] = payload["space"]
    payload["message"]["sender"]["name"] = sender
    payload["message"]["annotations"] = annotations if annotations is not None else _mention()
    return env


def _dm_envelope(sender=_USERS[0], text="hello", *, email="alice@example.com"):
    env = _make_chat_envelope(text=text, sender_email=email)
    env["chat"]["messagePayload"]["message"]["sender"]["name"] = sender
    return env


@pytest.fixture()
def parity_adapter(tmp_path):
    from plugins.platforms.google_chat.adapter import _ThreadCountStore

    a = GoogleChatAdapter(_parity_config())
    a._loop = asyncio.get_event_loop_policy().new_event_loop()
    a._chat_api = MagicMock()
    a._credentials = MagicMock()
    a.handle_message = AsyncMock()
    a._thread_count_store = _ThreadCountStore(tmp_path / "google_chat_thread_counts.json")
    yield a
    a._loop.close()


def _claims(**overrides):
    data = {
        "email": _CHAT_ISSUER,
        "email_verified": True,
        "sub": "unused",
    }
    data.update(overrides)
    return data


def _verify(adapter, claims, audience=_AUDIENCE):
    adapter._http_events_audience = audience

    def fake_verify(token, aud):
        if aud != audience:
            raise ValueError("audience mismatch")
        if token == "bad":
            raise ValueError("invalid token")
        return claims

    _gc_mod._verify_google_id_token = fake_verify
    return adapter.verify_http_event_request("Bearer good")


class TestHttpAppUrlAuth:
    def test_chat_issuer_verified_accepted(self, parity_adapter):
        ok, code = _verify(parity_adapter, _claims())
        assert ok is True
        assert code == ""

    def test_unverified_email_rejected(self, parity_adapter):
        ok, code = _verify(parity_adapter, _claims(email_verified=False))
        assert ok is False
        assert code == "google_email_not_verified"

    def test_config_truthy_email_verified_rejected(self, parity_adapter):
        for value in ("yes", "on", "1", 1):
            ok, code = _verify(parity_adapter, _claims(email_verified=value))
            assert ok is False, value
            assert code == "google_email_not_verified"

    def test_jwt_string_true_accepted(self, parity_adapter):
        ok, code = _verify(parity_adapter, _claims(email_verified="true"))
        assert ok is True

    def test_addon_matching_subject_accepted(self, parity_adapter):
        ok, code = _verify(parity_adapter, _claims(email=_ADDON, sub=_PRINCIPAL))
        assert ok is True
        assert code == ""

    def test_addon_wrong_subject_rejected(self, parity_adapter):
        ok, code = _verify(parity_adapter, _claims(email=_ADDON, sub="999"))
        assert ok is False
        assert code == "unexpected_google_addon_principal"

    def test_addon_missing_subject_rejected(self, parity_adapter):
        ok, code = _verify(parity_adapter, _claims(email=_ADDON, sub=""))
        assert ok is False
        assert code == "unexpected_google_addon_principal"

    def test_bot_sa_email_rejected_in_app_principal_mode(self, parity_adapter):
        ok, code = _verify(
            parity_adapter,
            _claims(email="charlie-google-chat@robbenmedia-charlie-openclaw.iam.gserviceaccount.com"),
        )
        assert ok is False
        assert code == "invalid_google_chat_issuer"

    def test_configured_relay_email_not_used_when_app_principal_set(self, tmp_path):
        cfg = _parity_config(http_events_service_account_email="relay@example.test")
        adapter = GoogleChatAdapter(cfg)
        ok, code = _verify(adapter, _claims(email="relay@example.test"))
        assert ok is False
        assert code == "invalid_google_chat_issuer"

    def test_wrong_audience_rejected(self, parity_adapter):
        def boom(token, aud):
            raise ValueError("bad audience")

        _gc_mod._verify_google_id_token = boom
        ok, code = parity_adapter.verify_http_event_request("Bearer good")
        assert ok is False
        assert code == "invalid_google_bearer"

    def test_missing_bearer_rejected(self, parity_adapter):
        ok, code = parity_adapter.verify_http_event_request("")
        assert ok is False
        assert code == "missing_google_bearer"


class TestLegacyRelayEmailMode:
    def _relay(self):
        cfg = _base_config(
            http_events_url=_AUDIENCE,
            http_events_audience=_AUDIENCE,
            http_events_service_account_email="relay@example.test",
        )
        return GoogleChatAdapter(cfg)

    def test_verified_configured_email_accepted(self):
        adapter = self._relay()
        ok, code = _verify(adapter, _claims(email="relay@example.test"))
        assert ok is True
        assert code == ""

    def test_unverified_configured_email_rejected(self):
        adapter = self._relay()
        ok, code = _verify(adapter, _claims(email="relay@example.test", email_verified=False))
        assert ok is False
        assert code == "google_email_not_verified"

    def test_other_email_rejected(self):
        adapter = self._relay()
        ok, code = _verify(adapter, _claims(email=_CHAT_ISSUER))
        assert ok is False
        assert code == "unexpected_google_bearer_identity"


class TestIntakePolicy:
    @pytest.mark.asyncio
    async def test_allowlisted_group_mention_dispatches(self, parity_adapter):
        await parity_adapter.dispatch_http_event(_group_envelope())
        parity_adapter.handle_message.assert_awaited_once()
        event = parity_adapter.handle_message.await_args.args[0]
        assert event.source.chat_id == _SPACE
        assert event.source.user_id == _USERS[0]
        assert event.source.chat_type == "group"
        assert getattr(event.source, "role_authorized", False) is not True

    @pytest.mark.asyncio
    async def test_other_space_dropped(self, parity_adapter):
        await parity_adapter.dispatch_http_event(_group_envelope(space="spaces/OTHER"))
        parity_adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_case_folded_space_dropped(self, parity_adapter):
        await parity_adapter.dispatch_http_event(_group_envelope(space=_SPACE.lower()))
        parity_adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unlisted_sender_dropped(self, parity_adapter):
        await parity_adapter.dispatch_http_event(_group_envelope(sender="users/000"))
        parity_adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_mention_dropped(self, parity_adapter):
        await parity_adapter.dispatch_http_event(_group_envelope(annotations=[]))
        parity_adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabled_space_dropped(self, tmp_path):
        groups = {_SPACE: {"enabled": False, "users": list(_USERS)}}
        adapter = GoogleChatAdapter(_parity_config(groups=groups))
        adapter.handle_message = AsyncMock()
        from plugins.platforms.google_chat.adapter import _ThreadCountStore
        adapter._thread_count_store = _ThreadCountStore(tmp_path / "t.json")
        await adapter.dispatch_http_event(_group_envelope())
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bot_sender_still_self_filtered(self, parity_adapter):
        env = _group_envelope(sender=_BOT, sender_type="BOT")
        await parity_adapter.dispatch_http_event(env)
        parity_adapter.handle_message.assert_not_awaited()

    def test_pubsub_bot_filter_still_acks_without_dispatch(self, parity_adapter):
        env = _group_envelope(sender=_BOT, sender_type="BOT")
        msg = _make_pubsub_message(env)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(parity_adapter, "_submit_on_loop", MagicMock())
            parity_adapter._on_pubsub_message(msg)
            parity_adapter._submit_on_loop.assert_not_called()
        msg.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_unpaired_dm_reaches_gateway_but_is_not_role_authorized(self, parity_adapter):
        await parity_adapter.dispatch_http_event(_dm_envelope())
        parity_adapter.handle_message.assert_awaited_once()
        event = parity_adapter.handle_message.await_args.args[0]
        assert event.source.chat_type == "dm"
        assert event.source.user_id == _USERS[0]
        assert getattr(event.source, "role_authorized", False) is not True


class TestGatewayAuthzSeam:
    def _runner(self, adapter, approved=False):
        from gateway.run import GatewayRunner
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
        runner = GatewayRunner(GatewayConfig())
        runner.adapters = {_GC: adapter}
        runner.pairing_store = MagicMock()
        runner.pairing_store.is_approved = MagicMock(return_value=approved)
        return runner

    def test_group_grant_does_not_authorize_dm(self, parity_adapter, monkeypatch):
        monkeypatch.delenv("GOOGLE_CHAT_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GOOGLE_CHAT_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        runner = self._runner(parity_adapter, approved=False)
        group = SessionSource(
            platform=_GC, chat_id=_SPACE, chat_type="group", user_id=_USERS[0], user_name="Alice",
        )
        dm = SessionSource(
            platform=_GC, chat_id="spaces/DM", chat_type="dm", user_id=_USERS[0], user_name="Alice",
        )
        assert runner._is_user_authorized(group) is True
        assert runner._is_user_authorized(dm) is False

    def test_paired_dm_uses_users_id(self, parity_adapter, monkeypatch):
        monkeypatch.delenv("GOOGLE_CHAT_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GOOGLE_CHAT_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        runner = self._runner(parity_adapter, approved=True)
        dm = SessionSource(
            platform=_GC, chat_id="spaces/DM", chat_type="dm", user_id=_USERS[0], user_name="Alice",
        )
        assert runner._is_user_authorized(dm) is True
        runner.pairing_store.is_approved.assert_called()
        args = runner.pairing_store.is_approved.call_args.args
        assert args[1] == _USERS[0]


class TestLegacyIdentityAndConnect:
    @pytest.mark.asyncio
    async def test_legacy_user_id_remains_email(self, tmp_path):
        from plugins.platforms.google_chat.adapter import _ThreadCountStore

        adapter = GoogleChatAdapter(_base_config())
        adapter._thread_count_store = _ThreadCountStore(tmp_path / "t.json")
        env = _make_chat_envelope()
        event = await adapter._build_message_event(env["chat"]["messagePayload"]["message"], env)
        assert event.source.user_id == "u@example.com"
        assert event.source.user_id_alt == "users/12345"
        assert adapter.enforces_own_access_policy is False

    @pytest.mark.asyncio
    async def test_connect_keeps_configured_bot_user_over_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cache = tmp_path / "google_chat_bot_id.json"
        cache.write_text(json.dumps({"bot_user_id": "users/STALE"}), encoding="utf-8")
        cfg = _parity_config()
        cfg.extra["http_events_url"] = "https://example.test/google-chat/events"
        adapter = GoogleChatAdapter(cfg)
        monkeypatch.setattr(_gc_mod, "_load_google_modules", lambda: True)
        monkeypatch.setattr(adapter, "_load_sa_credentials", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(_gc_mod, "build_service", MagicMock(return_value=MagicMock()))
        adapter._resolve_bot_user_id = AsyncMock(return_value="users/LOOKUP")
        assert await adapter.connect() is True
        assert adapter._bot_user_id == _BOT
        adapter._resolve_bot_user_id.assert_not_awaited()
        await adapter.disconnect()


class TestReplyToModeAll:
    def test_second_chunk_keeps_thread_when_all(self, parity_adapter):
        assert parity_adapter._reply_to_mode == "all"
        assert parity_adapter._should_thread_chunk(1, "spaces/S/threads/T") is True

    def test_second_chunk_drops_thread_when_first(self, tmp_path):
        adapter = GoogleChatAdapter(_parity_config(reply_to_mode="first"))
        assert adapter._source_parity_access is True
        assert adapter._should_thread_chunk(1, "spaces/S/threads/T") is False
        assert adapter._should_thread_chunk(0, "spaces/S/threads/T") is True

    def test_legacy_default_first_still_threads_later_chunks(self):
        adapter = GoogleChatAdapter(_base_config())
        assert adapter._reply_to_mode == "first"
        assert adapter._source_parity_access is False
        assert adapter._should_thread_chunk(1, "spaces/S/threads/T") is True

    def test_yaml_off_parses_false_and_maps_to_off_in_parity_mode(self):
        loaded = yaml.safe_load("reply_to_mode: off\n")
        assert loaded["reply_to_mode"] is False
        cfg = _parity_config()
        cfg.reply_to_mode = loaded["reply_to_mode"]
        adapter = GoogleChatAdapter(cfg)
        assert adapter._source_parity_access is True
        assert adapter._reply_to_mode == "off"
        assert adapter._should_thread_chunk(0, "spaces/S/threads/T") is False

    def test_parity_off_skips_cached_thread(self, parity_adapter):
        parity_adapter._reply_to_mode = "off"
        parity_adapter._last_inbound_thread["spaces/X"] = "spaces/X/threads/CACHED"
        assert parity_adapter._resolve_thread_id(None, None, chat_id="spaces/X") is None

    def test_parity_off_ignores_explicit_metadata_thread(self, parity_adapter):
        parity_adapter._reply_to_mode = "off"
        assert parity_adapter._resolve_thread_id(
            None,
            {"thread_id": "spaces/X/threads/EXPLICIT"},
            chat_id="spaces/X",
        ) is None

    @pytest.mark.asyncio
    async def test_parity_off_send_card_does_not_thread(self, parity_adapter):
        parity_adapter._reply_to_mode = "off"
        parity_adapter._create_message = AsyncMock(
            return_value=type("R", (), {"success": True, "message_id": "m/1", "error": None, "raw_response": None})()
        )
        await parity_adapter.send_card(
            "spaces/X",
            {"header": {"title": "card"}},
            metadata={"thread_id": "spaces/X/threads/EXPLICIT"},
        )
        body = parity_adapter._create_message.await_args.args[1]
        assert "thread" not in body

    def test_legacy_off_still_uses_cached_thread(self):
        adapter = GoogleChatAdapter(_base_config())
        adapter._reply_to_mode = "off"
        adapter._last_inbound_thread["spaces/X"] = "spaces/X/threads/CACHED"
        assert adapter._source_parity_access is False
        assert adapter._resolve_thread_id(None, None, chat_id="spaces/X") == "spaces/X/threads/CACHED"

    def test_legacy_off_still_honors_explicit_metadata_thread(self):
        adapter = GoogleChatAdapter(_base_config())
        adapter._reply_to_mode = "off"
        assert adapter._source_parity_access is False
        assert adapter._resolve_thread_id(
            None,
            {"thread_id": "spaces/X/threads/EXPLICIT"},
            chat_id="spaces/X",
        ) == "spaces/X/threads/EXPLICIT"



def _clear_auth_env(monkeypatch):
    for key in (
        "GOOGLE_CHAT_ALLOWED_USERS",
        "GOOGLE_CHAT_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "GOOGLE_CHAT_DM_POLICY",
        "GOOGLE_CHAT_GROUP_POLICY",
        "GOOGLE_CHAT_REQUIRE_MENTION",
        "GOOGLE_CHAT_BOT_USER",
        "GOOGLE_CHAT_HTTP_EVENTS_APP_PRINCIPAL",
    ):
        monkeypatch.delenv(key, raising=False)


def _auth_runner(adapter, store):
    from gateway.run import GatewayRunner
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    runner = GatewayRunner(GatewayConfig())
    runner.adapters = {_GC: adapter}
    runner.pairing_store = store
    return runner


async def _built_dm(adapter, sender, email):
    env = _dm_envelope(sender=sender, email=email)
    message = env["chat"]["messagePayload"]["message"]
    return await adapter._build_message_event(message, env)


class TestResourceIdentity:
    @pytest.mark.asyncio
    async def test_resource_id_keeps_case(self, parity_adapter):
        event = await _built_dm(parity_adapter, "users/AbC", "alice@example.com")
        assert event.source.user_id == "users/AbC"
        assert event.source.user_id_alt == "alice@example.com"
        assert getattr(event.source, "role_authorized", False) is not True

    def test_email_match_ignores_case_and_resource_match_does_not(self):
        from plugins.platforms.google_chat.adapter import _gchat_user_matches

        assert _gchat_user_matches("User@example.com", {"user@example.com"}) is True
        assert _gchat_user_matches("users/AbC", {"users/abc"}) is False
        email_adapter = GoogleChatAdapter(_parity_config(
            dm_policy="allowlist", group_policy="", groups={}, allow_from="user@example.com",
        ))
        resource_adapter = GoogleChatAdapter(_parity_config(
            dm_policy="allowlist", group_policy="", groups={}, allow_from="users/abc",
        ))
        assert email_adapter._is_dm_allowed("User@example.com") is True
        assert email_adapter._is_dm_allowed("users/AbC") is False
        assert resource_adapter._is_dm_allowed("users/abc") is True
        assert resource_adapter._is_dm_allowed("users/AbC") is False


class TestEmailAuthorizationBoundary:
    @pytest.mark.asyncio
    async def test_allowlisted_email_authorizes_exact_resource(self, parity_adapter, monkeypatch, tmp_path):
        from gateway.pairing import PairingStore

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", "alice@example.com")
        event = await _built_dm(parity_adapter, "users/AbC", "alice@example.com")
        runner = _auth_runner(parity_adapter, PairingStore())
        assert event.source.user_id == "users/AbC"
        assert getattr(event.source, "role_authorized", False) is not True
        assert runner._is_user_authorized(event.source) is True

    @pytest.mark.asyncio
    async def test_other_email_and_case_distinct_resource_denied(self, parity_adapter, monkeypatch, tmp_path):
        from gateway.pairing import PairingStore

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GOOGLE_CHAT_ALLOWED_USERS", "users/abc")
        other = await _built_dm(parity_adapter, "users/AbC", "bob@example.com")
        bare = await _built_dm(parity_adapter, "users/AbC", "")
        same = await _built_dm(parity_adapter, "users/abc", "")
        runner = _auth_runner(parity_adapter, PairingStore())
        assert runner._is_user_authorized(other.source) is False
        assert runner._is_user_authorized(bare.source) is False
        assert runner._is_user_authorized(same.source) is True

    @pytest.mark.asyncio
    async def test_paired_email_authorizes_only_that_email(self, parity_adapter, monkeypatch, tmp_path):
        from gateway.pairing import PairingStore

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = PairingStore()
        with store._lock:
            store._approve_user("google_chat", "alice@example.com", "Alice")
        paired = await _built_dm(parity_adapter, "users/AbC", "alice@example.com")
        other = await _built_dm(parity_adapter, "users/AbC", "bob@example.com")
        resource_only = await _built_dm(parity_adapter, "users/abc", "")
        runner = _auth_runner(parity_adapter, store)
        assert store.is_approved("google_chat", "users/AbC") is False
        assert store.is_approved("google_chat", "alice@example.com") is True
        assert getattr(paired.source, "role_authorized", False) is not True
        assert runner._is_user_authorized(paired.source) is True
        assert runner._is_user_authorized(other.source) is False
        assert runner._is_user_authorized(resource_only.source) is False

    def test_non_google_alt_email_is_not_an_allowlist_grant(self, monkeypatch, tmp_path):
        from gateway.pairing import PairingStore

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "alice@example.com")
        runner = _auth_runner(None, PairingStore())
        runner.adapters = {}
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="1",
            chat_type="dm",
            user_id="999",
            user_id_alt="alice@example.com",
        )
        assert runner._is_user_authorized(source) is False


class TestDmAllowlistShapes:
    def _allow_adapter(self, **extra):
        cfg = _parity_config(dm_policy="allowlist", group_policy="", groups={}, **extra)
        return GoogleChatAdapter(cfg)

    def test_scalar_allow_from_matches_whole_id(self, monkeypatch, tmp_path):
        from gateway.pairing import PairingStore

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = self._allow_adapter(allow_from="users/123")
        runner = _auth_runner(adapter, PairingStore())
        allowed = SessionSource(platform=_GC, chat_id="spaces/DM", chat_type="dm", user_id="users/123")
        denied = SessionSource(platform=_GC, chat_id="spaces/DM", chat_type="dm", user_id="users/999")
        character = SessionSource(platform=_GC, chat_id="spaces/DM", chat_type="dm", user_id="u")
        assert runner._is_user_authorized(allowed) is True
        assert runner._is_user_authorized(denied) is False
        assert runner._is_user_authorized(character) is False

    def test_allow_from_camel_case_and_email_scalar(self, monkeypatch, tmp_path):
        from gateway.pairing import PairingStore

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = GoogleChatAdapter(_parity_config(
            dm_policy="allowlist", group_policy="", groups={}, allow_from=None, allowFrom="alice@example.com",
        ))
        # _parity_extra copies overrides after the default groups, then _parity_config
        # passes them into _base_config. allow_from=None must not block allowFrom.
        adapter.config.extra.pop("allow_from", None)
        adapter.config.extra["allowFrom"] = "alice@example.com"
        runner = _auth_runner(adapter, PairingStore())
        allowed = SessionSource(
            platform=_GC, chat_id="spaces/DM", chat_type="dm",
            user_id="users/AbC", user_id_alt="alice@example.com",
        )
        denied = SessionSource(
            platform=_GC, chat_id="spaces/DM", chat_type="dm",
            user_id="users/abc", user_id_alt="bob@example.com",
        )
        assert runner._is_user_authorized(allowed) is True
        assert runner._is_user_authorized(denied) is False


class TestParityYamlPrecedence:
    def test_yaml_policy_wins_over_deprecated_env(self, monkeypatch):
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_DM_POLICY", "disabled")
        monkeypatch.setenv("GOOGLE_CHAT_BOT_USER", "users/from-env")
        adapter = GoogleChatAdapter(_parity_config(dm_policy="allowlist", bot_user="users/FromYaml"))
        assert adapter._dm_policy == "allowlist"
        assert adapter._bot_user == "users/FromYaml"

    def test_env_fallback_when_yaml_key_absent(self, monkeypatch):
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_DM_POLICY", "pairing")
        cfg = _base_config()
        adapter = GoogleChatAdapter(cfg)
        assert adapter._dm_policy == "pairing"
        assert adapter._source_parity_access is True

    def test_env_seed_does_not_carry_behavior_keys(self, monkeypatch):
        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("GOOGLE_CHAT_PROJECT_ID", "p")
        monkeypatch.setenv("GOOGLE_CHAT_SUBSCRIPTION_NAME", "projects/p/subscriptions/s")
        monkeypatch.setenv("GOOGLE_CHAT_DM_POLICY", "disabled")
        monkeypatch.setenv("GOOGLE_CHAT_REQUIRE_MENTION", "false")
        seed = _gc_mod._env_enablement() or {}
        assert "dm_policy" not in seed
        assert "require_mention" not in seed
        assert "bot_user" not in seed
        assert "group_policy" not in seed
        assert "http_events_app_principal" not in seed

    def test_loader_keeps_extra_policy(self, monkeypatch, tmp_path):
        from gateway.config import load_gateway_config

        _clear_auth_env(monkeypatch)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("GOOGLE_CHAT_DM_POLICY", "disabled")
        (tmp_path / "config.yaml").write_text(
            "platforms:\n"
            "  google_chat:\n"
            "    enabled: true\n"
            "    extra:\n"
            "      dm_policy: allowlist\n"
            "      group_policy: allowlist\n"
            "      require_mention: true\n"
            "      groups:\n"
            "        spaces/KEEPCASE:\n"
            "          enabled: true\n"
            "          users:\n"
            "            - users/AbC\n",
            encoding="utf-8",
        )
        cfg = load_gateway_config()
        extra = cfg.platforms[_GC].extra
        assert extra["dm_policy"] == "allowlist"
        assert extra["groups"]["spaces/KEEPCASE"]["users"] == ["users/AbC"]
