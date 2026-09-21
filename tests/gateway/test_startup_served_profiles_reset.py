"""A new gateway must publish its own served roster, never its predecessor's."""

import asyncio
from types import SimpleNamespace

from gateway import status
from gateway.run_adapters import GatewayAdapterLifecycleMixin
from gateway.run_startup import GatewayStartupMixin


def _start_over_old_roster(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.monitoring import gateway_health_export
    monkeypatch.setattr(
        gateway_health_export, "start_gateway_health_export",
        lambda config: SimpleNamespace(enabled=False),
    )
    status.write_runtime_status(gateway_state="stopped", served_profiles=["default", "old"])
    runner = SimpleNamespace(
        config=SimpleNamespace(sessions_dir=tmp_path / "sessions"),
        _start_log_systemd_timing_alignment=lambda: None,
        _start_loop_liveness_guards=lambda _loop: None,
        _log_agent_budget=lambda: None,
        _note_served_profiles=lambda _homes: None,
    )
    asyncio.run(GatewayStartupMixin._start_log_startup_environment(runner))
    return runner


def test_multiplex_to_single_profile_start_clears_old_roster(tmp_path, monkeypatch):
    runner = _start_over_old_roster(tmp_path, monkeypatch)
    runner._multiplex_on = lambda: False
    assert asyncio.run(GatewayAdapterLifecycleMixin._start_secondary_profile_adapters(runner)) == 0
    status.write_runtime_status(gateway_state="running")
    assert status.read_runtime_status()["served_profiles"] == []


def test_multiplex_restart_rebuilds_roster_after_empty_startup(tmp_path, monkeypatch):
    runner = _start_over_old_roster(tmp_path, monkeypatch)
    assert status.read_runtime_status()["served_profiles"] == []
    runner.pairing_stores = {"default": object(), "new": object()}
    GatewayAdapterLifecycleMixin._record_served_profiles(
        runner, "default", [("default", tmp_path), ("new", tmp_path / "profiles" / "new")],
    )
    status.write_runtime_status(gateway_state="running")
    assert status.read_runtime_status()["served_profiles"] == ["default", "new"]
