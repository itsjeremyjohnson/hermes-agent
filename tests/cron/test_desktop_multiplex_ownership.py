"""Desktop cron yields all store writes to profiles already owned by a gateway."""

import os
import json
from pathlib import Path
import threading


def test_desktop_gate_uses_verified_runtime_membership(monkeypatch, tmp_path):
    import hermes_constants
    from gateway import status
    from cron import scheduler_provider
    from hermes_cli import profiles, web_server

    root = tmp_path / "hermes"
    served = root / "profiles" / "served"
    unserved = root / "profiles" / "unserved"
    for home in (served, unserved):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("platforms: {}\n")
    (root / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n"
        "  multiplex_profile_allowlist: [served]\n"
    )
    # Actual runtime/config reads; only positive process-identity verification is
    # a fixture seam. A separate full native gateway probe covers that authority.
    (root / "gateway.pid").write_text(str(os.getpid()))
    runtime_record = {"pid": os.getpid(), "gateway_state": "running",
                      "served_profiles": ["default", "served"]}
    (root / "gateway_state.json").write_text(json.dumps(runtime_record))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    homes = list(profiles.profiles_to_serve(multiplex=True))
    assert ("served", served) in homes
    assert ("unserved", unserved) in homes
    assert not profiles._check_gateway_running(served)
    monkeypatch.setattr(
        status, "get_runtime_status_running_pid",
        lambda record, *, expected_home: (
            os.getpid() if record == runtime_record and expected_home == root else None
        ),
    )

    class RecordingBuiltin(scheduler_provider.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            self.start_kwargs = kwargs

    provider = RecordingBuiltin()
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: provider)
    web_server._start_desktop_cron_ticker(threading.Event())
    gate = provider.start_kwargs["profile_gate"]
    assert not gate("served", served)
    assert gate("unserved", unserved)
    # Config changes do not change the roster of the already-running gateway.
    (root / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n  multiplex_profile_allowlist: [unserved]\n"
    )
    assert not gate("served", served)
    assert gate("unserved", unserved)
    # The existing liveness authority must be consulted again on the next tick.
    (root / "gateway.pid").unlink()
    (root / "gateway_state.json").unlink()
    assert gate("served", served)


def test_desktop_gate_rejects_stale_record_pointing_at_unrelated_live_pid(monkeypatch, tmp_path):
    import hermes_constants
    from cron import scheduler_provider
    from hermes_cli import profiles, web_server

    root = tmp_path / "hermes"
    served = root / "profiles" / "served"
    served.mkdir(parents=True)
    (served / "config.yaml").write_text("platforms: {}\n")
    (root / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n  multiplex_profile_allowlist: [served]\n"
    )
    # A stale gateway record can refer to an unrelated process after PID reuse.
    # Use the actual test PID: alive, but neither a gateway command nor lock owner.
    (root / "gateway.pid").write_text(str(os.getpid()))
    (root / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "gateway_state": "running", "hermes_home": str(root),
        "served_profiles": ["default", "served"],
    }))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    assert not profiles._served_by_running_multiplexer("served")
    assert not profiles._check_gateway_running(root)

    class RecordingBuiltin(scheduler_provider.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            self.start_kwargs = kwargs

    provider = RecordingBuiltin()
    monkeypatch.setattr(scheduler_provider, "resolve_cron_scheduler", lambda: provider)
    web_server._start_desktop_cron_ticker(threading.Event())
    gate = provider.start_kwargs["profile_gate"]
    assert gate("served", served)
    assert gate("default", root)


def test_rejected_profile_has_no_startup_recovery_or_heartbeat(tmp_path):
    from cron.scheduler_provider import InProcessCronScheduler

    denied, allowed = tmp_path / "owned", tmp_path / "free"
    denied.mkdir()
    allowed.mkdir()
    stop = threading.Event()
    stop.set()  # Exercise actual startup recovery; no tick/agent execution needed.
    InProcessCronScheduler().start(
        stop,
        profile_homes=[("owned", denied), ("free", allowed)],
        profile_gate=lambda name, home: name != "owned",
    )
    assert not list(denied.iterdir()), "A rejected store must remain untouched at startup"
    assert (allowed / "cron" / "ticker_heartbeat").is_file()
    assert (allowed / "cron" / "executions.db").is_file()


def test_profile_gate_reopens_after_startup_without_changing_scope(monkeypatch, tmp_path):
    from cron import scheduler
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_constants import get_hermes_home

    home = tmp_path / "profile"
    home.mkdir()
    decisions = iter((False, True))
    stop = threading.Event()
    ticks = []

    def tick(**kwargs):
        ticks.append(Path(get_hermes_home()))
        stop.set()

    monkeypatch.setattr(scheduler, "tick", tick)
    InProcessCronScheduler().start(
        stop, interval=0, profile_homes=[("profile", home)],
        profile_gate=lambda name, profile_home: next(decisions),
    )
    assert ticks == [home]
    assert (home / "cron" / "ticker_heartbeat").is_file()
