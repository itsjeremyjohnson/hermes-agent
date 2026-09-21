"""Per-job script bounds exercise real storage, pipes and process-tree cleanup."""

import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pytest

from cron import jobs, scheduler, scheduler_script


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "scripts").mkdir()
    with jobs.use_cron_store(tmp_path):
        yield tmp_path


def test_limits_round_trip_and_invalid_updates_are_atomic(home):
    job = jobs.create_job(None, "every 1h", script="cache.py", no_agent=True,
                          paused=True, script_timeout_seconds=300,
                          script_max_output_bytes=20000)
    stored = jobs.get_job(job["id"])
    assert stored["script_timeout_seconds"] == 300
    assert stored["script_max_output_bytes"] == 20000
    before = (home / "cron/jobs.json").read_bytes()
    for field in ("script_timeout_seconds", "script_max_output_bytes"):
        for invalid in (0, -1, True, 1.5, "300", [], {}, 2**63):
            with pytest.raises(ValueError, match=field):
                jobs.update_job(job["id"], {field: invalid})
            with pytest.raises(ValueError, match=field):
                jobs.create_job(None, "every 1h", script="cache.py", no_agent=True,
                                **{field: invalid})
            assert (home / "cron/jobs.json").read_bytes() == before
    cleared = jobs.update_job(job["id"], {"script_timeout_seconds": None,
                                          "script_max_output_bytes": None})
    assert cleared.get("script_timeout_seconds") is None
    assert cleared.get("script_max_output_bytes") is None


def test_job_override_is_independent_of_default_and_other_jobs(home, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SCRIPT_TIMEOUT", "2")
    monkeypatch.setattr(scheduler, "_SCRIPT_TIMEOUT", scheduler._DEFAULT_SCRIPT_TIMEOUT)
    (home / "scripts/slow.py").write_text("import time; time.sleep(3); print('finished')\n")
    with ThreadPoolExecutor(max_workers=2) as pool:
        inherited = pool.submit(scheduler_script._run_job_script_with_claim_heartbeat,
                                {}, "slow.py")
        overridden = pool.submit(scheduler_script._run_job_script_with_claim_heartbeat,
                                 {"script_timeout_seconds": 6}, "slow.py")
        assert inherited.result(timeout=12)[0] is False
        assert overridden.result(timeout=12) == (True, "finished")
    assert os.environ["HERMES_CRON_SCRIPT_TIMEOUT"] == "2"
    job = jobs.create_job(None, "every 1h", script="slow.py", no_agent=True, paused=True)
    assert "script_timeout_seconds" not in job
    assert "script_max_output_bytes" not in job


@pytest.mark.parametrize("exit_code", [0, 17])
def test_output_limit_keeps_each_stream_tail_and_preserves_exit(home, monkeypatch, exit_code):
    cap = 13
    stdout = "é" * 40000 + "OUT-DONE"
    stderr = "☃" * 30000 + "ERR-DONE"
    marker = home / "finished"
    (home / "scripts/large.py").write_text(
        "import sys\n"
        "sys.stdout.buffer.write(('é'*40000+'OUT-DONE').encode()); sys.stdout.flush()\n"
        "sys.stderr.buffer.write(('☃'*30000+'ERR-DONE').encode()); sys.stderr.flush()\n"
        f"open({str(marker)!r}, 'w').write('finished')\n"
        f"sys.exit({exit_code})\n")
    captured = {}
    real_collect = scheduler_script._collect_bounded_script_output
    def observe_collection(proc, *args, **kwargs):
        result = real_collect(proc, *args, **kwargs)
        captured.update(stdout=result[0], stderr=result[1], error=result[2], exit=proc.returncode)
        return result
    monkeypatch.setattr(scheduler_script, "_collect_bounded_script_output", observe_collection)
    ok, output = scheduler_script._run_job_script_with_claim_heartbeat(
        {"script_max_output_bytes": cap}, "large.py")
    assert marker.read_text() == "finished"  # Large output does not interrupt the script.
    assert captured["error"] is None
    assert captured["exit"] == exit_code
    for stream, text in (("stdout", stdout), ("stderr", stderr)):
        assert captured[stream] == text.encode()[-cap:]
        assert len(captured[stream]) <= cap
    expected_stdout = stdout.encode()[-cap:].decode("utf-8", errors="replace")
    expected_stderr = stderr.encode()[-cap:].decode("utf-8", errors="replace")
    if exit_code == 0:
        assert (ok, output) == (True, expected_stdout)
    else:
        assert not ok and f"Script exited with code {exit_code}" in output
        assert expected_stdout in output and expected_stderr in output


def test_malformed_stored_limits_never_execute(home):
    marker = home / "executed"
    (home / "scripts/marker.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    for field in ("script_timeout_seconds", "script_max_output_bytes"):
        for invalid in (0, -1, True, 1.5, "300", [], {}, 2**63):
            ok, error = scheduler_script._run_job_script_with_claim_heartbeat(
                {field: invalid}, "marker.py")
            assert not ok and field in error
            assert not marker.exists()


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("cause", ["timeout", "cancel"])
def test_failure_stops_own_session_descendant_and_reader_threads(home, cause):
    pidfile = home / "descendant.pid"
    (home / "scripts/tree.py").write_text(
        "import subprocess, sys, time, os\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "os.write(2, b'x'*10000)\n"
        "time.sleep(30)\n")
    cancelled = threading.Event()
    def cancel_when_ready():
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        cancelled.set()
    cancel_thread = None
    if cause == "cancel":
        cancel_thread = threading.Thread(target=cancel_when_ready)
        cancel_thread.start()
    started = time.monotonic()
    try:
        ok, error = scheduler_script._run_job_script_with_claim_heartbeat(
            {"script_timeout_seconds": 2, "script_max_output_bytes": 100}, "tree.py",
            cancel_event=cancelled)
        assert not ok
        assert {"timeout": "timed out", "cancel": "ownership"}[cause] in error
        assert time.monotonic() - started < 12
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 3
        while psutil.pid_exists(pid) and time.monotonic() < deadline:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(.02)
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        assert not any(t.name.startswith("cron-script-output-") for t in threading.enumerate())
    finally:
        if cancel_thread is not None:
            cancel_thread.join(timeout=6)
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("detached", [False, True])
def test_timeout_releases_readers_after_leader_exits(home, detached):
    pidfile = home / "orphan.pid"
    (home / "scripts/orphan.py").write_text(
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        f"start_new_session={detached!r})\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n")
    try:
        ok, error = scheduler_script._run_job_script_with_claim_heartbeat(
            {"script_timeout_seconds": 2, "script_max_output_bytes": 100}, "orphan.py")
        assert not ok and "timed out" in error
        pid = int(pidfile.read_text())
        if not detached:
            deadline = time.monotonic() + 3
            while psutil.pid_exists(pid) and time.monotonic() < deadline:
                if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                    break
                time.sleep(.02)
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        # A detached, already-reparented process no longer has provable ancestry. The
        # runner must release its readers without claiming that process was terminated.
        assert not any(t.name.startswith("cron-script-output-") for t in threading.enumerate())
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("unsupported_pipe", [1, 2])
def test_nonblocking_setup_failure_reaps_child_and_started_reader(home, monkeypatch, unsupported_pipe):
    (home / "scripts/wait.py").write_text("import time; time.sleep(30)\n")
    processes = []
    real_popen, real_set_blocking = subprocess.Popen, os.set_blocking
    calls = 0
    def capture_spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process
    def unsupported(fd, blocking):
        nonlocal calls
        calls += 1
        if calls == unsupported_pipe:
            raise OSError("nonblocking pipes unsupported")
        return real_set_blocking(fd, blocking)
    monkeypatch.setattr(scheduler_script.subprocess, "Popen", capture_spawn)
    monkeypatch.setattr(scheduler_script.os, "set_blocking", unsupported)
    ok, error = scheduler_script._run_job_script_with_claim_heartbeat(
        {"script_max_output_bytes": 100}, "wait.py")
    assert not ok and "nonblocking pipes unsupported" in error
    assert processes and all(process.poll() is not None for process in processes)
    assert not any(t.name.startswith("cron-script-output-") for t in threading.enumerate())
