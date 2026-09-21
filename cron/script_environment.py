"""Cron scripts inherit their current profile's credentials, then normal scrub policy."""
import os

import re
import sqlite3

from hermes_constants import get_hermes_home

from agent.secret_scope import (
    UnscopedSecretError, _is_global_env, current_secret_scope, is_multiplex_active,
)
from tools.environments.local import build_subprocess_env


WORKER_MARKER = "_HERMES_CRON_EXTERNAL_WORKER"


class WorkerExecutionBindingError(UnscopedSecretError):
    """A worker marker cannot prove ownership of its current execution row."""


def _owned_worker_execution() -> str:
    value = os.environ.get(WORKER_MARKER)
    if value is None:
        return ""
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise WorkerExecutionBindingError("cron worker execution binding invalid")

    from cron import executions
    path = get_hermes_home().resolve() / "cron" / "executions.db"
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT status, process_id, pid, process_started_at FROM executions WHERE id=?",
                (value,)).fetchone()
    except sqlite3.Error as exc:
        raise WorkerExecutionBindingError("cron worker execution binding invalid") from exc

    expected_started = executions._process_start_time(os.getpid())
    owned = (
        row is not None
        and row["status"] == "running"
        and row["process_id"] == executions._PROCESS_ID
        and row["pid"] == os.getpid()
        and expected_started is not None
        and row["process_started_at"] == expected_started
    )
    if not owned:
        raise WorkerExecutionBindingError("cron worker execution binding invalid")
    return value


def build_cron_script_env() -> dict[str, str]:
    worker_execution = _owned_worker_execution()
    if not is_multiplex_active():
        return build_subprocess_env()
    scope = current_secret_scope()
    if scope is None:
        raise UnscopedSecretError('multiplex cron script requires its profile secret scope')
    # Worker processes may inherit default-profile .env values. Correct HOME
    # alone cannot make an unregistered CLI key safe: missing own keys must stay
    # missing, and keys absent from the parent's env must still reach this job.
    base = {key: value for key, value in os.environ.items() if _is_global_env(key)}
    # Raw profile dotenv scopes may contain deployment names. Those retain the
    # same process authority as get_secret; only profile-owned values overlay.
    base.update((key, value) for key, value in scope.items() if not _is_global_env(key))
    if worker_execution:
        base[WORKER_MARKER] = worker_execution
    return build_subprocess_env(base=base)
