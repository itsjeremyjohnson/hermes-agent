"""Profile-local original bot posts for Google Chat thread replies."""
from __future__ import annotations

import json
from contextlib import contextmanager
import os
import tempfile
from pathlib import Path
from typing import Any


class ThreadContextStore:
    """Bounded disk store shared by the gateway and standalone senders.

    Reads refresh from disk so a running gateway sees posts from cron. Writers lock
    across processes and replace atomically, preserving other concurrent sends.
    """
    def __init__(self, path: Path, limit: int = 1000):
        self.path = path
        self.limit = limit

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Google Chat thread context must be an object")
        return data

    @staticmethod
    def _key(space: str, thread: str) -> str:
        return json.dumps([space, thread], separators=(",", ":"))

    def get(self, space: str, thread: str | None) -> dict[str, Any] | None:
        if not thread:
            return None
        value = self._read().get(self._key(space, thread))
        if isinstance(value, dict) and isinstance(value.get("text"), str) and value.get("message_id"):
            return value
        return None

    def record(self, space: str, body: dict[str, Any], response: dict[str, Any]) -> bool:
        # Explicit replies must never replace the root or establish a new trusted thread.
        if body.get("thread"):
            return False
        thread = (response.get("thread") or {}).get("name")
        message_id = response.get("name")
        text = body.get("text")
        if not thread or not message_id or not isinstance(text, str) or not text:
            return False
        with self._locked_data() as data:
            key = self._key(space, thread)
            if key in data:
                return True
            data[key] = {"message_id": message_id, "text": text, "posted_at": response.get("createTime")}
            while len(data) > self.limit:
                del data[next(iter(data))]
            self._write(data)
        return True

    def record_related_message(self, space: str, thread: str, expected_root_message_id: str,
                               message_id: str, text: str) -> bool:
        """Append a confirmed bot reply to an explicitly identified existing root.

        The caller invokes this only after its outbound reply was accepted. The root
        pointer remains unchanged; repeated recording of the same receipt is idempotent.
        """
        if not expected_root_message_id or not message_id or not text:
            return False
        with self._locked_data() as data:
            root = data.get(self._key(space, thread))
            if not isinstance(root, dict) or root.get("message_id") != expected_root_message_id:
                return False
            related = root.setdefault("related_messages", {})
            if message_id in related:
                return True
            related[message_id] = text
            root["text"] += "\n\n" + text
            self._write(data)
        return True

    @contextmanager
    def _locked_data(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            os.chmod(lock.name, 0o600)
            if os.name == "nt":
                import msvcrt
                lock.write(" ")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX)
            yield self._read()

    def _write(self, data: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".google_chat_context_")
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(data, output, ensure_ascii=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def profile_thread_context_store() -> ThreadContextStore:
    from hermes_constants import get_hermes_home
    return ThreadContextStore(get_hermes_home() / "google_chat_thread_context.json")
