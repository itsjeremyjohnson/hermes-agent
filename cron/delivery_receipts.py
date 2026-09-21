"""Explicitly prepared, profile-local content bindings for actual cron sends.

This is an audit, not a retry queue. Unbound deliveries retain their existing path.
Only the transport boundary can finish a prepared binding; execution completion
and the queue's deferred-success return are never delivery evidence.
"""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import json
import os
import sqlite3

from cron import delivery_queue, executions
from hermes_time import now

MAX_RECEIPTS = 1000
_verification = ContextVar("cron_receipt_verification", default=None)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS delivery_receipts (
        execution_id TEXT PRIMARY KEY, job_id TEXT NOT NULL,
        started_at TEXT NOT NULL, content_sha256 TEXT NOT NULL,
        route_sha256 TEXT NOT NULL, metadata_json TEXT NOT NULL,
        outcome TEXT NOT NULL, prepared_at TEXT NOT NULL,
        finished_at TEXT, owner_pid INTEGER, owner_started_at INTEGER
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS delivery_receipt_pruning (
        job_id TEXT PRIMARY KEY, pruned_count INTEGER NOT NULL
    )""")
    conn.execute("CREATE TABLE IF NOT EXISTS delivery_receipt_tombstones (execution_id TEXT PRIMARY KEY)")
    conn.execute("CREATE INDEX IF NOT EXISTS receipt_job_start ON delivery_receipts(job_id,started_at)")


def _route(job, for_failure=False):
    from cron.scheduler_delivery import _resolve_delivery_targets
    targets = _resolve_delivery_targets(job, for_failure=for_failure)
    return digest(json.dumps(targets, sort_keys=True, separators=(",", ":"))), targets


def _identity(job, content, for_failure=False):
    execution_id = job.get("execution_id")
    record = executions.get_execution(execution_id) if execution_id else None
    if (not record or record["job_id"] != job.get("id") or not record.get("started_at")):
        raise ValueError("delivery receipt requires this job's durable started execution")
    route, configured = _route(job, for_failure)
    return (execution_id, record["job_id"], record["started_at"], digest(content), route), configured


def _lookup(execution_id):
    path = delivery_queue._path()
    if not execution_id or not path.exists():
        return None
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='delivery_receipts'").fetchone():
            return None
        if conn.execute("SELECT 1 FROM delivery_receipt_tombstones WHERE execution_id=?", (execution_id,)).fetchone():
            raise ValueError("delivery binding was retired; no replay")
        row = conn.execute("SELECT * FROM delivery_receipts WHERE execution_id=?", (execution_id,)).fetchone()
    return dict(row) if row else None


def prepare(job, content, *, metadata, quiet=False):
    """Bind an exact proposed result to a real current execution before returning it.

    Callers own domain validation. Metadata is immutable bounded JSON, not a
    credentials or card-content store. Quiet means the caller selected no send;
    if it later tries to send, the transport hook refuses that contradiction.
    """
    if not isinstance(content, str) or type(metadata) is not dict or type(quiet) is not bool:
        raise ValueError("invalid delivery binding")
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 2048:
        raise ValueError("delivery binding metadata exceeds limit")
    identity, configured = _identity(job, content)
    if not quiet and not configured:
        raise ValueError("delivery binding requires a configured external route")
    with delivery_queue._transaction() as conn:
        _schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM delivery_receipt_tombstones WHERE execution_id=?", (identity[0],)).fetchone():
            raise ValueError("delivery binding was retired; cannot rebind")
        prior = conn.execute("SELECT * FROM delivery_receipts WHERE execution_id=?", (identity[0],)).fetchone()
        if prior:
            actual = tuple(prior[key] for key in ("execution_id", "job_id", "started_at", "content_sha256", "route_sha256"))
            if actual != identity or prior["metadata_json"] != encoded or ((prior["outcome"] == "not_requested") != quiet):
                raise ValueError("execution delivery binding is immutable")
            return dict(prior)
        conn.execute("""INSERT INTO delivery_receipts
            (execution_id,job_id,started_at,content_sha256,route_sha256,metadata_json,outcome,prepared_at)
            VALUES (?,?,?,?,?,?,?,?)""", (*identity, encoded, "not_requested" if quiet else "prepared", now().isoformat()))
        _prune(conn)
    return _lookup(identity[0])


def note_verification(verified):
    """Called by the actual transport after all selected targets were attempted."""
    status = _verification.get()
    if status is not None:
        if verified is not True:
            status["unverified"] = True
        status["verified"] = verified is True and not status.get("unverified")


def note_unknown():
    """An in-flight send without confirmation must not establish delivery."""
    status = _verification.get()
    if status is not None:
        status["unknown"] = True


def observe_result(result):
    """Use the native confirmation contract without changing unbound transport behavior."""
    if _verification.get() is not None:
        from cron.scheduler_delivery import _confirm_adapter_delivery
        gaps = []
        confirmed = _confirm_adapter_delivery(result, unverified=gaps)
        note_verification(confirmed and not gaps)


def bound_targets():
    """Reuse the exact validated target snapshot at the actual send boundary."""
    status = _verification.get()
    return deepcopy(status["targets"]) if status is not None else None


def _finish(execution_id, outcome):
    with delivery_queue._transaction() as conn:
        _schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""UPDATE delivery_receipts SET outcome=?,finished_at=?
            WHERE execution_id=? AND outcome='sending' AND owner_pid=? AND owner_started_at IS ?""",
                     (outcome, now().isoformat(), execution_id, os.getpid(), executions._process_start_time(os.getpid())))
        _prune(conn)


def _prune(conn):
    stale = conn.execute("""SELECT execution_id,job_id FROM delivery_receipts
        WHERE outcome NOT IN ('prepared','sending') ORDER BY prepared_at DESC,execution_id DESC
        LIMIT -1 OFFSET ?""", (MAX_RECEIPTS,)).fetchall()
    for row in stale:
        conn.execute("""INSERT INTO delivery_receipt_pruning VALUES (?,1)
            ON CONFLICT(job_id) DO UPDATE SET pruned_count=pruned_count+1""", (row["job_id"],))
        conn.execute("DELETE FROM delivery_receipts WHERE execution_id=?", (row["execution_id"],))
        conn.execute("INSERT OR IGNORE INTO delivery_receipt_tombstones VALUES (?)", (row["execution_id"],))


def track_delivery(send):
    """Wrap the existing transport without configuring new jobs or changing unbound sends."""
    @wraps(send)
    def tracked(job, content, adapters=None, loop=None, *, for_failure=False):
        # A later failed execution has a different native summary/route. It is
        # not the prepared normal card, and must retain its existing alert path.
        binding = None if for_failure else _lookup(job.get("execution_id"))
        if binding is None:
            token = _verification.set(None)
            try:
                return send(job, content, adapters=adapters, loop=loop, for_failure=for_failure)
            finally:
                _verification.reset(token)
        # Preparation already bound the actual ledger start. Deferred delivery
        # can outlive normal execution-ledger retention; do not reread/prune that
        # independent authority while the immutable delivery binding survives.
        route, targets = _route(job, for_failure)
        identity = (job.get("execution_id"), job.get("id"), binding["started_at"], digest(content), route)
        expected = tuple(binding[key] for key in ("execution_id", "job_id", "started_at", "content_sha256", "route_sha256"))
        if identity != expected or not targets:
            raise ValueError("delivery does not match its prepared execution/content/route binding")
        # A non-Bot-Chat external worker only queues the send. None from that wait
        # means still pending, so this call must not claim or terminalize the receipt.
        # Bot Chat is delivered directly and has to record that outcome now.
        from cron.scheduler_delivery import BOT_CHAT_PLATFORM
        deferred = (
            adapters is None
            and os.environ.get("_HERMES_CRON_EXTERNAL_WORKER") == identity[0]
            and any(str(target.get("platform") or "") != BOT_CHAT_PLATFORM for target in targets)
        )
        if deferred:
            return send(job, content, adapters=adapters, loop=loop, for_failure=for_failure)
        with delivery_queue._transaction() as conn:
            _schema(conn)
            cur = conn.execute("""UPDATE delivery_receipts SET outcome='sending',owner_pid=?,owner_started_at=?
                WHERE execution_id=? AND outcome='prepared'""",
                               (os.getpid(), executions._process_start_time(os.getpid()), identity[0]))
            if cur.rowcount != 1:
                # Returning success here could misrepresent a concurrent/unknown
                # attempt. Do not call transport again, even after owner death.
                raise ValueError("bound delivery was already claimed or is terminal; no replay")
        verification = {"verified": False, "targets": deepcopy(targets)}
        token = _verification.set(verification)
        try:
            try:
                error = send(job, content, adapters=adapters, loop=loop, for_failure=for_failure)
            except BaseException:
                _finish(identity[0], "unknown")
                raise
            outcome = ("unknown" if verification.get("unknown") else "failed" if error
                       else "delivered" if verification["verified"] else "unverified")
            _finish(identity[0], outcome)
            return error
        finally:
            _verification.reset(token)
    return tracked


def history(job_id):
    """Return bound records and explicit loss of retained coverage, never guessed history."""
    path = delivery_queue._path()
    if not path.exists():
        return {"status": "unavailable", "records": []}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as conn:
        conn.row_factory = sqlite3.Row
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='delivery_receipts'").fetchone():
            return {"status": "unavailable", "records": []}
        # Read-only projection. The durable sending state already fences replay;
        # a history reader must not acquire scheduler mutation privileges.
        abandoned = set()
        for row in conn.execute("SELECT * FROM delivery_receipts WHERE job_id=? AND outcome='sending'", (job_id,)).fetchall():
            if not executions._owner_is_live(row["owner_pid"], row["owner_started_at"]):
                abandoned.add(row["execution_id"])
        lost = conn.execute("SELECT pruned_count FROM delivery_receipt_pruning WHERE job_id=?", (job_id,)).fetchone()
        rows = conn.execute("SELECT * FROM delivery_receipts WHERE job_id=? ORDER BY started_at,execution_id", (job_id,)).fetchall()
    records = [dict(row) for row in rows]
    for row in records:
        if row["execution_id"] in abandoned:
            row["outcome"] = "unknown"
    return {"status": "truncated" if lost else "available", "records": records}
