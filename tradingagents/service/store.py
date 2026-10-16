"""Transactional SQLite queue and compare-and-set run lifecycle."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

from tradingagents.service.models import pending_run, utc_now


class QueueFull(Exception):
    pass


class StoreOwner:
    """Hold the data directory's OS lock for the server's full lifetime."""

    def __init__(self, database_path: Path):
        path = Path(database_path).resolve().parent / ".service-owner.lock"
        self.file = path.open("a+b")
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise OSError("service data directory is already owned") from exc

    def close(self):
        if self.file.closed:
            return
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()


class RunStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, request TEXT NOT NULL,
                response TEXT NOT NULL, status TEXT NOT NULL, claim_token TEXT,
                created_at TEXT NOT NULL)""")
            db.execute(
                "CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _valid_id(job_id: str) -> bool:
        try:
            return str(UUID(job_id)) == job_id
        except (ValueError, TypeError, AttributeError):
            return False

    def create(self, kind: str, request: dict, max_pending: int) -> dict:
        job_id = str(uuid4())
        response = pending_run(kind, job_id, request)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT COUNT(*) FROM runs WHERE status='pending'").fetchone()[0]
                >= max_pending
            ):
                raise QueueFull
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, 'pending', NULL, ?)",
                (job_id, kind, json.dumps(request), json.dumps(response), response["createdAt"]),
            )
        return response

    def get(self, job_id: str) -> dict | None:
        if not self._valid_id(job_id):
            return None
        with self._connect() as db:
            row = db.execute("SELECT response FROM runs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["response"]) if row else None

    def get_job(self, job_id: str) -> dict | None:
        if not self._valid_id(job_id):
            return None
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (job_id,)).fetchone()
        return (
            {
                "id": row["id"],
                "kind": row["kind"],
                "request": json.loads(row["request"]),
                "response": json.loads(row["response"]),
                "status": row["status"],
                "claim_token": row["claim_token"],
            }
            if row
            else None
        )

    def get_report(self, job_id: str) -> dict | None:
        if not self._valid_id(job_id):
            return None
        with self._connect() as db:
            row = db.execute("SELECT payload FROM reports WHERE id=?", (job_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def claim_next(self) -> dict | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM runs WHERE status='pending' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if not row:
                return None
            response = json.loads(row["response"])
            response["status"] = "running"
            if row["kind"] == "analysis":
                response["startedAt"] = utc_now()
            token = secrets.token_hex(32)
            db.execute(
                "UPDATE runs SET status='running', response=?, claim_token=? WHERE id=?",
                (json.dumps(response), token, row["id"]),
            )
            return {
                "id": row["id"],
                "kind": row["kind"],
                "request": json.loads(row["request"]),
                "response": response,
                "claim_token": token,
            }

    def _terminal(
        self, job_id: str, token: str, status: str, result: dict | None, error: str | None
    ) -> bool:
        if status not in {"completed", "failed"} or not self._valid_id(job_id):
            return False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT response FROM runs WHERE id=? AND status='running' AND claim_token=?",
                (job_id, token),
            ).fetchone()
            if not row:
                return False
            response = json.loads(row["response"])
            result = dict(result or {})
            report = result.pop("_report", None)
            response.update(result)
            response["status"] = status
            response["completedAt"] = utc_now()
            if error:
                response["error"] = error
            db.execute(
                "UPDATE runs SET status=?, response=?, claim_token=NULL WHERE id=?",
                (status, json.dumps(response), job_id),
            )
            if report is not None and status == "completed":
                db.execute("INSERT INTO reports VALUES (?, ?)", (job_id, json.dumps(report)))
        return True

    def finish(
        self, job_id: str, claim_token: str, status: str, result: dict | None, error: str | None
    ) -> bool:
        return self._terminal(job_id, claim_token, status, result, error)

    def terminate(self, job_id: str, claim_token: str, code: str) -> bool:
        if code not in {"RUN_INTERRUPTED", "RUN_TIMED_OUT"}:
            raise ValueError("invalid termination code")
        return self._terminal(job_id, claim_token, "failed", None, code)

    def update_progress(self, job_id: str, claim_token: str, attempted: int) -> bool:
        if not self._valid_id(job_id) or type(attempted) is not int or attempted < 0:
            return False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT response FROM runs WHERE id=? AND kind='backtest' "
                "AND status='running' AND claim_token=?",
                (job_id, claim_token),
            ).fetchone()
            if not row:
                return False
            response = json.loads(row["response"])
            if attempted < response["cellsAttempted"] or attempted > response["cellsTotal"]:
                return False
            response["cellsAttempted"] = attempted
            db.execute("UPDATE runs SET response=? WHERE id=?", (json.dumps(response), job_id))
        return True

    def recover_interrupted(self) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id, response FROM runs WHERE status='running'").fetchall()
            for row in rows:
                response = json.loads(row["response"])
                response.update(status="failed", error="RUN_INTERRUPTED", completedAt=utc_now())
                db.execute(
                    "UPDATE runs SET status='failed', response=?, claim_token=NULL WHERE id=?",
                    (json.dumps(response), row["id"]),
                )
        return len(rows)
