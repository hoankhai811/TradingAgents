"""Loopback HTTP routes and a single bounded child-process dispatcher."""

from __future__ import annotations

import copy
import json
import logging
import multiprocessing
import os
import socket
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from pydantic import ValidationError

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.service.market import (
    MAX_RESULT_BYTES,
    read_result,
    validate_market_request,
    validate_query,
)
from tradingagents.service.models import AnalysisRequest, BacktestRequest
from tradingagents.service.parent_guard import WindowsChildJob, guarded_call, guarded_run
from tradingagents.service.store import QueueFull, RunStore, StoreOwner
from tradingagents.service.worker import ProviderNotConfigured, ensure_provider_configured

logger = logging.getLogger(__name__)
MAX_HTTP_HANDLERS = 16
REQUEST_READ_SECONDS = 5
MAX_READ_CHILDREN = 2
READ_SECONDS = 45

MESSAGES = {
    "INVALID_REQUEST": (400, "Invalid request", False),
    "RUN_NOT_FOUND": (404, "Run not found", False),
    "REPORT_NOT_READY": (409, "Report is not available for this run", False),
    "QUEUE_FULL": (429, "Pending queue is full", True),
    "READ_BUSY": (429, "Market reads are busy", True),
    "DEPENDENCY_TIMEOUT": (504, "Market dependency timed out", True),
    "PROVIDER_NOT_CONFIGURED": (503, "Provider is not configured", False),
    "DEPENDENCY_UNAVAILABLE": (503, "Required dependency is unavailable", True),
    "INTERNAL_ERROR": (500, "Service error", True),
}


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(
        self,
        address,
        store: RunStore,
        config: dict,
        max_pending: int,
        max_body_bytes: int,
        worker_timeout: float,
        process_factory,
        start_dispatcher: bool,
        read_target,
        read_timeout: float,
    ):
        self.store = store
        self.config = copy.deepcopy(config)
        self.max_pending = max_pending
        self.max_body_bytes = max_body_bytes
        self.worker_timeout = worker_timeout
        self.process_factory = process_factory
        self.read_target = read_target
        self.read_timeout = read_timeout
        self.read_slots = threading.BoundedSemaphore(MAX_READ_CHILDREN)
        self.read_lock = threading.Lock()
        self.read_children = {}
        self.read_state = store.path.parent / "read-state"
        self.stopping = threading.Event()
        self.accept_lock = threading.Lock()
        self.active_lock = threading.Lock()
        self.http_slots = threading.BoundedSemaphore(MAX_HTTP_HANDLERS)
        self.active = None
        self.dispatcher = None
        self.process_parent_pid = os.getpid()
        self.owner = StoreOwner(store.path)
        self.child_job = None
        try:
            self.child_job = WindowsChildJob()
            super().__init__(address, ServiceHandler)
            self.store.recover_interrupted()
            if start_dispatcher:
                self.dispatcher = threading.Thread(
                    target=self._dispatch, name="tradingagents-dispatcher", daemon=True
                )
                self.dispatcher.start()
        except BaseException:
            self.stopping.set()
            if hasattr(self, "socket"):
                self.socket.close()
            if self.child_job is not None:
                self.child_job.close()
            self.owner.close()
            raise

    def process_request(self, request, client_address):
        if not self.http_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.http_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.http_slots.release()

    def _stop_active(self, job_id, token, process, code):
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
        self.store.terminate(job_id, token, code)

    def _dispatch(self):
        while not self.stopping.is_set():
            job = self.store.claim_next()
            if job is None:
                self.stopping.wait(0.05)
                continue
            job_id, token = job["id"], job["claim_token"]
            process = None
            try:
                gate = multiprocessing.Event()
                process = self.process_factory(
                    target=guarded_run,
                    args=(gate, self.process_parent_pid, self.store, job_id, self.config),
                )
                with self.active_lock:
                    self.active = (job_id, token, process)
                process.start()
                if isinstance(process, multiprocessing.Process):
                    self.child_job.assign(process)
                with self.active_lock:
                    if not self.stopping.is_set():
                        gate.set()
                if not gate.is_set():
                    self._stop_active(job_id, token, process, "RUN_INTERRUPTED")
                    continue
                deadline = time.monotonic() + self.worker_timeout
                while (
                    process.is_alive()
                    and not self.stopping.is_set()
                    and time.monotonic() < deadline
                ):
                    process.join(0.05)
                if process.is_alive():
                    self._stop_active(
                        job_id,
                        token,
                        process,
                        "RUN_INTERRUPTED" if self.stopping.is_set() else "RUN_TIMED_OUT",
                    )
                else:
                    process.join()
                    self.store.terminate(job_id, token, "RUN_INTERRUPTED")
            except Exception:
                logger.error("Could not launch or reap run %s", job_id)
                if process is not None and getattr(process, "pid", None) is not None:
                    self._stop_active(job_id, token, process, "RUN_INTERRUPTED")
                else:
                    self.store.terminate(job_id, token, "RUN_INTERRUPTED")
            finally:
                with self.active_lock:
                    self.active = None

    def stop_workers(self):
        with self.accept_lock, self.active_lock, self.read_lock:
            self.stopping.set()
            active = self.active
            reads = list(self.read_children.values())
        if active is not None:
            self._stop_active(*active, "RUN_INTERRUPTED")
        for process in reads:
            self._reap_read(process)
        if self.dispatcher:
            self.dispatcher.join()
        self.child_job.close()

    @staticmethod
    def _reap_read(process):
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
        if process.pid is not None:
            process.join(2)
        return not process.is_alive()

    def read_market(self, kind: str, args: tuple) -> tuple[str | None, dict | None]:
        if not self.read_slots.acquire(blocking=False):
            return "READ_BUSY", None
        result_path = self.read_state / f"{uuid4()}.json"
        process = None
        try:
            self.read_state.mkdir(mode=0o700, parents=True, exist_ok=True)
            gate = multiprocessing.Event()
            process = multiprocessing.Process(
                target=guarded_call,
                args=(
                    gate,
                    self.process_parent_pid,
                    self.read_target,
                    (kind, args, self.config, self.store.path.parent, result_path),
                ),
            )
            deadline = time.monotonic() + self.read_timeout
            with self.read_lock:
                if self.stopping.is_set():
                    return "DEPENDENCY_UNAVAILABLE", None
                self.read_children[result_path] = process
                process.start()
                self.child_job.assign(process)
                if not self.stopping.is_set():
                    gate.set()
            while process.is_alive() and not self.stopping.is_set() and time.monotonic() < deadline:
                process.join(min(0.05, max(0, deadline - time.monotonic())))
            if process.is_alive() or self.stopping.is_set() or time.monotonic() >= deadline:
                self._reap_read(process)
                return (
                    "DEPENDENCY_UNAVAILABLE" if self.stopping.is_set() else "DEPENDENCY_TIMEOUT"
                ), None
            process.join()
            if not result_path.is_file() or result_path.stat().st_size > MAX_RESULT_BYTES:
                return "DEPENDENCY_UNAVAILABLE", None
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return "DEPENDENCY_UNAVAILABLE", None
            if payload.get("error") in {"INVALID_REQUEST", "DEPENDENCY_UNAVAILABLE"}:
                return payload["error"], None
            if not isinstance(payload.get("result"), dict):
                return "DEPENDENCY_UNAVAILABLE", None
            return None, payload["result"]
        except Exception:
            logger.error("Market read failed")
            if process is not None and process.pid is not None:
                self._reap_read(process)
            return "DEPENDENCY_UNAVAILABLE", None
        finally:
            if process is None or not process.is_alive():
                with self.read_lock:
                    self.read_children.pop(result_path, None)
                result_path.unlink(missing_ok=True)
                for temp in self.read_state.glob(f"{result_path.name}.*.tmp"):
                    temp.unlink(missing_ok=True)
                self.read_slots.release()
            else:
                logger.error("Market read child could not be reaped; capacity remains reserved")

    def shutdown(self):
        self.stop_workers()
        super().shutdown()
        self.server_close()

    def server_close(self):
        self.stop_workers()
        super().server_close()
        self.owner.close()


class ServiceHandler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer

    def setup(self):
        self.read_deadline = time.monotonic() + REQUEST_READ_SECONDS
        self.request.settimeout(REQUEST_READ_SECONDS)
        super().setup()
        self.read_timer = threading.Timer(REQUEST_READ_SECONDS, self._expire_read)
        self.read_timer.daemon = True
        self.read_timer.start()

    def _expire_read(self):
        with suppress(OSError):
            self.connection.shutdown(socket.SHUT_RD)

    def finish(self):
        self.read_timer.cancel()
        super().finish()

    def log_message(self, fmt, *args):
        logger.info("HTTP %s", self.command)

    def _send(self, status: int, body: dict):
        payload = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True

    def _read_body(self, length: int) -> bytes:
        chunks = []
        while length:
            remaining = self.read_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request read deadline exceeded")
            self.connection.settimeout(remaining)
            chunk = self.rfile.read1(min(length, self.server.max_body_bytes))
            if not chunk:
                raise ValueError("incomplete request body")
            chunks.append(chunk)
            length -= len(chunk)
        return b"".join(chunks)

    def _error(self, code: str):
        status, message, retryable = MESSAGES[code]
        self._send(
            status,
            {
                "error": message,
                "message": message,
                "statusCode": status,
                "code": code,
                "requestId": str(uuid4()),
                "retryable": retryable,
            },
        )

    def _safe_host(self):
        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _safe_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
            return (
                parsed.scheme == "http"
                and parsed.hostname in {"127.0.0.1", "localhost"}
                and parsed.port == self.server.server_address[1]
                and not (
                    parsed.path
                    or parsed.query
                    or parsed.fragment
                    or parsed.username
                    or parsed.password
                )
            )
        except ValueError:
            return False

    def _reject_post(self, code: str):
        # Consume small fixed-length bodies so Windows clients receive the JSON error before close.
        try:
            length = int(self.headers.get("Content-Length", ""))
            if (
                not self.headers.get("Transfer-Encoding")
                and 0 < length <= self.server.max_body_bytes
            ):
                self._read_body(length)
        except (ValueError, TimeoutError, ConnectionError):
            pass
        self._error(code)

    def do_POST(self):
        if not self._safe_host() or not self._safe_origin():
            return self._reject_post("INVALID_REQUEST")
        if self.server.stopping.is_set():
            return self._reject_post("DEPENDENCY_UNAVAILABLE")
        path = urlsplit(self.path).path
        if path in {"/analysis", "/api/v1/analysis"}:
            kind, model = "analysis", AnalysisRequest
        elif path in {"/backtests", "/api/v1/backtest"}:
            kind, model = "backtest", BacktestRequest
        else:
            return self._error("RUN_NOT_FOUND")
        if (
            self.headers.get("Transfer-Encoding")
            or self.headers.get_content_type() != "application/json"
        ):
            return self._error("INVALID_REQUEST")
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length <= 0 or length > self.server.max_body_bytes:
                return self._error("INVALID_REQUEST")
            request = model.model_validate_json(self._read_body(length)).model_dump(
                mode="json", exclude_none=True
            )
            ensure_provider_configured(self.server.config)
            with self.server.accept_lock:
                run = (
                    None
                    if self.server.stopping.is_set()
                    else self.server.store.create(kind, request, self.server.max_pending)
                )
        except (
            ValueError,
            ValidationError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TimeoutError,
            ConnectionError,
        ):
            return self._error("INVALID_REQUEST")
        except ProviderNotConfigured:
            return self._error("PROVIDER_NOT_CONFIGURED")
        except QueueFull:
            return self._error("QUEUE_FULL")
        except Exception:
            logger.error("HTTP acceptance failed")
            return self._error("INTERNAL_ERROR")
        if run is None:
            return self._error("DEPENDENCY_UNAVAILABLE")
        self._send(202, run)

    def do_GET(self):
        if not self._safe_host():
            return self._error("INVALID_REQUEST")
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/assets/search" or path.startswith("/market-data/"):
            if self.server.stopping.is_set():
                return self._error("DEPENDENCY_UNAVAILABLE")
            try:
                params = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                if path == "/assets/search":
                    if set(params) != {"q"} or len(params["q"]) != 1:
                        raise ValueError("invalid query")
                    kind, args = "search", (validate_query(params["q"][0]),)
                else:
                    parts = path.strip("/").split("/")
                    if (
                        len(parts) != 3
                        or set(params) != {"startDate", "endDate"}
                        or any(len(values) != 1 for values in params.values())
                    ):
                        raise ValueError("invalid market request")
                    kind, args = (
                        "market",
                        (parts[1], unquote(parts[2]), params["startDate"][0], params["endDate"][0]),
                    )
                    validate_market_request(*args)
            except ValueError:
                return self._error("INVALID_REQUEST")
            code, response = self.server.read_market(kind, args)
            return self._error(code) if code else self._send(200, response)
        if path == "/api/v1/health":
            if (
                self.server.stopping.is_set()
                or self.server.dispatcher is None
                or not self.server.dispatcher.is_alive()
            ):
                return self._error("DEPENDENCY_UNAVAILABLE")
            return self._send(200, {"status": "ok", "framework": "TradingAgents"})
        parts = path.strip("/").split("/")
        kind = None
        report = False
        if len(parts) >= 2 and parts[0] == "runs":
            kind, job_id = "analysis", parts[1]
            report = len(parts) == 3 and parts[2] == "report"
            if len(parts) != 2 and not report:
                return self._error("RUN_NOT_FOUND")
        elif len(parts) == 2 and parts[0] == "backtests":
            kind, job_id = "backtest", parts[1]
        elif len(parts) == 4 and parts[:3] == ["api", "v1", "analysis"]:
            kind, job_id = "analysis", parts[3]
        elif len(parts) == 4 and parts[:3] == ["api", "v1", "backtest"]:
            kind, job_id = "backtest", parts[3]
        if kind is None:
            return self._error("RUN_NOT_FOUND")
        job = self.server.store.get_job(job_id)
        if not job or job["kind"] != kind:
            return self._error("RUN_NOT_FOUND")
        if report:
            if job["status"] != "completed":
                return self._error("REPORT_NOT_READY")
            projection = self.server.store.get_report(job_id)
            if projection is None:
                return self._error("REPORT_NOT_READY")
            return self._send(200, projection)
        self._send(200, job["response"])


def create_server(
    host: str,
    port: int,
    store: RunStore,
    config: dict,
    *,
    max_pending: int = 16,
    max_body_bytes: int = 65536,
    worker_timeout: float = 900,
    process_factory=multiprocessing.Process,
    start_dispatcher: bool = True,
    read_target=read_result,
    read_timeout: float = READ_SECONDS,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("thin service must bind to loopback")
    if max_pending < 1 or max_body_bytes < 1 or worker_timeout <= 0 or read_timeout <= 0:
        raise ValueError("service limits must be positive")
    effective = copy.deepcopy(DEFAULT_CONFIG)
    effective.update(copy.deepcopy(config))
    return ServiceHTTPServer(
        (host, port),
        store,
        effective,
        max_pending,
        max_body_bytes,
        worker_timeout,
        process_factory,
        start_dispatcher,
        read_target,
        read_timeout,
    )
