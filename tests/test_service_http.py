"""Real loopback HTTP and durable lifecycle; only the child process is fake."""

import http.client
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

CONNECT = socket.socket.connect
CONNECT_EX = socket.socket.connect_ex
GETADDRINFO = socket.getaddrinfo
ANALYSIS = {"ticker": "NVDA", "tradeDate": "2026-01-05"}


@pytest.fixture(autouse=True)
def _allow_only_loopback(monkeypatch):
    def allowed_connect(sock, address):
        if address[0] not in {"127.0.0.1", "::1"}:
            raise OSError("external connection denied")
        return CONNECT(sock, address)

    def allowed_connect_ex(sock, address):
        if address[0] not in {"127.0.0.1", "::1"}:
            raise OSError("external connection denied")
        return CONNECT_EX(sock, address)

    def allowed_lookup(host, *args, **kwargs):
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError("external lookup denied")
        return GETADDRINFO(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", allowed_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", allowed_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", allowed_lookup)


@contextmanager
def live_server(tmp_path, **options):
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")
    server = create_server("127.0.0.1", 0, store, {"llm_provider": "ollama"}, **options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, store
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def request(server, method, path, body=None, headers=None, timeout=3):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = json.dumps(body).encode() if body is not None else None
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, payload, all_headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, json.loads(data)


def blocked_market_read(kind, args, config, cache_dir, result_path):
    Path(str(result_path) + ".started").write_text("started", encoding="utf-8")
    time.sleep(30)


def fixed_market_read(kind, args, config, cache_dir, result_path):
    from tradingagents.dataflows.files import replace_file

    if kind == "search":
        result = {"query": args[0], "assets": [{"ticker": "NVDA", "assetType": "stock"}]}
    else:
        result = {
            "asset": {"ticker": args[1], "assetType": args[0]},
            "startDate": args[2],
            "endDate": args[3],
            "bars": [],
        }
    replace_file(
        result_path,
        lambda path: Path(path).write_text(json.dumps({"result": result}), encoding="utf-8"),
    )


def test_market_get_routes_return_child_results_without_creating_paid_runs(tmp_path):
    from datetime import date, timedelta

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    with live_server(tmp_path, read_target=fixed_market_read) as (server, store):
        assert request(server, "GET", "/assets/search?q=%20nvda%20") == (
            200,
            {"query": "nvda", "assets": [{"ticker": "NVDA", "assetType": "stock"}]},
        )
        assert request(
            server, "GET", f"/market-data/crypto/BTC-USD?startDate={yesterday}&endDate={yesterday}"
        ) == (
            200,
            {
                "asset": {"ticker": "BTC-USD", "assetType": "crypto"},
                "startDate": yesterday,
                "endDate": yesterday,
                "bars": [],
            },
        )
        assert store.claim_next() is None


def test_stock_market_http_rejects_nonstocks_before_read_child_and_keeps_equities(tmp_path):
    from datetime import date, timedelta
    from urllib.parse import quote

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    with live_server(tmp_path, read_target=fixed_market_read) as (server, store):
        for symbol in (
            "BTC-USDT",
            "ETH-USDC",
            "BTCUSD",
            "BTC-USD",
            "PEPE-USDT",
            "PEPE-USDT+",
            "PEPE-USDC+",
            "PEPE-USDT++",
            "PEPE-USDC+++",
            "XAUUSD",
            "EURUSD",
            "SPX500",
            "GC=F",
            "EURUSD=X",
            "^GSPC",
        ):
            path = f"/market-data/stock/{quote(symbol, safe='')}?startDate={yesterday}&endDate={yesterday}"
            status, body = request(server, "GET", path)
            assert (status, body["code"]) == (400, "INVALID_REQUEST"), symbol
            assert store.claim_next() is None
        assert not list((tmp_path / "read-state").glob("*.json"))
        for symbol in ("BTC", "BRK-B", "NVDA", "700.HK", "600519.SH"):
            path = f"/market-data/stock/{quote(symbol, safe='')}?startDate={yesterday}&endDate={yesterday}"
            assert request(server, "GET", path)[0] == 200
        assert store.claim_next() is None


def test_market_reads_have_two_killable_slots_without_blocking_health(tmp_path):
    from urllib.parse import quote

    with live_server(tmp_path, read_target=blocked_market_read, read_timeout=2) as (server, store):
        outcomes = []
        calls = [
            threading.Thread(
                target=lambda: outcomes.append(
                    request(server, "GET", "/assets/search?q=nvda", timeout=5)
                )
            )
            for _ in range(2)
        ]
        for call in calls:
            call.start()
        deadline = time.monotonic() + 3
        while (
            len(list((tmp_path / "read-state").glob("*.started"))) < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert len(list((tmp_path / "read-state").glob("*.started"))) == 2
        assert request(server, "GET", "/api/v1/health")[0] == 200
        assert request(server, "GET", "/runs/00000000-0000-4000-8000-000000000001")[0] == 404
        status, body = request(server, "GET", "/assets/search?q=third")
        assert (status, body["code"]) == (429, "READ_BUSY")
        for call in calls:
            call.join(timeout=5)
            assert not call.is_alive()
        assert len(outcomes) == 2
        assert all(
            status == 504 and body["code"] == "DEPENDENCY_TIMEOUT" for status, body in outcomes
        )
        assert store.claim_next() is None
        status, body = request(
            server, "GET", "/market-data/crypto/BTC-USDT?startDate=2026-01-01&endDate=2026-01-02"
        )
        assert (status, body["code"]) == (400, "INVALID_REQUEST")
        assert request(server, "GET", "/assets/search?q=one&q=two")[0] == 400
        assert request(server, "GET", "/assets/search?q=one&unknown=x")[0] == 400
        assert request(server, "GET", "/assets/search?q=" + quote("a" * 81))[0] == 400


def test_shutdown_reaps_stalled_market_reads_and_releases_store(tmp_path):
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")
    server = create_server(
        "127.0.0.1",
        0,
        store,
        {"llm_provider": "ollama"},
        read_target=blocked_market_read,
        read_timeout=20,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    call = threading.Thread(
        target=lambda: request(server, "GET", "/assets/search?q=nvda", timeout=5)
    )
    call.start()
    deadline = time.monotonic() + 3
    while not list((tmp_path / "read-state").glob("*.started")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list((tmp_path / "read-state").glob("*.started"))
    started = time.monotonic()
    server.shutdown()
    serving.join(timeout=3)
    server.server_close()
    call.join(timeout=3)
    assert time.monotonic() - started < 5
    assert not call.is_alive()
    reopened = create_server(
        "127.0.0.1", 0, RunStore(store.path), {"llm_provider": "ollama"}, start_dispatcher=False
    )
    try:
        assert reopened.store.claim_next() is None
    finally:
        reopened.server_close()


def test_abrupt_parent_death_kills_market_read_child(tmp_path):
    controller = tmp_path / "read_controller.py"
    started, finished = tmp_path / "started", tmp_path / "finished"
    controller.write_text(
        """import http.client
import os
import threading
import time
from pathlib import Path
from tradingagents.service.server import create_server
from tradingagents.service.store import RunStore

def slow_read(kind, args, config, cache_dir, result_path):
    Path(os.environ['T16_STARTED']).write_text('started')
    time.sleep(1)
    Path(os.environ['T16_FINISHED']).write_text('finished')

def request(server):
    conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=5)
    conn.request('GET', '/assets/search?q=nvda')
    conn.getresponse().read()

if __name__ == '__main__':
    server = create_server('127.0.0.1', 0, RunStore(Path(os.environ['T16_DB'])),
                           {'llm_provider': 'ollama'}, read_target=slow_read)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=request, args=(server,), daemon=True).start()
    deadline = time.monotonic() + 8
    while not Path(os.environ['T16_STARTED']).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    os._exit(0 if Path(os.environ['T16_STARTED']).exists() else 2)
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        T16_STARTED=str(started),
        T16_FINISHED=str(finished),
        T16_DB=str(tmp_path / "runs.db"),
        PYTHONPATH=os.pathsep.join(
            (str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", ""))
        ),
    )
    child = subprocess.run(
        [sys.executable, str(controller)], env=env, capture_output=True, text=True, timeout=15
    )
    assert child.returncode == 0, child.stderr
    time.sleep(1.5)
    assert not finished.exists()


def test_aliases_share_contract(tmp_path):
    with live_server(tmp_path, start_dispatcher=False) as (server, _):
        root_status, root = request(server, "POST", "/analysis", ANALYSIS)
        alias_status, alias = request(server, "POST", "/api/v1/analysis", ANALYSIS)
        assert (root_status, alias_status) == (202, 202)
        assert root.keys() == alias.keys() and root["runId"] != alias["runId"]
        assert request(server, "GET", "/runs/" + root["runId"])[1] == root
        assert request(server, "GET", "/api/v1/analysis/" + root["runId"])[1] == root
        status, back = request(
            server,
            "POST",
            "/api/v1/backtest",
            {"tickers": ["NVDA"], "startDate": "2026-01-05", "endDate": "2026-01-05"},
        )
        assert status == 202 and back["cellsTotal"] == 1
        assert request(server, "GET", "/backtests/" + back["backtestId"])[1] == back
        assert request(server, "GET", "/api/v1/backtest/" + back["backtestId"])[1] == back


def test_health_requires_a_live_dispatcher_and_reports_safe_503(tmp_path):
    with live_server(tmp_path, start_dispatcher=False) as (server, _):
        status, error = request(server, "GET", "/api/v1/health")
        assert status == 503
        assert error["code"] == "DEPENDENCY_UNAVAILABLE"
        assert set(error) == {"error", "message", "statusCode", "code", "requestId", "retryable"}
        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()
        server.dispatcher = dead
        assert request(server, "GET", "/api/v1/health")[0] == 503

    with live_server(tmp_path / "healthy") as (server, _):
        assert request(server, "GET", "/api/v1/health") == (
            200,
            {"status": "ok", "framework": "TradingAgents"},
        )
        server.stopping.set()
        assert request(server, "GET", "/api/v1/health")[0] == 503


def test_valid_post_after_stopping_is_unavailable_but_bad_origin_is_invalid(tmp_path):
    with live_server(tmp_path, start_dispatcher=False) as (server, store):
        server.stopping.set()
        status, error = request(server, "POST", "/analysis", ANALYSIS)
        assert (status, error["code"], error["retryable"]) == (
            503,
            "DEPENDENCY_UNAVAILABLE",
            True,
        )
        for headers in ({"Host": "evil.example"}, {"Origin": "http://evil.example"}):
            status, error = request(server, "POST", "/analysis", ANALYSIS, headers)
            assert (status, error["code"]) == (400, "INVALID_REQUEST")
        assert store.claim_next() is None


def test_accepted_job_is_persisted_before_dispatch(tmp_path):
    with live_server(tmp_path, start_dispatcher=False) as (server, store):
        status, body = request(server, "POST", "/analysis", ANALYSIS)
        assert status == 202 and store.get(body["runId"]) == body
        assert store.get_job(body["runId"])["status"] == "pending"


def test_report_pending_unknown_and_completed(tmp_path):
    with live_server(tmp_path, start_dispatcher=False) as (server, store):
        _, body = request(server, "POST", "/analysis", ANALYSIS)
        job_id = body["runId"]
        assert request(server, "GET", f"/runs/{job_id}/report")[0] == 409
        assert request(server, "GET", "/runs/00000000-0000-4000-8000-000000000001/report")[0] == 404
        claim = store.claim_next()
        report = {"runId": job_id, "tradeDate": "2026-01-05", "sections": {"market": "safe text"}}
        store.finish(job_id, claim["claim_token"], "completed", {"_report": report}, None)
        assert request(server, "GET", f"/runs/{job_id}/report") == (200, report)
        assert type(store)(store.path).get_report(job_id) == report
        assert request(server, "GET", "/runs/../../etc/passwd")[0] == 404


def test_large_body_chunked_body_and_host_origin_rejected(tmp_path):
    with live_server(tmp_path, start_dispatcher=False, max_body_bytes=128) as (server, store):
        assert (
            request(server, "POST", "/analysis", {"ticker": "N" * 200, "tradeDate": "2026-01-05"})[
                0
            ]
            == 400
        )
        assert (
            request(server, "POST", "/analysis", ANALYSIS, {"Origin": "https://evil.example"})[0]
            == 400
        )
        own_port = server.server_address[1]
        assert (
            request(
                server, "POST", "/analysis", ANALYSIS, {"Origin": f"https://127.0.0.1:{own_port}"}
            )[0]
            == 400
        )
        assert request(server, "POST", "/analysis", ANALYSIS, {"Host": "evil.example"})[0] == 400
        assert (
            request(server, "POST", "/analysis", ANALYSIS, {"Transfer-Encoding": "chunked"})[0]
            == 400
        )
        assert store.claim_next() is None


def test_incomplete_body_expires_while_health_and_status_remain_available(tmp_path, monkeypatch):
    from tradingagents.service import server as service_server

    monkeypatch.setattr(service_server, "REQUEST_READ_SECONDS", 0.5, raising=False)
    with live_server(tmp_path, process_factory=HangingProcess, worker_timeout=30) as (
        server,
        store,
    ):
        _, accepted = request(server, "POST", "/analysis", ANALYSIS)
        stalled = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=1)
        try:
            stalled.sendall(
                (
                    "POST /analysis HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{server.server_address[1]}\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: 100\r\n\r\n{"
                ).encode()
            )
            assert request(server, "GET", "/api/v1/health", timeout=0.3)[0] == 200
            assert request(server, "GET", "/runs/" + accepted["runId"], timeout=0.3)[0] == 200
            stalled.settimeout(2)
            assert b"400" in stalled.recv(4096)
            assert store.get(accepted["runId"])["status"] in {"running", "completed"}
        finally:
            with suppress(OSError):
                stalled.shutdown(socket.SHUT_RDWR)
            stalled.close()


def test_trickling_body_cannot_extend_absolute_read_deadline(tmp_path, monkeypatch):
    from tradingagents.service import server as service_server

    monkeypatch.setattr(service_server, "REQUEST_READ_SECONDS", 0.4)
    with live_server(tmp_path) as (server, _):
        client = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=1)
        stop = threading.Event()
        sender = None
        try:
            client.sendall(
                (
                    "POST /analysis HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{server.server_address[1]}\r\n"
                    "Content-Type: application/json\r\n"
                    "Content-Length: 100\r\n\r\n{"
                ).encode()
            )

            def trickle():
                while not stop.wait(0.05):
                    try:
                        client.sendall(b" ")
                    except OSError:
                        return

            sender = threading.Thread(target=trickle)
            sender.start()
            client.settimeout(1)
            began = time.monotonic()
            assert b"400" in client.recv(4096)
            assert time.monotonic() - began < 0.9
        finally:
            stop.set()
            if sender is not None:
                sender.join(timeout=1)
            with suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()


def test_trickling_headers_cannot_hold_a_handler_past_read_deadline(tmp_path, monkeypatch):
    from tradingagents.service import server as service_server

    monkeypatch.setattr(service_server, "REQUEST_READ_SECONDS", 0.4)
    with live_server(tmp_path) as (server, _):
        client = socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=1)
        stop = threading.Event()
        sender = None
        try:
            client.sendall(b"POST /analysis HTTP/1.1\r\nX-Pad: ")

            def trickle():
                while not stop.wait(0.05):
                    try:
                        client.sendall(b" ")
                    except OSError:
                        return

            sender = threading.Thread(target=trickle)
            sender.start()
            client.settimeout(1)
            began = time.monotonic()
            try:
                reply = client.recv(4096)
            except (ConnectionResetError, ConnectionAbortedError):
                reply = b""
            assert reply == b"" or b"400" in reply
            assert time.monotonic() - began < 0.9
        finally:
            stop.set()
            if sender is not None:
                sender.join(timeout=1)
            with suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()


def test_http_handler_limit_closes_excess_connection(tmp_path, monkeypatch):
    from tradingagents.service import server as service_server

    monkeypatch.setattr(service_server, "MAX_HTTP_HANDLERS", 2)
    monkeypatch.setattr(service_server, "REQUEST_READ_SECONDS", 1.5)
    with live_server(tmp_path) as (server, _):
        sockets = []
        try:
            for _ in range(2):
                client = socket.create_connection(("127.0.0.1", server.server_address[1]))
                client.sendall(
                    (
                        "POST /analysis HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{server.server_address[1]}\r\n"
                        "Content-Type: application/json\r\n"
                        "Content-Length: 100\r\n\r\n{"
                    ).encode()
                )
                sockets.append(client)
            deadline = time.monotonic() + 1
            while server.http_slots._value and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.http_slots._value == 0
            third = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=0.5)
            try:
                third.request("GET", "/api/v1/health")
                with pytest.raises(
                    (http.client.RemoteDisconnected, ConnectionResetError, ConnectionAbortedError)
                ):
                    third.getresponse()
            finally:
                third.close()
        finally:
            for client in sockets:
                with suppress(OSError):
                    client.shutdown(socket.SHUT_RDWR)
                client.close()


def test_overload_is_429(tmp_path):
    with live_server(tmp_path, start_dispatcher=False, max_pending=1) as (server, _):
        assert request(server, "POST", "/analysis", ANALYSIS)[0] == 202
        status, error = request(server, "POST", "/analysis", ANALYSIS)
        assert (status, error["code"], error["retryable"]) == (429, "QUEUE_FULL", True)


class HangingProcess:
    def __init__(self, target, args):
        self.alive = False

    def start(self):
        self.alive = True

    def join(self, timeout=None):
        time.sleep(min(timeout or 0, 0.01))

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False


def _offline_child(store, job_id, _config):
    claim = store.get_job(job_id)
    store.finish(
        job_id,
        claim["claim_token"],
        "completed",
        {
            "decision": {
                "runId": job_id,
                "ticker": "NVDA",
                "tradeDate": "2026-01-05",
                "rating": "Buy",
                "finalTradeDecision": "offline fixture",
                "generatedAt": "2026-01-05T00:00:00Z",
            },
            "_report": {
                "runId": job_id,
                "tradeDate": "2026-01-05",
                "sections": {"decision": "offline fixture"},
            },
        },
        None,
    )


def test_dispatcher_runs_one_real_child_process_with_offline_boundary(tmp_path):
    def factory(target, args):
        return multiprocessing.Process(target=_offline_child, args=args[2:])

    with live_server(tmp_path, process_factory=factory, worker_timeout=10) as (server, store):
        _, body = request(server, "POST", "/analysis", ANALYSIS)
        deadline = time.monotonic() + 8
        while (
            store.get(body["runId"])["status"] not in {"completed", "failed"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        got = store.get(body["runId"])
        assert got["status"] == "completed" and got["decision"]["rating"] == "Buy"
        assert request(server, "GET", f"/runs/{body['runId']}/report")[1]["sections"] == {
            "decision": "offline fixture"
        }


def test_shutdown_timeout_and_restart_do_not_reexecute_running_job(tmp_path):
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    with live_server(tmp_path, worker_timeout=0.05, process_factory=HangingProcess) as (
        server,
        store,
    ):
        _, body = request(server, "POST", "/analysis", ANALYSIS)
        deadline = time.monotonic() + 2
        while (
            store.get(body["runId"])["status"] not in {"failed", "completed"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert store.get(body["runId"])["error"] == "RUN_TIMED_OUT"
    reopened = RunStore(tmp_path / "runs.db")
    server = create_server(
        "127.0.0.1", 0, reopened, {"llm_provider": "ollama"}, start_dispatcher=False
    )
    try:
        assert reopened.get(body["runId"])["error"] == "RUN_TIMED_OUT"
        assert reopened.claim_next() is None
    finally:
        server.server_close()


@pytest.mark.parametrize("same_port", [True, False])
def test_second_server_cannot_recover_or_dispatch_owned_store(tmp_path, same_port):
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    with live_server(tmp_path, worker_timeout=30, process_factory=HangingProcess) as (
        first,
        store,
    ):
        _, active = request(first, "POST", "/analysis", ANALYSIS)
        deadline = time.monotonic() + 2
        while store.get(active["runId"])["status"] != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(active["runId"])["status"] == "running"
        _, pending = request(first, "POST", "/analysis", ANALYSIS)
        second = None
        try:
            try:
                second = create_server(
                    "127.0.0.1",
                    first.server_address[1] if same_port else 0,
                    RunStore(store.path),
                    {"llm_provider": "ollama"},
                )
            except OSError:
                pass
            else:
                pytest.fail("second server acquired an active data directory")
        finally:
            if second is not None:
                second.server_close()
        assert store.get(active["runId"])["status"] == "running"
        assert store.get(pending["runId"])["status"] == "pending"


def test_shutdown_cannot_accept_a_request_paused_before_store_create(tmp_path, monkeypatch):
    from tradingagents.service import server as service_server
    from tradingagents.service.store import RunStore

    entered = threading.Event()
    release = threading.Event()

    def paused_preflight(_config):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(service_server, "ensure_provider_configured", paused_preflight)
    with live_server(tmp_path, start_dispatcher=False) as (server, store):
        outcome = {}
        submit = threading.Thread(
            target=lambda: outcome.update(
                zip(("status", "body"), request(server, "POST", "/analysis", ANALYSIS), strict=True)
            )
        )
        submit.start()
        stopping = None
        try:
            assert entered.wait(2)
            stopping = threading.Thread(target=server.shutdown)
            stopping.start()
            assert server.stopping.wait(2)
            with pytest.raises(OSError):
                service_server.create_server(
                    "127.0.0.1",
                    0,
                    RunStore(store.path),
                    {"llm_provider": "ollama"},
                    start_dispatcher=False,
                )
        finally:
            release.set()
            submit.join(timeout=3)
            if stopping is not None:
                stopping.join(timeout=3)
        assert not submit.is_alive()
        assert stopping is not None and not stopping.is_alive()
        assert outcome["status"] == 503
        assert outcome["body"]["code"] == "DEPENDENCY_UNAVAILABLE"
        assert store.claim_next() is None
        reopened = service_server.create_server(
            "127.0.0.1", 0, RunStore(store.path), {"llm_provider": "ollama"}, start_dispatcher=False
        )
        reopened.server_close()


def test_startup_failure_releases_store_ownership(tmp_path, monkeypatch):
    from http.server import ThreadingHTTPServer

    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")

    def failed_bind(self, *args, **kwargs):
        raise OSError("offline bind failure")

    with monkeypatch.context() as patch:
        patch.setattr(ThreadingHTTPServer, "__init__", failed_bind)
        with pytest.raises(OSError, match="offline bind failure"):
            create_server("127.0.0.1", 0, store, {"llm_provider": "ollama"})
    server = create_server(
        "127.0.0.1", 0, store, {"llm_provider": "ollama"}, start_dispatcher=False
    )
    server.server_close()


def test_data_directory_lock_is_exclusive_across_processes(tmp_path):
    with live_server(tmp_path, start_dispatcher=False) as (_, store):
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path\n"
                "from tradingagents.service.server import create_server\n"
                "from tradingagents.service.store import RunStore\n"
                "import sys\n"
                "try:\n"
                "    server = create_server('127.0.0.1', 0, RunStore(Path(sys.argv[1])), "
                "{'llm_provider': 'ollama'}, start_dispatcher=False)\n"
                "except OSError:\n"
                "    raise SystemExit(0)\n"
                "server.server_close()\n"
                "raise SystemExit(2)\n",
                str(store.path),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert child.returncode == 0, child.stderr


def test_shutdown_interrupts_active_job_before_restart(tmp_path):
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    store = RunStore(tmp_path / "runs.db")
    server = create_server(
        "127.0.0.1",
        0,
        store,
        {"llm_provider": "ollama"},
        worker_timeout=30,
        process_factory=HangingProcess,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        _, body = request(server, "POST", "/analysis", ANALYSIS)
        deadline = time.monotonic() + 2
        while store.get(body["runId"])["status"] != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(body["runId"])["status"] == "running"
    finally:
        server.shutdown()
        serving.join(timeout=3)
        server.server_close()
    assert RunStore(tmp_path / "runs.db").get(body["runId"])["error"] == "RUN_INTERRUPTED"


def test_shutdown_during_child_start_never_opens_paid_work_gate(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    gates = []

    class PausedStart(HangingProcess):
        def __init__(self, target, args):
            super().__init__(target, args)
            self.gate = args[0]
            gates.append(self.gate)

        def start(self):
            entered.set()
            assert release.wait(3)
            super().start()

    with live_server(tmp_path, process_factory=PausedStart) as (server, store):
        _, body = request(server, "POST", "/analysis", ANALYSIS)
        assert entered.wait(2)
        stopping = threading.Thread(target=server.stop_workers)
        stopping.start()
        assert server.stopping.wait(2)
        release.set()
        stopping.join(timeout=3)
        assert not stopping.is_alive()
        assert not gates[0].is_set()
        assert store.get(body["runId"])["error"] == "RUN_INTERRUPTED"


def test_crash_between_claim_and_launch_is_terminal(tmp_path):
    class FailedLaunch(HangingProcess):
        def start(self):
            raise OSError("launch failed")

    with live_server(tmp_path, process_factory=FailedLaunch) as (server, store):
        _, body = request(server, "POST", "/analysis", ANALYSIS)
        deadline = time.monotonic() + 2
        while store.get(body["runId"])["status"] != "failed" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(body["runId"])["error"] == "RUN_INTERRUPTED"


def test_missing_provider_rejected_before_acceptance(tmp_path, monkeypatch):
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    store = RunStore(tmp_path / "runs.db")
    server = create_server(
        "127.0.0.1", 0, store, {"llm_provider": "google"}, start_dispatcher=False
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        status, error = request(server, "POST", "/analysis", ANALYSIS)
        assert (status, error["code"]) == (503, "PROVIDER_NOT_CONFIGURED")
        assert store.claim_next() is None
    finally:
        server.shutdown()
        serving.join(timeout=3)
        server.server_close()


def test_emitted_payloads_conform_to_real_shared_schemas(tmp_path, monkeypatch):
    import yaml
    from jsonschema import Draft202012Validator, FormatChecker, ValidationError
    from referencing import Registry, Resource

    contracts = next(
        parent / "trading-contracts"
        for parent in Path(__file__).resolve().parents
        if (parent / "trading-contracts" / "schemas").exists()
    )
    registry = Registry()
    schemas = {}
    for path in (contracts / "schemas").glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(path.name, resource).with_resource(
            schema["$id"], resource
        )
        schemas[path.name] = schema
    openapi = yaml.safe_load(
        (contracts / "openapi" / "trading-agents.yaml").read_text(encoding="utf-8")
    )
    checker = FormatChecker()
    assert "date-time" in checker.checkers
    assert checker.conforms("2026-01-01T12:00:00", "date-time") is False
    assert checker.conforms("2026-01-01", "date-time") is False

    def validate(name, value):
        schema = schemas.get(name) or openapi["components"]["schemas"][name]
        Draft202012Validator(schema, registry=registry, format_checker=checker).validate(value)

    with pytest.raises(ValidationError):
        validate(
            "portfolio-context.schema.json",
            {"positions": [], "asOf": "2026-01-01T12:00:00"},
        )

    with live_server(tmp_path, start_dispatcher=False) as (server, store):
        from tradingagents.service import server as service_server

        with monkeypatch.context() as guard:
            guard.setattr(
                service_server,
                "ensure_provider_configured",
                lambda _: pytest.fail("invalid portfolio reached provider preflight"),
            )
            for as_of in ("2026-01-01T12:00:00", "2026-01-01"):
                status, error = request(
                    server,
                    "POST",
                    "/analysis",
                    ANALYSIS | {"portfolio": {"positions": [], "asOf": as_of}},
                )
                assert (status, error["code"]) == (400, "INVALID_REQUEST")
        assert store.claim_next() is None
        portfolio = {"positions": [], "asOf": "2026-01-01T12:00:00+07:00"}
        validate("portfolio-context.schema.json", portfolio)
        status, analysis = request(server, "POST", "/analysis", ANALYSIS | {"portfolio": portfolio})
        assert status == 202
        assert analysis["portfolio"] == portfolio
        validate("analysis-run.schema.json", analysis)
        _, backtest = request(
            server,
            "POST",
            "/backtests",
            {"tickers": ["NVDA"], "startDate": "2026-01-05", "endDate": "2026-01-05"},
        )
        validate("backtest-run.schema.json", backtest)
        from tradingagents.service import worker

        class OfflineGraph:
            def __init__(self, analysts, config):
                pass

            def propagate(self, ticker, date, asset_type, portfolio):
                return {
                    "final_trade_decision": "buy after evidence",
                    "market_report": "market evidence",
                }, "Buy"

        monkeypatch.setattr(worker, "TradingAgentsGraph", OfflineGraph)
        store.claim_next()
        worker.run_job(store, analysis["runId"], {"llm_provider": "ollama"})
        emitted = request(server, "GET", "/runs/" + analysis["runId"])[1]
        assert emitted["portfolio"] == portfolio
        validate("analysis-run.schema.json", emitted)
        decision = emitted["decision"]
        validate("trading-decision.schema.json", decision)
        validate("RunReport", request(server, "GET", f"/runs/{analysis['runId']}/report")[1])
        from tradingagents.backtest import BacktestResult, BacktestSummary

        monkeypatch.setattr(
            worker,
            "run_backtest",
            lambda *args, **kwargs: BacktestResult(
                run_id="offline",
                log_path=tmp_path / "unused",
                failures=[("NVDA", "2026-01-05", "private vendor error")],
            ),
        )
        monkeypatch.setattr(
            worker, "summarize", lambda result: BacktestSummary(resolved=0, pending=1, by_rating={})
        )
        store.claim_next()
        worker.run_job(store, backtest["backtestId"], {"llm_provider": "ollama"})
        validate(
            "backtest-run.schema.json",
            request(server, "GET", "/backtests/" + backtest["backtestId"])[1],
        )
        error = request(server, "GET", "/runs/00000000-0000-4000-8000-000000000001")[1]
        validate("ErrorResponse", error)
        with pytest.raises(ValidationError):
            validate("trading-decision.schema.json", decision | {"rating": "STRONG_BUY"})
        with pytest.raises(ValidationError):
            validate("backtest-run.schema.json", backtest | {"cellsRun": -1})
        with pytest.raises(ValidationError):
            validate(
                "ErrorResponse", {key: value for key, value in error.items() if key != "requestId"}
            )


def test_abrupt_parent_death_kills_spawned_child_before_later_work(tmp_path):
    """An orphan child must not outlive its parent and overlap a restarted dispatcher."""
    controller = tmp_path / "controller.py"
    started = tmp_path / "started"
    finished = tmp_path / "finished"
    controller.write_text(
        """import multiprocessing
import os
import time
from pathlib import Path
from tradingagents.service.server import create_server
from tradingagents.service.store import RunStore

def slow_child(started, finished):
    Path(started).write_text('started')
    time.sleep(1)
    Path(finished).write_text('finished')

def factory(target, args):
    return multiprocessing.Process(target=slow_child, args=(STARTED, FINISHED))

if __name__ == '__main__':
    STARTED, FINISHED, DB = os.environ['T14_STARTED'], os.environ['T14_FINISHED'], os.environ['T14_DB']
    store = RunStore(Path(DB))
    server = create_server('127.0.0.1', 0, store, {'llm_provider': 'ollama'},
                           process_factory=factory, worker_timeout=10)
    store.create('analysis', {'ticker': 'NVDA', 'tradeDate': '2026-01-05',
                 'assetType': 'stock', 'selectedAnalysts': ['market'], 'researchDepth': 'quick'}, 2)
    deadline = time.monotonic() + 8
    while not Path(STARTED).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not Path(STARTED).exists():
        os._exit(2)
    os._exit(0)
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        T14_STARTED=str(started),
        T14_FINISHED=str(finished),
        T14_DB=str(tmp_path / "runs.db"),
        PYTHONPATH=os.pathsep.join(
            (str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", ""))
        ),
    )
    controller_result = subprocess.run(
        [sys.executable, str(controller)], env=env, capture_output=True, text=True, timeout=15
    )
    assert controller_result.returncode == 0, controller_result.stderr
    time.sleep(1.5)
    assert not finished.exists()
    from tradingagents.service.server import create_server
    from tradingagents.service.store import RunStore

    reopened = RunStore(tmp_path / "runs.db")
    resumed = create_server(
        "127.0.0.1", 0, reopened, {"llm_provider": "ollama"}, start_dispatcher=False
    )
    try:
        assert reopened.claim_next() is None
    finally:
        resumed.server_close()


@pytest.mark.skipif(os.name != "nt", reason="Windows CTRL_BREAK_EVENT lifecycle")
def test_cli_interrupt_exits_and_closes_local_store(tmp_path):
    command = [
        sys.executable,
        "-m",
        "tradingagents.service",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--data-dir",
        str(tmp_path / "service"),
    ]
    child = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "service" / "runs.sqlite3").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "service" / "runs.sqlite3").exists()
        child.send_signal(signal.CTRL_BREAK_EVENT)
        _, errors = child.communicate(timeout=10)
        assert child.returncode == 0, errors
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
