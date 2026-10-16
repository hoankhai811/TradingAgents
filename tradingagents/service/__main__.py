"""Start the local thin service with server-owned engine settings."""

import argparse
import os
import signal
from pathlib import Path

from tradingagents.default_config import build_default_config
from tradingagents.service.server import create_server
from tradingagents.service.store import RunStore


def main():
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    parser = argparse.ArgumentParser(description="TradingAgents loopback thin service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-pending", type=int, default=16)
    parser.add_argument("--max-body-bytes", type=int, default=65536)
    parser.add_argument("--worker-timeout", type=float, default=900)
    args = parser.parse_args()
    server = None
    try:
        server = create_server(
            args.host,
            args.port,
            RunStore(args.data_dir / "runs.sqlite3"),
            build_default_config(),
            max_pending=args.max_pending,
            max_body_bytes=args.max_body_bytes,
            worker_timeout=args.worker_timeout,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.stop_workers()
            server.server_close()


if __name__ == "__main__":
    main()
