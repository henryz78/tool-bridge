"""HTTP server and request handler."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .config import Settings
from .router import dispatch


class BridgeServer(ThreadingHTTPServer):
    """Threading HTTP server that holds application settings."""

    def __init__(self, address: tuple[str, int], handler_cls: type, settings: Settings):
        self.settings = settings
        super().__init__(address, handler_cls)


class BridgeHandler(BaseHTTPRequestHandler):
    """Thin request handler that delegates to the router."""

    server: BridgeServer

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:
        dispatch(self, self.server.settings, "GET", self.path, None)

    def do_HEAD(self) -> None:
        dispatch(self, self.server.settings, "HEAD", self.path, None)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else None
        dispatch(self, self.server.settings, "POST", self.path, body)

    def log_message(self, format: str, *args: Any) -> None:
        # Minimal logging to stdout
        print(f"[bridge] {format % args}")


def create_server(settings: Settings) -> BridgeServer:
    """Create and return a BridgeServer instance."""
    return BridgeServer(
        (settings.listen_host, settings.listen_port),
        BridgeHandler,
        settings,
    )


def run_server(settings: Settings) -> None:
    """Create and start the server. Blocks until interrupted."""
    srv = create_server(settings)
    print(f"toolbridge listening on {settings.listen_host}:{settings.listen_port}")
    print(f"  upstream: {settings.upstream_url}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        srv.shutdown()


# ---------------------------------------------------------------------------
# Threaded server management (for GUI mode)
# ---------------------------------------------------------------------------

_server_instance: BridgeServer | None = None
_server_thread: threading.Thread | None = None


def start_server_threaded(settings: Settings) -> None:
    """Start the server in a daemon thread. Safe to call from any thread."""
    global _server_instance, _server_thread
    stop_server()
    _server_instance = create_server(settings)
    _server_thread = threading.Thread(target=_server_instance.serve_forever, daemon=True)
    _server_thread.start()


def stop_server() -> None:
    """Stop the running server if any."""
    global _server_instance, _server_thread
    if _server_instance is not None:
        _server_instance.shutdown()
        _server_instance.server_close()
        _server_instance = None
    _server_thread = None


def is_server_running() -> bool:
    return _server_instance is not None


def get_server_port() -> int | None:
    if _server_instance is not None:
        return _server_instance.settings.listen_port
    return None
