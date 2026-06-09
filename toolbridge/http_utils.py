"""HTTP handler utilities shared by route modules."""

from __future__ import annotations

import json
from typing import Any


def read_json_body(handler: Any) -> dict | None:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.startswith("application/json"):
        return None
    length = int(handler.headers.get("Content-Length", 0))
    if not length:
        return None
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def send_json(handler: Any, status: int, payload: dict, cors: bool = True) -> None:
    if getattr(handler, "sse_started", False):
        error_msg = json.dumps({"error": payload.get("error", "internal error")}, ensure_ascii=False)
        try:
            handler.wfile.write(f"event: error\ndata: {error_msg}\n\n".encode("utf-8"))
            handler.wfile.flush()
        except Exception:
            pass
        return

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    if cors:
        handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
