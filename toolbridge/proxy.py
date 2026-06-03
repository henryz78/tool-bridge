"""Upstream HTTP client — connection management and request helpers."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import urllib.error
import http.client
from typing import Any

from .config import Settings
from .errors import UpstreamError


def _parse_url(url: str) -> tuple[str, int, str]:
    """Parse an upstream URL into (host, port, base_path)."""
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    base = parsed.path.rstrip("/")
    return host, port, base


def open_upstream_connection(settings: Settings) -> http.client.HTTPConnection:
    """Open a connection to the upstream API."""
    host, port, _ = _parse_url(settings.upstream_url)
    parsed = urllib.parse.urlparse(settings.upstream_url)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=settings.upstream_timeout)
    return http.client.HTTPConnection(host, port, timeout=settings.upstream_timeout)


def build_upstream_headers(request_headers: dict | None, settings: Settings) -> dict[str, str]:
    """Construct headers for an upstream request."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if settings.upstream_auth:
        headers["Authorization"] = settings.upstream_auth
    return headers


def fetch_upstream(
    method: str,
    path: str,
    body: bytes | None,
    settings: Settings,
) -> tuple[int, bytes, dict[str, str]]:
    """Make a non-streaming request to the upstream API.

    Returns (status_code, response_body, response_headers).
    """
    url = settings.upstream_url.rstrip("/") + path
    req = urllib.request.Request(url, data=body, method=method)
    for k, v in build_upstream_headers(None, settings).items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=settings.upstream_timeout) as resp:
            body = resp.read()
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, body, hdrs
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp else b""
        raise UpstreamError(exc.code, body) from exc


def fetch_upstream_chat(payload: dict, settings: Settings) -> dict:
    """Send a Chat Completions request and return the parsed JSON response."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, resp_body, _ = fetch_upstream("POST", "/v1/chat/completions", body, settings)
    return json.loads(resp_body)


def stream_upstream_chat(payload: dict, settings: Settings) -> http.client.HTTPResponse:
    """Open a streaming connection to the upstream and return the raw response
    for the caller to read SSE chunks from."""
    conn = open_upstream_connection(settings)
    _, _, base = _parse_url(settings.upstream_url)
    path = base + "/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = build_upstream_headers(None, settings)
    conn.request("POST", path, body=body, headers=headers)
    return conn.getresponse()
