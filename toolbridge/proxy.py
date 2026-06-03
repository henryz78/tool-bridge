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


def open_upstream_connection(
    settings: Settings,
    url: str | None = None,
    timeout: int | None = None,
) -> http.client.HTTPConnection:
    """Open a connection to the upstream API."""
    target_url = url or settings.upstream_url
    target_timeout = timeout if timeout is not None else settings.upstream_timeout
    host, port, _ = _parse_url(target_url)
    parsed = urllib.parse.urlparse(target_url)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=target_timeout)
    return http.client.HTTPConnection(host, port, timeout=target_timeout)


def build_upstream_headers(
    request_headers: dict | None,
    settings: Settings,
    auth: str | None = None,
) -> dict[str, str]:
    """Construct headers for an upstream request."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    target_auth = auth if auth is not None else settings.upstream_auth
    if target_auth:
        headers["Authorization"] = target_auth
    return headers


def _extract_model_from_body(body: bytes | None) -> str | None:
    if not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
        if isinstance(data, dict):
            return data.get("model")
    except Exception:
        pass
    return None


def fetch_upstream(
    method: str,
    path: str,
    body: bytes | None,
    settings: Settings,
) -> tuple[int, bytes, dict[str, str]]:
    """Make a non-streaming request to the upstream API.

    Returns (status_code, response_body, response_headers).
    """
    model = _extract_model_from_body(body)
    url, auth, timeout = settings.get_upstream_config(model)

    full_url = url.rstrip("/") + path
    req = urllib.request.Request(full_url, data=body, method=method)
    for k, v in build_upstream_headers(None, settings, auth=auth).items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_bytes = resp.read()
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, body_bytes, hdrs
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read() if exc.fp else b""
        raise UpstreamError(exc.code, body_bytes) from exc


def fetch_upstream_chat(payload: dict, settings: Settings) -> dict:
    """Send a Chat Completions request and return the parsed JSON response."""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, resp_body, _ = fetch_upstream("POST", "/v1/chat/completions", body, settings)
    return json.loads(resp_body)


def stream_upstream_chat(payload: dict, settings: Settings) -> http.client.HTTPResponse:
    """Open a streaming connection to the upstream and return the raw response
    for the caller to read SSE chunks from."""
    model = payload.get("model", "")
    url, auth, timeout = settings.get_upstream_config(model)

    conn = open_upstream_connection(settings, url=url, timeout=timeout)
    _, _, base = _parse_url(url)
    path = base + "/v1/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = build_upstream_headers(None, settings, auth=auth)
    conn.request("POST", path, body=body, headers=headers)
    return conn.getresponse()
