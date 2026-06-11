"""Request authentication helpers."""

from __future__ import annotations

import hmac
import ipaddress
from typing import Any

from .config import Settings


def admin_authorized(handler: Any, settings: Settings) -> bool:
    return _authorized(handler, settings.admin_token, "X-Admin-Token")


def bridge_authorized(handler: Any, settings: Settings) -> bool:
    if not settings.bridge_api_key and is_public_bind_host(settings.listen_host):
        return False
    return _authorized(handler, settings.bridge_api_key, "X-Bridge-Api-Key")


def is_public_bind_host(host: str) -> bool:
    raw = str(host or "").strip().lower()
    if raw in {"127.0.0.1", "::1", "localhost"}:
        return False
    if not raw:
        return True

    candidate = raw
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]

    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return True
    return not address.is_loopback


def validate_public_bind_auth(settings: Settings) -> None:
    if not is_public_bind_host(settings.listen_host):
        return

    missing = []
    if not settings.admin_token:
        missing.append("ADMIN_TOKEN")
    if not settings.bridge_api_key:
        missing.append("BRIDGE_API_KEY")
    if 0 < len(missing) < 2:
        names = ", ".join(missing)
        raise ValueError(
            f"Public HOST setup is incomplete; missing: {names}. "
            "Set both ADMIN_TOKEN and BRIDGE_API_KEY, or leave both empty for first-run setup mode."
        )


def _authorized(handler: Any, expected: str, header_name: str) -> bool:
    if not expected:
        return True
    presented = _token_from_headers(handler, header_name)
    return bool(presented) and hmac.compare_digest(presented, expected)


def _token_from_headers(handler: Any, header_name: str) -> str:
    direct = handler.headers.get(header_name, "").strip()
    if direct:
        return direct

    auth = handler.headers.get("Authorization", "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""
