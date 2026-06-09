"""Request authentication helpers."""

from __future__ import annotations

import hmac
from typing import Any

from .config import Settings


def admin_authorized(handler: Any, settings: Settings) -> bool:
    return _authorized(handler, settings.admin_token, "X-Admin-Token")


def bridge_authorized(handler: Any, settings: Settings) -> bool:
    return _authorized(handler, settings.bridge_api_key, "X-Bridge-Api-Key")


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
