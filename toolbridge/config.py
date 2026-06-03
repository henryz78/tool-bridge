"""Application settings read from environment variables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import ClassVar


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.lower() in ("true", "1", "yes", "on")


def _env_int(key: str, default: int = 0) -> int:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_json_dict(key: str, default: dict | None = None) -> dict[str, str]:
    raw = os.environ.get(key, "")
    if not raw:
        return default or {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _env_json_list(key: str, default: list | None = None) -> list[str]:
    raw = os.environ.get(key, "")
    if not raw:
        return default or []
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


@dataclass
class Settings:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    upstream_url: str = "http://127.0.0.1:3000"
    upstream_timeout: int = 240
    upstream_auth: str = ""
    upstream_extra_fields: dict = field(default_factory=dict)
    name_mapping: dict[str, str] = field(default_factory=dict)
    allow_unmapped: bool = True
    native_tool_model_ids: set[str] = field(default_factory=set)
    exposed_model_ids: list[str] = field(default_factory=list)
    tool_instruction_intro: str = ""
    retry_on_parse_failure: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 0.0
    upstreams: list[dict] = field(default_factory=list)
    model_routes: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Model resolution helpers
    # ------------------------------------------------------------------

    def resolve_model_name(self, requested: str) -> str | None:
        from .model_map import resolve_model, strip_model_hints

        clean = strip_model_hints(requested)
        return resolve_model(clean, self.name_mapping, self.allow_unmapped)

    def is_native_tool_model(self, requested: str) -> bool:
        from .model_map import is_native_model, strip_model_hints

        clean = strip_model_hints(requested)
        return is_native_model(clean, self.name_mapping, self.native_tool_model_ids, self.allow_unmapped)

    def get_exposed_models(self) -> list[str]:
        from .model_map import collect_exposed_ids

        base = collect_exposed_ids(self.name_mapping, self.native_tool_model_ids, self.exposed_model_ids)
        seen = set(base)
        result = list(base)
        for m in self.model_routes:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def get_upstream_config(self, model: str) -> tuple[str, str, int]:
        """Return (url, auth, timeout) for the given model."""
        if not model:
            return self.upstream_url, self.upstream_auth, self.upstream_timeout
        provider_id = self.model_routes.get(model)
        if provider_id:
            for p in self.upstreams:
                if p.get("id") == provider_id:
                    return (
                        str(p.get("url", self.upstream_url)),
                        str(p.get("auth", self.upstream_auth)),
                        int(p.get("timeout", self.upstream_timeout))
                    )
        return self.upstream_url, self.upstream_auth, self.upstream_timeout

    # ------------------------------------------------------------------
    # Serialization (for GUI config file)
    # ------------------------------------------------------------------

    _FIELD_TO_ENV: ClassVar[dict[str, str]] = {
        "listen_host": "HOST",
        "listen_port": "PORT",
        "upstream_url": "UPSTREAM_BASE_URL",
        "upstream_timeout": "UPSTREAM_TIMEOUT_SECONDS",
        "upstream_auth": "UPSTREAM_AUTH_HEADER",
        "upstream_extra_fields": "UPSTREAM_EXTRA_BODY_JSON",
        "name_mapping": "MODEL_MAP_JSON",
        "allow_unmapped": "ALLOW_UNMAPPED_MODEL_PASSTHROUGH",
        "native_tool_model_ids": "NATIVE_TOOL_MODELS_JSON",
        "exposed_model_ids": "PUBLIC_MODEL_IDS_JSON",
        "tool_instruction_intro": "TOOL_PROMPT_PREAMBLE",
        "retry_on_parse_failure": "FC_ERROR_RETRY",
        "max_retry_attempts": "FC_ERROR_RETRY_MAX_ATTEMPTS",
        "retry_delay_seconds": "RETRY_DELAY_SECONDS",
        "upstreams": "UPSTREAMS_JSON",
        "model_routes": "MODEL_ROUTES_JSON",
    }

    def to_dict(self) -> dict:
        """Export settings as a dict using env-var-style keys."""
        out: dict = {}
        for field_name, env_key in self._FIELD_TO_ENV.items():
            val = getattr(self, field_name)
            if isinstance(val, set):
                val = sorted(val)
            out[env_key] = val
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Settings:
        """Create Settings from a dict using env-var-style keys."""
        return cls(
            listen_host=str(data.get("HOST", "0.0.0.0")),
            listen_port=int(data.get("PORT", 8080)),
            upstream_url=str(data.get("UPSTREAM_BASE_URL", "http://127.0.0.1:3000")),
            upstream_timeout=int(data.get("UPSTREAM_TIMEOUT_SECONDS", 240)),
            upstream_auth=str(data.get("UPSTREAM_AUTH_HEADER", "")),
            upstream_extra_fields=data.get("UPSTREAM_EXTRA_BODY_JSON", {}),
            name_mapping=data.get("MODEL_MAP_JSON", {}),
            allow_unmapped=bool(data.get("ALLOW_UNMAPPED_MODEL_PASSTHROUGH", True)),
            native_tool_model_ids=set(data.get("NATIVE_TOOL_MODELS_JSON", [])),
            exposed_model_ids=list(data.get("PUBLIC_MODEL_IDS_JSON", [])),
            tool_instruction_intro=str(data.get("TOOL_PROMPT_PREAMBLE", "")),
            retry_on_parse_failure=bool(data.get("FC_ERROR_RETRY", True)),
            max_retry_attempts=int(data.get("FC_ERROR_RETRY_MAX_ATTEMPTS", 3)),
            retry_delay_seconds=float(data.get("RETRY_DELAY_SECONDS", 0)),
            upstreams=list(data.get("UPSTREAMS_JSON", [])),
            model_routes=dict(data.get("MODEL_ROUTES_JSON", {})),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            listen_host=_env_str("HOST", "0.0.0.0"),
            listen_port=_env_int("PORT", 8080),
            upstream_url=_env_str("UPSTREAM_BASE_URL", "http://127.0.0.1:3000"),
            upstream_timeout=_env_int("UPSTREAM_TIMEOUT_SECONDS", 240),
            upstream_auth=_env_str("UPSTREAM_AUTH_HEADER"),
            upstream_extra_fields=_env_json_dict("UPSTREAM_EXTRA_BODY_JSON"),
            name_mapping=_env_json_dict("MODEL_MAP_JSON"),
            allow_unmapped=_env_bool("ALLOW_UNMAPPED_MODEL_PASSTHROUGH", True),
            native_tool_model_ids=set(_env_json_list("NATIVE_TOOL_MODELS_JSON")),
            exposed_model_ids=_env_json_list("PUBLIC_MODEL_IDS_JSON"),
            tool_instruction_intro=_env_str("TOOL_PROMPT_PREAMBLE"),
            retry_on_parse_failure=_env_bool("FC_ERROR_RETRY", True),
            max_retry_attempts=_env_int("FC_ERROR_RETRY_MAX_ATTEMPTS", 3),
            retry_delay_seconds=float(os.environ.get("RETRY_DELAY_SECONDS", "0")),
            upstreams=_env_json_list("UPSTREAMS_JSON"),
            model_routes=_env_json_dict("MODEL_ROUTES_JSON"),
        )
