"""Tests for dashboard/settings route handlers."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import toolbridge.routes_dashboard as routes_dashboard
from toolbridge.config import Settings
from toolbridge.routes_dashboard import (
    handle_api_settings_get,
    handle_api_settings_post,
    handle_api_status,
    handle_api_upstream_models,
)


class FakeHandler:
    def __init__(self, payload: dict | None = None, path: str = "/") -> None:
        raw = json.dumps(payload or {}).encode("utf-8")
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        }
        self.path = path
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.response_status: int | None = None
        self.response_headers: list[tuple[str, str]] = []
        self.server = SimpleNamespace(settings=None)

    def send_response(self, status: int) -> None:
        self.response_status = status

    def send_header(self, key: str, value: str) -> None:
        self.response_headers.append((key, value))

    def end_headers(self) -> None:
        pass

    def json_response(self) -> dict:
        return json.loads(self.wfile.getvalue().decode("utf-8"))


class FakeModelsResponse:
    def __enter__(self) -> "FakeModelsResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({
            "object": "list",
            "data": [{"id": "provider-model"}],
        }).encode("utf-8")


class TestSettingsRoutes(unittest.TestCase):
    def test_settings_get_returns_serialized_settings_without_cors(self) -> None:
        handler = FakeHandler()
        settings = Settings(listen_host="127.0.0.1", listen_port=9876)

        handle_api_settings_get(handler, settings)

        self.assertEqual(handler.response_status, 200)
        self.assertEqual(handler.json_response()["PORT"], 9876)
        header_names = {name for name, _value in handler.response_headers}
        self.assertNotIn("Access-Control-Allow-Origin", header_names)

    def test_settings_post_saves_config_and_swaps_server_settings(self) -> None:
        payload = {
            "HOST": "127.0.0.1",
            "PORT": 9876,
            "UPSTREAM_BASE_URL": "http://provider.example/api",
            "UPSTREAM_TIMEOUT_SECONDS": "12",
            "UPSTREAM_AUTH_HEADER": "provider-token",
            "MODEL_MAP_JSON": {"public-chat": "provider-chat"},
            "NATIVE_TOOL_MODELS_JSON": ["provider-chat"],
            "FC_ERROR_RETRY": False,
            "FC_ERROR_RETRY_MAX_ATTEMPTS": "2",
            "RETRY_DELAY_SECONDS": "0.5",
        }
        handler = FakeHandler(payload)
        settings = Settings(listen_host="127.0.0.1", listen_port=9876)
        handler.server.settings = settings
        saved_configs: list[dict] = []

        with mock.patch("toolbridge.config_file.save_config", side_effect=saved_configs.append):
            handle_api_settings_post(handler, settings)

        self.assertEqual(handler.response_status, 200)
        self.assertEqual(handler.json_response(), {"ok": True, "port_changed": False})
        self.assertEqual(handler.server.settings.upstream_url, "http://provider.example/api")
        self.assertEqual(handler.server.settings.upstream_timeout, 12)
        self.assertEqual(handler.server.settings.upstream_auth, "provider-token")
        self.assertEqual(handler.server.settings.name_mapping, {"public-chat": "provider-chat"})
        self.assertEqual(handler.server.settings.native_tool_model_ids, {"provider-chat"})
        self.assertFalse(handler.server.settings.retry_on_parse_failure)
        self.assertEqual(handler.server.settings.max_retry_attempts, 2)
        self.assertEqual(handler.server.settings.retry_delay_seconds, 0.5)
        self.assertEqual(saved_configs[0]["UPSTREAM_BASE_URL"], "http://provider.example/api")
        header_names = {name for name, _value in handler.response_headers}
        self.assertNotIn("Access-Control-Allow-Origin", header_names)


class TestStatusRoute(unittest.TestCase):
    def test_status_returns_cached_latency_and_autostart_state(self) -> None:
        handler = FakeHandler()
        settings = Settings(listen_host="127.0.0.1", listen_port=9876)
        with routes_dashboard._latency_lock:
            routes_dashboard._latency_cache["default"] = 42
            routes_dashboard._latency_cache["providers"] = {"provider-a": 9}

        with mock.patch("toolbridge.routes_dashboard.start_latency_monitor") as start_monitor, \
             mock.patch("toolbridge.autostart.is_autostart_enabled", return_value=True):
            handle_api_status(handler, settings)

        self.assertEqual(handler.response_status, 200)
        self.assertEqual(handler.json_response(), {
            "status": "running",
            "port": 9876,
            "host": "127.0.0.1",
            "upstream_latency_ms": 42,
            "provider_latencies": {"provider-a": 9},
            "autostart_enabled": True,
        })
        start_monitor.assert_called_once_with(settings)
        header_names = {name for name, _value in handler.response_headers}
        self.assertNotIn("Access-Control-Allow-Origin", header_names)


class TestUpstreamModelsRoute(unittest.TestCase):
    def test_upstream_models_uses_selected_provider_config(self) -> None:
        handler = FakeHandler(path="/api/upstream/models?provider_id=provider-a")
        settings = Settings(
            upstream_url="http://default.example/base",
            upstream_auth="default-token",
            upstream_timeout=3,
            upstreams=[{
                "id": "provider-a",
                "url": "http://provider.example/api",
                "auth": "provider-token",
                "timeout": 9,
            }],
        )
        captured: dict = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            return FakeModelsResponse()

        with mock.patch("toolbridge.routes_dashboard.urllib.request.urlopen", side_effect=fake_urlopen):
            handle_api_upstream_models(handler, settings)

        self.assertEqual(handler.response_status, 200)
        self.assertEqual(captured["url"], "http://provider.example/api/v1/models")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer provider-token")
        self.assertEqual(captured["timeout"], 9)
        self.assertEqual(handler.json_response()["data"][0]["id"], "provider-model")


if __name__ == "__main__":
    unittest.main()
