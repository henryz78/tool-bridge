"""Tests for toolbridge.server threaded management and Settings serialization."""

import json
import time
import unittest
from unittest import mock

import requests

from toolbridge.config import Settings
from toolbridge.server import (
    create_server,
    start_server_threaded,
    stop_server,
    is_server_running,
    get_server_port,
    run_server,
)


class TestThreadedServer(unittest.TestCase):
    def setUp(self) -> None:
        stop_server()

    def tearDown(self) -> None:
        stop_server()

    def test_start_stop(self) -> None:
        s = Settings(listen_port=18999)
        start_server_threaded(s)
        self.assertTrue(is_server_running())
        self.assertEqual(get_server_port(), 18999)
        stop_server()
        self.assertFalse(is_server_running())
        self.assertIsNone(get_server_port())

    def test_http_responds(self) -> None:
        s = Settings(listen_port=18998)
        start_server_threaded(s)
        try:
            resp = requests.get("http://127.0.0.1:18998/health", timeout=2)
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {"ok": True})
        finally:
            stop_server()

    def test_restart_changes_port(self) -> None:
        s1 = Settings(listen_port=18997)
        start_server_threaded(s1)
        self.assertEqual(get_server_port(), 18997)
        s2 = Settings(listen_port=18996)
        start_server_threaded(s2)
        self.assertEqual(get_server_port(), 18996)
        stop_server()
        self.assertFalse(is_server_running())


class TestSettingsSerialization(unittest.TestCase):
    def test_roundtrip(self) -> None:
        original = Settings(
            listen_port=9999,
            upstream_url="http://example.com",
            name_mapping={"gpt-4": "deepseek"},
            native_tool_model_ids={"m1", "m2"},
            retry_on_parse_failure=False,
        )
        data = original.to_dict()
        self.assertIn("PORT", data)
        self.assertIn("UPSTREAM_BASE_URL", data)
        self.assertIn("MODEL_MAP_JSON", data)
        self.assertIn("NATIVE_TOOL_MODELS_JSON", data)
        self.assertEqual(data["PORT"], 9999)
        self.assertEqual(data["UPSTREAM_BASE_URL"], "http://example.com")
        self.assertEqual(data["MODEL_MAP_JSON"], {"gpt-4": "deepseek"})
        self.assertIsInstance(data["NATIVE_TOOL_MODELS_JSON"], list)
        self.assertEqual(sorted(data["NATIVE_TOOL_MODELS_JSON"]), ["m1", "m2"])

        restored = Settings.from_dict(data)
        self.assertEqual(restored.listen_port, 9999)
        self.assertEqual(restored.upstream_url, "http://example.com")
        self.assertEqual(restored.name_mapping, {"gpt-4": "deepseek"})
        self.assertEqual(restored.native_tool_model_ids, {"m1", "m2"})
        self.assertFalse(restored.retry_on_parse_failure)

    def test_from_dict_defaults(self) -> None:
        s = Settings.from_dict({})
        self.assertEqual(s.listen_port, 8080)
        self.assertEqual(s.upstream_url, "http://127.0.0.1:3000")
        self.assertTrue(s.retry_on_parse_failure)

    def test_from_dict_partial(self) -> None:
        s = Settings.from_dict({"PORT": 5555, "FC_ERROR_RETRY": False})
        self.assertEqual(s.listen_port, 5555)
        self.assertFalse(s.retry_on_parse_failure)
        self.assertEqual(s.upstream_url, "http://127.0.0.1:3000")


if __name__ == "__main__":
    unittest.main()
