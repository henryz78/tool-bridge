"""Tests for toolbridge.config."""

import os
import unittest
from toolbridge.config import Settings, _env_bool, _env_int, _env_json_dict, _env_json_list, _env_str


class TestEnvHelpers(unittest.TestCase):
    def test_bool_true_values(self):
        for val in ("true", "True", "1", "yes", "on"):
            os.environ["_TEST_BOOL"] = val
            self.assertTrue(_env_bool("_TEST_BOOL"))
            del os.environ["_TEST_BOOL"]

    def test_bool_false_values(self):
        for val in ("false", "0", "no", "off", ""):
            os.environ["_TEST_BOOL"] = val if val else ""
            if not val:
                os.environ.pop("_TEST_BOOL", None)
            self.assertFalse(_env_bool("_TEST_BOOL", default=False))
            if "_TEST_BOOL" in os.environ:
                del os.environ["_TEST_BOOL"]

    def test_bool_default(self):
        self.assertTrue(_env_bool("_NOT_SET_XYZ", default=True))
        self.assertFalse(_env_bool("_NOT_SET_XYZ", default=False))

    def test_int_parsing(self):
        os.environ["_TEST_INT"] = "42"
        self.assertEqual(_env_int("_TEST_INT"), 42)
        del os.environ["_TEST_INT"]

    def test_int_default(self):
        self.assertEqual(_env_int("_NOT_SET_XYZ", 99), 99)

    def test_int_invalid(self):
        os.environ["_TEST_INT"] = "abc"
        self.assertEqual(_env_int("_TEST_INT", 7), 7)
        del os.environ["_TEST_INT"]

    def test_json_dict(self):
        os.environ["_TEST_DICT"] = '{"a":"b"}'
        self.assertEqual(_env_json_dict("_TEST_DICT"), {"a": "b"})
        del os.environ["_TEST_DICT"]

    def test_json_list(self):
        os.environ["_TEST_LIST"] = '["x","y"]'
        self.assertEqual(_env_json_list("_TEST_LIST"), ["x", "y"])
        del os.environ["_TEST_LIST"]

    def test_str(self):
        os.environ["_TEST_STR"] = "  hello  "
        self.assertEqual(_env_str("_TEST_STR"), "hello")
        del os.environ["_TEST_STR"]


class TestSettings(unittest.TestCase):
    def test_from_environment_defaults(self):
        # Clear all relevant env vars
        for key in ["HOST", "PORT", "UPSTREAM_BASE_URL", "UPSTREAM_TIMEOUT_SECONDS",
                     "UPSTREAM_AUTH_HEADER", "UPSTREAM_EXTRA_BODY_JSON", "MODEL_MAP_JSON",
                     "ALLOW_UNMAPPED_MODEL_PASSTHROUGH", "NATIVE_TOOL_MODELS_JSON",
                     "PUBLIC_MODEL_IDS_JSON", "TOOL_PROMPT_PREAMBLE",
                     "FC_ERROR_RETRY", "FC_ERROR_RETRY_MAX_ATTEMPTS", "RETRY_DELAY_SECONDS"]:
            os.environ.pop(key, None)
        s = Settings.from_environment()
        self.assertEqual(s.listen_host, "127.0.0.1")
        self.assertEqual(s.listen_port, 8080)
        self.assertEqual(s.upstream_url, "http://127.0.0.1:3000")
        self.assertTrue(s.allow_unmapped)

    def test_from_environment_custom(self):
        os.environ["PORT"] = "9999"
        os.environ["UPSTREAM_BASE_URL"] = "http://example.com:4000"
        os.environ["MODEL_MAP_JSON"] = '{"a":"b"}'
        os.environ["NATIVE_TOOL_MODELS_JSON"] = '["b"]'
        os.environ["FC_ERROR_RETRY"] = "false"
        os.environ["FC_ERROR_RETRY_MAX_ATTEMPTS"] = "5"
        os.environ["ADMIN_TOKEN"] = "admin-secret"
        os.environ["BRIDGE_API_KEY"] = "bridge-secret"
        s = Settings.from_environment()
        self.assertEqual(s.listen_port, 9999)
        self.assertEqual(s.upstream_url, "http://example.com:4000")
        self.assertEqual(s.name_mapping, {"a": "b"})
        self.assertEqual(s.native_tool_model_ids, {"b"})
        self.assertFalse(s.retry_on_parse_failure)
        self.assertEqual(s.max_retry_attempts, 5)
        self.assertEqual(s.admin_token, "admin-secret")
        self.assertEqual(s.bridge_api_key, "bridge-secret")
        for key in ["PORT", "UPSTREAM_BASE_URL", "MODEL_MAP_JSON",
                     "NATIVE_TOOL_MODELS_JSON", "FC_ERROR_RETRY",
                     "FC_ERROR_RETRY_MAX_ATTEMPTS", "ADMIN_TOKEN",
                     "BRIDGE_API_KEY"]:
            del os.environ[key]

    def test_resolve_model_name(self):
        s = Settings(name_mapping={"chat": "upstream-chat"}, allow_unmapped=True)
        self.assertEqual(s.resolve_model_name("chat"), "upstream-chat")
        self.assertEqual(s.resolve_model_name("unknown"), "unknown")

    def test_resolve_model_name_unmapped_disabled(self):
        s = Settings(name_mapping={"chat": "upstream-chat"}, allow_unmapped=False)
        self.assertIsNone(s.resolve_model_name("unknown"))

    def test_is_native_tool_model(self):
        s = Settings(name_mapping={"pro": "upstream-pro"}, native_tool_model_ids={"upstream-pro"}, allow_unmapped=True)
        self.assertTrue(s.is_native_tool_model("pro"))
        self.assertFalse(s.is_native_tool_model("chat"))

    def test_get_exposed_models(self):
        s = Settings(
            name_mapping={"a": "up-a", "b": "up-b"},
            native_tool_model_ids={"up-a"},
            exposed_model_ids=["a", "c"],
        )
        models = s.get_exposed_models()
        self.assertIn("a", models)
        self.assertIn("c", models)
        self.assertIn("b", models)

    def test_to_dict_can_redact_secrets(self):
        s = Settings(
            upstream_auth="provider-token",
            admin_token="admin-secret",
            bridge_api_key="bridge-secret",
            upstreams=[{"id": "p1", "auth": "provider-token"}],
        )
        data = s.to_dict(redact_secrets=True)
        self.assertEqual(data["UPSTREAM_AUTH_HEADER"], "********")
        self.assertEqual(data["ADMIN_TOKEN"], "********")
        self.assertEqual(data["BRIDGE_API_KEY"], "********")
        self.assertEqual(data["UPSTREAMS_JSON"][0]["auth"], "********")


if __name__ == "__main__":
    unittest.main()
