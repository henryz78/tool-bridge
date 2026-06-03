"""Tests for toolbridge.model_map."""

import unittest
from toolbridge.model_map import strip_model_hints, resolve_model, is_native_model, collect_exposed_ids


class TestStripModelHints(unittest.TestCase):
    def test_strips_bracket_suffix(self):
        self.assertEqual(strip_model_hints("model[1m]"), "model")
        self.assertEqual(strip_model_hints("model[128k]"), "model")
        self.assertEqual(strip_model_hints("model[32K]"), "model")

    def test_no_suffix(self):
        self.assertEqual(strip_model_hints("model"), "model")

    def test_whitespace(self):
        self.assertEqual(strip_model_hints("  model  "), "model")


class TestResolveModel(unittest.TestCase):
    def test_mapped(self):
        self.assertEqual(resolve_model("chat", {"chat": "upstream"}, True), "upstream")

    def test_unmapped_allowed(self):
        self.assertEqual(resolve_model("other", {}, True), "other")

    def test_unmapped_blocked(self):
        self.assertIsNone(resolve_model("other", {}, False))

    def test_hint_stripped(self):
        self.assertEqual(resolve_model("chat[1m]", {"chat": "upstream"}, True), "upstream")


class TestIsNativeModel(unittest.TestCase):
    def test_native(self):
        self.assertTrue(is_native_model("pro", {"pro": "up-pro"}, {"up-pro"}, True))

    def test_not_native(self):
        self.assertFalse(is_native_model("chat", {"chat": "up-chat"}, {"up-pro"}, True))

    def test_unmapped_not_native(self):
        self.assertFalse(is_native_model("unknown", {}, {"up-pro"}, False))


class TestCollectExposedIds(unittest.TestCase):
    def test_combines_sources(self):
        result = collect_exposed_ids({"a": "up-a"}, {"up-a"}, ["a", "c"])
        self.assertEqual(result, ["a", "c", "up-a"])

    def test_dedup(self):
        result = collect_exposed_ids({"a": "up-a"}, set(), ["a"])
        self.assertEqual(result, ["a", "up-a"])

    def test_empty(self):
        result = collect_exposed_ids({}, set(), [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
