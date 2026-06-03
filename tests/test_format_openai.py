"""Tests for toolbridge.format_openai."""

import unittest
import json
from toolbridge.format_openai import (
    serialize_content,
    normalize_tool_calls,
    normalize_messages,
    has_tool_calls_in_history,
)


class TestSerializeContent(unittest.TestCase):
    def test_string(self):
        self.assertEqual(serialize_content("hello"), "hello")

    def test_none(self):
        self.assertEqual(serialize_content(None), "")

    def test_list_of_text(self):
        result = serialize_content([{"type": "text", "text": "hi"}])
        self.assertEqual(result, "hi")

    def test_dict(self):
        result = serialize_content({"key": "val"})
        self.assertIn("key", result)


class TestNormalizeToolCalls(unittest.TestCase):
    def test_standard_format(self):
        calls = [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]
        result = normalize_tool_calls(calls)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "run")

    def test_flat_format(self):
        calls = [{"id": "c1", "name": "run", "arguments": "{}"}]
        result = normalize_tool_calls(calls)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "run")

    def test_dict_arguments(self):
        calls = [{"id": "c1", "name": "run", "arguments": {"key": "val"}}]
        result = normalize_tool_calls(calls)
        parsed = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(parsed, {"key": "val"})


class TestNormalizeMessages(unittest.TestCase):
    def test_tool_role_converted(self):
        msgs = [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]
        result = normalize_messages(msgs)
        self.assertEqual(result[0]["role"], "user")
        self.assertIn("t1", result[0]["content"])

    def test_assistant_with_tool_calls(self):
        msgs = [{"role": "assistant", "content": "text", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        ]}]
        result = normalize_messages(msgs)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertIn("tool_calls", result[0])
        self.assertIn("content", result[0])


class TestHasToolCallsInHistory(unittest.TestCase):
    def test_true(self):
        msgs = [{"role": "assistant", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "a", "arguments": "{}"}}]}]
        self.assertTrue(has_tool_calls_in_history(msgs))

    def test_false(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        self.assertFalse(has_tool_calls_in_history(msgs))


if __name__ == "__main__":
    unittest.main()
