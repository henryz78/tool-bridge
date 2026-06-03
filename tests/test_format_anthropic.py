"""Tests for toolbridge.format_anthropic."""

import unittest
import json
from toolbridge.format_anthropic import (
    convert_anthropic_to_openai,
    convert_openai_to_anthropic,
    anthropic_stop_reason,
    anthropic_usage,
)


class TestConvertAnthropicToOpenAI(unittest.TestCase):
    def test_basic_message(self):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1024,
        }
        result = convert_anthropic_to_openai(payload)
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "user")

    def test_with_system(self):
        payload = {
            "model": "m",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        result = convert_anthropic_to_openai(payload)
        self.assertEqual(result["messages"][0]["role"], "system")

    def test_with_tools(self):
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"name": "run", "description": "Run", "input_schema": {}}],
        }
        result = convert_anthropic_to_openai(payload)
        self.assertIn("tools", result)
        self.assertEqual(result["tools"][0]["function"]["name"], "run")

    def test_tool_use_in_assistant(self):
        payload = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "run", "input": {"k": "v"}}
                ]},
            ],
            "max_tokens": 100,
        }
        result = convert_anthropic_to_openai(payload)
        asst = [m for m in result["messages"] if m["role"] == "assistant"][0]
        self.assertIn("tool_calls", asst)

    def test_tool_choice_required(self):
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"name": "run", "description": "", "input_schema": {}}],
            "tool_choice": "any",
        }
        result = convert_anthropic_to_openai(payload)
        self.assertEqual(result["tool_choice"], "required")


class TestConvertOpenAIToAnthropic(unittest.TestCase):
    def test_text_response(self):
        openai_resp = {
            "choices": [{
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_openai_to_anthropic(openai_resp, "test-model", False)
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["stop_reason"], "end_turn")
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        self.assertEqual(text_blocks[0]["text"], "hi there")

    def test_tool_call_response(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "run", "arguments": '{"k":"v"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = convert_openai_to_anthropic(openai_resp, "test-model", True)
        self.assertEqual(result["stop_reason"], "tool_use")
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "run")

    def test_thinking_block(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "result",
                    "reasoning_content": "thinking...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        result = convert_openai_to_anthropic(openai_resp, "m", False)
        think_blocks = [b for b in result["content"] if b["type"] == "thinking"]
        self.assertEqual(len(think_blocks), 1)


class TestAnthropicStopReason(unittest.TestCase):
    def test_stop(self):
        self.assertEqual(anthropic_stop_reason("stop", []), "end_turn")

    def test_length(self):
        self.assertEqual(anthropic_stop_reason("length", []), "max_tokens")

    def test_tool_calls(self):
        self.assertEqual(anthropic_stop_reason("tool_calls", []), "tool_use")

    def test_tool_use_block(self):
        self.assertEqual(anthropic_stop_reason("stop", [{"type": "tool_use"}]), "tool_use")


class TestAnthropicUsage(unittest.TestCase):
    def test_mapping(self):
        result = anthropic_usage({"prompt_tokens": 10, "completion_tokens": 5})
        self.assertEqual(result, {"input_tokens": 10, "output_tokens": 5})


if __name__ == "__main__":
    unittest.main()
