"""Tests for toolbridge.virtual_tools."""

import unittest
from toolbridge.virtual_tools import (
    generate_activation_marker,
    build_tool_directive,
    parse_tool_invocation,
    classify_failure,
    FailureKind,
    ToolInvocation,
    ParseOutcome,
    validate_tool_params,
    _strip_think_blocks,
    _fallback_extract,
)


class TestGenerateActivationMarker(unittest.TestCase):
    def test_format(self):
        marker = generate_activation_marker()
        self.assertTrue(marker.startswith("[[CALL-"))
        self.assertTrue(marker.endswith("]]"))
        hex_part = marker[7:-2]
        self.assertEqual(len(hex_part), 6)
        self.assertTrue(all(c in "0123456789abcdef" for c in hex_part))

    def test_unique(self):
        markers = {generate_activation_marker() for _ in range(50)}
        # Should produce unique markers (extremely unlikely to collide with 6 hex chars)
        self.assertGreater(len(markers), 40)


class TestBuildToolDirective(unittest.TestCase):
    def test_contains_marker(self):
        marker = "[[CALL-test12]]"
        tools = [{"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {}}}]
        directive = build_tool_directive(tools, marker)
        self.assertIn(marker, directive)
        self.assertIn("<<<TOOLS>>>", directive)
        self.assertIn("<<<END_TOOLS>>>", directive)
        self.assertIn("read_file", directive)

    def test_required_choice(self):
        marker = "[[CALL-test12]]"
        tools = [{"type": "function", "function": {"name": "run", "description": "", "parameters": {}}}]
        directive = build_tool_directive(tools, marker, tool_choice="required")
        self.assertIn("MUST", directive)

    def test_no_parallel(self):
        marker = "[[CALL-test12]]"
        tools = [{"type": "function", "function": {"name": "run", "description": "", "parameters": {}}}]
        directive = build_tool_directive(tools, marker, parallel_tool_calls=False)
        self.assertIn("only one tool", directive)

    def test_intro(self):
        marker = "[[CALL-test12]]"
        tools = [{"type": "function", "function": {"name": "run", "description": "", "parameters": {}}}]
        directive = build_tool_directive(tools, marker, intro="Custom intro")
        self.assertIn("Custom intro", directive)


class TestParseToolInvocation(unittest.TestCase):
    def test_single_tool(self):
        marker = "[[CALL-test12]]"
        text = f"Some text {marker}\n<<<TOOLS>>>\n[{{\"name\":\"read\",\"params\":{{\"path\":\"/etc\"}}}}]\n<<<END_TOOLS>>>"
        outcome = parse_tool_invocation(text, marker)
        self.assertEqual(len(outcome.invocations), 1)
        self.assertEqual(outcome.invocations[0].name, "read")
        self.assertEqual(outcome.invocations[0].parameters, {"path": "/etc"})
        self.assertEqual(outcome.text_before_marker, "Some text")

    def test_multiple_tools(self):
        marker = "[[CALL-test12]]"
        text = f"{marker}\n<<<TOOLS>>>\n[{{\"name\":\"a\",\"params\":{{}}}},{{\"name\":\"b\",\"params\":{{}}}}]\n<<<END_TOOLS>>>"
        outcome = parse_tool_invocation(text, marker)
        self.assertEqual(len(outcome.invocations), 2)

    def test_no_marker(self):
        text = "Just a normal response without any tools."
        outcome = parse_tool_invocation(text, "[[CALL-test12]]")
        self.assertEqual(len(outcome.invocations), 0)
        self.assertIsNone(outcome.text_before_marker)

    def test_incomplete_block(self):
        marker = "[[CALL-test12]]"
        text = f"{marker}\n<<<TOOLS>>>\n[{{\"name\":\"a\""
        from toolbridge.errors import ParseError
        with self.assertRaises(ParseError):
            parse_tool_invocation(text, marker)

    def test_think_blocks_stripped(self):
        marker = "[[CALL-test12]]"
        text = f" 科学院 some thinking 科学院 {marker}\n<<<TOOLS>>>\n[{{\"name\":\"a\",\"params\":{{}}}}]\n<<<END_TOOLS>>>"
        outcome = parse_tool_invocation(text, marker)
        self.assertEqual(len(outcome.invocations), 1)

    def test_thinking_tags_stripped(self):
        marker = "[[CALL-test12]]"
        text = f"<thinking>internal</thinking>{marker}\n<<<TOOLS>>>\n[{{\"name\":\"a\",\"params\":{{}}}}]\n<<<END_TOOLS>>>"
        outcome = parse_tool_invocation(text, marker)
        self.assertEqual(len(outcome.invocations), 1)


class TestClassifyFailure(unittest.TestCase):
    def test_no_marker(self):
        kind = classify_failure("plain text", "[[CALL-abc123]]")
        self.assertEqual(kind, FailureKind.NO_MARKER)

    def test_incomplete_block(self):
        marker = "[[CALL-abc123]]"
        kind = classify_failure(f"{marker}\n<<<TOOLS>>>\npartial", marker)
        self.assertEqual(kind, FailureKind.INCOMPLETE_BLOCK)

    def test_malformed_payload(self):
        marker = "[[CALL-abc123]]"
        kind = classify_failure(f"{marker}\n<<<TOOLS>>>\nbad json\n<<<END_TOOLS>>>", marker)
        self.assertEqual(kind, FailureKind.MALFORMED_PAYLOAD)


class TestValidateToolParams(unittest.TestCase):
    def test_unknown_tool(self):
        errors = validate_tool_params(
            [ToolInvocation(name="nonexistent", parameters={})],
            [{"type": "function", "function": {"name": "real", "parameters": {}}}],
        )
        self.assertTrue(any("Unknown" in e for e in errors))

    def test_missing_required(self):
        errors = validate_tool_params(
            [ToolInvocation(name="read", parameters={})],
            [{"type": "function", "function": {"name": "read", "parameters": {"required": ["path"]}}}],
        )
        self.assertTrue(any("Missing" in e for e in errors))

    def test_valid(self):
        errors = validate_tool_params(
            [ToolInvocation(name="read", parameters={"path": "/tmp"})],
            [{"type": "function", "function": {"name": "read", "parameters": {"required": ["path"]}}}],
        )
        self.assertEqual(len(errors), 0)


class TestFallbackExtract(unittest.TestCase):
    def test_basic_name_params(self):
        text = '{"name": "read", "params": {"path": "/etc"}}'
        invocations = _fallback_extract(text)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].name, "read")
        self.assertEqual(invocations[0].parameters, {"path": "/etc"})

    def test_parameters_key(self):
        text = '{"name": "run", "parameters": {"cmd": "ls"}}'
        invocations = _fallback_extract(text)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].name, "run")
        self.assertEqual(invocations[0].parameters, {"cmd": "ls"})

    def test_arguments_key(self):
        text = '{"name": "exec", "arguments": {"shell": true}}'
        invocations = _fallback_extract(text)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].name, "exec")
        self.assertEqual(invocations[0].parameters, {"shell": True})

    def test_no_params(self):
        text = '{"name": "list"}'
        invocations = _fallback_extract(text)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].parameters, {})

    def test_malformed_json_multiple_names(self):
        text = '"name": "a", ... "name": "b"'
        invocations = _fallback_extract(text)
        self.assertEqual(len(invocations), 2)

    def test_malformed_params_json(self):
        text = '"name": "read", "params": {broken}'
        invocations = _fallback_extract(text)
        self.assertEqual(len(invocations), 1)
        self.assertEqual(invocations[0].name, "read")
        self.assertEqual(invocations[0].parameters, {})


class TestParseWithFallback(unittest.TestCase):
    def test_malformed_json_triggers_fallback(self):
        marker = "[[CALL-test12]]"
        # Malformed JSON: missing closing bracket, but name/params are extractable
        text = f'{marker}\n<<<TOOLS>>>\n{{"name":"read","params":{{"path":"/etc"}}\n<<<END_TOOLS>>>'
        outcome = parse_tool_invocation(text, marker)
        self.assertEqual(len(outcome.invocations), 1)
        self.assertEqual(outcome.invocations[0].name, "read")

    def test_valid_json_no_fallback(self):
        marker = "[[CALL-test12]]"
        text = f'{marker}\n<<<TOOLS>>>\n[{{"name":"read","params":{{"path":"/etc"}}}}]\n<<<END_TOOLS>>>'
        outcome = parse_tool_invocation(text, marker)
        self.assertEqual(len(outcome.invocations), 1)
        self.assertEqual(outcome.invocations[0].parameters, {"path": "/etc"})


if __name__ == "__main__":
    unittest.main()
