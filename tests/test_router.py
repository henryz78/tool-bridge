"""Tests for toolbridge.router request behavior."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from toolbridge.config import Settings
from toolbridge.errors import UpstreamError
from toolbridge.router import dispatch, handle_chat, _stream_anthropic_response


class FakeHandler:
    def __init__(self, payload: dict):
        raw = json.dumps(payload).encode("utf-8")
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        }
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


def parse_sse_events(raw: bytes) -> list[dict]:
    events: list[dict] = []
    for block in raw.decode("utf-8").strip().split("\n\n"):
        if not block:
            continue
        event_type = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_lines.append(line[6:])
        data = "\n".join(data_lines)
        try:
            parsed_data = json.loads(data) if data else None
        except json.JSONDecodeError:
            parsed_data = data
        events.append({
            "event": event_type,
            "data": parsed_data,
        })
    return events


class FakeUrlopenResponse:
    status = 200
    headers = {"content-type": "application/json"}

    def __enter__(self) -> "FakeUrlopenResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
        }).encode("utf-8")


class FakeStreamingResponse:
    status = 200

    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class FakeConnection:
    def __init__(self, response: FakeStreamingResponse):
        self.response = response
        self.requests: list[dict] = []
        self.closed = False

    def request(self, method: str, path: str, body: bytes, headers: dict) -> None:
        self.requests.append({
            "method": method,
            "path": path,
            "body": json.loads(body.decode("utf-8")),
            "headers": headers,
        })

    def getresponse(self) -> FakeStreamingResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


class TestHandleChatRouting(unittest.TestCase):
    def test_mapped_model_uses_route_from_requested_model(self) -> None:
        settings = Settings(
            upstream_url="http://default.example/v1",
            upstream_auth="default-token",
            name_mapping={"public-chat": "provider-chat"},
            upstreams=[{
                "id": "provider-a",
                "url": "http://provider.example/api",
                "auth": "provider-token",
                "timeout": 7,
            }],
            model_routes={"public-chat": "provider-a"},
        )
        handler = FakeHandler({
            "model": "public-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        captured: dict = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeUrlopenResponse()

        with mock.patch("toolbridge.proxy.urllib.request.urlopen", side_effect=fake_urlopen):
            handle_chat(handler, settings)

        self.assertEqual(handler.response_status, 200)
        self.assertEqual(captured["url"], "http://provider.example/api/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer provider-token")
        self.assertEqual(captured["timeout"], 7)
        self.assertEqual(captured["body"]["model"], "provider-chat")
        self.assertEqual(handler.json_response()["choices"][0]["message"]["content"], "ok")

    def test_streaming_virtual_tool_call_emits_openai_tool_deltas(self) -> None:
        marker = "[[CALL-fixed]]"
        tool_payload = (
            f"Before lookup {marker}\n"
            "<<<TOOLS>>>\n"
            '[{"name":"lookup","params":{"query":"mars"}}]\n'
            "<<<END_TOOLS>>>"
        )
        upstream_chunk = json.dumps({
            "choices": [{"delta": {"content": tool_payload}, "finish_reason": None}]
        })
        response = FakeStreamingResponse([
            f"data: {upstream_chunk}\n\n".encode("utf-8"),
            b"data: [DONE]\n\n",
        ])
        connection = FakeConnection(response)
        settings = Settings(name_mapping={"public-chat": "provider-chat"})
        handler = FakeHandler({
            "model": "public-chat",
            "messages": [{"role": "user", "content": "lookup mars"}],
            "stream": True,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {"type": "object", "required": ["query"]},
                },
            }],
        })

        with mock.patch("toolbridge.router.generate_activation_marker", return_value=marker), \
             mock.patch("toolbridge.proxy.open_upstream_connection", return_value=connection):
            handle_chat(handler, settings)

        events = parse_sse_events(handler.wfile.getvalue())
        deltas = [
            choice["delta"]
            for event in events
            if isinstance(event["data"], dict)
            for choice in event["data"].get("choices", [])
        ]
        tool_deltas = [delta for delta in deltas if "tool_calls" in delta]
        self.assertEqual(handler.response_status, 200)
        self.assertEqual(connection.requests[0]["body"]["model"], "provider-chat")
        self.assertNotIn("tools", connection.requests[0]["body"])
        self.assertIn(marker, connection.requests[0]["body"]["messages"][0]["content"])
        self.assertEqual(tool_deltas[0]["tool_calls"][0]["function"]["name"], "lookup")
        self.assertEqual(
            json.loads(tool_deltas[1]["tool_calls"][0]["function"]["arguments"]),
            {"query": "mars"},
        )
        self.assertEqual(events[-1]["data"], "[DONE]")


class TestAnthropicStreamingResponse(unittest.TestCase):
    def test_content_block_start_uses_empty_streaming_payloads(self) -> None:
        handler = FakeHandler({})
        response = {
            "id": "msg_test",
            "usage": {"input_tokens": 3, "output_tokens": 5},
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Need a lookup."},
                {"type": "tool_use", "id": "toolu_test", "name": "lookup", "input": {"query": "mars"}},
            ],
        }

        _stream_anthropic_response(handler, response, "public-chat")

        starts = [
            event["data"]["content_block"]
            for event in parse_sse_events(handler.wfile.getvalue())
            if event["event"] == "content_block_start"
        ]
        self.assertEqual(starts[0], {"type": "text", "text": ""})
        self.assertEqual(starts[1], {"type": "tool_use", "id": "toolu_test", "name": "lookup", "input": {}})


class TestDispatchErrorHandling(unittest.TestCase):
    def test_upstream_json_error_preserves_status_and_body(self) -> None:
        handler = FakeHandler({
            "model": "public-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })
        settings = Settings(name_mapping={"public-chat": "provider-chat"})
        error_body = json.dumps({
            "error": {"type": "rate_limit_error", "message": "too many requests"}
        }).encode("utf-8")

        with mock.patch("toolbridge.router.fetch_upstream_chat", side_effect=UpstreamError(429, error_body)):
            dispatch(handler, settings, "POST", "/v1/chat/completions", None)

        self.assertEqual(handler.response_status, 429)
        self.assertEqual(handler.json_response(), {
            "error": {"type": "rate_limit_error", "message": "too many requests"}
        })


if __name__ == "__main__":
    unittest.main()
