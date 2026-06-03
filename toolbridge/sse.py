"""SSE reading, writing, and streaming trigger-signal detection."""

from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass
from typing import Any, Generator


# ---------------------------------------------------------------------------
# Trigger scanner (streaming detection state machine)
# ---------------------------------------------------------------------------

class _State(enum.Enum):
    SCANNING = "scanning"
    IN_THINK_BLOCK = "in_think_block"
    MARKER_FOUND = "marker_found"


@dataclass
class ScanResult:
    """Result from a single feed of the trigger scanner."""
    activated: bool
    prefix_text: str


_THINK_OPEN_RE = re.compile(r"<(thinking|think)>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</(thinking|think)>", re.IGNORECASE)


class TriggerScanner:
    """State-machine scanner that detects the activation marker in a
    streaming text flow while ignoring content inside think blocks.

    Processes the full text available after each feed, handling think-block
    nesting and re-scanning after think blocks close.
    """

    def __init__(self, marker: str):
        self._marker = marker
        self._state = _State.SCANNING
        self._buffer = ""
        self._text_before_marker = ""

    def feed(self, text: str) -> ScanResult:
        """Process an incoming text chunk.  Returns a ScanResult indicating
        whether the marker was found and any text before it."""
        self._buffer += text
        return self._process()

    @property
    def found(self) -> bool:
        return self._state == _State.MARKER_FOUND

    @property
    def accumulated(self) -> str:
        """All text buffered after the marker was found."""
        return self._buffer

    @property
    def pending_prefix(self) -> str:
        """Alias for text_before_marker for backward compat."""
        return self._text_before_marker

    def _process(self) -> ScanResult:
        """Core processing loop — runs until a stable state is reached."""
        while True:
            if self._state == _State.MARKER_FOUND:
                return ScanResult(activated=True, prefix_text="")

            if self._state == _State.SCANNING:
                result = self._search_buffer()
                if self._state == _State.MARKER_FOUND:
                    return result
                if self._state == _State.SCANNING:
                    return result
                # State changed to IN_THINK_BLOCK — loop to handle it
                continue

            if self._state == _State.IN_THINK_BLOCK:
                self._skip_think_block()
                # After skipping, state may be SCANNING or still IN_THINK_BLOCK
                continue

            return ScanResult(activated=False, prefix_text="")

    def _search_buffer(self) -> ScanResult:
        """Scan the buffer for the marker, respecting think-block boundaries."""
        buf = self._text_before_marker + self._buffer
        self._text_before_marker = ""
        self._buffer = buf

        pos = 0
        while pos < len(buf):
            # Check for think block opening at current position
            think_match = _THINK_OPEN_RE.search(buf, pos)
            # Check for marker at current position
            marker_pos = buf.find(self._marker, pos)

            # If no marker found at all
            if marker_pos < 0:
                # Check if there's a think block ahead
                if think_match is not None:
                    self._text_before_marker = buf[:think_match.start()]
                    # Buffer starts AFTER the <thinking> tag so _skip_think_block
                    # doesn't need to re-parse it
                    self._buffer = buf[think_match.end():]
                    self._state = _State.IN_THINK_BLOCK
                    return ScanResult(activated=False, prefix_text="")

                # Nothing found — keep buffer for partial matching
                keep_len = len(self._marker) + len("<thinking>") + 5
                if len(buf) <= keep_len:
                    self._text_before_marker = buf
                    self._buffer = ""
                    return ScanResult(activated=False, prefix_text="")

                safe_text = buf[:-keep_len]
                self._buffer = buf[-keep_len:]
                return ScanResult(activated=False, prefix_text=safe_text)

            # Marker found — but check if a think block opens before it
            if think_match is not None and think_match.start() < marker_pos:
                # Enter think block first
                self._text_before_marker = buf[:think_match.start()]
                # Buffer starts AFTER the <thinking> tag
                self._buffer = buf[think_match.end():]
                self._state = _State.IN_THINK_BLOCK
                return ScanResult(activated=False, prefix_text="")

            # Marker found outside any think block
            self._state = _State.MARKER_FOUND
            self._text_before_marker = buf[:marker_pos].strip()
            self._buffer = buf[marker_pos + len(self._marker):]
            return ScanResult(activated=True, prefix_text=self._text_before_marker)

        # Empty buffer
        return ScanResult(activated=False, prefix_text="")

    def _skip_think_block(self) -> None:
        """Skip past content inside a think block until the closing tag.

        The buffer starts right after the ``<thinking>`` tag that was already
        consumed. We search for ``</thinking>`` while tracking any nested
        ``<thinking>`` opens.
        """
        buf = self._buffer
        depth = 0  # nested depth relative to the already-consumed opener
        pos = 0

        while pos < len(buf):
            close_match = _THINK_CLOSE_RE.search(buf, pos)
            if close_match is None:
                # No close tag — keep tail for partial matching
                keep_len = len("</thinking>") + 5
                if len(buf) > keep_len:
                    self._buffer = buf[-keep_len:]
                return

            # Check if a nested open appears before the close
            open_match = _THINK_OPEN_RE.search(buf, pos)
            if open_match is not None and open_match.start() < close_match.start():
                depth += 1
                pos = open_match.end()
                continue

            # Found a close tag
            if depth > 0:
                depth -= 1
                pos = close_match.end()
                continue

            # depth == 0: this close matches our original opener
            remaining = buf[close_match.end():]
            self._buffer = remaining
            self._state = _State.SCANNING
            return

        # Ran out of buffer inside think block
        keep_len = len("</thinking>") + 5
        if len(self._buffer) > keep_len:
            self._buffer = self._buffer[-keep_len:]


# ---------------------------------------------------------------------------
# SSE reader (parse upstream chunks)
# ---------------------------------------------------------------------------

def read_sse_chunks(response: Any) -> Generator[dict, None, None]:
    """Read an upstream HTTP response and yield parsed SSE data objects.

    Each yielded dict has at least a ``data`` key.  ``event`` may also
    be present for named events.
    """
    buf = ""
    for raw_line in response:
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8", errors="replace")
        buf += raw_line

        while "\n\n" in buf:
            event_block, buf = buf.split("\n\n", 1)
            event_type = "message"
            data_lines: list[str] = []
            for line in event_block.split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:])
                # ignore comments and other fields
            data_str = "\n".join(data_lines)
            if data_str:
                yield {"event": event_type, "data": data_str}

    if buf.strip():
        event_type = "message"
        data_lines: list[str] = []
        for line in buf.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:])
        data_str = "\n".join(data_lines)
        if data_str:
            yield {"event": event_type, "data": data_str}


# ---------------------------------------------------------------------------
# SSE writer helpers
# ---------------------------------------------------------------------------

def begin_sse_response(handler: Any) -> None:
    """Send SSE headers to the downstream client."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def write_sse_event(handler: Any, data: str) -> None:
    """Write an unnamed SSE event to the client."""
    handler.wfile.write(f"data: {data}\n\n".encode("utf-8"))
    handler.wfile.flush()


def write_sse_named_event(handler: Any, event: str, data: str) -> None:
    """Write a named SSE event to the client."""
    handler.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
    handler.wfile.flush()


# ---------------------------------------------------------------------------
# OpenAI streaming emission
# ---------------------------------------------------------------------------

def emit_openai_text_delta(handler: Any, base: dict, text: str) -> None:
    """Emit an OpenAI streaming chunk with a text content delta."""
    chunk = {
        **base,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None,
        }],
    }
    write_sse_event(handler, json.dumps(chunk, ensure_ascii=False))


def emit_openai_tool_call_delta(handler: Any, base: dict, tool_calls: list[dict]) -> None:
    """Emit OpenAI streaming chunks for tool calls."""
    for i, call in enumerate(tool_calls):
        # First chunk: name + id
        chunk = {
            **base,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "id": call.get("id", ""),
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }],
        }
        write_sse_event(handler, json.dumps(chunk, ensure_ascii=False))

        # Argument chunks
        args = call["function"].get("arguments", "{}")
        chunk2 = {
            **base,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "function": {"arguments": args},
                    }]
                },
                "finish_reason": None,
            }],
        }
        write_sse_event(handler, json.dumps(chunk2, ensure_ascii=False))

    # Final chunk with finish_reason
    final = {
        **base,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    }
    write_sse_event(handler, json.dumps(final, ensure_ascii=False))


def emit_openai_done(handler: Any) -> None:
    """Emit the [DONE] sentinel for OpenAI streaming."""
    write_sse_event(handler, "[DONE]")


# ---------------------------------------------------------------------------
# Anthropic streaming emission
# ---------------------------------------------------------------------------

def emit_anthropic_message_start(handler: Any, msg_id: str, model: str, usage: dict) -> None:
    payload = json.dumps({
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": model, "stop_reason": None, "stop_sequence": None,
            "usage": usage,
        },
    }, ensure_ascii=False)
    write_sse_named_event(handler, "message_start", payload)


def emit_anthropic_content_block_start(handler: Any, index: int, block: dict) -> None:
    payload = json.dumps({
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    }, ensure_ascii=False)
    write_sse_named_event(handler, "content_block_start", payload)


def emit_anthropic_content_block_delta(handler: Any, index: int, delta: dict) -> None:
    payload = json.dumps({
        "type": "content_block_delta",
        "index": index,
        "delta": delta,
    }, ensure_ascii=False)
    write_sse_named_event(handler, "content_block_delta", payload)


def emit_anthropic_content_block_stop(handler: Any, index: int) -> None:
    payload = json.dumps({"type": "content_block_stop", "index": index})
    write_sse_named_event(handler, "content_block_stop", payload)


def emit_anthropic_message_delta(handler: Any, stop_reason: str, usage: dict) -> None:
    payload = json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage,
    }, ensure_ascii=False)
    write_sse_named_event(handler, "message_delta", payload)


def emit_anthropic_message_stop(handler: Any) -> None:
    write_sse_named_event(handler, "message_stop", "{}")
