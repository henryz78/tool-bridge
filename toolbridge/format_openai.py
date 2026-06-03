"""OpenAI Chat Completions message normalization."""

from __future__ import annotations

import json
from typing import Any


def serialize_content(content: Any) -> str:
    """Convert any content shape to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                typ = item.get("type", "")
                if typ == "text":
                    parts.append(item.get("text", ""))
                elif typ in ("image_url", "image"):
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_tool_calls(raw_calls: list[dict] | None) -> list[dict]:
    """Normalize tool_call objects to a consistent shape.

    Supports both ``{"function": {"name": ..., "arguments": ...}}`` and
    ``{"name": ..., "arguments": ...}`` variants.
    """
    if not raw_calls:
        return []
    result: list[dict] = []
    for raw in raw_calls:
        if "function" in raw:
            fn = raw["function"]
            result.append({
                "id": raw.get("id", ""),
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}") if isinstance(fn.get("arguments"), str) else json.dumps(fn["arguments"], ensure_ascii=False),
                },
            })
        else:
            result.append({
                "id": raw.get("id", ""),
                "type": "function",
                "function": {
                    "name": raw.get("name", ""),
                    "arguments": raw.get("arguments", "{}") if isinstance(raw.get("arguments"), str) else json.dumps(raw.get("arguments", {}), ensure_ascii=False),
                },
            })
    return result


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Rewrite messages so that tool-related roles are compatible with upstream.

    - ``tool`` role messages become ``user`` role with serialized content.
    - Assistant messages with ``tool_calls`` get their calls serialized into
      the text content as well as preserved in the ``tool_calls`` field.
    """
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "tool":
            tool_id = msg.get("tool_call_id", "")
            text = serialize_content(content)
            result.append({
                "role": "user",
                "content": f"[Tool result for {tool_id}]\n{text}" if tool_id else text,
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            calls = normalize_tool_calls(msg["tool_calls"])
            call_text = _format_calls_as_text(calls)
            base_text = serialize_content(content)
            combined = f"{base_text}\n{call_text}" if base_text else call_text
            result.append({
                "role": "assistant",
                "content": combined,
                "tool_calls": calls,
            })
            continue

        result.append({
            "role": role,
            "content": serialize_content(content),
        })

    return result


def has_tool_calls_in_history(messages: list[dict]) -> bool:
    """Return True if any assistant message in *messages* carries tool_calls."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    return False


def _format_calls_as_text(calls: list[dict]) -> str:
    """Format tool_calls into a human-readable block for the upstream model."""
    lines: list[str] = []
    for call in calls:
        fn = call.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", "{}")
        lines.append(f"  Called {name}({args})")
    return "Tool calls:\n" + "\n".join(lines)
