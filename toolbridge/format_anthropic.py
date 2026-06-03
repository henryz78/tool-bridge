"""Anthropic Messages API ↔ OpenAI Chat Completions conversion."""

from __future__ import annotations

import hashlib
import base64
import json
import time
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Anthropic → OpenAI (request)
# ---------------------------------------------------------------------------

def convert_anthropic_to_openai(payload: dict) -> dict:
    """Convert an Anthropic Messages API request body to an OpenAI Chat
    Completions request body."""
    messages_raw = payload.get("messages", [])
    system = payload.get("system", "")
    tools = payload.get("tools", [])
    tool_choice = payload.get("tool_choice", None)
    model = payload.get("model", "")
    max_tokens = payload.get("max_tokens", 4096)
    stream = payload.get("stream", False)

    messages = _flatten_anthropic_messages(messages_raw, system)
    openai_tools = _convert_anthropic_tools(tools) if tools else []
    openai_choice = _convert_anthropic_tool_choice(tool_choice, openai_tools) if openai_tools else None

    result: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if openai_tools:
        result["tools"] = openai_tools
    if openai_choice is not None:
        result["tool_choice"] = openai_choice
    return result


def _flatten_anthropic_messages(messages: list[dict], system: Any) -> list[dict]:
    """Turn Anthropic multi-block messages into flat OpenAI role/content pairs."""
    result: list[dict] = []

    if system:
        sys_text = _serialize_blocks(system) if isinstance(system, list) else str(system)
        result.append({"role": "system", "content": sys_text})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            text = _serialize_blocks(content) if isinstance(content, list) else str(content)
            result.append({"role": "user", "content": text})

        elif role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for block in (content if isinstance(content, list) else [{"type": "text", "text": str(content)}]):
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", str(uuid.uuid4())),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    })
                elif btype in ("thinking", "redacted_thinking"):
                    pass  # dropped
            assistant: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            result.append(assistant)

    return result


def _serialize_blocks(content: Any) -> str:
    """Serialize Anthropic content blocks to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_result":
                tool_id = block.get("tool_use_id", "")
                inner = block.get("content", "")
                inner_text = _serialize_blocks(inner) if isinstance(inner, list) else str(inner)
                parts.append(f"[Tool result for {tool_id}]\n{inner_text}")
            elif btype == "image":
                parts.append("[image]")
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def _convert_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function tool format."""
    result: list[dict] = []
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})
        result.append({
            "type": "function",
            "function": {"name": name, "description": desc, "parameters": schema},
        })
    return result


def _convert_anthropic_tool_choice(choice: Any, openai_tools: list[dict]) -> Any:
    """Map Anthropic tool_choice to OpenAI tool_choice."""
    if choice is None or choice == "auto":
        return "auto"
    if choice == "any":
        return "required"
    if choice == "none":
        return "none"
    if isinstance(choice, dict):
        name = choice.get("name", "")
        return {"type": "function", "function": {"name": name}}
    return "auto"


# ---------------------------------------------------------------------------
# OpenAI → Anthropic (response)
# ---------------------------------------------------------------------------

def convert_openai_to_anthropic(
    openai_resp: dict,
    requested_model: str,
    had_tools: bool,
    marker: str = "",
) -> dict:
    """Convert an OpenAI Chat Completions response to an Anthropic Messages
    response."""
    choices = openai_resp.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")
    usage = openai_resp.get("usage", {})

    content_blocks: list[dict] = []

    # Thinking block
    reasoning = message.get("reasoning_content", "")
    if reasoning:
        sig = _thinking_signature(reasoning)
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": sig,
        })

    # Text block
    text = message.get("content", "")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool-use blocks
    for call in (message.get("tool_calls") or []):
        fn = call.get("function", {})
        content_blocks.append({
            "type": "tool_use",
            "id": call.get("id", str(uuid.uuid4())),
            "name": fn.get("name", ""),
            "input": _parse_json_args(fn.get("arguments", "{}")),
        })

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    stop = anthropic_stop_reason(finish, content_blocks)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": requested_model,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": anthropic_usage(usage),
    }


def anthropic_stop_reason(finish: str, content_blocks: list[dict]) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    has_tool_use = any(b.get("type") == "tool_use" for b in content_blocks)
    if finish == "tool_calls" or has_tool_use:
        return "tool_use"
    if finish == "length":
        return "max_tokens"
    return "end_turn"


def anthropic_usage(openai_usage: dict) -> dict:
    """Map OpenAI usage to Anthropic usage."""
    return {
        "input_tokens": openai_usage.get("prompt_tokens", 0),
        "output_tokens": openai_usage.get("completion_tokens", 0),
    }


def _parse_json_args(raw: str) -> dict:
    """Safely parse tool call arguments JSON."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _thinking_signature(text: str) -> str:
    """Create a base64-encoded SHA-256 signature for a thinking block."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")
