"""Virtual tool calling engine — prompt injection, output parsing, retry."""

from __future__ import annotations

import enum
import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from .errors import ParseError


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolInvocation:
    """A single parsed tool invocation."""
    name: str
    parameters: dict


@dataclass
class ParseOutcome:
    """Result of parsing a virtual tool call output."""
    text_before_marker: str | None
    invocations: list[ToolInvocation] = field(default_factory=list)


class FailureKind(enum.Enum):
    """Classification of a tool-call parse failure."""
    NO_MARKER = "no_marker"
    INCOMPLETE_BLOCK = "incomplete_block"
    MALFORMED_PAYLOAD = "malformed_payload"


# ---------------------------------------------------------------------------
# Activation marker
# ---------------------------------------------------------------------------

def generate_activation_marker() -> str:
    """Produce a per-request activation marker like ``[[CALL-a1b2c3]]``."""
    suffix = "".join(secrets.choice("0123456789abcdef") for _ in range(6))
    return f"[[CALL-{suffix}]]"


# ---------------------------------------------------------------------------
# Tool directive (system prompt injection)
# ---------------------------------------------------------------------------

_TOOL_DIRECTIVE_TEMPLATE = """\
This conversation has access to external tools. When you need to invoke a tool, \
follow this protocol:

1. Output the activation marker exactly: {marker}
2. Immediately after, output the tool invocations in this format:
<<<TOOLS>>>
[{{"name":"tool_name","params":{{"key":"value"}}}}]
<<<END_TOOLS>>>

Tool definitions:
{tool_schemas}

Guidelines:
- Only output the activation marker when you intend to call tools.
- The "params" value must be a valid JSON object matching the tool's input schema.
- Use only tool names from the definitions above.
- To call multiple tools, add more objects to the JSON array.
- If you can answer without tools, respond normally — do not emit the marker.
{extra}"""


def build_tool_directive(
    tools: list[dict],
    marker: str,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = None,
    intro: str = "",
) -> str:
    """Build the system-prompt directive that teaches the model how to call tools."""
    schemas = json.dumps(
        [
            {"name": t["function"]["name"], "description": t["function"].get("description", ""), "parameters": t["function"].get("parameters", {})}
            for t in tools
        ],
        indent=2,
        ensure_ascii=False,
    )

    extra_parts: list[str] = []
    if tool_choice == "required" or tool_choice == "any":
        extra_parts.append(
            "- You MUST call at least one tool in this response — do not reply without a tool invocation."
        )
    elif isinstance(tool_choice, dict):
        fname = tool_choice.get("function", {}).get("name", "")
        if fname:
            extra_parts.append(f'- You MUST call the tool "{fname}" — no other tool is acceptable.')
    if parallel_tool_calls is False:
        extra_parts.append("- Call only one tool per response — do not invoke multiple tools at once.")

    extra = ("\n" + "\n".join(extra_parts)) if extra_parts else ""
    preamble = f"{intro}\n\n" if intro else ""

    return preamble + _TOOL_DIRECTIVE_TEMPLATE.format(marker=marker, tool_schemas=schemas, extra=extra)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_BEGIN = "<<<TOOLS>>>"
_END = "<<<END_TOOLS>>>"

# Fallback regex: try to extract name/params pairs even when JSON is broken
_FALLBACK_RE = re.compile(
    r'"name"\s*:\s*"([^"]+)"'
    r'(?:.*?"params"\s*:\s*)?'
    r'(?:\{[^}]*\})?',
    re.DOTALL,
)


def parse_tool_invocation(text: str, marker: str) -> ParseOutcome:
    """Parse the model output looking for the marker and the tool block."""
    cleaned = _strip_think_blocks(text)

    marker_pos = cleaned.find(marker)
    if marker_pos < 0:
        return ParseOutcome(text_before_marker=None, invocations=[])

    pre = cleaned[:marker_pos].strip()
    after_marker = cleaned[marker_pos + len(marker):]

    begin_pos = after_marker.find(_BEGIN)
    if begin_pos < 0:
        return ParseOutcome(text_before_marker=pre, invocations=[])

    end_pos = after_marker.find(_END, begin_pos + len(_BEGIN))
    if end_pos < 0:
        raise ParseError("incomplete tool block")

    json_text = after_marker[begin_pos + len(_BEGIN):end_pos].strip()

    # Primary: JSON parse
    try:
        items = json.loads(json_text)
        if not isinstance(items, list):
            items = [items]
        invocations = _extract_invocations(items)
        if invocations:
            return ParseOutcome(text_before_marker=pre or None, invocations=invocations)
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction from malformed JSON
    invocations = _fallback_extract(json_text)
    if invocations:
        return ParseOutcome(text_before_marker=pre or None, invocations=invocations)

    raise ParseError("invalid JSON in tool block")


def _extract_invocations(items: list) -> list[ToolInvocation]:
    """Extract ToolInvocation objects from parsed JSON items."""
    result: list[ToolInvocation] = []
    for item in items:
        if isinstance(item, dict) and "name" in item:
            params = item.get("params", item.get("parameters", item.get("arguments", {})))
            if not isinstance(params, dict):
                params = {}
            result.append(ToolInvocation(name=item["name"], parameters=params))
    return result


def _fallback_extract(text: str) -> list[ToolInvocation]:
    """Attempt regex-based extraction when JSON parsing fails."""
    invocations: list[ToolInvocation] = []
    # Try to find {"name": "...", ...} patterns
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
    for m in name_pattern.finditer(text):
        name = m.group(1)
        # Try to find a params/parameters object following the name
        after = text[m.end():]
        params = {}
        params_match = re.search(
            r'"(?:params|parameters|arguments)"\s*:\s*(\{[^}]*\})',
            after[:500],
        )
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}
        invocations.append(ToolInvocation(name=name, parameters=params))
    return invocations


def classify_failure(text: str, marker: str) -> FailureKind:
    """Classify why tool invocation parsing failed."""
    cleaned = _strip_think_blocks(text)
    if marker not in cleaned:
        return FailureKind.NO_MARKER

    marker_pos = cleaned.find(marker)
    after_marker = cleaned[marker_pos + len(marker):]

    if _BEGIN in after_marker and _END not in after_marker[after_marker.find(_BEGIN):]:
        return FailureKind.INCOMPLETE_BLOCK

    return FailureKind.MALFORMED_PAYLOAD


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

_CONTINUATION_PROMPT = (
    "Your previous response was interrupted. Here is what you produced so far:\n"
    "{partial}\n\n"
    "Please continue and complete the tool invocation block."
)

_CORRECTION_PROMPT = (
    "The tool invocation in your previous response could not be parsed:\n"
    "{diagnostic}\n\n"
    "Your prior output:\n{prior}\n\n"
    "Please try again using the exact format specified above."
)


def retry_parse(
    prior_text: str,
    marker: str,
    tools: list[dict],
    messages: list[dict],
    resolved_model: str,
    settings: Any,
) -> ParseOutcome | None:
    """Attempt to retry parsing up to *max_retry_attempts* times."""
    from .proxy import fetch_upstream_chat

    if not settings.retry_on_parse_failure:
        return None

    max_attempts = settings.max_retry_attempts
    for attempt in range(max_attempts):
        kind = classify_failure(prior_text, marker)
        if kind == FailureKind.NO_MARKER:
            return None

        if kind == FailureKind.INCOMPLETE_BLOCK:
            prompt = _CONTINUATION_PROMPT.format(partial=prior_text)
            messages = messages + [
                {"role": "assistant", "content": prior_text},
                {"role": "user", "content": prompt},
            ]
        else:
            diagnostic = "The JSON between <<<TOOLS>>> and <<<END_TOOLS>>> could not be parsed."
            prompt = _CORRECTION_PROMPT.format(diagnostic=diagnostic, prior=prior_text)
            messages = messages + [
                {"role": "assistant", "content": prior_text},
                {"role": "user", "content": prompt},
            ]

        payload = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
        }
        resp = fetch_upstream_chat(payload, settings)
        prior_text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")

        try:
            outcome = parse_tool_invocation(prior_text, marker)
            if outcome.invocations:
                return outcome
        except ParseError:
            continue

    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tool_params(invocations: list[ToolInvocation], tool_schemas: list[dict]) -> list[str]:
    """Validate parsed tool invocations against schemas. Returns list of error strings."""
    errors: list[str] = []
    schema_by_name: dict[str, dict] = {}
    for t in tool_schemas:
        fn = t.get("function", {})
        schema_by_name[fn.get("name", "")] = fn.get("parameters", {})

    for inv in invocations:
        if inv.name not in schema_by_name:
            errors.append(f"Unknown tool: {inv.name}")
            continue
        schema = schema_by_name[inv.name]
        required = schema.get("required", [])
        for param in required:
            if param not in inv.parameters:
                errors.append(f"Missing required parameter '{param}' in tool {inv.name}")
    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"科学研究院.*?科院", re.DOTALL | re.IGNORECASE)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """Remove 科学研究院...科院 and <thinking>...</thinking> blocks."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINKING_TAG_RE.sub("", text)
    return text
