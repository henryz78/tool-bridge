"""Route dispatch — maps request paths to handler functions."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .config import Settings
from .errors import BridgeError, UpstreamError, ParseError
from .format_openai import normalize_messages, normalize_tool_calls
from .format_anthropic import convert_anthropic_to_openai, convert_openai_to_anthropic, anthropic_usage
from .virtual_tools import (
    generate_activation_marker,
    build_tool_directive,
    parse_tool_invocation,
    classify_failure,
    FailureKind,
    ToolInvocation,
)
from .proxy import fetch_upstream, fetch_upstream_chat
from .sse import (
    TriggerScanner,
    read_sse_chunks,
    begin_sse_response,
    write_sse_event,
    emit_openai_text_delta,
    emit_openai_tool_call_delta,
    emit_openai_done,
    emit_anthropic_message_start,
    emit_anthropic_content_block_start,
    emit_anthropic_content_block_delta,
    emit_anthropic_content_block_stop,
    emit_anthropic_message_delta,
    emit_anthropic_message_stop,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def handle_health(handler: Any, settings: Settings) -> None:
    _send_json(handler, 200, {"ok": True})


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------

def handle_models(handler: Any, settings: Settings) -> None:
    models = settings.get_exposed_models()
    payload = {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "owned_by": "bridge", "permission": []}
            for mid in models
        ],
    }
    _send_json(handler, 200, payload)


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------

def handle_chat(handler: Any, settings: Settings) -> None:
    body = _read_json_body(handler)
    if body is None:
        _send_json(handler, 400, {"error": "invalid JSON body"})
        return

    if settings.upstream_extra_fields:
        body = {**settings.upstream_extra_fields, **body}

    requested_model = body.get("model", "")
    resolved = settings.resolve_model_name(requested_model)
    if resolved is None:
        _send_json(handler, 400, {"error": f"unknown model: {requested_model}"})
        return

    body["model"] = resolved
    stream = body.get("stream", False)
    tools = body.get("tools", [])

    # Native passthrough
    if not tools or settings.is_native_tool_model(requested_model):
        _passthrough_chat(handler, body, stream, settings)
        return

    # Virtual tool calling
    _virtual_tool_chat(handler, body, settings, stream)


def _passthrough_chat(handler: Any, body: dict, stream: bool, settings: Settings) -> None:
    if stream:
        from .proxy import open_upstream_connection, build_upstream_headers, resolve_request_path
        model = body.get("model", "")
        url, auth, timeout = settings.get_upstream_config(model)
        conn = open_upstream_connection(settings, url=url, timeout=timeout)
        try:
            path = resolve_request_path(url, "/v1/chat/completions")
            raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers = build_upstream_headers(None, settings, auth=auth)
            conn.request("POST", path, body=raw_body, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                err_body = resp.read()
                try:
                    err_json = json.loads(err_body.decode("utf-8", errors="replace"))
                except Exception:
                    err_json = {"error": err_body.decode("utf-8", errors="replace")}
                _send_json(handler, resp.status, err_json, cors=True)
                return
            begin_sse_response(handler)
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
        finally:
            conn.close()
    else:
        result = fetch_upstream_chat(body, settings)
        _send_json(handler, 200, result)


def _virtual_tool_chat(handler: Any, body: dict, settings: Settings, stream: bool) -> None:
    marker = generate_activation_marker()
    tools = body.pop("tools", [])
    tool_choice = body.pop("tool_choice", None)
    parallel = body.pop("parallel_tool_calls", None)

    # Inject tool directive as a system message
    directive = build_tool_directive(
        tools, marker, tool_choice=tool_choice,
        parallel_tool_calls=parallel, intro=settings.tool_instruction_intro,
    )
    messages = body.get("messages", [])
    _inject_system_message(messages, directive)
    body["messages"] = messages

    if stream:
        # Real streaming: send stream=True upstream, detect marker in real-time
        body["stream"] = True
        _stream_virtual_tool_call(handler, body, marker, tools, settings)
    else:
        body["stream"] = False
        _nonstream_virtual_tool_call(handler, body, marker, tools, settings)


def _nonstream_virtual_tool_call(
    handler: Any, body: dict, marker: str, tools: list[dict], settings: Settings,
) -> None:
    result = fetch_upstream_chat(body, settings)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    valid_names = [t["function"]["name"] for t in tools]
    try:
        outcome = parse_tool_invocation(content, marker, valid_names)
        if outcome and outcome.invocations:
            from .virtual_tools import validate_tool_params
            errors = validate_tool_params(outcome.invocations, tools)
            if errors:
                raise ParseError("; ".join(errors))
    except ParseError:
        outcome = None

    if outcome and outcome.invocations:
        response = _build_openai_tool_response(body["model"], outcome, settings)
        _send_json(handler, 200, response)
        return

    # Retry
    if settings.retry_on_parse_failure:
        outcome = _retry_virtual_parse(content, marker, tools, body.get("messages", []), body["model"], settings)
        if outcome and outcome.invocations:
            response = _build_openai_tool_response(body["model"], outcome, settings)
            _send_json(handler, 200, response)
            return

    # No tool calls found — passthrough the original response
    _send_json(handler, 200, result)


def _stream_virtual_tool_call(
    handler: Any, body: dict, marker: str, tools: list[dict], settings: Settings,
) -> None:
    from .proxy import open_upstream_connection, build_upstream_headers, resolve_request_path
    import urllib.parse

    model = body.get("model", "")
    url, auth, timeout = settings.get_upstream_config(model)

    conn = open_upstream_connection(settings, url=url, timeout=timeout)
    try:
        path = resolve_request_path(url, "/v1/chat/completions")

        raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = build_upstream_headers(None, settings, auth=auth)
        conn.request("POST", path, body=raw_body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read()
            try:
                err_json = json.loads(err_body.decode("utf-8", errors="replace"))
            except Exception:
                err_json = {"error": err_body.decode("utf-8", errors="replace")}
            _send_json(handler, resp.status, err_json, cors=True)
            return

        scanner = TriggerScanner(marker)
        all_content = ""
        marker_found = False
        post_marker_buf = ""  # content after marker, buffered for tool parsing
        base_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "model": body.get("model", ""),
        }

        begin_sse_response(handler)

        for sse_event in read_sse_chunks(resp):
            data = sse_event.get("data", "")
            if data == "[DONE]":
                break
            try:
                parsed_chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta_content = ""
            for choice in parsed_chunk.get("choices", []):
                delta = choice.get("delta", {})
                if "content" in delta:
                    delta_content += delta["content"]
                # Check for finish_reason from upstream
                finish = choice.get("finish_reason")

            if delta_content:
                all_content += delta_content

                if marker_found:
                    # After marker: buffer for tool call parsing, don't emit
                    post_marker_buf += delta_content
                else:
                    # Before marker: scan and emit real-time text deltas
                    scan_result = scanner.feed(delta_content)
                    if scan_result.prefix_text:
                        emit_openai_text_delta(handler, base_chunk, scan_result.prefix_text)
                    if scanner.found:
                        marker_found = True
                        # Content accumulated in scanner after marker
                        post_marker_buf = scanner.accumulated

        valid_names = [t["function"]["name"] for t in tools]
        # Stream complete — decide what to emit
        if marker_found:
            # Try to parse tool invocations from the full content
            try:
                outcome = parse_tool_invocation(all_content, marker, valid_names)
                if outcome and outcome.invocations:
                    from .virtual_tools import validate_tool_params
                    errors = validate_tool_params(outcome.invocations, tools)
                    if errors:
                        raise ParseError("; ".join(errors))
            except ParseError:
                outcome = None

            if outcome and outcome.invocations:
                tool_calls = _invocations_to_openai_calls(outcome.invocations)
                emit_openai_tool_call_delta(handler, base_chunk, tool_calls)
            else:
                # Marker found but no valid tool calls — emit remaining text
                if post_marker_buf:
                    emit_openai_text_delta(handler, base_chunk, post_marker_buf)
        else:
            # No marker found — emit any buffered text from scanner
            remaining = scanner.pending_prefix + scanner.accumulated
            if remaining:
                emit_openai_text_delta(handler, base_chunk, remaining)

        emit_openai_done(handler)
    finally:
        conn.close()


def _build_openai_tool_response(model: str, outcome: Any, settings: Settings) -> dict:
    tool_calls = _invocations_to_openai_calls(outcome.invocations)
    msg: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": tool_calls,
    }
    if outcome.text_before_marker:
        msg["content"] = outcome.text_before_marker
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "created": int(time.time()),
    }


def _invocations_to_openai_calls(invocations: list[ToolInvocation]) -> list[dict]:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": inv.name,
                "arguments": json.dumps(inv.parameters, ensure_ascii=False),
            },
        }
        for inv in invocations
    ]


def _retry_virtual_parse(
    prior_text: str,
    marker: str,
    tools: list[dict],
    messages: list[dict],
    resolved_model: str,
    settings: Settings,
) -> Any:
    from .virtual_tools import retry_parse
    return retry_parse(prior_text, marker, tools, messages, resolved_model, settings)


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------

def handle_anthropic(handler: Any, settings: Settings) -> None:
    body = _read_json_body(handler)
    if body is None:
        _send_json(handler, 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "invalid JSON body"}})
        return

    openai_payload = convert_anthropic_to_openai(body)
    if settings.upstream_extra_fields:
        openai_payload = {**settings.upstream_extra_fields, **openai_payload}

    requested_model = body.get("model", "")
    resolved = settings.resolve_model_name(requested_model)
    if resolved is None:
        _send_json(handler, 400, {"type": "error", "error": {"type": "not_found_error", "message": f"unknown model: {requested_model}"}})
        return

    openai_payload["model"] = resolved
    stream = body.get("stream", False)
    tools = body.get("tools", [])

    if not tools or settings.is_native_tool_model(requested_model):
        if stream:
            _passthrough_anthropic_stream(handler, openai_payload, requested_model, settings)
        else:
            result = fetch_upstream_chat(openai_payload, settings)
            anthropic_resp = convert_openai_to_anthropic(result, requested_model, bool(tools))
            _send_json(handler, 200, anthropic_resp)
        return

    # Virtual tool calling for Anthropic
    marker = generate_activation_marker()
    openai_tools = openai_payload.pop("tools", [])
    openai_choice = openai_payload.pop("tool_choice", None)
    had_tools = True  # we know tools were present

    directive = build_tool_directive(
        openai_tools, marker, tool_choice=openai_choice,
        intro=settings.tool_instruction_intro,
    )
    messages = openai_payload.get("messages", [])
    _inject_system_message(messages, directive)
    openai_payload["messages"] = messages
    openai_payload["stream"] = False

    valid_names = [t["function"]["name"] for t in openai_tools]
    result = fetch_upstream_chat(openai_payload, settings)
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    try:
        outcome = parse_tool_invocation(content, marker, valid_names)
        if outcome and outcome.invocations:
            from .virtual_tools import validate_tool_params
            errors = validate_tool_params(outcome.invocations, openai_tools)
            if errors:
                raise ParseError("; ".join(errors))
    except ParseError:
        outcome = None

    if not outcome or not outcome.invocations:
        if settings.retry_on_parse_failure:
            outcome = _retry_virtual_parse(content, marker, openai_tools, openai_payload.get("messages", []), requested_model, settings)

    # Build Anthropic response directly from parsed outcome
    if outcome and outcome.invocations:
        tool_blocks: list[dict] = []
        if outcome.text_before_marker:
            tool_blocks.append({"type": "text", "text": outcome.text_before_marker})
        for inv in outcome.invocations:
            tool_blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": inv.name,
                "input": inv.parameters,
            })
        if not tool_blocks:
            tool_blocks.append({"type": "text", "text": ""})

        usage = result.get("usage", {})
        anthropic_resp = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": tool_blocks,
            "model": requested_model,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": anthropic_usage(usage),
        }
    else:
        # No tool calls — convert the original upstream response
        anthropic_resp = convert_openai_to_anthropic(result, requested_model, had_tools)

    if stream:
        _stream_anthropic_response(handler, anthropic_resp, requested_model)
    else:
        _send_json(handler, 200, anthropic_resp)


def _passthrough_anthropic_stream(
    handler: Any, openai_payload: dict, requested_model: str, settings: Settings,
) -> None:
    from .proxy import open_upstream_connection, build_upstream_headers, resolve_request_path
    import urllib.parse

    model = requested_model
    url, auth, timeout = settings.get_upstream_config(model)

    conn = open_upstream_connection(settings, url=url, timeout=timeout)
    try:
        path = resolve_request_path(url, "/v1/chat/completions")

        raw_body = json.dumps(openai_payload, ensure_ascii=False).encode("utf-8")
        headers = build_upstream_headers(None, settings, auth=auth)
        conn.request("POST", path, body=raw_body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read()
            try:
                err_json = json.loads(err_body.decode("utf-8", errors="replace"))
            except Exception:
                err_json = {"error": err_body.decode("utf-8", errors="replace")}
            _send_json(handler, resp.status, err_json, cors=True)
            return

        # Convert OpenAI SSE stream to Anthropic SSE stream
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        usage = {"input_tokens": 0, "output_tokens": 0}
        block_index = 0
        in_text_block = False
        in_tool_block = False
        current_tool_index = -1

        begin_sse_response(handler)
        emit_anthropic_message_start(handler, msg_id, requested_model, usage)

        for sse_event in read_sse_chunks(resp):
            data = sse_event.get("data", "")
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                if "content" in delta and delta["content"]:
                    if in_tool_block:
                        emit_anthropic_content_block_stop(handler, block_index)
                        block_index += 1
                        in_tool_block = False
                    if not in_text_block:
                        emit_anthropic_content_block_start(handler, block_index, {"type": "text", "text": ""})
                        in_text_block = True
                    emit_anthropic_content_block_delta(handler, block_index, {"type": "text_delta", "text": delta["content"]})

                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        tc_index = tc.get("index", 0)

                        # New tool call starts — close previous block
                        if tc_index != current_tool_index:
                            if in_text_block:
                                emit_anthropic_content_block_stop(handler, block_index)
                                block_index += 1
                                in_text_block = False
                            if in_tool_block:
                                emit_anthropic_content_block_stop(handler, block_index)
                                block_index += 1
                                in_tool_block = False
                            current_tool_index = tc_index

                        fn = tc.get("function", {})
                        if fn.get("name"):
                            if in_text_block:
                                emit_anthropic_content_block_stop(handler, block_index)
                                block_index += 1
                                in_text_block = False
                            emit_anthropic_content_block_start(handler, block_index, {
                                "type": "tool_use", "id": tc.get("id", ""), "name": fn["name"], "input": {},
                            })
                            in_tool_block = True
                        if fn.get("arguments"):
                            emit_anthropic_content_block_delta(handler, block_index, {
                                "type": "input_json_delta", "partial_json": fn["arguments"],
                            })

                if finish:
                    if in_text_block:
                        emit_anthropic_content_block_stop(handler, block_index)
                        block_index += 1
                        in_text_block = False
                    if in_tool_block:
                        emit_anthropic_content_block_stop(handler, block_index)
                        block_index += 1
                        in_tool_block = False
                    stop = "end_turn" if finish == "stop" else "tool_use" if finish == "tool_calls" else "max_tokens"
                    emit_anthropic_message_delta(handler, stop, {"output_tokens": 0})

        emit_anthropic_message_stop(handler)
    finally:
        conn.close()


def _stream_anthropic_response(handler: Any, anthropic_resp: dict, requested_model: str) -> None:
    """Emit a pre-built Anthropic response as SSE events."""
    msg_id = anthropic_resp.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    usage = anthropic_resp.get("usage", {"input_tokens": 0, "output_tokens": 0})

    begin_sse_response(handler)
    emit_anthropic_message_start(handler, msg_id, requested_model, usage)

    for i, block in enumerate(anthropic_resp.get("content", [])):
        emit_anthropic_content_block_start(handler, i, block)
        btype = block.get("type", "")
        if btype == "text":
            emit_anthropic_content_block_delta(handler, i, {"type": "text_delta", "text": block.get("text", "")})
        elif btype == "tool_use":
            args_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            emit_anthropic_content_block_delta(handler, i, {"type": "input_json_delta", "partial_json": args_json})
        emit_anthropic_content_block_stop(handler, i)

    stop = anthropic_resp.get("stop_reason", "end_turn")
    emit_anthropic_message_delta(handler, stop, {"output_tokens": usage.get("output_tokens", 0)})
    emit_anthropic_message_stop(handler)


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------

def handle_passthrough(handler: Any, settings: Settings, method: str, path: str, body: bytes | None) -> None:
    status, resp_body, hdrs = fetch_upstream(method, path, body, settings)
    handler.send_response(status)
    for k, v in hdrs.items():
        if k.lower() not in ("transfer-encoding", "connection"):
            handler.send_header(k, v)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(resp_body)


# ---------------------------------------------------------------------------
# Dashboard and Settings APIs
# ---------------------------------------------------------------------------

def handle_dashboard(handler: Any, settings: Settings) -> None:
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    html_path = os.path.join(dir_path, "dashboard.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read().encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(content)))
        handler.end_headers()
        handler.wfile.write(content)
    except Exception as exc:
        _send_json(handler, 500, {"error": f"Failed to load dashboard: {exc}"}, cors=False)


def handle_api_settings_get(handler: Any, settings: Settings) -> None:
    _send_json(handler, 200, settings.to_dict(), cors=False)


def handle_api_settings_post(handler: Any, settings: Settings) -> None:
    body = _read_json_body(handler)
    if body is None:
        _send_json(handler, 400, {"error": "Invalid JSON body"}, cors=False)
        return

    from .config_file import save_config
    from .autostart import enable_autostart, disable_autostart

    autostart = body.get("autostart")
    if autostart is not None:
        try:
            if autostart:
                enable_autostart()
            else:
                disable_autostart()
        except Exception:
            pass

    try:
        host = str(body.get("HOST", settings.listen_host))
        old_port = settings.listen_port
        
        try:
            new_port = int(body.get("PORT", settings.listen_port))
        except (ValueError, TypeError):
            new_port = old_port

        try:
            upstream_timeout = int(body.get("UPSTREAM_TIMEOUT_SECONDS", settings.upstream_timeout))
        except (ValueError, TypeError):
            upstream_timeout = settings.upstream_timeout

        try:
            retry_delay_seconds = float(body.get("RETRY_DELAY_SECONDS", settings.retry_delay_seconds))
        except (ValueError, TypeError):
            retry_delay_seconds = settings.retry_delay_seconds

        try:
            max_retry_attempts = int(body.get("FC_ERROR_RETRY_MAX_ATTEMPTS", settings.max_retry_attempts))
        except (ValueError, TypeError):
            max_retry_attempts = settings.max_retry_attempts

        port_changed = (old_port != new_port)
        if port_changed:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((host, new_port))
            except Exception as e:
                _send_json(handler, 400, {"error": f"端口 {new_port} 已被占用或无法绑定: {e}"}, cors=False)
                return
            finally:
                sock.close()

        # Build clean merged dict for Settings instantiation
        merged = settings.to_dict()
        merged.update({
            "HOST": host,
            "PORT": new_port,
            "UPSTREAM_BASE_URL": str(body.get("UPSTREAM_BASE_URL", settings.upstream_url)),
            "UPSTREAM_TIMEOUT_SECONDS": upstream_timeout,
            "UPSTREAM_AUTH_HEADER": str(body.get("UPSTREAM_AUTH_HEADER", settings.upstream_auth)),
            "UPSTREAM_EXTRA_BODY_JSON": body.get("UPSTREAM_EXTRA_BODY_JSON", settings.upstream_extra_fields),
            "MODEL_MAP_JSON": body.get("MODEL_MAP_JSON", settings.name_mapping),
            "ALLOW_UNMAPPED_MODEL_PASSTHROUGH": bool(body.get("ALLOW_UNMAPPED_MODEL_PASSTHROUGH", settings.allow_unmapped)),
            "NATIVE_TOOL_MODELS_JSON": list(body.get("NATIVE_TOOL_MODELS_JSON", settings.native_tool_model_ids)),
            "PUBLIC_MODEL_IDS_JSON": list(body.get("PUBLIC_MODEL_IDS_JSON", settings.exposed_model_ids)),
            "TOOL_PROMPT_PREAMBLE": str(body.get("TOOL_PROMPT_PREAMBLE", settings.tool_instruction_intro)),
            "FC_ERROR_RETRY": bool(body.get("FC_ERROR_RETRY", settings.retry_on_parse_failure)),
            "FC_ERROR_RETRY_MAX_ATTEMPTS": max_retry_attempts,
            "RETRY_DELAY_SECONDS": retry_delay_seconds,
            "UPSTREAMS_JSON": list(body.get("UPSTREAMS_JSON", settings.upstreams)),
            "MODEL_ROUTES_JSON": dict(body.get("MODEL_ROUTES_JSON", settings.model_routes)),
        })

        new_settings = Settings.from_dict(merged)
        save_config(new_settings.to_dict())

        # Swap settings atomically on the server
        handler.server.settings = new_settings

        if port_changed:
            from .server import is_server_running
            if is_server_running():
                import threading
                from .server import start_server_threaded
                threading.Timer(0.5, lambda: start_server_threaded(new_settings)).start()

        _send_json(handler, 200, {"ok": True, "port_changed": port_changed}, cors=False)
    except Exception as exc:
        _send_json(handler, 500, {"error": f"Failed to save settings: {exc}"}, cors=False)


def _ping_upstream(url: str, timeout: int = 2) -> float | None:
    import time
    import urllib.request
    import urllib.error
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        return round((time.perf_counter() - start) * 1000)
    except Exception:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout):
                pass
            return round((time.perf_counter() - start) * 1000)
        except urllib.error.HTTPError:
            return round((time.perf_counter() - start) * 1000)
        except Exception:
            return None


import threading

_latency_lock = threading.Lock()
_latency_cache: dict[str, Any] = {"default": None, "providers": {}}
_latency_thread_started = False


def start_latency_monitor(settings: Settings) -> None:
    global _latency_thread_started
    with _latency_lock:
        if _latency_thread_started:
            return
        _latency_thread_started = True

    def monitor_loop():
        while True:
            try:
                from .server import _server_instance
                current_settings = settings
                if _server_instance is not None:
                    current_settings = _server_instance.settings

                # 1. Ping default upstream
                default_url = current_settings.upstream_url
                default_latency = _ping_upstream(default_url)

                # 2. Ping other providers
                upstreams = list(current_settings.upstreams)
                provider_latencies = {}
                for p in upstreams:
                    p_id = p.get("id")
                    p_url = p.get("url")
                    if p_id and p_url:
                        provider_latencies[p_id] = _ping_upstream(p_url)

                with _latency_lock:
                    _latency_cache["default"] = default_latency
                    _latency_cache["providers"] = provider_latencies
            except Exception as e:
                print(f"[bridge] Latency monitor error: {e}")
            time.sleep(10)

    t = threading.Thread(target=monitor_loop, daemon=True, name="LatencyMonitor")
    t.start()


def handle_api_status(handler: Any, settings: Settings) -> None:
    from .autostart import is_autostart_enabled
    start_latency_monitor(settings)

    with _latency_lock:
        latency = _latency_cache.get("default")
        provider_latencies = dict(_latency_cache.get("providers", {}))

    _send_json(handler, 200, {
        "status": "running",
        "port": settings.listen_port,
        "host": settings.listen_host,
        "upstream_latency_ms": latency,
        "provider_latencies": provider_latencies,
        "autostart_enabled": is_autostart_enabled()
    }, cors=False)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ROUTE_TABLE: dict[tuple[str, str], Any] = {
    ("GET", "/"): handle_dashboard,
    ("GET", "/dashboard"): handle_dashboard,
    ("GET", "/api/settings"): handle_api_settings_get,
    ("POST", "/api/settings"): handle_api_settings_post,
    ("GET", "/api/status"): handle_api_status,
    ("GET", "/health"): handle_health,
    ("GET", "/v1/models"): handle_models,
    ("POST", "/v1/chat/completions"): handle_chat,
    ("POST", "/v1/messages"): handle_anthropic,
}


def dispatch(handler: Any, settings: Settings, method: str, path: str, body: bytes | None) -> None:
    """Route a request to the appropriate handler."""
    import urllib.parse
    import urllib.error
    clean_path = urllib.parse.urlparse(path).path
    key = (method, clean_path.rstrip("/") if clean_path != "/" else clean_path)
    handler_fn = _ROUTE_TABLE.get(key)

    try:
        if handler_fn:
            handler_fn(handler, settings)
        else:
            handle_passthrough(handler, settings, method, path, body)
    except UpstreamError as exc:
        try:
            err_json = json.loads(exc.body.decode("utf-8", errors="replace"))
        except Exception:
            err_json = {"error": f"upstream returned {exc.status}"}
        _send_json(handler, exc.status, err_json)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            _send_json(handler, 504, {"error": f"gateway timeout: {exc.reason}"})
        else:
            _send_json(handler, 502, {"error": f"bad gateway: {exc.reason}"})
    except TimeoutError as exc:
        _send_json(handler, 504, {"error": f"gateway timeout: {exc}"})
    except ConnectionError as exc:
        _send_json(handler, 502, {"error": f"connection error: {exc}"})
    except BridgeError as exc:
        _send_json(handler, 500, {"error": str(exc)})
    except Exception as exc:
        _send_json(handler, 500, {"error": f"internal error: {exc}"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json_body(handler: Any) -> dict | None:
    content_type = handler.headers.get("Content-Type", "")
    if not content_type.startswith("application/json"):
        return None
    length = int(handler.headers.get("Content-Length", 0))
    if not length:
        return None
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _send_json(handler: Any, status: int, payload: dict, cors: bool = True) -> None:
    if getattr(handler, "sse_started", False):
        error_msg = json.dumps({"error": payload.get("error", "internal error")}, ensure_ascii=False)
        try:
            handler.wfile.write(f"event: error\ndata: {error_msg}\n\n".encode("utf-8"))
            handler.wfile.flush()
        except Exception:
            pass
        return

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    if cors:
        handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _inject_system_message(messages: list[dict], text: str) -> None:
    """Prepend or merge a system message at the beginning of the messages list."""
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0].get("content", "") + "\n\n" + text
    else:
        messages.insert(0, {"role": "system", "content": text})
