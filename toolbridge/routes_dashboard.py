"""Dashboard, settings, status, and upstream-model route handlers."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .config import SECRET_MASK, Settings
from .http_utils import read_json_body, send_json


def handle_dashboard(handler: Any, settings: Settings) -> None:
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
        send_json(handler, 500, {"error": f"Failed to load dashboard: {exc}"}, cors=False)


def handle_api_settings_get(handler: Any, settings: Settings) -> None:
    send_json(handler, 200, settings.to_dict(redact_secrets=True), cors=False)


def handle_api_settings_post(handler: Any, settings: Settings) -> None:
    body = read_json_body(handler)
    if body is None:
        send_json(handler, 400, {"error": "Invalid JSON body"}, cors=False)
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
                send_json(handler, 400, {"error": f"端口 {new_port} 已被占用或无法绑定: {e}"}, cors=False)
                return
            finally:
                sock.close()

        merged = settings.to_dict()
        upstream_auth = _secret_value(body, "UPSTREAM_AUTH_HEADER", settings.upstream_auth)
        admin_token = _secret_value(body, "ADMIN_TOKEN", settings.admin_token)
        bridge_api_key = _secret_value(body, "BRIDGE_API_KEY", settings.bridge_api_key)
        upstreams = _merge_upstream_secrets(body.get("UPSTREAMS_JSON", settings.upstreams), settings.upstreams)
        merged.update({
            "HOST": host,
            "PORT": new_port,
            "UPSTREAM_BASE_URL": str(body.get("UPSTREAM_BASE_URL", settings.upstream_url)),
            "UPSTREAM_TIMEOUT_SECONDS": upstream_timeout,
            "UPSTREAM_AUTH_HEADER": upstream_auth,
            "UPSTREAM_EXTRA_BODY_JSON": body.get("UPSTREAM_EXTRA_BODY_JSON", settings.upstream_extra_fields),
            "MODEL_MAP_JSON": body.get("MODEL_MAP_JSON", settings.name_mapping),
            "ALLOW_UNMAPPED_MODEL_PASSTHROUGH": bool(body.get("ALLOW_UNMAPPED_MODEL_PASSTHROUGH", settings.allow_unmapped)),
            "NATIVE_TOOL_MODELS_JSON": list(body.get("NATIVE_TOOL_MODELS_JSON", settings.native_tool_model_ids)),
            "PUBLIC_MODEL_IDS_JSON": list(body.get("PUBLIC_MODEL_IDS_JSON", settings.exposed_model_ids)),
            "TOOL_PROMPT_PREAMBLE": str(body.get("TOOL_PROMPT_PREAMBLE", settings.tool_instruction_intro)),
            "FC_ERROR_RETRY": bool(body.get("FC_ERROR_RETRY", settings.retry_on_parse_failure)),
            "FC_ERROR_RETRY_MAX_ATTEMPTS": max_retry_attempts,
            "RETRY_DELAY_SECONDS": retry_delay_seconds,
            "UPSTREAMS_JSON": upstreams,
            "MODEL_ROUTES_JSON": dict(body.get("MODEL_ROUTES_JSON", settings.model_routes)),
            "ADMIN_TOKEN": admin_token,
            "BRIDGE_API_KEY": bridge_api_key,
        })

        new_settings = Settings.from_dict(merged)
        save_config(new_settings.to_dict())

        handler.server.settings = new_settings

        if port_changed:
            from .server import is_server_running
            if is_server_running():
                from .server import start_server_threaded
                threading.Timer(0.5, lambda: start_server_threaded(new_settings)).start()

        send_json(handler, 200, {"ok": True, "port_changed": port_changed}, cors=False)
    except Exception as exc:
        send_json(handler, 500, {"error": f"Failed to save settings: {exc}"}, cors=False)


def _secret_value(body: dict, key: str, current: str) -> str:
    if key not in body:
        return current
    value = str(body.get(key, ""))
    if value == SECRET_MASK:
        return current
    return value


def _merge_upstream_secrets(incoming: Any, current: list[dict]) -> list[dict]:
    current_by_id = {str(p.get("id", "")): p for p in current if isinstance(p, dict)}
    result: list[dict] = []
    for provider in list(incoming or []):
        if not isinstance(provider, dict):
            continue
        item = dict(provider)
        current_provider = current_by_id.get(str(item.get("id", "")), {})
        if item.get("auth") == SECRET_MASK:
            item["auth"] = str(current_provider.get("auth", ""))
        result.append(item)
    return result


def _ping_upstream(url: str, timeout: int = 2) -> float | None:
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


_latency_lock = threading.Lock()
_latency_cache: dict[str, Any] = {"default": None, "providers": {}}
_latency_thread_started = False


def start_latency_monitor(settings: Settings) -> None:
    global _latency_thread_started
    with _latency_lock:
        if _latency_thread_started:
            return
        _latency_thread_started = True

    def monitor_loop() -> None:
        while True:
            try:
                from .server import _server_instance
                current_settings = settings
                if _server_instance is not None:
                    current_settings = _server_instance.settings

                default_url = current_settings.upstream_url
                default_latency = _ping_upstream(default_url)

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

    send_json(handler, 200, {
        "status": "running",
        "port": settings.listen_port,
        "host": settings.listen_host,
        "upstream_latency_ms": latency,
        "provider_latencies": provider_latencies,
        "autostart_enabled": is_autostart_enabled()
    }, cors=False)


def handle_api_upstream_models(handler: Any, settings: Settings) -> None:
    parsed = urllib.parse.urlparse(handler.path)
    q = urllib.parse.parse_qs(parsed.query)
    provider_id = q.get("provider_id", [None])[0]

    url = settings.upstream_url
    auth = settings.upstream_auth
    timeout = settings.upstream_timeout

    if provider_id:
        for p in settings.upstreams:
            if p.get("id") == provider_id:
                url = p.get("url", url)
                auth = p.get("auth", auth)
                timeout = p.get("timeout", timeout)
                break

    from .proxy import build_upstream_headers, merge_url_path
    full_url = merge_url_path(url, "/v1/models")
    req = urllib.request.Request(full_url, method="GET")

    headers = build_upstream_headers(None, settings, auth=auth)
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            send_json(handler, 200, data, cors=False)
    except Exception as exc:
        send_json(handler, 500, {"error": f"Failed to fetch models from upstream: {exc}"}, cors=False)
