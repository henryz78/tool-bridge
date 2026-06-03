# toolbridge

OpenAI / Anthropic 兼容的 tool-calling 桥接代理。

将不支持原生 tool calling 的上游 API 包装为支持 `tools` / `tool_choice` / `parallel_tool_calls` 的 OpenAI 兼容端点，同时支持 Anthropic Messages API (`/v1/messages`) 的完整转换。

## 功能

- **虚拟 tool calling**：将工具定义注入 prompt，使用激活标记和 JSON 格式解析模型输出的工具调用
- **原生 tool calling 直通**：对支持原生 tool calling 的模型直接透传，零延迟
- **Anthropic Messages API**：完整支持 `/v1/messages`，包括 `tool_use` / `tool_result` / `thinking` block 转换
- **模型映射**：将外部模型名映射到上游模型名
- **SSE 流式**：支持 OpenAI 和 Anthropic 两种流式格式
- **零依赖**：纯 Python 标准库，`http.server` + `urllib`

## 快速开始

### Docker

```bash
docker build -t toolbridge .
docker run -d \
  -e UPSTREAM_BASE_URL=http://your-upstream:3000 \
  -e UPSTREAM_AUTH_HEADER="Bearer your-key" \
  -e MODEL_MAP_JSON='{"deepseek-chat":"deepseek-v4-flash"}' \
  -p 8080:8080 \
  toolbridge
```

### 直接运行

```bash
UPSTREAM_BASE_URL=http://127.0.0.1:3000 \
UPSTREAM_AUTH_HEADER="Bearer your-key" \
MODEL_MAP_JSON='{"deepseek-chat":"deepseek-v4-flash"}' \
python -m toolbridge
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8080` | 监听端口 |
| `UPSTREAM_BASE_URL` | `http://127.0.0.1:3000` | 上游 API 地址 |
| `UPSTREAM_AUTH_HEADER` | (空) | 上游认证头 |
| `UPSTREAM_TIMEOUT_SECONDS` | `240` | 上游请求超时 |
| `UPSTREAM_EXTRA_BODY_JSON` | `{}` | 追加到上游请求的额外字段 |
| `MODEL_MAP_JSON` | `{}` | 模型名映射 |
| `ALLOW_UNMAPPED_MODEL_PASSTHROUGH` | `true` | 未映射模型是否直通 |
| `NATIVE_TOOL_MODELS_JSON` | `[]` | 支持原生 tool calling 的模型列表 |
| `PUBLIC_MODEL_IDS_JSON` | `[]` | `/v1/models` 返回的模型 ID 列表 |
| `TOOL_PROMPT_PREAMBLE` | (内置) | 虚拟 tool calling 的 prompt 前言 |
| `FC_ERROR_RETRY` | `true` | 解析失败时是否自动重试 |
| `FC_ERROR_RETRY_MAX_ATTEMPTS` | `3` | 最大重试次数 |
| `RETRY_DELAY_SECONDS` | `0` | 重试间隔秒数 |

## 端点

- `GET /v1/models` — 列出可用模型
- `POST /v1/chat/completions` — OpenAI Chat Completions
- `POST /v1/messages` — Anthropic Messages API
- `GET /health` — 健康检查
- 其他路径 — 原样转发到上游

## 许可证

MIT
