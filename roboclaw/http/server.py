"""FastAPI server for the RoboClaw web chat UI.

Runs the full gateway runtime (AgentLoop, CronService, HeartbeatService,
ChannelManager) so the web UI has feature parity with ``roboclaw gateway``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.http.runtime import WebRuntime

import httpx
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from roboclaw.channels.web import WebChannel
from roboclaw.config.loader import get_config_path, load_config, load_runtime_config, save_config
from roboclaw.providers.factory import build_provider
from roboclaw.providers.registry import PROVIDERS
from roboclaw.utils.helpers import sync_workspace_templates


# ------------------------------------------------------------------
# Settings helpers
# ------------------------------------------------------------------


def _mask_api_key(api_key: str) -> str:
    if _api_key_validation_error(api_key):
        return "疑似误填，需重新保存"
    if len(api_key) >= 10:
        return f"{api_key[:6]}...{api_key[-4:]}"
    return "已保存" if api_key else ""


def _api_key_validation_error(api_key: str) -> str:
    """Return a user-facing error when the API key looks like pasted shell text."""
    if not api_key:
        return ""
    if any(ch.isspace() for ch in api_key):
        return "API key 里不能包含空格或换行；请只粘贴服务商给出的 key 本身。"
    shell_fragments = ("rm -rf", "2>&1", "&&", "||", "$(", "`", "export ")
    if any(fragment in api_key for fragment in shell_fragments):
        return "API key 看起来像终端命令或日志内容；请重新粘贴真实 key。"
    return ""


def _provider_options(config: Any) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for spec in PROVIDERS:
        provider_config = getattr(config.providers, spec.name, None)
        api_key = provider_config.api_key if provider_config and provider_config.api_key else ""
        api_key_error = _api_key_validation_error(api_key)
        configured = _is_provider_configured(spec, provider_config)
        options.append({
            "name": spec.name,
            "label": spec.label,
            "oauth": spec.is_oauth,
            "local": spec.is_local,
            "direct": spec.is_direct,
            "configured": configured,
            "api_base": provider_config.api_base if provider_config and provider_config.api_base else "",
            "has_api_key": bool(api_key),
            "masked_api_key": _mask_api_key(api_key),
            "api_key_warning": api_key_error,
            "extra_headers": provider_config.extra_headers if provider_config and provider_config.extra_headers else {},
        })
    return options


def _is_provider_configured(spec: Any, provider_config: Any) -> bool:
    if provider_config and provider_config.api_key and _api_key_validation_error(provider_config.api_key):
        return False
    if spec.is_oauth:
        return _is_oauth_provider_configured(spec.name)
    if spec.name == "azure_openai":
        return bool(provider_config and provider_config.api_key and provider_config.api_base)
    if spec.is_local or spec.name == "custom":
        return bool(provider_config and provider_config.api_base)
    return bool(provider_config and provider_config.api_key)


def _is_oauth_provider_configured(provider_name: str) -> bool:
    if provider_name == "openai_codex":
        from roboclaw.providers.openai_codex_provider import has_codex_oauth_token

        return has_codex_oauth_token()
    return False


def _provider_status_payload(config: Any) -> dict[str, Any]:
    providers = _provider_options(config)
    active_provider = config.get_provider_name(config.agents.defaults.model)
    default_model = config.agents.defaults.model
    if active_provider == "openai_codex":
        from roboclaw.providers.openai_codex_provider import normalize_codex_model

        default_model = normalize_codex_model(default_model)
    active_option = next((item for item in providers if item["name"] == active_provider), None)
    custom_option = next((item for item in providers if item["name"] == "custom"), None)
    return {
        "default_model": default_model,
        "default_provider": config.agents.defaults.provider,
        "active_provider": active_provider,
        "active_provider_configured": bool(active_option and active_option["configured"]),
        "custom_provider": custom_option or {
            "name": "custom",
            "label": "Custom",
            "configured": False,
            "api_base": "",
            "has_api_key": False,
            "masked_api_key": "",
            "api_key_warning": "",
            "extra_headers": {},
        },
        "providers": providers,
    }


def _default_model_for_provider(provider_name: str) -> str:
    defaults = {
        "custom": "claude-sonnet-4-5-20250929",
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-4.1",
        "openrouter": "anthropic/claude-sonnet-4-5",
        "aihubmix": "claude-sonnet-4-5-20250929",
        "siliconflow": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "volcengine": "doubao-seed-1-6",
        "deepseek": "deepseek-chat",
        "dashscope": "qwen-plus",
        "gemini": "gemini-2.5-pro",
        "zhipu": "glm-4.5",
        "moonshot": "kimi-k2-0711-preview",
        "minimax": "abab6.5s-chat",
        "github_copilot": "github-copilot/gpt-4o",
    }
    return defaults.get(provider_name, "")


def _is_oauth_model_name(model: str) -> bool:
    lower = model.lower()
    return lower.startswith("openai-codex/") or lower.startswith("openai_codex/") or lower.startswith("github-copilot/")


def _classify_provider_error(content: str | None) -> dict[str, str]:
    text = (content or "").strip()
    lower = text.lower()
    if not text:
        return {"code": "", "message": ""}
    if "invalid token" in lower or "invalid api key" in lower or "401" in lower or "unauthorized" in lower:
        return {
            "code": "invalid_token",
            "message": "API key/token 无效，或者中转站没有接受这个 key。",
        }
    if "insufficient balance" in lower or "quota" in lower or "402" in lower or "403" in lower:
        return {
            "code": "insufficient_balance",
            "message": "该 AI provider 账户余额、额度或权限不足。注意这里不是银行卡余额，也不是 AutoDL 余额。",
        }
    if "model not found" in lower or "404" in lower:
        return {
            "code": "model_not_found",
            "message": "模型名不被当前 provider 支持，请改成该平台实际支持的模型名。",
        }
    if "rate limit" in lower or "429" in lower:
        return {
            "code": "rate_limited",
            "message": "请求频率或用量额度被限制，请稍后重试或切换 provider。",
        }
    if (
        "unsupported server-side tools" in lower
        or "tool_use/tool_result" in lower
        or "mismatched tool" in lower
        or "improperly formed" in lower
        or "tool_choice" in lower
        or "oversized tool" in lower
    ):
        return {
            "code": "tool_calling_unsupported",
            "message": "该中转/模型不兼容 RoboClaw Agent 工具调用，只适合普通聊天或生成训练方案。",
        }
    return {
        "code": "provider_error",
        "message": text[:500],
    }


async def _provider_test_config(payload: dict[str, Any]) -> Any:
    config = load_config(get_config_path()).model_copy(deep=True)
    provider_name = payload.get("provider") or config.agents.defaults.provider or "custom"
    section = getattr(config.providers, provider_name, None)
    if section is None:
        raise ValueError(f"Unknown provider: {provider_name}")

    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        api_key = api_key.strip()
        api_key_error = _api_key_validation_error(api_key)
        if api_key_error:
            raise ValueError(api_key_error)
        section.api_key = api_key

    api_base = payload.get("api_base")
    if isinstance(api_base, str):
        section.api_base = api_base.strip() or None

    extra_headers = payload.get("extra_headers")
    if isinstance(extra_headers, dict):
        section.extra_headers = extra_headers or None

    api_key_error = _api_key_validation_error(section.api_key or "")
    if api_key_error:
        raise ValueError(api_key_error)

    requested_model = payload.get("model")
    requested_model = requested_model.strip() if isinstance(requested_model, str) else ""
    if provider_name not in {"openai_codex", "github_copilot"} and _is_oauth_model_name(requested_model):
        requested_model = ""
    if provider_name == "openai_codex":
        from roboclaw.providers.openai_codex_provider import DEFAULT_CODEX_MODEL, normalize_codex_model

        config.agents.defaults.model = normalize_codex_model(requested_model or config.agents.defaults.model) or DEFAULT_CODEX_MODEL
    elif provider_name == "github_copilot":
        config.agents.defaults.model = requested_model or "github-copilot/gpt-4o"
    elif requested_model:
        config.agents.defaults.model = requested_model
    elif section.api_base:
        discovered_model = await _discover_custom_model(section.api_base, section.api_key or None)
        if discovered_model:
            config.agents.defaults.model = discovered_model
        else:
            config.agents.defaults.model = _default_model_for_provider(provider_name) or config.agents.defaults.model
    elif _is_oauth_model_name(config.agents.defaults.model):
        config.agents.defaults.model = _default_model_for_provider(provider_name) or config.agents.defaults.model
    config.agents.defaults.provider = provider_name
    return config


async def _run_provider_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    config = await _provider_test_config(payload)
    provider_name = config.get_provider_name(config.agents.defaults.model) or config.agents.defaults.provider
    provider = build_provider(config)
    model = provider.get_default_model()

    text_response = await provider.chat_with_retry(
        messages=[
            {"role": "system", "content": "Reply with exactly: OK"},
            {"role": "user", "content": "connection test"},
        ],
        tools=None,
        model=model,
        max_tokens=16,
        temperature=0,
    )
    text_error = _classify_provider_error(text_response.content) if text_response.finish_reason == "error" else {"code": "", "message": ""}
    text_ok = text_response.finish_reason != "error"

    tool_payload = [{
        "type": "function",
        "function": {
            "name": "report_provider_status",
            "description": "Report provider status for a connection test.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "message": {"type": "string"},
                },
                "required": ["ok"],
                "additionalProperties": False,
            },
        },
    }]
    tool_response = await provider.chat_with_retry(
        messages=[
            {"role": "system", "content": "Call the report_provider_status tool with ok=true."},
            {"role": "user", "content": "Test tool calling."},
        ],
        tools=tool_payload,
        model=model,
        max_tokens=64,
        temperature=0,
        tool_choice="auto",
    )
    tool_error = _classify_provider_error(tool_response.content) if tool_response.finish_reason == "error" else {"code": "", "message": ""}
    tool_calls_returned = bool(tool_response.tool_calls)
    tool_ok = tool_response.finish_reason != "error" and tool_calls_returned

    skill_matrix = []
    if text_ok:
        for spec in _provider_skill_probe_specs():
            skill_matrix.append(await _run_provider_skill_probe(provider, model, spec))

    if text_ok and tool_ok:
        capability = "agent"
        passed = sum(1 for item in skill_matrix if item["ok"])
        total = len(skill_matrix)
        recommendation = (
            f"该 provider 可以用于 RoboClaw Agent。Skill/tool 兼容性 {passed}/{total}；"
            "未通过的 skill 会退化为规划或需要换 provider。"
        )
    elif text_ok:
        capability = "planner"
        recommendation = "该 provider 可以普通聊天和生成训练方案，但未验证通过工具调用；自动执行类任务建议切换到支持 function calling 的 provider。"
    else:
        capability = "unavailable"
        recommendation = text_error["message"] or "当前 provider 暂不可用。"

    return {
        "status": "ok",
        "provider": provider_name,
        "model": model,
        "capability": capability,
        "recommendation": recommendation,
        "text": {
            "ok": text_ok,
            "finishReason": text_response.finish_reason,
            "errorCode": text_error["code"],
            "message": text_error["message"] or (text_response.content or "")[:200],
        },
        "tools": {
            "ok": tool_ok,
            "accepted": tool_response.finish_reason != "error",
            "toolCallsReturned": tool_calls_returned,
            "finishReason": tool_response.finish_reason,
            "errorCode": tool_error["code"],
            "message": tool_error["message"] or (
                "工具调用验证通过。"
                if tool_ok
                else "模型没有返回工具调用；此 provider 可能只能用于普通规划。"
            ),
        },
        "skillMatrix": skill_matrix,
    }


def _provider_skill_probe_specs() -> list[dict[str, Any]]:
    """Representative skill/tool probes. They do not execute tools."""
    return [
        {
            "id": "skill_file_context",
            "label": "Skill 文件读取",
            "description": "读取 SKILL.md / repo 文件，是 OpenClaw 风格 skill 的入口。",
            "expectedTool": "read_file",
            "prompt": "Call read_file to read roboclaw/skills/robot-training/SKILL.md.",
            "tool": {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file with optional pagination.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },
        },
        {
            "id": "shell_automation",
            "label": "Shell / 环境调试",
            "description": "安装、检查环境、读取日志、修复云端任务通常需要 exec。",
            "expectedTool": "exec",
            "prompt": "Call exec to run: python --version",
            "tool": {
                "type": "function",
                "function": {
                    "name": "exec",
                    "description": "Execute a shell command and return output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "working_dir": {"type": "string"},
                            "timeout": {"type": "integer"},
                        },
                        "required": ["command"],
                    },
                },
            },
        },
        {
            "id": "web_research",
            "label": "网页检索",
            "description": "查开源模型、论文、数据集和云平台文档。",
            "expectedTool": "web_search",
            "prompt": "Call web_search to search: OpenVLA-OFT LIBERO benchmark.",
            "tool": {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web and return titles, URLs, and snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            },
        },
        {
            "id": "cloud_training",
            "label": "Evo Studio 总控",
            "description": "OpenClaw 风格总控入口：自然语言任务委托给后端，再由后端做云训练、runtime match、source preflight。",
            "expectedTool": "evo_studio_agent_consult",
            "prompt": "Call evo_studio_agent_consult with mode=plan for an rlinf_vla smoke test. Do not start a paid job.",
            "tool": {
                "type": "function",
                "function": {
                    "name": "evo_studio_agent_consult",
                    "description": "Delegate Evo Studio training/data/cloud operations to RoboClaw's backend control-plane agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "mode": {"type": "string", "enum": ["plan", "execute", "repair", "status"]},
                            "workflow": {"type": "string"},
                            "provider": {"type": "string"},
                            "params": {"type": "object"},
                            "confirmed": {"type": "boolean"},
                        },
                        "required": ["task"],
                    },
                },
            },
        },
        {
            "id": "robotics_diagnostics",
            "label": "机器人诊断",
            "description": "硬件接入、校准、采集、回放、部署前检查。",
            "expectedTool": "doctor",
            "prompt": "Call doctor to inspect the embodied robot environment.",
            "tool": {
                "type": "function",
                "function": {
                    "name": "doctor",
                    "description": "Check embodied environment health and summarize the current setup.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "verbose": {"type": "boolean"},
                        },
                    },
                },
            },
        },
        {
            "id": "subagent_delegation",
            "label": "子任务代理",
            "description": "复杂任务拆给后台 agent，例如长时间训练排障或代码审查。",
            "expectedTool": "spawn",
            "prompt": "Call spawn to delegate: inspect cloud training logs and summarize failures.",
            "tool": {
                "type": "function",
                "function": {
                    "name": "spawn",
                    "description": "Spawn a subagent to handle a background task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            },
        },
    ]


async def _run_provider_skill_probe(provider: Any, model: str, spec: dict[str, Any]) -> dict[str, Any]:
    expected_tool = str(spec["expectedTool"])
    response = await provider.chat_with_retry(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are testing RoboClaw/OpenClaw skill compatibility. "
                    "Call the requested tool. Do not answer in prose unless tool calling is impossible."
                ),
            },
            {"role": "user", "content": spec["prompt"]},
        ],
        tools=[spec["tool"]],
        model=model,
        max_tokens=96,
        temperature=0,
        tool_choice="auto",
    )
    error = _classify_provider_error(response.content) if response.finish_reason == "error" else {"code": "", "message": ""}
    called = [tool_call.name for tool_call in response.tool_calls]
    ok = expected_tool in called
    if ok:
        message = "通过"
    elif response.finish_reason == "error":
        message = error["message"]
    elif called:
        message = f"返回了其他工具：{', '.join(called)}"
    else:
        message = "没有返回工具调用，可能只能规划不能执行。"
    return {
        "id": spec["id"],
        "label": spec["label"],
        "description": spec["description"],
        "expectedTool": expected_tool,
        "ok": ok,
        "finishReason": response.finish_reason,
        "errorCode": error["code"],
        "message": message,
        "calledTools": called,
    }


async def _discover_custom_model(api_base: str, api_key: str | None) -> str | None:
    if not api_base:
        return None
    url = api_base.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Failed to auto-discover models from {}: {}", url, exc)
        return None

    data = payload.get("data", [])
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            return str(item["id"])
    return None


# ------------------------------------------------------------------
# System routes
# ------------------------------------------------------------------


def _register_system_routes(app: FastAPI, runtime: WebRuntime) -> None:
    @app.get("/api/system/provider-status")
    async def provider_status() -> dict[str, Any]:
        config = load_config(get_config_path())
        return _provider_status_payload(config)

    @app.get("/api/system/runtime-info")
    async def runtime_info() -> dict[str, Any]:
        return {
            "web_runtime_version": 2,
            "features": {
                "provider_settings": True,
                "chat_session_bootstrap": True,
                "dict_allow_from": True,
            },
        }

    @app.post("/api/system/provider-config")
    async def save_provider_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        result = await _handle_save_provider(payload, runtime)
        if result.get("status") == "ok":
            app.state.llm_provider = runtime.provider
        return result

    @app.post("/api/system/provider-test")
    async def test_provider_config(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
        try:
            return await _run_provider_diagnostic(payload)
        except Exception as exc:
            error = _classify_provider_error(str(exc))
            return {
                "status": "error",
                "provider": payload.get("provider") or "",
                "model": payload.get("model") or "",
                "capability": "unavailable",
                "recommendation": error["message"] or "Provider 测试失败。",
                "text": {
                    "ok": False,
                    "finishReason": "error",
                    "errorCode": error["code"],
                    "message": error["message"],
                },
                "tools": {
                    "ok": False,
                    "accepted": False,
                    "toolCallsReturned": False,
                    "finishReason": "error",
                    "errorCode": error["code"],
                    "message": error["message"],
                },
            }

    @app.get("/api/system/hf-config")
    async def hf_config_status() -> dict[str, Any]:
        config = load_config(get_config_path())
        hf = config.huggingface
        return {
            "endpoint": hf.endpoint,
            "masked_token": _mask_api_key(hf.token),
            "proxy": hf.proxy,
        }

    @app.post("/api/system/hf-config")
    async def save_hf_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        config = load_config(get_config_path())
        hf = config.huggingface
        endpoint = payload.get("endpoint")
        if isinstance(endpoint, str):
            hf.endpoint = endpoint.strip()
        if payload.get("clear_token"):
            hf.token = ""
        else:
            token = payload.get("token")
            if isinstance(token, str) and token.strip():
                hf.token = token.strip()
        proxy = payload.get("proxy")
        if isinstance(proxy, str):
            hf.proxy = proxy.strip()
        save_config(config, get_config_path())
        return {
            "status": "ok",
            "endpoint": hf.endpoint,
            "masked_token": _mask_api_key(hf.token),
            "proxy": hf.proxy,
        }

    @app.get("/api/system/control-record-config")
    async def control_record_config() -> dict[str, Any]:
        config = load_config(get_config_path())
        return config.control_center.record.model_dump()

    @app.post("/api/system/control-record-config")
    async def save_control_record_config(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        config = load_config(get_config_path())
        record = config.control_center.record
        task = payload.get("task")
        if isinstance(task, str):
            record.task = task
        num_episodes = payload.get("num_episodes")
        if isinstance(num_episodes, int):
            record.num_episodes = num_episodes
        episode_time_s = payload.get("episode_time_s")
        if isinstance(episode_time_s, int):
            record.episode_time_s = episode_time_s
        reset_time_s = payload.get("reset_time_s")
        if isinstance(reset_time_s, int):
            record.reset_time_s = reset_time_s
        dataset_name = payload.get("dataset_name")
        if isinstance(dataset_name, str):
            record.dataset_name = dataset_name
        fps = payload.get("fps")
        if isinstance(fps, int):
            record.fps = fps
        use_cameras = payload.get("use_cameras")
        if isinstance(use_cameras, bool):
            record.use_cameras = use_cameras
        save_config(config, get_config_path())
        return {"status": "ok", **record.model_dump()}


async def _handle_save_provider(payload: dict[str, Any], runtime: WebRuntime) -> dict[str, Any]:
    """Apply provider config changes, swap provider atomically, refresh agent."""
    config = load_config(get_config_path())

    provider_name = payload.get("provider", "custom")
    section = getattr(config.providers, provider_name, None)
    if section is None:
        return {"status": "error", "message": f"Unknown provider: {provider_name}"}

    if payload.get("clear_api_key"):
        section.api_key = ""

    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        api_key = api_key.strip()
        api_key_error = _api_key_validation_error(api_key)
        if api_key_error:
            return {"status": "error", "message": api_key_error}
        section.api_key = api_key

    api_base = payload.get("api_base")
    if isinstance(api_base, str):
        section.api_base = api_base.strip() or None

    error = _apply_extra_headers(payload, section)
    if error:
        return error

    api_key_error = _api_key_validation_error(section.api_key or "")
    if api_key_error:
        return {"status": "error", "message": api_key_error}

    requested_model = payload.get("model")
    requested_model = requested_model.strip() if isinstance(requested_model, str) else ""
    if provider_name not in {"openai_codex", "github_copilot"} and _is_oauth_model_name(requested_model):
        requested_model = ""

    if provider_name == "openai_codex":
        from roboclaw.providers.openai_codex_provider import DEFAULT_CODEX_MODEL, normalize_codex_model

        config.agents.defaults.model = normalize_codex_model(config.agents.defaults.model) or DEFAULT_CODEX_MODEL
    elif provider_name == "github_copilot":
        config.agents.defaults.model = "github-copilot/gpt-4o"
    elif requested_model:
        config.agents.defaults.model = requested_model
    # Auto-discover model for providers that use a custom base URL
    elif section.api_base:
        discovered_model = await _discover_custom_model(section.api_base, section.api_key or None)
        if discovered_model:
            config.agents.defaults.model = discovered_model
        elif _is_oauth_model_name(config.agents.defaults.model):
            config.agents.defaults.model = _default_model_for_provider(provider_name) or config.agents.defaults.model
    elif _is_oauth_model_name(config.agents.defaults.model):
        config.agents.defaults.model = _default_model_for_provider(provider_name) or config.agents.defaults.model

    config.agents.defaults.provider = provider_name if provider_name != "auto" else "auto"

    save_config(config, get_config_path())

    # Atomic provider swap
    new_provider = build_provider(config)
    runtime.swap_provider(new_provider, config)

    return {"status": "ok", **_provider_status_payload(config)}


def _apply_extra_headers(payload: dict[str, Any], section: Any) -> dict[str, Any] | None:
    """Parse and apply extra_headers from payload. Returns error dict on failure."""
    extra_headers = payload.get("extra_headers")
    if isinstance(extra_headers, str):
        try:
            extra_headers = json.loads(extra_headers) if extra_headers.strip() else {}
        except json.JSONDecodeError:
            return {"status": "error", "message": "extra_headers must be valid JSON."}
    if isinstance(extra_headers, dict):
        section.extra_headers = extra_headers or None
    return None


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def create_app(
    *,
    config_path: str | None = None,
    workspace: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> FastAPI:
    """Build the FastAPI app with the full gateway runtime."""
    from roboclaw.http.runtime import WebRuntime

    config = load_runtime_config(config_path, workspace)
    sync_workspace_templates(config.workspace_path)

    runtime = WebRuntime.build(config, host=host, port=port)

    app = FastAPI(title="RoboClaw Web UI")

    # CORS middleware
    web_cfg = config.channels.web
    web_defaults = WebChannel.default_config()
    cors_origins = web_cfg.get("cors_origins", web_defaults.get("cors_origins", []))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    web_ch = runtime.channel_manager.get_channel("web")
    if web_ch is not None:
        web_ch.register_routes(app)
    _register_system_routes(app, runtime)

    # Dashboard routes
    if web_ch is not None:
        from roboclaw.http.routes import register_all_routes

        app.state.hardware_monitor = runtime.hw_monitor
        app.state.embodied_service = runtime.embodied_service
        app.state.llm_provider = runtime.provider

        # Wire the service into the agent's embodied tool groups
        from roboclaw.embodied.toolkit.tools import EmbodiedToolGroup

        runtime.agent.embodied_service = runtime.embodied_service
        for tool in runtime.agent.tools.iter_tools():
            if hasattr(tool, "embodied_service"):
                tool.embodied_service = runtime.embodied_service
            if hasattr(tool, "llm_provider"):
                tool.llm_provider = runtime.provider
            if isinstance(tool, EmbodiedToolGroup):
                tool.embodied_service = runtime.embodied_service

        register_all_routes(
            app,
            web_ch,
            runtime.embodied_service,
            get_config=lambda: (web_cfg["host"], web_cfg["port"]),
            llm_provider=runtime.provider,
        )

    # Serve built frontend in production (ui/dist/)
    ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"
    if ui_dist.is_dir():
        from starlette.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        app.mount("/assets", StaticFiles(directory=str(ui_dist / "assets")), name="ui-assets")

        @app.get("/{full_path:path}")
        async def _spa_fallback(full_path: str):
            file_path = ui_dist / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))
            # no-cache: browser must revalidate index.html every time,
            # so it picks up new asset hashes after a frontend rebuild.
            return FileResponse(
                str(ui_dist / "index.html"),
                headers={"Cache-Control": "no-cache"},
            )

    # Store state for host/port access
    app.state.web_host = web_cfg["host"]
    app.state.web_port = web_cfg["port"]

    @app.on_event("startup")
    async def _startup() -> None:
        await runtime.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await runtime.shutdown()

    return app


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------


def _check_device_permissions() -> None:
    """Check serial/camera device permissions at startup, auto-fix if possible."""
    import os
    import sys

    if sys.platform != "linux":
        return
    from roboclaw.embodied.embodiment.hardware.scan import list_serial_device_paths
    devices = list_serial_device_paths()
    if not devices:
        return
    denied = [d for d in devices if not os.access(d, os.R_OK | os.W_OK)]
    if not denied:
        return
    logger.warning("Serial devices without permission: {}", denied)
    from roboclaw.embodied.embodiment.hardware.scan import fix_serial_permissions
    if fix_serial_permissions():
        logger.info("Auto-fixed serial device permissions")
    else:
        logger.warning(
            "Cannot auto-fix serial permissions. Run: bash scripts/setup-udev.sh"
        )


def _ensure_ui_build() -> None:
    """Rebuild frontend if ui/src is newer than ui/dist."""
    import shutil
    import subprocess

    ui_root = Path(__file__).resolve().parent.parent.parent / "ui"
    ui_src = ui_root / "src"
    ui_dist = ui_root / "dist"

    if not ui_src.is_dir():
        return

    needs_build = False

    # Check 1: git commit hash — survives git reset --hard which resets mtimes
    build_hash_file = ui_dist / ".build_commit"
    current_hash = _git_head_hash(ui_root.parent)
    if current_hash:
        saved_hash = build_hash_file.read_text().strip() if build_hash_file.is_file() else ""
        if saved_hash != current_hash:
            needs_build = True

    # Check 2: mtime fallback for non-git or dirty working tree
    if not needs_build:
        def _newest_mtime(directory: Path) -> float:
            return max((f.stat().st_mtime for f in directory.rglob("*") if f.is_file()), default=0)

        src_mtime = _newest_mtime(ui_src)
        dist_mtime = _newest_mtime(ui_dist) if ui_dist.is_dir() else 0
        if src_mtime > dist_mtime:
            needs_build = True

    if not needs_build:
        return

    npm = shutil.which("npm")
    if not npm:
        logger.warning("Frontend outdated but npm not found — skipping rebuild")
        return

    logger.info("Frontend source newer than build, rebuilding ui …")
    node_modules = ui_root / "node_modules"
    if not node_modules.is_dir():
        logger.info("Installing frontend dependencies …")
        subprocess.run([npm, "install"], cwd=str(ui_root), check=True)
    result = subprocess.run([npm, "run", "build"], cwd=str(ui_root))
    if result.returncode != 0:
        logger.warning("Frontend build failed (exit {}), serving stale dist", result.returncode)
    else:
        logger.info("Frontend rebuild complete")
        if current_hash:
            build_hash_file.write_text(current_hash)


def _git_head_hash(repo_root: Path) -> str:
    """Return short HEAD hash, or empty string if not a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def main(
    *,
    config_path: str | None = None,
    workspace: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the web server with uvicorn."""
    import uvicorn

    _check_device_permissions()
    _ensure_ui_build()
    app = create_app(config_path=config_path, workspace=workspace, host=host, port=port)
    logger.info("Starting RoboClaw Web UI at http://{}:{}", app.state.web_host, app.state.web_port)
    uvicorn.run(app, host=app.state.web_host, port=app.state.web_port, log_level="info")


def _main_cli() -> None:
    """CLI entrypoint for ``python -m roboclaw.http.server``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the RoboClaw Web UI server.")
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    main(
        config_path=args.config_path,
        workspace=args.workspace,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    _main_cli()
