"""FastAPI server for the RoboClaw web chat UI.

Runs the full gateway runtime (AgentLoop, CronService, HeartbeatService,
ChannelManager) so the web UI has feature parity with ``roboclaw gateway``.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from roboclaw.agent.loop import AgentLoop
from roboclaw.bus.queue import MessageBus
from roboclaw.channels.manager import ChannelManager
from roboclaw.channels.web import WebChannel
from roboclaw.config.loader import get_config_path, load_config, load_runtime_config, save_config
from roboclaw.config.paths import get_cron_dir
from roboclaw.cron.service import CronService
from roboclaw.cron.types import CronJob
from roboclaw.heartbeat.service import HeartbeatService
from roboclaw.providers.base import GenerationSettings
from roboclaw.providers.factory import ProviderConfigurationError, UnconfiguredProvider, build_provider
from roboclaw.providers.registry import PROVIDERS
from roboclaw.session.manager import SessionManager
from roboclaw.utils.helpers import sync_workspace_templates


# ------------------------------------------------------------------
# Settings helpers
# ------------------------------------------------------------------


def _mask_api_key(api_key: str) -> str:
    if len(api_key) >= 10:
        return f"{api_key[:6]}...{api_key[-4:]}"
    return "已保存" if api_key else ""


def _provider_options(config: Any) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for spec in PROVIDERS:
        provider_config = getattr(config.providers, spec.name, None)
        api_key = provider_config.api_key if provider_config and provider_config.api_key else ""
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
            "extra_headers": provider_config.extra_headers if provider_config and provider_config.extra_headers else {},
        })
    return options


def _is_provider_configured(spec: Any, provider_config: Any) -> bool:
    if spec.is_oauth:
        return False
    if spec.name == "azure_openai":
        return bool(provider_config and provider_config.api_key and provider_config.api_base)
    if spec.is_local or spec.name == "custom":
        return bool(provider_config and provider_config.api_base)
    return bool(provider_config and provider_config.api_key)


def _provider_status_payload(config: Any) -> dict[str, Any]:
    providers = _provider_options(config)
    active_provider = config.get_provider_name(config.agents.defaults.model)
    active_option = next((item for item in providers if item["name"] == active_provider), None)
    custom_option = next((item for item in providers if item["name"] == "custom"), None)
    return {
        "default_model": config.agents.defaults.model,
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
            "extra_headers": {},
        },
        "providers": providers,
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


def _refresh_agent_defaults(agent: AgentLoop, config: Any) -> None:
    agent.model = config.agents.defaults.model
    agent.provider.generation = GenerationSettings(
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        reasoning_effort=config.agents.defaults.reasoning_effort,
    )
    agent.memory_consolidator.model = config.agents.defaults.model
    agent.subagents.model = config.agents.defaults.model


# ------------------------------------------------------------------
# System routes
# ------------------------------------------------------------------


def _register_system_routes(app: FastAPI, agent: AgentLoop) -> None:
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
        return await _handle_save_provider(payload, agent)


async def _handle_save_provider(payload: dict[str, Any], agent: AgentLoop) -> dict[str, Any]:
    """Apply provider config changes, swap provider atomically, refresh agent."""
    config = load_config(get_config_path())
    section = config.providers.custom

    if payload.get("clear_api_key"):
        section.api_key = ""

    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        section.api_key = api_key.strip()

    api_base = payload.get("api_base")
    if isinstance(api_base, str):
        section.api_base = api_base.strip() or None

    error = _apply_extra_headers(payload, section)
    if error:
        return error

    discovered_model = await _discover_custom_model(section.api_base or "", section.api_key or None)
    if discovered_model:
        config.agents.defaults.model = discovered_model
    config.agents.defaults.provider = "custom" if section.api_base else "auto"

    save_config(config, get_config_path())

    # Atomic provider swap
    new_provider = build_provider(config)
    agent.provider = new_provider
    _refresh_agent_defaults(agent, config)

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
# Cron callback (copied from gateway command pattern)
# ------------------------------------------------------------------


async def _on_cron_job(agent: AgentLoop, bus: MessageBus, provider: Any, job: CronJob) -> str | None:
    from roboclaw.agent.tools.cron import CronTool
    from roboclaw.agent.tools.message import MessageTool
    from roboclaw.utils.evaluator import evaluate_response

    reminder_note = (
        "[Scheduled Task] Timer finished.\n\n"
        f"Task '{job.name}' has been triggered.\n"
        f"Scheduled instruction: {job.payload.message}"
    )

    cron_tool = agent.tools.get("cron")
    cron_token = None
    if isinstance(cron_tool, CronTool):
        cron_token = cron_tool.set_cron_context(True)
    try:
        response = await agent.process_direct(
            reminder_note,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
    finally:
        if isinstance(cron_tool, CronTool) and cron_token is not None:
            cron_tool.reset_cron_context(cron_token)

    message_tool = agent.tools.get("message")
    if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
        return response

    if not (job.payload.deliver and job.payload.to and response):
        return response

    should_notify = await evaluate_response(response, job.payload.message, provider, agent.model)
    if should_notify:
        from roboclaw.bus.events import OutboundMessage
        await bus.publish_outbound(OutboundMessage(
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to,
            content=response,
        ))
    return response


# ------------------------------------------------------------------
# Heartbeat helpers
# ------------------------------------------------------------------


def _pick_heartbeat_target(channels: ChannelManager, session_manager: SessionManager) -> tuple[str, str]:
    """Pick a routable channel/chat target for heartbeat-triggered messages."""
    enabled = set(channels.enabled_channels)
    for item in session_manager.list_sessions():
        key = item.get("key") or ""
        if ":" not in key:
            continue
        channel, chat_id = key.split(":", 1)
        if channel in {"cli", "system"}:
            continue
        if channel in enabled and chat_id:
            return channel, chat_id
    return "cli", "direct"


async def _cancel_background_tasks(app: FastAPI) -> None:
    """Cancel all background tasks created during startup."""
    for task_name in ("heartbeat_task", "cron_task", "channels_task", "agent_task"):
        task = getattr(app.state, task_name, None)
        if task is None:
            continue
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


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

    # 1. Load config
    config = load_runtime_config(config_path, workspace)
    sync_workspace_templates(config.workspace_path)

    # 2. Shared infra
    bus = MessageBus()
    sessions = SessionManager(config.workspace_path)

    # 3. Provider (graceful fallback for unconfigured state)
    try:
        provider = build_provider(config)
    except ProviderConfigurationError as exc:
        logger.warning("Provider not configured at startup: {}. Configure via Settings.", exc)
        provider = UnconfiguredProvider(str(exc))

    # 4. Cron service
    cron_store_path = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store_path)

    # 5. Agent loop
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=sessions,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )
    _refresh_agent_defaults(agent, config)

    # Cron callback (needs agent + bus + provider)
    cron.on_job = lambda job: _on_cron_job(agent, bus, provider, job)

    # 6. Heartbeat service
    hb_cfg = config.gateway.heartbeat

    async def on_heartbeat_execute(tasks: str) -> str:
        channel, chat_id = _pick_heartbeat_target(channel_manager, sessions)

        async def _silent(*_args: Any, **_kwargs: Any) -> None:
            pass

        return await agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

    async def on_heartbeat_notify(response: str) -> None:
        from roboclaw.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target(channel_manager, sessions)
        if channel == "cli":
            return
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    # 7. Force-enable web channel in config
    web_defaults = WebChannel.default_config()
    web_cfg = {**web_defaults, "enabled": True, "host": "0.0.0.0"}
    if host is not None:
        web_cfg["host"] = host
    if port is not None:
        web_cfg["port"] = port
    config.channels.web = web_cfg

    # 8. Channel manager (discovers and creates WebChannel via registry)
    channel_manager = ChannelManager(config, bus)

    # 9. Post-configure: inject session manager into WebChannel
    web_ch = channel_manager.get_channel("web")
    if web_ch is not None:
        web_ch.sessions = sessions

    # 10. Create FastAPI app (server owns it, not the channel)
    app = FastAPI(title="RoboClaw Web UI")

    # 11. CORS middleware
    cors_origins = web_cfg.get("cors_origins", web_defaults.get("cors_origins", []))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 12. Register routes
    if web_ch is not None:
        web_ch.register_routes(app)
    _register_system_routes(app, agent)

    # 12b. Dashboard routes + hardware monitor
    if web_ch is not None:
        from roboclaw.embodied.hardware_monitor import HardwareMonitor
        from roboclaw.web.dashboard import register_dashboard_routes

        async def _on_hw_fault(fault: Any) -> None:
            await web_ch.broadcast_dashboard_event({
                "type": "dashboard.fault", **fault.to_dict(),
            })

        async def _on_hw_fault_resolved(fault: Any) -> None:
            await web_ch.broadcast_dashboard_event({
                "type": "dashboard.fault.resolved",
                "fault_type": fault.fault_type.value,
                "device_alias": fault.device_alias,
            })

        hw_monitor = HardwareMonitor(
            on_fault=_on_hw_fault,
            on_fault_resolved=_on_hw_fault_resolved,
        )
        app.state.hardware_monitor = hw_monitor

        register_dashboard_routes(
            app,
            web_ch,
            get_config=lambda: (web_cfg["host"], web_cfg["port"]),
        )

    # 13. Mount embodied data-collection routes
    from roboclaw.embodied.web.routes import router as embodied_router
    app.include_router(embodied_router)

    # 14. Mount workflow pipeline routes
    from roboclaw.embodied.web.workflow_routes import router as workflow_router
    app.include_router(workflow_router)

    # 14b. Mount dataset explorer routes
    from roboclaw.embodied.web.explorer_routes import router as explorer_router
    app.include_router(explorer_router)

    # 15. Serve built frontend in production (roboclaw-web/ui/dist/)
    ui_dist = Path(__file__).resolve().parent.parent.parent / "roboclaw-web" / "ui" / "dist"
    if ui_dist.is_dir():
        from starlette.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        # Static assets (js, css, images)
        app.mount("/assets", StaticFiles(directory=str(ui_dist / "assets")), name="ui-assets")

        # SPA fallback: any non-API path returns index.html for client-side routing
        @app.get("/{full_path:path}")
        async def _spa_fallback(full_path: str):
            file_path = ui_dist / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(ui_dist / "index.html"))

    # Store state for host/port access
    app.state.web_host = web_cfg["host"]
    app.state.web_port = web_cfg["port"]

    # 15. Startup: launch all background tasks
    @app.on_event("startup")
    async def _startup() -> None:
        app.state.agent_task = asyncio.create_task(agent.run(), name="roboclaw-agent")
        app.state.channels_task = asyncio.create_task(channel_manager.start_all(), name="roboclaw-channels")
        app.state.cron_task = asyncio.create_task(cron.start(), name="roboclaw-cron")
        app.state.heartbeat_task = asyncio.create_task(heartbeat.start(), name="roboclaw-heartbeat")
        hw_mon = getattr(app.state, "hardware_monitor", None)
        if hw_mon is not None:
            app.state.hardware_monitor_task = asyncio.create_task(
                hw_mon.run(), name="roboclaw-hw-monitor",
            )

    # 16. Shutdown: tear down gracefully
    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Stop active recording if any
        active_rec = getattr(app.state, "active_recording", None)
        if active_rec is not None and active_rec.active:
            active_rec.stop()

        # Stop hardware monitor
        hw_mon = getattr(app.state, "hardware_monitor", None)
        if hw_mon is not None:
            hw_mon.stop()
        hw_task = getattr(app.state, "hardware_monitor_task", None)
        if hw_task is not None:
            hw_task.cancel()
            with suppress(asyncio.CancelledError):
                await hw_task

        agent.stop()
        await channel_manager.stop_all()
        heartbeat.stop()
        cron.stop()
        await _cancel_background_tasks(app)
        await agent.close_mcp()

    return app


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------


def main(
    *,
    config_path: str | None = None,
    workspace: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the web server with uvicorn."""
    import uvicorn

    app = create_app(config_path=config_path, workspace=workspace, host=host, port=port)
    logger.info("Starting RoboClaw Web UI at http://{}:{}", app.state.web_host, app.state.web_port)
    uvicorn.run(app, host=app.state.web_host, port=app.state.web_port, log_level="info")
