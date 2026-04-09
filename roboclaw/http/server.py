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
        return await _handle_save_provider(payload, runtime)


async def _handle_save_provider(payload: dict[str, Any], runtime: WebRuntime) -> dict[str, Any]:
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

        # Wire the service into the agent's embodied tool groups
        from roboclaw.embodied.toolkit.tools import EmbodiedToolGroup

        runtime.agent.embodied_service = runtime.embodied_service
        for tool in runtime.agent.tools.iter_tools():
            if isinstance(tool, EmbodiedToolGroup):
                tool.embodied_service = runtime.embodied_service

        register_all_routes(
            app,
            web_ch,
            runtime.embodied_service,
            get_config=lambda: (web_cfg["host"], web_cfg["port"]),
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

    # Find newest mtime in src/ vs dist/
    def _newest_mtime(directory: Path) -> float:
        return max((f.stat().st_mtime for f in directory.rglob("*") if f.is_file()), default=0)

    src_mtime = _newest_mtime(ui_src)
    dist_mtime = _newest_mtime(ui_dist) if ui_dist.is_dir() else 0

    if dist_mtime >= src_mtime:
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
