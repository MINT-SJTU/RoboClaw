"""FastAPI server for the RoboClaw web chat UI.

Runs the full gateway runtime (AgentLoop, CronService, HeartbeatService,
ChannelManager) so the web UI has feature parity with ``roboclaw gateway``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.http.runtime import WebRuntime

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from roboclaw.channels.web import WebChannel
from roboclaw.config.loader import get_config_path, load_config, load_runtime_config, save_config
from roboclaw.providers.factory import ProviderConfigurationError, build_provider
from roboclaw.providers.registry import PROVIDERS, find_by_name
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
            "keywords": list(spec.keywords),
            "default_model": spec.default_model,
            "model_presets": list(spec.model_presets),
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
            "keywords": [],
            "default_model": find_by_name("custom").default_model if find_by_name("custom") else "",
            "model_presets": list(find_by_name("custom").model_presets) if find_by_name("custom") else [],
            "configured": False,
            "api_base": "",
            "has_api_key": False,
            "masked_api_key": "",
            "extra_headers": {},
        },
        "providers": providers,
    }


_VERSIONED_PATH = re.compile(r"/v\d+(?:beta\d*)?/?$", re.IGNORECASE)


async def _discover_provider_models(
    api_base: str,
    api_key: str | None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[list[str], str]:
    if not api_base:
        return [], ""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    base = api_base.rstrip("/")
    candidates = [base + "/models"]
    # If user forgot the /vN segment, fall back to /v1/models — the OpenAI-compatible
    # convention that virtually every LLM gateway speaks.
    if not _VERSIONED_PATH.search(base):
        candidates.append(base + "/v1/models")

    last_error = ""
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in candidates:
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.warning("Failed to auto-discover models from {}: {}", url, exc)
                continue

            data = payload.get("data", []) if isinstance(payload, dict) else payload
            if not isinstance(data, list):
                return [], "Model endpoint returned an unsupported response shape."
            models: list[str] = []
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
                elif isinstance(item, str) and item.strip():
                    models.append(item.strip())
            return sorted(set(models), key=str.lower), ""

    return [], last_error


async def _discover_custom_model(api_base: str, api_key: str | None) -> str | None:
    models, _error = await _discover_provider_models(api_base, api_key)
    return models[0] if models else None


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

    @app.post("/api/system/provider-models")
    async def provider_models(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return await _handle_provider_models(payload)

    @app.post("/api/system/provider-test")
    async def provider_test(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return await _handle_provider_test(payload)

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


async def _handle_save_provider(payload: dict[str, Any], runtime: WebRuntime) -> dict[str, Any]:
    """Apply provider config changes, swap provider atomically, refresh agent."""
    config = load_config(get_config_path())
    provider_name, section = _resolve_provider_section(config, payload)
    _apply_credential_payload(payload, section)
    _apply_extra_headers(payload, section)
    config.agents.defaults.provider = provider_name
    await _resolve_default_model(config, payload, section, allow_discovery=True)

    try:
        new_provider = build_provider(config)
    except ProviderConfigurationError as exc:
        raise _provider_config_http_error(exc) from exc

    save_config(config, get_config_path())
    runtime.swap_provider(new_provider, config)

    return {"status": "ok", **_provider_status_payload(config)}


async def _handle_provider_models(payload: dict[str, Any]) -> dict[str, Any]:
    """Discover model ids for the selected provider without saving settings."""
    config = load_config(get_config_path())
    provider_name = str(payload.get("provider") or config.agents.defaults.provider or "custom")
    provider_config = getattr(config.providers, provider_name, None)
    if provider_config is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

    api_base = payload.get("api_base")
    if isinstance(api_base, str) and api_base.strip():
        resolved_api_base = api_base.strip()
    else:
        resolved_api_base = provider_config.api_base or ""

    if not resolved_api_base:
        spec = find_by_name(provider_name)
        if spec and spec.default_api_base:
            resolved_api_base = spec.default_api_base

    if not resolved_api_base:
        raise HTTPException(status_code=400, detail="Model discovery requires an API base URL.")

    api_key = payload.get("api_key")
    resolved_api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else provider_config.api_key
    extra_headers = payload.get("extra_headers")
    resolved_headers = extra_headers if isinstance(extra_headers, dict) else provider_config.extra_headers
    models, error = await _discover_provider_models(
        resolved_api_base,
        resolved_api_key or None,
        resolved_headers,
    )
    return {"models": models, "error": error}


async def _handle_provider_test(payload: dict[str, Any]) -> dict[str, Any]:
    """Send a minimal chat request with unsaved provider settings."""
    config = load_config(get_config_path())
    provider_name, section = _resolve_provider_section(config, payload)
    _apply_credential_payload(payload, section)
    _apply_extra_headers(payload, section)
    config.agents.defaults.provider = provider_name
    await _resolve_default_model(config, payload, section, allow_discovery=False)

    try:
        provider = build_provider(config)
    except ProviderConfigurationError as exc:
        raise _provider_config_http_error(exc) from exc

    test_input = payload.get("input")
    if not isinstance(test_input, str) or not test_input.strip():
        test_input = "Reply with OK if the RoboClaw AI provider test reaches you."

    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": test_input.strip()}],
        model=config.agents.defaults.model,
    )
    if response.finish_reason == "error":
        return {
            "ok": False,
            "finish_reason": response.finish_reason,
            "error": response.content or "Provider returned an error.",
        }
    return {
        "ok": True,
        "finish_reason": response.finish_reason,
        "content": response.content or "",
    }


def _resolve_provider_section(config: Any, payload: dict[str, Any]) -> tuple[str, Any]:
    """Resolve the providers.<name> section from payload, raising 400 if unknown."""
    provider_name = str(payload.get("provider") or config.agents.defaults.provider or "custom")
    section = getattr(config.providers, provider_name, None)
    if section is None:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")
    return provider_name, section


def _apply_credential_payload(payload: dict[str, Any], section: Any) -> None:
    """Apply api_key / api_base / clear_api_key fields onto the provider section."""
    if payload.get("clear_api_key"):
        section.api_key = ""
    api_key = payload.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        section.api_key = api_key.strip()
    api_base = payload.get("api_base")
    if isinstance(api_base, str):
        section.api_base = api_base.strip() or None


async def _resolve_default_model(
    config: Any,
    payload: dict[str, Any],
    section: Any,
    *,
    allow_discovery: bool,
) -> None:
    """Pick config.agents.defaults.model, in priority order:
       explicit payload > spec.default_model > (optional) /models discovery.
    """
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        config.agents.defaults.model = model.strip()
        return

    provider_name = config.agents.defaults.provider
    spec = find_by_name(provider_name) if provider_name else None
    if spec and spec.default_model:
        config.agents.defaults.model = spec.default_model
        return

    if allow_discovery and section.api_base:
        discovered = await _discover_custom_model(section.api_base, section.api_key or None)
        if discovered:
            config.agents.defaults.model = discovered


def _apply_extra_headers(payload: dict[str, Any], section: Any) -> None:
    """Parse and apply extra_headers from payload. Raises HTTP 400 on bad JSON."""
    extra_headers = payload.get("extra_headers")
    if isinstance(extra_headers, str):
        try:
            extra_headers = json.loads(extra_headers) if extra_headers.strip() else {}
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="extra_headers must be valid JSON.",
            ) from exc
    if isinstance(extra_headers, dict):
        section.extra_headers = extra_headers or None


def _provider_config_http_error(exc: ProviderConfigurationError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "code": "provider_configuration_error",
            "message": str(exc),
            "hint": exc.hint,
        },
    )


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

    app = FastAPI(title="RoboClaw")

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
    logger.info("Starting RoboClaw at http://{}:{}", app.state.web_host, app.state.web_port)
    uvicorn.run(app, host=app.state.web_host, port=app.state.web_port, log_level="info")
