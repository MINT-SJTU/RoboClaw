#!/usr/bin/env python3
"""Standalone demo-only agent runner for simulation/navigation."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.markdown import Markdown

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}
DEMO_REQUIRED_MODULES = [
    "pydantic_settings",
    "loguru",
    "httpx",
    "tiktoken",
    "litellm",
    "json_repair",
]


def _missing_demo_modules() -> list[str]:
    return [name for name in DEMO_REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _import_runtime() -> tuple[object, type, type, object, object, object]:
    from roboclaw import __logo__
    from roboclaw.agent.loop_nav import NavigationDemoAgentLoop
    from roboclaw.bus.queue import MessageBus
    from roboclaw.config.loader import load_runtime_config
    from roboclaw.providers.factory import ProviderConfigurationError, build_provider
    from roboclaw.utils.helpers import sync_workspace_templates

    return (
        __logo__,
        NavigationDemoAgentLoop,
        MessageBus,
        load_runtime_config,
        ProviderConfigurationError,
        build_provider,
        sync_workspace_templates,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated simulation/navigation demo agent.",
    )
    parser.add_argument("-m", "--message", help="Single message to send to the demo agent.")
    parser.add_argument("-s", "--session", help="Session ID to resume; omit to start a new one.")
    parser.add_argument("-w", "--workspace", help="Workspace directory override.")
    parser.add_argument("-c", "--config", help="Path to config file.")
    parser.add_argument(
        "--markdown",
        dest="markdown",
        action="store_true",
        default=True,
        help="Render assistant output as Markdown.",
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="Print plain text instead of Markdown rendering.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show RoboClaw info/debug logs during the demo.",
    )
    return parser.parse_args()


def _render_response(content: str, *, markdown: bool, logo: str) -> None:
    console.print()
    console.print(f"[cyan]{logo} RoboClaw Demo[/cyan]")
    if markdown:
        console.print(Markdown(content or ""))
    else:
        console.print(content or "")
    console.print()


def _new_session_id() -> str:
    return f"cli:{uuid.uuid4().hex[:12]}"


def _should_exit(command: str) -> bool:
    normalized = re.sub(r"\s+", " ", command.strip().lower())
    if normalized in EXIT_COMMANDS:
        return True
    return bool(re.fullmatch(r"(?:ok|okay)[,\s]+(?:exit|quit)", normalized))


def _effective_provider_name(config) -> str:
    provider = config.agents.defaults.provider or "auto"
    model = config.agents.defaults.model or ""
    if provider == "auto" and "/" in model:
        provider = model.split("/", 1)[0]
    return provider


def _apply_demo_env_overrides(config) -> None:
    env_provider = os.environ.get("ROBOCLAW_DEMO_PROVIDER")
    env_model = os.environ.get("ROBOCLAW_DEMO_MODEL")
    env_api_key = os.environ.get("ROBOCLAW_DEMO_API_KEY")
    env_api_base = os.environ.get("ROBOCLAW_DEMO_API_BASE")
    env_workspace = os.environ.get("ROBOCLAW_DEMO_WORKSPACE")

    if env_workspace:
        config.agents.defaults.workspace = env_workspace
    if env_provider:
        config.agents.defaults.provider = env_provider
    if env_model:
        config.agents.defaults.model = env_model

    provider_name = _effective_provider_name(config)
    provider_cfg = getattr(config.providers, provider_name, None)
    if provider_cfg is None:
        return
    if env_api_key:
        provider_cfg.api_key = env_api_key
    if env_api_base:
        provider_cfg.api_base = env_api_base


def _ensure_demo_credentials(config) -> bool:
    provider_name = _effective_provider_name(config)
    provider_cfg = getattr(config.providers, provider_name, None)
    api_key = getattr(provider_cfg, "api_key", "") if provider_cfg is not None else ""
    api_base = getattr(provider_cfg, "api_base", None) if provider_cfg is not None else None

    if provider_name in {"ollama", "vllm"}:
        return True
    if provider_name == "custom" and api_base:
        return True
    if api_key:
        return True

    if not sys.stdin.isatty():
        console.print(
            f"[red]Error:[/red] Missing API key for provider '{provider_name}'. "
            "Set ROBOCLAW_DEMO_API_KEY in the environment before running the demo."
        )
        return False

    console.print(
        f"[yellow]Demo config does not store your API key.[/yellow] "
        f"Enter the key for provider '{provider_name}' to use it only for this run."
    )
    try:
        entered = getpass.getpass("Demo API key (not echoed, not saved): ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[red]Cancelled before API key entry.[/red]")
        return False
    if not entered:
        console.print("[red]Error:[/red] Empty API key.")
        return False
    if provider_cfg is not None:
        provider_cfg.api_key = entered
    return True


async def _run_once(loop, message: str, session_id: str) -> str:
    return await loop.process_direct(message, session_key=session_id)


async def _run_interactive(loop, *, session_id: str, markdown: bool, logo: str) -> None:
    console.print(f"{logo} Demo navigation mode\n")
    console.print(f"[dim]Session: {session_id}[/dim]")
    console.print("[dim]Type exit or Ctrl+C to quit.[/dim]\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            command = user_input.strip()
            if not command:
                continue
            if _should_exit(command):
                break
            response = await loop.process_direct(command, session_key=session_id)
            _render_response(response, markdown=markdown, logo=logo)
    except (KeyboardInterrupt, EOFError):
        console.print()
    finally:
        console.print(
            "[dim]Resume this session:[/dim] "
            f"[bold]python robotics/scripts/run_demo_agent.py --session {session_id}[/bold]"
        )


async def _cleanup_demo_runtime(loop) -> None:
    if loop is None:
        return
    try:
        simulation_tool = loop.tools.get("embodied_simulation")
    except Exception:
        return
    if simulation_tool is None:
        return
    try:
        result = await simulation_tool.execute(action="shutdown")
    except Exception as exc:
        console.print(f"[yellow]Demo cleanup warning:[/yellow] failed to stop simulation runtime: {exc}")
        return

    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("ok"):
            console.print("[dim]Demo cleanup:[/dim] simulation shutdown requested.")


def _configure_demo_logging(*, verbose: bool) -> None:
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO" if verbose else "WARNING",
        colorize=False,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )


async def _main_async(args: argparse.Namespace) -> int:
    missing = _missing_demo_modules()
    if missing:
        console.print("[red]Error:[/red] Demo runner dependencies are missing.")
        console.print("Install them in the current Python environment:")
        console.print(
            "[bold]python -m pip install "
            + " ".join(missing)
            + "[/bold]"
        )
        return 1

    _configure_demo_logging(verbose=args.verbose)

    (
        logo,
        NavigationDemoAgentLoop,
        MessageBus,
        load_runtime_config,
        ProviderConfigurationError,
        build_provider,
        sync_workspace_templates,
    ) = _import_runtime()

    config = load_runtime_config(args.config, args.workspace)
    _apply_demo_env_overrides(config)
    sync_workspace_templates(config.workspace_path)

    if not _ensure_demo_credentials(config):
        return 1

    try:
        provider = build_provider(config)
    except ProviderConfigurationError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        if exc.hint:
            console.print(exc.hint)
        return 1

    loop = NavigationDemoAgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    session_id = args.session or _new_session_id()
    try:
        if args.message:
            response = await _run_once(loop, args.message, session_id)
            _render_response(response, markdown=args.markdown, logo=logo)
        else:
            await _run_interactive(loop, session_id=session_id, markdown=args.markdown, logo=logo)
    finally:
        await _cleanup_demo_runtime(loop)
        await loop.close_mcp()
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        console.print("\n[dim]Demo runner interrupted.[/dim]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
