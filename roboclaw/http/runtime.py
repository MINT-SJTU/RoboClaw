"""WebRuntime — owns all background services and their lifecycle."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from loguru import logger

from roboclaw.agent.loop import AgentLoop
from roboclaw.bus.queue import MessageBus
from roboclaw.channels.manager import ChannelManager
from roboclaw.channels.web import WebChannel
from roboclaw.config.paths import get_cron_dir
from roboclaw.cron.service import CronService
from roboclaw.cron.types import CronJob
from roboclaw.heartbeat.service import HeartbeatService
from roboclaw.providers.base import GenerationSettings
from roboclaw.providers.factory import ProviderConfigurationError, UnconfiguredProvider, build_provider
from roboclaw.session.manager import SessionManager


class WebRuntime:
    """Holds all services, manages their lifecycle."""

    def __init__(self) -> None:
        self.bus: MessageBus | None = None
        self.sessions: SessionManager | None = None
        self.provider: Any = None
        self.cron: CronService | None = None
        self.agent: AgentLoop | None = None
        self.heartbeat: HeartbeatService | None = None
        self.channel_manager: ChannelManager | None = None
        self.hw_monitor: Any | None = None
        self.embodied_service: Any | None = None
        self._tasks: list[asyncio.Task] = []

    @classmethod
    def build(cls, config: Any, *, host: str | None = None, port: int | None = None) -> WebRuntime:
        """Construct all services from config."""
        rt = cls()

        # Shared infra
        rt.bus = MessageBus()
        rt.sessions = SessionManager(config.workspace_path)

        # Provider (graceful fallback for unconfigured state)
        try:
            rt.provider = build_provider(config)
        except ProviderConfigurationError as exc:
            logger.warning("Provider not configured at startup: {}. Configure via Settings.", exc)
            rt.provider = UnconfiguredProvider(str(exc))

        # Cron service
        cron_store_path = get_cron_dir() / "jobs.json"
        rt.cron = CronService(cron_store_path)

        # Agent loop
        rt.agent = AgentLoop(
            bus=rt.bus,
            provider=rt.provider,
            workspace=config.workspace_path,
            model=config.agents.defaults.model,
            max_iterations=config.agents.defaults.max_tool_iterations,
            context_window_tokens=config.agents.defaults.context_window_tokens,
            web_search_config=config.tools.web.search,
            web_proxy=config.tools.web.proxy or None,
            exec_config=config.tools.exec,
            cron_service=rt.cron,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            session_manager=rt.sessions,
            mcp_servers=config.tools.mcp_servers,
            channels_config=config.channels,
        )
        rt._refresh_agent_defaults(config)

        # Cron callback
        rt.cron.on_job = lambda job: rt._on_cron_job(job)

        # Force-enable web channel in config
        web_defaults = WebChannel.default_config()
        web_cfg = {**web_defaults, "enabled": True, "host": "0.0.0.0"}
        if host is not None:
            web_cfg["host"] = host
        if port is not None:
            web_cfg["port"] = port
        config.channels.web = web_cfg

        # Heartbeat service
        hb_cfg = config.gateway.heartbeat
        rt.heartbeat = HeartbeatService(
            workspace=config.workspace_path,
            provider=rt.provider,
            model=rt.agent.model,
            on_execute=rt._on_heartbeat_execute,
            on_notify=rt._on_heartbeat_notify,
            interval_s=hb_cfg.interval_s,
            enabled=hb_cfg.enabled,
        )

        # Channel manager
        rt.channel_manager = ChannelManager(config, rt.bus)

        # Inject session manager into WebChannel
        web_ch = rt.channel_manager.get_channel("web")
        if web_ch is not None:
            web_ch.sessions = rt.sessions

        # Hardware monitor + EmbodiedService (only if web channel available)
        if web_ch is not None:
            rt._build_embodied(web_ch)

        return rt

    def _build_embodied(self, web_ch: Any) -> None:
        """Build HardwareMonitor and EmbodiedService with Board."""
        from roboclaw.embodied.board import Board, WS_TYPES
        from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
        from roboclaw.embodied.embodiment.manifest import Manifest
        from roboclaw.embodied.service import EmbodiedService

        board = Board()

        async def _board_subscriber(channel: str, data: dict[str, Any]) -> None:
            ws_type = WS_TYPES.get(channel)
            if ws_type:
                await web_ch.broadcast({"type": ws_type, **data})

        board.on(None, _board_subscriber)

        manifest = Manifest(board=board)
        self.hw_monitor = HardwareMonitor(board=board, manifest=manifest)
        self.embodied_service = EmbodiedService(
            hardware_monitor=self.hw_monitor,
            board=board,
            manifest=manifest,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch all background tasks."""
        self._tasks.append(asyncio.create_task(self.agent.run(), name="roboclaw-agent"))
        self._tasks.append(asyncio.create_task(self.channel_manager.start_all(), name="roboclaw-channels"))
        self._tasks.append(asyncio.create_task(self.cron.start(), name="roboclaw-cron"))
        self._tasks.append(asyncio.create_task(self.heartbeat.start(), name="roboclaw-heartbeat"))
        if self.hw_monitor is not None:
            self._tasks.append(asyncio.create_task(self.hw_monitor.run(), name="roboclaw-hw-monitor"))

    async def shutdown(self) -> None:
        """Gracefully stop all services."""
        if self.embodied_service is not None:
            await self.embodied_service.shutdown()
        if self.hw_monitor is not None:
            self.hw_monitor.stop()
        self.agent.stop()
        await self.channel_manager.stop_all()
        self.heartbeat.stop()
        self.cron.stop()
        for task in self._tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self.agent.close_mcp()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def swap_provider(self, new_provider: Any, config: Any) -> None:
        """Atomically swap provider and refresh agent defaults."""
        self.provider = new_provider
        self.agent.provider = new_provider
        self._refresh_agent_defaults(config)

    def _refresh_agent_defaults(self, config: Any) -> None:
        self.agent.model = config.agents.defaults.model
        self.agent.provider.generation = GenerationSettings(
            temperature=config.agents.defaults.temperature,
            max_tokens=config.agents.defaults.max_tokens,
            reasoning_effort=config.agents.defaults.reasoning_effort,
        )
        self.agent.memory_consolidator.model = config.agents.defaults.model
        self.agent.subagents.model = config.agents.defaults.model

    def _pick_heartbeat_target(self) -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(self.channel_manager.enabled_channels)
        for item in self.sessions.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        return "cli", "direct"

    async def _on_heartbeat_execute(self, tasks: str) -> str:
        channel, chat_id = self._pick_heartbeat_target()

        async def _silent(*_args: Any, **_kwargs: Any) -> None:
            pass

        return await self.agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

    async def _on_heartbeat_notify(self, response: str) -> None:
        from roboclaw.bus.events import OutboundMessage
        channel, chat_id = self._pick_heartbeat_target()
        if channel == "cli":
            return
        await self.bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    async def _on_cron_job(self, job: CronJob) -> str | None:
        from roboclaw.agent.tools.cron import CronTool
        from roboclaw.agent.tools.message import MessageTool
        from roboclaw.utils.evaluator import evaluate_response

        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        cron_tool = self.agent.tools.get("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)
        try:
            response = await self.agent.process_direct(
                reminder_note,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
        finally:
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)

        message_tool = self.agent.tools.get("message")
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return response

        if not (job.payload.deliver and job.payload.to and response):
            return response

        should_notify = await evaluate_response(response, job.payload.message, self.provider, self.agent.model)
        if should_notify:
            from roboclaw.bus.events import OutboundMessage
            await self.bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response,
            ))
        return response
