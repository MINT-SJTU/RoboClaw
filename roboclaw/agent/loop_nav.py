"""Demo-only AgentLoop for the isolated simulation/navigation slice."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from roboclaw.agent.context_nav import NavigationDemoContextBuilder
from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.skills import BUILTIN_SKILLS_DIR
from roboclaw.agent.tools.cron import CronTool
from roboclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from roboclaw.agent.tools.message import MessageTool
from roboclaw.agent.tools.shell import ExecTool
from roboclaw.agent.tools.spawn import SpawnTool
from roboclaw.agent.tools.web import WebFetchTool, WebSearchTool
from roboclaw.embodied.demo_tools import register_demo_tools

if TYPE_CHECKING:
    from roboclaw.bus.queue import MessageBus
    from roboclaw.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from roboclaw.cron.service import CronService
    from roboclaw.providers.base import LLMProvider
    from roboclaw.session.manager import SessionManager


class NavigationDemoAgentLoop(AgentLoop):
    """Agent loop that only exposes demo navigation tools."""

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        tty_handoff: Any = None,
    ) -> None:
        super().__init__(
            bus=bus,
            provider=provider,
            workspace=workspace,
            model=model,
            max_iterations=max_iterations,
            context_window_tokens=context_window_tokens,
            web_search_config=web_search_config,
            web_proxy=web_proxy,
            exec_config=exec_config,
            cron_service=cron_service,
            restrict_to_workspace=restrict_to_workspace,
            session_manager=session_manager,
            mcp_servers=mcp_servers,
            channels_config=channels_config,
            tty_handoff=tty_handoff,
            embodied_service=None,
        )
        self.context = NavigationDemoContextBuilder(workspace)
        self.memory_consolidator._build_messages = self.context.build_messages

    def _register_default_tools(self) -> None:
        """Register base tools plus demo navigation tools, but not arm embodied tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read)
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            )
        )
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        register_demo_tools(self.tools)
