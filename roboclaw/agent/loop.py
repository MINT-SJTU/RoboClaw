"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from roboclaw.agent.context import ContextBuilder
from roboclaw.agent.memory import MemoryStore
from roboclaw.agent.subagent import SubagentManager
from roboclaw.agent.tools.cron import CronTool
from roboclaw.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from roboclaw.agent.tools.message import MessageTool
from roboclaw.agent.tools.registry import ToolRegistry
from roboclaw.agent.tools.shell import ExecTool
from roboclaw.agent.tools.spawn import SpawnTool
from roboclaw.agent.tools.web import WebFetchTool, WebSearchTool
from roboclaw.bus.events import InboundMessage, OutboundMessage
from roboclaw.embodied.builtins import list_ros2_profiles
from roboclaw.embodied.catalog import build_catalog
from roboclaw.embodied.execution.controller import EmbodiedExecutionController
from roboclaw.embodied.execution.tools import EmbodiedControlTool, EmbodiedStatusTool
from roboclaw.embodied.execution.orchestration.runtime import RuntimeManager
from roboclaw.embodied.localization import choose_language
from roboclaw.bus.queue import MessageBus
from roboclaw.embodied.onboarding import OnboardingController
from roboclaw.embodied.onboarding.model import PREFERRED_LANGUAGE_KEY, OnboardingIntent
from roboclaw.providers.base import LLMProvider
from roboclaw.session.manager import Session, SessionManager
from roboclaw.utils.helpers import strip_code_fences

if TYPE_CHECKING:
    from roboclaw.config.schema import ChannelsConfig, ExecToolConfig
    from roboclaw.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from roboclaw.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.embodied_runtime = RuntimeManager()
        self.embodied_execution = EmbodiedExecutionController(
            workspace=workspace,
            tools=self.tools,
            runtime_manager=self.embodied_runtime,
        )
        self.onboarding = OnboardingController(
            workspace=workspace,
            tools=self.tools,
            intent_parser=self._parse_onboarding_intent,
            calibration_starter=self._start_onboarding_calibration,
        )
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        self.tools.register(EmbodiedStatusTool(self.embodied_execution))
        self.tools.register(EmbodiedControlTool(self.embodied_execution))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from roboclaw.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    def _set_embodied_tool_context(
        self,
        session: Session,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """Bind the current session/progress callback to embodied tools."""
        if tool := self.tools.get("embodied_status"):
            if hasattr(tool, "set_context"):
                tool.set_context(session)
        if tool := self.tools.get("embodied_control"):
            if hasattr(tool, "set_context"):
                tool.set_context(session, on_progress)

    def _embodied_runtime_context(self, session: Session, catalog: Any | None = None) -> str:
        """Render the current embodied snapshot into runtime metadata."""
        snapshot = self.embodied_execution.build_agent_snapshot(session, catalog=catalog)
        language = choose_language(session.metadata.get(PREFERRED_LANGUAGE_KEY))
        return (
            f"Preferred Response Language: {language}\n"
            + "[Embodied Context]\n"
            + json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True)
        )

    async def _start_onboarding_calibration(
        self,
        *,
        session: Session,
        action: str,
        setup_id: str,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> Any:
        return await self.embodied_execution.execute_action(
            session,
            action=action,
            setup_id=setup_id,
            on_progress=on_progress,
        )

    async def _parse_onboarding_intent(
        self,
        session: Session,
        state: Any,
        content: str,
    ) -> OnboardingIntent | None:
        supported_robots: list[str] = []
        for robot in self.onboarding.catalog.robots.list():
            if robot.id not in supported_robots:
                supported_robots.append(robot.id)

        for profile in list_ros2_profiles():
            if profile.robot_id not in supported_robots:
                supported_robots.append(profile.robot_id)
        example_robot = supported_robots[0] if supported_robots else "robot"

        prompt = (
            "You extract embodied onboarding intent as JSON.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Use this schema:\n"
            "{"
            '"robot_ids": string[], "connected": true|false|null, "serial_path": string|null, '
            '"ros2_install_profile": string|null, "ros2_state": true|false|null, '
            '"ros2_install_requested": boolean, "ros2_step_advance": boolean, '
            '"calibration_requested": boolean, "preferred_language": "en"|"zh"|null'
            "}.\n"
            "Do not infer facts that the user did not imply.\n"
            f"Supported robot models (canonical IDs): {', '.join(supported_robots)}.\n"
            "When the user mentions a robot, always normalize to the closest canonical ID from this list.\n"
            f"For example: '{example_robot.upper()}', '{example_robot.replace('-', '_')}', "
            f"'{example_robot.title()}', '{example_robot.replace('-', ' ')}' should all map to '{example_robot}'.\n"
            "If the user's robot does not match any supported model, return the user's text as-is in robot_ids.\n"
            "Examples:\n"
            '- "帮我标定" -> {"calibration_requested": true, "preferred_language": "zh"}\n'
            '- "已经接好了" -> {"connected": true, "preferred_language": "zh"}\n'
            '- "not connected yet" -> {"connected": false, "preferred_language": "en"}\n'
            '- "这一步做完了，继续" -> {"ros2_step_advance": true, "preferred_language": "zh"}'
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "session_key": session.key,
                        "current_stage": getattr(state, "stage", None).value if getattr(state, "stage", None) else None,
                        "missing_facts": list(getattr(state, "missing_facts", [])),
                        "robots": list(getattr(state, "robot_attachments", [])),
                        "sensors": list(getattr(state, "sensor_attachments", [])),
                        "user_message": content,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = await self.provider.chat(
            messages=messages,
            tools=None,
            model=self.model,
            max_tokens=300,
            temperature=0.0,
            reasoning_effort="low",
        )
        raw = self._strip_think(response.content) or ""
        if not raw:
            return None
        raw = strip_code_fences(raw)
        try:
            payload = json.loads(raw)
        except Exception:
            logger.warning("Failed to decode onboarding intent JSON: {}", raw)
            return None
        if not isinstance(payload, dict):
            return None
        robot_ids = payload.get("robot_ids")
        return OnboardingIntent(
            robot_ids=tuple(item for item in robot_ids if isinstance(item, str)) if isinstance(robot_ids, list) else (),
            connected=payload.get("connected") if isinstance(payload.get("connected"), bool) else None,
            serial_path=payload.get("serial_path") if isinstance(payload.get("serial_path"), str) else None,
            ros2_install_profile=payload.get("ros2_install_profile") if isinstance(payload.get("ros2_install_profile"), str) else None,
            ros2_state=payload.get("ros2_state") if isinstance(payload.get("ros2_state"), bool) else None,
            ros2_install_requested=bool(payload.get("ros2_install_requested")),
            ros2_step_advance=bool(payload.get("ros2_step_advance")),
            calibration_requested=bool(payload.get("calibration_requested")),
            preferred_language=payload.get("preferred_language") if isinstance(payload.get("preferred_language"), str) else None,
        )

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            self._set_embodied_tool_context(session)
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                channel=channel,
                chat_id=chat_id,
                extra_runtime_context=self._embodied_runtime_context(session),
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🤖 RoboClaw commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        self._set_embodied_tool_context(session, on_progress or _bus_progress)

        if self.embodied_execution.has_pending_calibration(session):
            response = await self.embodied_execution.handle_pending_calibration_message(
                msg,
                session,
                on_progress=on_progress or _bus_progress,
            )
            self.sessions.save(session)
            return response

        if self.onboarding.has_active_onboarding(session) or self.onboarding.should_handle_setup_edit(session, msg.content):
            response = await self.onboarding.handle_message(
                msg,
                session,
                on_progress=on_progress or _bus_progress,
            )
            self.sessions.save(session)
            return response

        catalog = build_catalog(self.workspace)
        snapshot = self.embodied_execution.build_agent_snapshot(session, catalog=catalog)
        if (
            snapshot.selected_setup_id is None
            and not snapshot.candidates
            and self.embodied_execution.looks_like_embodied_request(msg.content, catalog=catalog)
        ):
            response = await self.onboarding.handle_message(
                msg,
                session,
                on_progress=on_progress or _bus_progress,
            )
            self.sessions.save(session)
            return response

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            extra_runtime_context=self._embodied_runtime_context(session, catalog=catalog),
        )

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
