"""Agent tools for embodied status and control."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from roboclaw.agent.tools.base import Tool
from roboclaw.embodied.execution.controller import EmbodiedExecutionController
from roboclaw.session.manager import Session

ProgressCallback = Callable[[str], Awaitable[None]]


class EmbodiedStatusTool(Tool):
    """Read-only embodied status tool exposed to the main agent."""

    def __init__(self, controller: EmbodiedExecutionController):
        self._controller = controller
        self._session: Session | None = None

    def set_context(self, session: Session) -> None:
        """Bind the current conversation session."""
        self._session = session

    @property
    def name(self) -> str:
        return "embodied_status"

    @property
    def description(self) -> str:
        return (
            "Inspect the current embodied setup, runtime state, calibration state, and available control surface. "
            "Use this first when setup choice or readiness is unclear."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "setup_id": {
                    "type": "string",
                    "description": "Optional explicit embodied setup id to inspect.",
                },
            },
        }

    async def execute(self, setup_id: str | None = None, **kwargs: Any) -> str:
        if self._session is None:
            return "Error: embodied_status has no active session context"
        snapshot = self._controller.build_agent_snapshot(self._session, setup_id=setup_id)
        return json.dumps(snapshot.to_dict(), ensure_ascii=False, sort_keys=True)


class EmbodiedControlTool(Tool):
    """Strong-constrained embodied control tool exposed to the main agent."""

    def __init__(self, controller: EmbodiedExecutionController):
        self._controller = controller
        self._session: Session | None = None
        self._on_progress: ProgressCallback | None = None

    def set_context(self, session: Session, on_progress: ProgressCallback | None = None) -> None:
        """Bind the current conversation session and optional progress callback."""
        self._session = session
        self._on_progress = on_progress

    @property
    def name(self) -> str:
        return "embodied_control"

    @property
    def description(self) -> str:
        return (
            "Execute one embodied action through the strong procedure pipeline. "
            "Supported actions: connect, calibrate, debug, reset, run_primitive."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["connect", "calibrate", "debug", "reset", "run_primitive"],
                    "description": "Embodied action to execute.",
                },
                "setup_id": {
                    "type": "string",
                    "description": "Optional explicit embodied setup id.",
                },
                "primitive_name": {
                    "type": "string",
                    "description": "Required when action is run_primitive.",
                },
                "primitive_args": {
                    "type": "object",
                    "description": "Optional primitive arguments for run_primitive.",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        setup_id: str | None = None,
        primitive_name: str | None = None,
        primitive_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        if self._session is None:
            return "Error: embodied_control has no active session context"
        result = await self._controller.execute_action(
            self._session,
            action=action,
            setup_id=setup_id,
            primitive_name=primitive_name,
            primitive_args=primitive_args,
            on_progress=self._on_progress,
        )
        return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
