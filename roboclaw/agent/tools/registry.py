"""Tool registry for dynamic tool management."""

import json
import uuid
from typing import Any

from loguru import logger

from roboclaw.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.

    Allows dynamic registration and execution of tools.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def iter_tools(self):
        """Iterate over all registered Tool instances."""
        return self._tools.values()

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]

    @staticmethod
    def _error_payload(
        *,
        message: str,
        retry_hint: str,
        trace_id: str | None = None,
        error_type: str | None = None,
    ) -> str:
        """Return a structured tool error as JSON text for model consumption."""
        payload: dict[str, Any] = {
            "tool_error": {
                "message": message,
                "retry_hint": retry_hint.strip(),
            }
        }
        if trace_id:
            payload["tool_error"]["trace_id"] = trace_id
        if error_type:
            payload["tool_error"]["error_type"] = error_type
        return json.dumps(payload, ensure_ascii=False)

    async def execute(self, name: str, params: dict[str, Any]) -> str | list:
        """Execute a tool by name with given parameters."""
        retry_hint = "Analyze the error above and try a different approach."

        tool = self._tools.get(name)
        if not tool:
            return self._error_payload(
                message=f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}",
                retry_hint=retry_hint,
                error_type="ToolNotFound",
            )

        try:
            # Attempt to cast parameters to match schema types
            params = tool.cast_params(params)

            # Validate parameters
            errors = tool.validate_params(params)
            if errors:
                return self._error_payload(
                    message=f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors),
                    retry_hint=retry_hint,
                    error_type="ToolValidationError",
                )
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return self._error_payload(
                    message=result,
                    retry_hint=retry_hint,
                    error_type="ToolReturnedError",
                )
            return result
        except Exception as e:
            trace_id = uuid.uuid4().hex[:12]
            logger.exception(
                "Tool execution failed trace_id={} tool={} error_type={}",
                trace_id,
                name,
                e.__class__.__name__,
            )
            return self._error_payload(
                message=(
                    f"Error executing {name} "
                    f"[trace_id={trace_id}, error_type={e.__class__.__name__}]: {str(e)}"
                ),
                retry_hint=retry_hint,
                trace_id=trace_id,
                error_type=e.__class__.__name__,
            )

    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
