"""Demo-only context builder for simulation navigation."""

from __future__ import annotations

from pathlib import Path

from roboclaw.agent.context import ContextBuilder
from roboclaw.agent.demo_navigation_prompt import DEMO_NAVIGATION_PROMPT


class NavigationDemoContextBuilder(ContextBuilder):
    """Context builder that appends demo navigation guidance to the system prompt."""

    def __init__(self, workspace: Path, *, extra_system_prompt: str | None = None) -> None:
        super().__init__(workspace)
        self._extra_system_prompt = extra_system_prompt or DEMO_NAVIGATION_PROMPT

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        base_prompt = super().build_system_prompt(skill_names)
        if not self._extra_system_prompt:
            return base_prompt
        return "\n\n---\n\n".join(
            [base_prompt, f"# Extra System Guidance\n\n{self._extra_system_prompt}"]
        )
