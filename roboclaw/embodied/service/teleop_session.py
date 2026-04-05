"""TeleopSession — CLI teleoperation via OperationEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.embodied.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class TeleopSession:
    def __init__(self, parent: EmbodiedService):
        self._parent = parent

    async def teleoperate(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.adapters.cli import run_cli_session

        return await run_cli_session(self._parent, "teleoperate", manifest, kwargs, tty_handoff)
