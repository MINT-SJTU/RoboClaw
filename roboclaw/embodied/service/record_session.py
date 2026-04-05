"""RecordSession — CLI dataset recording via OperationEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roboclaw.embodied.engine.helpers import _validate_dataset_name

if TYPE_CHECKING:
    from roboclaw.embodied.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class RecordSession:
    def __init__(self, parent: EmbodiedService):
        self._parent = parent

    async def record(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        from roboclaw.embodied.adapters.cli import run_cli_session

        dataset_name = kwargs.get("dataset_name")
        if dataset_name:
            error = _validate_dataset_name(dataset_name)
            if error:
                return error
        return await run_cli_session(self._parent, "record", manifest, kwargs, tty_handoff)
