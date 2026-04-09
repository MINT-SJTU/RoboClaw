"""DoctorService - embodied environment health checks."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class DoctorService:
    def __init__(self, parent: EmbodiedService):
        self._parent = parent

    async def check(
        self,
        manifest: Manifest,
        kwargs: dict[str, Any],
        tty_handoff: Any,
    ) -> str:
        import asyncio

        environment = await asyncio.to_thread(self._check_environment_sync)
        manifest_snapshot = manifest.snapshot
        hardware_status = self._parent.get_hardware_status(manifest)
        result = {
            "environment": environment,
            "manifest": manifest_snapshot,
            "hardware_status": hardware_status,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)

    @staticmethod
    def _check_environment_sync() -> dict[str, Any]:
        """Check LeRobot and SDK availability (may block on first import)."""
        env: dict[str, Any] = {
            "lerobot_installed": False,
            "lerobot_version": None,
            "feetech_sdk": False,
            "dynamixel_sdk": False,
        }
        try:
            import lerobot
            env["lerobot_installed"] = True
            env["lerobot_version"] = getattr(lerobot, "__version__", "unknown")
        except (ImportError, OSError):
            pass
        try:
            import scservo_sdk  # noqa: F401
            env["feetech_sdk"] = True
        except (ImportError, OSError):
            pass
        try:
            import dynamixel_sdk  # noqa: F401
            env["dynamixel_sdk"] = True
        except (ImportError, OSError):
            pass
        return env
