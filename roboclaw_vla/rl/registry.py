"""RLinf registry injection point for RoboClaw VLA models.

This module intentionally keeps the integration thin. Importing it must be
safe during EVO_Train preflight, while real model registration happens only
when RLinf is available in the remote training image.
"""

from __future__ import annotations

from typing import Any


def register() -> bool:
    """Register RoboClaw model adapters with RLinf when the API is present.

    This follows the RLinf/Dexbotic model registry entrypoint without making
    RoboClaw import RLinf at normal application startup.
    """

    try:
        from rlinf.models import register_model  # type: ignore
    except Exception:
        return False

    register_model("roboclaw-placeholder", _PlaceholderPolicy)
    return True


def register_all() -> bool:
    return register()


class _PlaceholderPolicy:
    """Placeholder policy used only to prove the registry hook is importable."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "roboclaw-placeholder is a registry sentinel. Provide a real RoboClaw "
            "VLA policy adapter before launching full RLinf training."
        )
