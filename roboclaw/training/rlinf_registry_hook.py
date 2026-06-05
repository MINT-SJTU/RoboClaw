"""RLinf ``RLINF_EXT_MODULE`` hook for RoboClaw-owned VLA adapters."""

from __future__ import annotations

from roboclaw.training.rlinf_catalog import register_all_rlinf_models


def register() -> None:
    """Called once per RLinf worker process."""

    register_all_rlinf_models()
