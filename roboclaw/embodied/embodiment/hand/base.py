"""HandSpec — static specification for dexterous hand types."""

from __future__ import annotations

from dataclasses import dataclass

from roboclaw.embodied.embodiment.base import EmbodimentSpec


@dataclass(frozen=True)
class HandSpec(EmbodimentSpec):
    """Static specification for a dexterous hand model.

    Captures the fixed hardware characteristics: finger count, labels,
    communication parameters, and probe settings. The controller class
    is referenced by module path for lazy import.
    """

    finger_labels: tuple[str, ...] = ()
    num_fingers: int = 0
    baudrate: int = 0
    default_slave_id: int = 1
    open_positions: tuple[int, ...] = ()
    close_positions: tuple[int, ...] = ()
    probe_register: int = 0
    probe_register_count: int = 0
    probe_candidates: tuple[int, ...] | None = None
    controller_module: str = ""
    controller_class: str = ""
