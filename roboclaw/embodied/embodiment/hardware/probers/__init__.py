"""Protocol-specific hardware probers."""
from __future__ import annotations

from typing import Protocol


class PortProber(Protocol):
    """Interface for protocol-specific port probing."""

    def probe(self, port_path: str, baudrate: int = 1_000_000, motor_ids: list[int] | None = None) -> list[int]:
        """Probe port, return responding motor IDs. Empty list = no match."""
        ...

    def read_positions(
        self, port_path: str, motor_ids: list[int], baudrate: int = 1_000_000,
    ) -> dict[int, int]:
        """Read current positions for given motor IDs."""
        ...


_REGISTRY: dict[str, type[PortProber]] = {}


def register_prober(protocol: str, cls: type[PortProber]) -> None:
    _REGISTRY[protocol] = cls


def get_prober(protocol: str) -> PortProber:
    if protocol not in _REGISTRY:
        raise ValueError(f"Unknown probe protocol: {protocol}")
    return _REGISTRY[protocol]()


# Eager import to trigger registration
from roboclaw.embodied.embodiment.hardware.probers import feetech as _feetech  # noqa: F401
from roboclaw.embodied.embodiment.hardware.probers import dynamixel as _dynamixel  # noqa: F401
