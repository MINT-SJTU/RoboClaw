from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Interface:
    """Base class for all hardware interfaces."""

    interface_type: str  # "serial" / "video" / "can"

    @property
    def address(self) -> str:
        """Canonical address used for persistence, equality, and guard keying."""
        raise NotImplementedError

    @property
    def stable_id(self) -> str:
        """Most stable identifier for persistence and Guard keying."""
        raise NotImplementedError

    @property
    def runtime_address(self) -> str:
        """Current endpoint used to talk to the device."""
        return self.address

    @property
    def exists(self) -> bool:
        """Whether the physical device is currently present."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Interface:
        raise NotImplementedError
