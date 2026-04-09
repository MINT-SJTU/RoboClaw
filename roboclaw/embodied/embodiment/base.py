"""EmbodimentSpec — base class for all static embodiment specifications."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbodimentSpec:
    """All static embodiment objects inherit from this base.

    A "static spec" describes the fixed characteristics of a hardware type
    (motor count, protocol, baudrate, register addresses) — not runtime
    state (connected, current position).
    """

    name: str  # Unique identifier, e.g. "so101", "inspire_rh56", "opencv"
    roles: tuple[str, ...] = ()  # e.g. ("follower", "leader") for paired devices
    device_patterns: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "linux": ("ttyACM*", "ttyUSB*"),
        "darwin": ("cu.usb*", "cu.usbserial*"),
    })

    def spec_name_for(self, role: str) -> str:
        """Construct full spec name for a given role (e.g. 'so101' + 'follower' → 'so101_follower')."""
        return f"{self.name}_{role}" if role else self.name
