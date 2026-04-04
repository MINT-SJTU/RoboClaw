"""EmbodimentSpec — base class for all static embodiment specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbodimentSpec:
    """All static embodiment objects inherit from this base.

    A "static spec" describes the fixed characteristics of a hardware type
    (motor count, protocol, baudrate, register addresses) — not runtime
    state (connected, current position).
    """

    name: str  # Unique identifier, e.g. "so101", "inspire_rh56", "opencv"
