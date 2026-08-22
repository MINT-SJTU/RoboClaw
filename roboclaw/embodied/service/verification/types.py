"""Shared types for embodied verification backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Violation:
    """A user-facing verification failure or warning."""

    code: str
    message: str
    field: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Result returned by a verifier."""

    violations: tuple[Violation, ...] = ()
    warnings: tuple[Violation, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations

    def format_violations(self) -> str:
        return " · ".join(v.message for v in self.violations)


@dataclass(frozen=True)
class VerificationRequest:
    """Host-visible information available before launching LeRobot."""

    argv: Sequence[str]
    manifest: Any
    mode: str = "infer"
    checkpoint_path: str | Path | None = None
    dataset: Any | None = None
    num_episodes: int = 1
    episode_time_s: int = 60
    episode: int = 0
    fps: int = 30
    use_cameras: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
