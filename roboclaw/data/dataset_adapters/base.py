"""Dataset adapter contracts for curation and quality validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


CanonicalEpisode = dict[str, Any]


class DatasetAdapter(Protocol):
    """Read arbitrary dataset layouts as RoboClaw canonical episodes."""

    dataset_path: Path

    def list_episodes(self) -> list[int]:
        """Return available episode indices."""
        ...

    def load_episode(self, episode_index: int) -> CanonicalEpisode:
        """Return one episode in the canonical shape used by validators."""
        ...
