"""Persistence for cross-provider training job metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from roboclaw.embodied.command import logs_dir
from roboclaw.embodied.training.types import TrainingJobRecord

logger = logging.getLogger(__name__)


class TrainingJobStore:
    """Persist lightweight job metadata alongside embodied job logs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or logs_dir()) / "training_meta"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def save(self, record: TrainingJobRecord) -> None:
        self.path(record.job_id).write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, job_id: str) -> TrainingJobRecord | None:
        path = self.path(job_id)
        if not path.exists():
            return None
        return TrainingJobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[TrainingJobRecord]:
        records: list[TrainingJobRecord] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                record = TrainingJobRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping corrupt training metadata record %s: %s", path.name, exc)
                continue
            records.append(record)
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records
