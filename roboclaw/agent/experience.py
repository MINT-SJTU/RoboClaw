"""Structured experience storage and retrieval for agent adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from roboclaw.utils.helpers import ensure_dir

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:-]*")


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _normalize_key(value: str | None) -> str:
    return _normalize_text(value).lower()


def _tokenize(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for match in _TOKEN_RE.finditer(value.lower()):
            token = match.group(0)
            if len(token) >= 2:
                tokens.add(token)
    return tokens


@dataclass(frozen=True)
class ExperienceRecord:
    timestamp: str
    task_type: str
    summary: str
    outcome: str
    lesson: str = ""
    dataset: str = ""
    replay_datasets: str = ""
    policy: str = ""
    provider: str = ""
    job_id: str = ""
    source: str = ""
    error: str = ""
    checkpoint_path: str = ""
    dataset_path: str = ""
    task_name: str = ""

    @classmethod
    def create(
        cls,
        *,
        task_type: str,
        summary: str,
        outcome: str,
        lesson: str = "",
        dataset: str = "",
        replay_datasets: str = "",
        policy: str = "",
        provider: str = "",
        job_id: str = "",
        source: str = "",
        error: str = "",
        checkpoint_path: str = "",
        dataset_path: str = "",
        task_name: str = "",
    ) -> "ExperienceRecord":
        return cls(
            timestamp=datetime.now(UTC).isoformat(),
            task_type=_normalize_text(task_type),
            summary=_normalize_text(summary),
            outcome=_normalize_text(outcome),
            lesson=_normalize_text(lesson),
            dataset=_normalize_text(dataset),
            replay_datasets=_normalize_text(replay_datasets),
            policy=_normalize_text(policy),
            provider=_normalize_text(provider),
            job_id=_normalize_text(job_id),
            source=_normalize_text(source),
            error=_normalize_text(error),
            checkpoint_path=_normalize_text(checkpoint_path),
            dataset_path=_normalize_text(dataset_path),
            task_name=_normalize_text(task_name),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceRecord":
        return cls(
            timestamp=_normalize_text(str(payload.get("timestamp") or "")),
            task_type=_normalize_text(str(payload.get("task_type") or "")),
            summary=_normalize_text(str(payload.get("summary") or "")),
            outcome=_normalize_text(str(payload.get("outcome") or "")),
            lesson=_normalize_text(str(payload.get("lesson") or "")),
            dataset=_normalize_text(str(payload.get("dataset") or "")),
            replay_datasets=_normalize_text(str(payload.get("replay_datasets") or "")),
            policy=_normalize_text(str(payload.get("policy") or "")),
            provider=_normalize_text(str(payload.get("provider") or "")),
            job_id=_normalize_text(str(payload.get("job_id") or "")),
            source=_normalize_text(str(payload.get("source") or "")),
            error=_normalize_text(str(payload.get("error") or "")),
            checkpoint_path=_normalize_text(str(payload.get("checkpoint_path") or "")),
            dataset_path=_normalize_text(str(payload.get("dataset_path") or "")),
            task_name=_normalize_text(str(payload.get("task_name") or "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        parts = (
            _normalize_key(self.task_type),
            _normalize_key(self.summary),
            _normalize_key(self.outcome),
            _normalize_key(self.dataset),
            _normalize_key(self.replay_datasets),
            _normalize_key(self.policy),
            _normalize_key(self.provider),
            _normalize_key(self.job_id),
            _normalize_key(self.error),
            _normalize_key(self.lesson),
        )
        return "|".join(parts)


class ExperienceStore:
    """Append-only JSONL store plus lightweight experience retrieval."""

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.experience_file = self.memory_dir / "EXPERIENCES.jsonl"

    def read_all(self) -> list[ExperienceRecord]:
        if not self.experience_file.exists():
            return []
        records: list[ExperienceRecord] = []
        for raw_line in self.experience_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(ExperienceRecord.from_dict(payload))
        return records

    def append(self, record: ExperienceRecord) -> bool:
        records = self.read_all()
        fingerprint = record.fingerprint()
        if any(existing.fingerprint() == fingerprint for existing in records):
            return False
        with self.experience_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return True

    def search(
        self,
        *,
        query: str = "",
        task_type: str = "",
        dataset: str = "",
        policy: str = "",
        provider: str = "",
        outcomes: frozenset[str] | None = None,
        limit: int = 3,
    ) -> list[ExperienceRecord]:
        query_tokens = _tokenize(query, dataset, policy, provider, task_type)
        scored: list[tuple[int, ExperienceRecord]] = []
        for record in self.read_all():
            if outcomes is not None and record.outcome not in outcomes:
                continue
            score = 0
            if task_type and _normalize_key(record.task_type) == _normalize_key(task_type):
                score += 6
            if dataset and _normalize_key(record.dataset) == _normalize_key(dataset):
                score += 8
            if policy and _normalize_key(record.policy) == _normalize_key(policy):
                score += 6
            if provider and _normalize_key(record.provider) == _normalize_key(provider):
                score += 4

            record_tokens = _tokenize(
                record.summary,
                record.lesson,
                record.dataset,
                record.policy,
                record.provider,
                record.task_name,
                record.error,
            )
            score += len(query_tokens & record_tokens)
            if score <= 0:
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        return [record for _, record in scored[:limit]]

    def build_context(
        self,
        *,
        query: str = "",
        task_type: str = "",
        dataset: str = "",
        policy: str = "",
        provider: str = "",
        limit: int = 3,
    ) -> str:
        records = self.search(
            query=query,
            task_type=task_type,
            dataset=dataset,
            policy=policy,
            provider=provider,
            limit=limit,
        )
        if not records:
            return ""

        lines = [
            "Use these past outcomes as hints. Reuse what worked and avoid repeating failures."
        ]
        for record in records:
            fields = [record.outcome]
            if record.dataset:
                fields.append(f"dataset={record.dataset}")
            if record.replay_datasets:
                fields.append(f"replay={record.replay_datasets}")
            if record.policy:
                fields.append(f"policy={record.policy}")
            if record.provider:
                fields.append(f"provider={record.provider}")
            summary = record.lesson or record.summary
            lines.append(f"- [{record.timestamp[:19]}] {'; '.join(fields)} -> {summary}")
        return "\n".join(lines)

    def get_replay_datasets(
        self,
        current_dataset: str,
        policy: str,
        max_datasets: int = 3,
    ) -> list[str]:
        current_key = _normalize_key(current_dataset)
        policy_key = _normalize_key(policy)
        unique: list[str] = []
        seen: set[str] = set()
        records = sorted(self.read_all(), key=lambda record: record.timestamp, reverse=True)
        for record in records:
            dataset_name = _normalize_text(record.dataset)
            dataset_key = _normalize_key(dataset_name)
            if record.task_type != "train" or record.outcome != "success":
                continue
            if not dataset_name or dataset_key == current_key or dataset_key in seen:
                continue
            if policy_key and _normalize_key(record.policy) != policy_key:
                continue
            seen.add(dataset_key)
            unique.append(dataset_name)
            if len(unique) >= max_datasets:
                break
        return unique
