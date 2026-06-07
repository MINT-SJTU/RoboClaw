"""Config-driven dataset adapter for simple JSONL row datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboclaw.data.curation.state import load_dataset_info

from .base import CanonicalEpisode


class MappingAdapter:
    """Map user dataset fields into RoboClaw canonical episode rows.

    This is intentionally small: it handles JSONL row datasets and nested field
    paths. Platform AI can propose the mapping, but this adapter executes it in
    a deterministic and testable way.
    """

    def __init__(self, dataset_path: Path, mapping: dict[str, Any]) -> None:
        self.dataset_path = dataset_path
        self.mapping = mapping
        fields = mapping.get("fields", mapping)
        if not isinstance(fields, dict) or not fields:
            raise ValueError("mapping adapter requires a non-empty fields mapping")
        self.fields = {str(target): str(source) for target, source in fields.items()}
        rows_file = mapping.get("rows_file") or mapping.get("rowsFile")
        self.rows_path = self._resolve_rows_path(str(rows_file)) if rows_file else self._default_rows_path()

    def list_episodes(self) -> list[int]:
        rows = self._read_rows()
        episode_key = self.mapping.get("episode_index_field") or self.mapping.get("episodeIndexField") or "episode_index"
        indices: set[int] = set()
        for row in rows:
            value = _get_path(row, str(episode_key))
            if value is None:
                indices.add(0)
                continue
            try:
                indices.add(int(value))
            except (TypeError, ValueError):
                continue
        return sorted(indices)

    def load_episode(self, episode_index: int) -> CanonicalEpisode:
        rows = self._episode_rows(episode_index)
        mapped_rows = [_map_row(row, self.fields) for row in rows]
        info = load_dataset_info(self.dataset_path) or _infer_info(mapped_rows, episode_count=len(self.list_episodes()))
        episode_meta = _episode_meta(mapped_rows, episode_index)
        video_files = _collect_video_files(self.dataset_path, mapped_rows)
        return {
            "info": info,
            "episode_meta": episode_meta,
            "rows": mapped_rows,
            "parquet_path": self.rows_path,
            "video_dir": self.rows_path.parent,
            "video_files": video_files,
            "chunk": "000",
        }

    def _episode_rows(self, episode_index: int) -> list[dict[str, Any]]:
        rows = self._read_rows()
        episode_key = self.mapping.get("episode_index_field") or self.mapping.get("episodeIndexField") or "episode_index"
        grouped: list[dict[str, Any]] = []
        has_episode_key = False
        for row in rows:
            value = _get_path(row, str(episode_key))
            if value is None:
                continue
            has_episode_key = True
            try:
                if int(value) == episode_index:
                    grouped.append(row)
            except (TypeError, ValueError):
                continue
        if has_episode_key:
            return grouped
        if episode_index == 0:
            return rows
        return []

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.rows_path.exists():
            raise FileNotFoundError(f"mapping rows file not found: {self.rows_path}")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.rows_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row at line {line_number} must be an object")
            rows.append(payload)
        return rows

    def _resolve_rows_path(self, rows_file: str) -> Path:
        path = Path(rows_file)
        if path.is_absolute():
            return path
        return self.dataset_path / path

    def _default_rows_path(self) -> Path:
        for relative_path in ("data/episodes.jsonl", "episodes.jsonl", "data.jsonl"):
            path = self.dataset_path / relative_path
            if path.exists():
                return path
        return self.dataset_path / "data" / "episodes.jsonl"


def _map_row(row: dict[str, Any], fields: dict[str, str]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    missing: list[str] = []
    for target, source in fields.items():
        value = _get_path(row, source)
        if value is None:
            missing.append(source)
            continue
        mapped[target] = value
    if missing:
        raise ValueError(f"missing mapped source fields: {', '.join(sorted(missing))}")
    return mapped


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def _infer_info(rows: list[dict[str, Any]], *, episode_count: int) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {}
    if rows:
        sample = rows[0]
        if "action" in sample:
            features["action"] = {"dtype": "float32"}
        if "observation.state" in sample:
            features["observation.state"] = {"dtype": "float32"}
        for key in sample:
            if key.startswith("observation.images."):
                features[key] = {"dtype": "image"}
    return {
        "total_episodes": episode_count,
        "robot_type": "",
        "fps": 0,
        "features": features,
    }


def _episode_meta(rows: list[dict[str, Any]], episode_index: int) -> dict[str, Any]:
    timestamps = [float(row["timestamp"]) for row in rows if _is_number(row.get("timestamp"))]
    length = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    task = ""
    for row in rows:
        value = row.get("task") or row.get("language_instruction")
        if value:
            task = str(value)
            break
    return {"episode_index": episode_index, "length": length, "task": task}


def _collect_video_files(dataset_path: Path, rows: list[dict[str, Any]]) -> list[Path]:
    paths: set[Path] = set()
    for row in rows:
        for key, value in row.items():
            if not key.startswith("observation.images."):
                continue
            if not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = dataset_path / path
            if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"} and path.exists():
                paths.add(path)
    return sorted(paths)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
