"""Local dataset push planning helpers."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboclaw.data.datasets import DatasetCatalog, DatasetRef

_DATASET_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class DatasetPushSummary:
    dataset_id: str
    source_path: Path
    file_count: int
    total_bytes: int
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str
    has_manifest: bool

    def to_manifest(self, *, username: str, visibility: str = "private") -> dict[str, Any]:
        return {
            "datasetId": self.dataset_id,
            "ownerUsername": username,
            "visibility": visibility,
            "sourcePath": str(self.source_path),
            "fileCount": self.file_count,
            "totalBytes": self.total_bytes,
            "totalEpisodes": self.total_episodes,
            "totalFrames": self.total_frames,
            "fps": self.fps,
            "robotType": self.robot_type,
            "hasManifest": self.has_manifest,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "nextHandle": dataset_handle(username, self.dataset_id),
        }


def infer_dataset_id(path: Path) -> str:
    """Infer a stable dataset id from a local path name."""
    raw = path.expanduser().resolve().name.strip().lower()
    slug = _DATASET_ID_RE.sub("-", raw).strip("-_")
    slug = re.sub(r"[-_]{2,}", "-", slug)
    if not slug:
        raise ValueError(f"Cannot infer dataset id from path: {path}")
    return slug


def dataset_handle(username: str, dataset_id: str) -> str:
    owner = username.strip() or "current-user"
    return f"evo://{owner}/{dataset_id.strip()}"


def format_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def push_dataset_to_local_catalog(
    summary: DatasetPushSummary,
    *,
    username: str,
    visibility: str = "private",
    force: bool = False,
    catalog: DatasetCatalog | None = None,
) -> DatasetRef:
    """Copy a local dataset into the RoboClaw dataset catalog."""
    dataset_catalog = catalog or DatasetCatalog()
    target = dataset_catalog.resolve_local_path(summary.dataset_id)
    source = summary.source_path.resolve()
    if target.exists() and source != target.resolve():
        if not force:
            raise ValueError(f"Dataset target already exists: {target}")
        shutil.rmtree(target)

    if source != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
    _write_catalog_metadata(target, summary, username=username, visibility=visibility)
    return dataset_catalog.require_local_dataset(summary.dataset_id)


def scan_dataset_path(path: Path, *, dataset_id: str = "") -> DatasetPushSummary:
    """Scan a local dataset directory for a push summary."""
    source = path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    if not source.is_dir():
        raise ValueError(f"Dataset push expects a directory: {source}")

    info = _read_info_json(source)
    total_episodes = _int(info.get("total_episodes"))
    total_frames = _int(info.get("total_frames"))
    fps = _int(info.get("fps"))
    robot_type = str(info.get("robot_type") or info.get("robotType") or "").strip()

    episodes_path = source / "meta" / "episodes.jsonl"
    if total_episodes <= 0 and episodes_path.is_file():
        total_episodes = _count_jsonl_rows(episodes_path)

    if total_episodes <= 0:
        total_episodes = _count_episode_like_entries(source)

    file_count, total_bytes = _walk_files(source)
    resolved_id = dataset_id.strip() or infer_dataset_id(source)
    return DatasetPushSummary(
        dataset_id=resolved_id,
        source_path=source,
        file_count=file_count,
        total_bytes=total_bytes,
        total_episodes=total_episodes,
        total_frames=total_frames,
        fps=fps,
        robot_type=robot_type,
        has_manifest=(source / "meta" / "info.json").is_file(),
    )


def _write_catalog_metadata(
    dataset_path: Path,
    summary: DatasetPushSummary,
    *,
    username: str,
    visibility: str,
) -> None:
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = _read_info_json(dataset_path) if info_path.is_file() else {}
    info.setdefault("dataset_id", summary.dataset_id)
    info.setdefault("source_dataset", summary.dataset_id)
    info["ownerUsername"] = username
    info["contributionSource"] = "self_collected"
    info["visibility"] = visibility
    info["sourceKind"] = "local_path"
    info["sourceUri"] = str(summary.source_path)
    info["storageMode"] = "managed"
    info["uploadStatus"] = "uploaded"
    info["total_episodes"] = summary.total_episodes
    info["total_frames"] = summary.total_frames
    if summary.fps:
        info["fps"] = summary.fps
    if summary.robot_type:
        info["robot_type"] = summary.robot_type
    now = datetime.now(timezone.utc).isoformat()
    info.setdefault("createdAt", now)
    info["ingestedAt"] = now
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_info_json(source: Path) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset metadata is not valid JSON: {info_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {info_path}")
    return payload


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_episode_like_entries(source: Path) -> int:
    candidates = (
        source / "episodes",
        source / "data" / "episodes",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return sum(1 for child in candidate.iterdir() if child.is_dir() or child.is_file())
    data_jsonl = source / "data" / "episodes.jsonl"
    if data_jsonl.is_file():
        return _count_jsonl_rows(data_jsonl)
    root_jsonl = source / "episodes.jsonl"
    if root_jsonl.is_file():
        return _count_jsonl_rows(root_jsonl)
    return 0


def _walk_files(source: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for entry in source.rglob("*"):
        if not entry.is_file():
            continue
        file_count += 1
        total_bytes += entry.stat().st_size
    return file_count, total_bytes


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
