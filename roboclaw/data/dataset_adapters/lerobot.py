"""LeRobot/HuggingFace dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from loguru import logger

from roboclaw.data.curation.bridge import read_parquet_rows
from roboclaw.data.curation.state import load_dataset_info

from .base import CanonicalEpisode


class LeRobotAdapter:
    """Read a LeRobot-style dataset workspace as canonical episodes."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path

    def list_episodes(self) -> list[int]:
        info = load_dataset_info(self.dataset_path)
        total = int(info.get("total_episodes", 0) or 0)
        if total > 0:
            return list(range(total))
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            return []
        indices: list[int] = []
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                indices.append(int(entry["episode_index"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(set(indices))

    def load_episode(self, episode_index: int) -> CanonicalEpisode:
        info = load_dataset_info(self.dataset_path)
        episode_meta = _load_episode_meta(self.dataset_path, episode_index)
        chunk = _resolve_chunk(info, episode_index)
        parquet_relative_path = Path("data") / f"chunk-{chunk}" / f"episode_{episode_index:06d}.parquet"
        parquet_path = self.dataset_path / parquet_relative_path
        if parquet_path.exists():
            rows = _read_parquet_rows(parquet_path)
        else:
            remote_dataset_id = _resolve_remote_dataset_id(self.dataset_path, info)
            parquet_path = _download_remote_file(
                remote_dataset_id,
                parquet_relative_path,
                local_root=self.dataset_path,
            )
            rows = _read_parquet_rows(parquet_path)

        video_dir = self.dataset_path / "videos" / f"chunk-{chunk}" / f"episode_{episode_index:06d}"
        if video_dir.exists():
            video_files = _list_video_files(video_dir)
        else:
            remote_dataset_id = _resolve_remote_dataset_id(self.dataset_path, info)
            video_files = _download_remote_videos(
                remote_dataset_id,
                info,
                episode_index,
                local_root=self.dataset_path,
            )

        return {
            "info": info,
            "episode_meta": episode_meta,
            "rows": rows,
            "parquet_path": parquet_path,
            "video_dir": video_dir,
            "video_files": video_files,
            "chunk": chunk,
        }


def _resolve_remote_dataset_id(dataset_path: Path, info: dict[str, Any]) -> str:
    source_dataset = info.get("source_dataset") or info.get("repo_id") or info.get("dataset_id")
    if isinstance(source_dataset, str) and source_dataset.strip():
        return source_dataset.strip()
    try:
        from roboclaw.data.curation.paths import datasets_root

        root = datasets_root().resolve()
        resolved = dataset_path.resolve()
        if str(resolved).startswith(str(root) + "/"):
            return resolved.relative_to(root).as_posix()
    except Exception:
        logger.debug("Failed to resolve remote dataset id", exc_info=True)
    return dataset_path.name


def _load_episode_meta(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {}
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("episode_index") == episode_index:
            return entry
    return {}


def _resolve_chunk(info: dict[str, Any], episode_index: int) -> str:
    chunks_size = info.get("chunks_size", 1000)
    if chunks_size <= 0:
        chunks_size = 1000
    return f"{episode_index // chunks_size:03d}"


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_parquet_rows(path)


def _download_remote_file(
    dataset_id: str,
    relative_path: Path,
    *,
    local_root: Path | None = None,
) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": dataset_id,
        "filename": relative_path.as_posix(),
        "repo_type": "dataset",
    }
    if local_root is not None:
        kwargs["local_dir"] = str(local_root)
    cached_path = hf_hub_download(**kwargs)
    return Path(cached_path)


def _extract_video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    keys: list[str] = []
    for name, config in features.items():
        if not isinstance(config, dict):
            continue
        if config.get("dtype") == "video":
            keys.append(str(name))
    return keys


def _download_remote_videos(
    dataset_id: str,
    info: dict[str, Any],
    episode_index: int,
    *,
    local_root: Path | None = None,
) -> list[Path]:
    template = info.get("video_path")
    if not isinstance(template, str) or not template:
        return []

    chunk = _resolve_chunk(info, episode_index)
    chunk_index = int(chunk)
    video_keys = _extract_video_keys(info)
    results: list[Path] = []
    for video_key in video_keys:
        try:
            relative_path = template.format(
                episode_chunk=chunk_index,
                video_key=video_key,
                episode_index=episode_index,
            )
            results.append(
                _download_remote_file(
                    dataset_id,
                    Path(relative_path),
                    local_root=local_root,
                ),
            )
        except Exception:
            logger.warning("Failed to download video {}", video_key, exc_info=True)
            continue
    return results


def _list_video_files(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(video_dir.glob("*.mp4"))
