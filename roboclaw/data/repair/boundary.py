from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from roboclaw.data.paths import datasets_root

from .io import load_info
from .lerobot_adapter import LeRobotDatasetAdapter


def resolve_dataset_root(dataset: str, root: Path | None) -> Path:
    dataset_path = Path(dataset).expanduser()
    if dataset_path.exists():
        return dataset_path.resolve()

    base_root = root.expanduser().resolve() if root is not None else datasets_root().resolve()
    candidates = [base_root / dataset]
    if "/" not in dataset.strip("/"):
        candidates.append(base_root / "local" / dataset)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Dataset path not found. Searched:\n{searched}")


def resolve_repo_id(dataset_root: Path, root: Path | None) -> str:
    if root is None:
        return f"local/{dataset_root.name}"

    base_root = root.expanduser().resolve()
    if dataset_root.is_relative_to(base_root):
        return dataset_root.relative_to(base_root).as_posix()
    return f"local/{dataset_root.name}"


def get_camera_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in sorted(info.get("features", {}).items())
        if key.startswith("observation.images.") and value.get("dtype") in {"image", "video"}
    ]


def get_pixel_count(feature: dict[str, Any]) -> int:
    shape = feature.get("shape", (0, 0, 0))
    names = list(feature.get("names") or [])
    if names:
        h_idx = names.index("height") if "height" in names else None
        w_idx = names.index("width") if "width" in names else None
        if h_idx is not None and w_idx is not None:
            return int(shape[h_idx]) * int(shape[w_idx])
    return int(shape[0]) * int(shape[1]) if len(shape) >= 2 else 0


def select_camera_key(info: dict[str, Any], camera_key: str | None) -> str:
    camera_keys = get_camera_keys(info)
    if not camera_keys:
        raise ValueError("Dataset does not contain any image or video observation keys.")
    if camera_key is not None:
        if camera_key not in camera_keys:
            available = "\n".join(f"- {key}" for key in camera_keys)
            raise ValueError(f"Unknown camera key: {camera_key}\nAvailable cameras:\n{available}")
        return camera_key

    features = info["features"]
    return sorted(
        camera_keys,
        key=lambda key: (0 if "front" in key else 1, -get_pixel_count(features[key]), key),
    )[0]


def parse_episode_indices(spec: str, total_episodes: int) -> list[int]:
    if total_episodes <= 0:
        raise ValueError("Dataset has no episodes to export.")
    if spec.strip().lower() == "all":
        return list(range(total_episodes))

    episode_indices: list[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for episode_index in values:
            if episode_index < 0 or episode_index >= total_episodes:
                raise ValueError(
                    f"Episode index {episode_index} is out of range for dataset with {total_episodes} episodes."
                )
            if episode_index not in seen:
                episode_indices.append(episode_index)
                seen.add(episode_index)
    if not episode_indices:
        raise ValueError("No episodes were selected.")
    return episode_indices


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def format_episode_spec(episode_indices: list[int]) -> str:
    if not episode_indices:
        return ""
    groups: list[str] = []
    start = episode_indices[0]
    prev = start
    for index in episode_indices[1:]:
        if index == prev + 1:
            prev = index
            continue
        groups.append(str(start) if start == prev else f"{start}-{prev}")
        start = index
        prev = index
    groups.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(groups)


@dataclass(frozen=True)
class BoundaryFrameExportRequest:
    dataset: str
    output_dir: Path
    episodes: str = "all"
    camera_key: str | None = None
    root: Path | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class BoundaryFrameExportResult:
    dataset_root: Path
    repo_id: str
    camera_key: str
    episodes_exported: int
    manifest_path: Path


def export_episode_boundary_frames(
    *,
    dataset: Any,
    output_dir: Path,
    episode_indices: list[int],
    camera_key: str,
) -> Path:
    pad_width = max(3, len(str(max(episode_indices))))
    episodes = dataset.meta.episodes
    manifest_rows: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        from_index = int(episodes["dataset_from_index"][episode_index])
        to_index = int(episodes["dataset_to_index"][episode_index])
        length = int(episodes["length"][episode_index])
        success = str(episodes["episode_success"][episode_index]) if "episode_success" in episodes.column_names else ""

        first_name = f"episode_{episode_index:0{pad_width}d}_first.png"
        last_name = f"episode_{episode_index:0{pad_width}d}_last.png"
        frame_to_pil_image(dataset[from_index][camera_key]).save(output_dir / first_name)
        frame_to_pil_image(dataset[to_index - 1][camera_key]).save(output_dir / last_name)

        manifest_rows.append(
            {
                "episode_index": episode_index,
                "length": length,
                "episode_success": success,
                "first_dataset_index": from_index,
                "last_dataset_index": to_index - 1,
                "first_file": first_name,
                "last_file": last_name,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_index",
                "length",
                "episode_success",
                "first_dataset_index",
                "last_dataset_index",
                "first_file",
                "last_file",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    (output_dir / "README.txt").write_text(
        f"Dataset root: {dataset.root}\n"
        f"Dataset repo_id: {dataset.repo_id}\n"
        f"Camera: {camera_key}\n"
        f"Episodes exported: {format_episode_spec(episode_indices)} ({len(episode_indices)} total)\n"
        "Files per episode: episode_XXX_first.png, episode_XXX_last.png\n",
        encoding="utf-8",
    )
    return manifest_path


def frame_to_pil_image(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.copy()
    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    else:
        array = np.asarray(frame)

    if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


class BoundaryFrameExporter:
    def __init__(self, adapter: LeRobotDatasetAdapter | None = None) -> None:
        self._adapter = adapter or LeRobotDatasetAdapter()

    def export(self, request: BoundaryFrameExportRequest) -> BoundaryFrameExportResult:
        dataset_root = resolve_dataset_root(request.dataset, request.root)
        repo_id = resolve_repo_id(dataset_root, request.root)
        info = load_info(dataset_root)
        camera_key = select_camera_key(info, request.camera_key)
        episode_indices = parse_episode_indices(request.episodes, int(info["total_episodes"]))
        prepare_output_dir(request.output_dir, request.overwrite)
        dataset = self._adapter.open_dataset(repo_id=repo_id, root=dataset_root)
        manifest_path = export_episode_boundary_frames(
            dataset=dataset,
            output_dir=request.output_dir,
            episode_indices=episode_indices,
            camera_key=camera_key,
        )
        return BoundaryFrameExportResult(
            dataset_root=dataset_root,
            repo_id=repo_id,
            camera_key=camera_key,
            episodes_exported=len(episode_indices),
            manifest_path=manifest_path,
        )
