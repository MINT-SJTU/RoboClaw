"""Dataset catalog — unified identity, storage, and runtime resolution."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import HTTPException
from loguru import logger

from roboclaw.data.curation.features import extract_action_names, extract_state_names
from roboclaw.data.curation.paths import datasets_root
from roboclaw.data.dataset_sessions import (
    get_dataset_summary,
    is_session_handle,
    resolve_dataset_handle_or_workspace,
)

DatasetKind = Literal["local", "remote"]
ImportStatus = Literal["queued", "running", "completed", "error"]

_DATASET_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

__all__ = [
    "DatasetCapabilities",
    "DatasetCatalog",
    "DatasetImportJobRef",
    "DatasetRef",
    "DatasetRuntimeRef",
    "DatasetStats",
    "datasets_root_from_manifest",
    "delete_dataset",
    "extract_action_names",
    "extract_state_names",
    "get_dataset_info",
    "list_datasets",
    "resolve_dataset_path",
    "validate_dataset_slug",
]


def resolve_dataset_path(name: str) -> Path:
    """Resolve a dataset name to its full path on disk."""
    if is_session_handle(name):
        return resolve_dataset_handle_or_workspace(name)

    root = datasets_root()
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Datasets root '{root}' does not exist")
    resolved_root = root.resolve()

    def _is_safe(path: Path) -> bool:
        rp = path.resolve()
        return rp.is_dir() and str(rp).startswith(str(resolved_root) + "/")

    direct = root / name
    if _is_safe(direct):
        return direct.resolve()

    for parent in root.iterdir():
        if not parent.is_dir():
            continue
        candidate = parent / name
        if _is_safe(candidate):
            return candidate.resolve()

    raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")


def validate_dataset_slug(slug: str) -> None:
    """Raise ValueError if *slug* is not a valid runtime dataset slug."""
    if not slug or not _DATASET_SLUG_RE.fullmatch(slug):
        raise ValueError(
            "dataset_name must be a non-empty ASCII slug "
            "(letters, numbers, underscores, hyphens)."
        )


def datasets_root_from_manifest(manifest: Any) -> Path:
    """Return the datasets workspace root for *manifest*."""
    configured = manifest.snapshot.get("datasets", {}).get("root", "")
    if configured:
        return Path(configured).expanduser()
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets"


@dataclass(frozen=True)
class DatasetStats:
    total_episodes: int = 0
    total_frames: int = 0
    fps: int = 0
    robot_type: str = ""
    features: tuple[str, ...] = ()
    episode_lengths: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "features": list(self.features),
            "episode_lengths": list(self.episode_lengths),
        }


@dataclass(frozen=True)
class DatasetCapabilities:
    can_replay: bool = False
    can_train: bool = False
    can_delete: bool = False
    can_push: bool = False
    can_pull: bool = False
    can_curate: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "can_replay": self.can_replay,
            "can_train": self.can_train,
            "can_delete": self.can_delete,
            "can_push": self.can_push,
            "can_pull": self.can_pull,
            "can_curate": self.can_curate,
        }


@dataclass(frozen=True)
class DatasetRuntimeRef:
    name: str
    repo_id: str
    local_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "local_path": str(self.local_path),
        }


@dataclass(frozen=True)
class DatasetRef:
    id: str
    kind: DatasetKind
    label: str
    slug: str
    source_dataset: str
    stats: DatasetStats
    capabilities: DatasetCapabilities
    runtime: DatasetRuntimeRef | None = None
    local_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "slug": self.slug,
            "source_dataset": self.source_dataset,
            "stats": self.stats.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "runtime": self.runtime.to_dict() if self.runtime else None,
        }


@dataclass(frozen=True)
class DatasetImportJobRef:
    job_id: str
    dataset_id: str
    status: ImportStatus
    include_videos: bool
    message: str = ""
    dataset: DatasetRef | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "include_videos": self.include_videos,
            "message": self.message,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "imported_dataset_id": self.dataset.id if self.dataset else None,
            "local_path": str(self.dataset.local_path) if self.dataset and self.dataset.local_path else None,
        }


class DatasetCatalog:
    """Single source of truth for dataset identity, storage, and runtime resolution."""

    def __init__(self, root_resolver: Callable[[], Path] | None = None) -> None:
        self._root_resolver = root_resolver or datasets_root
        self._import_jobs: dict[str, DatasetImportJobRef] = {}

    @property
    def root(self) -> Path:
        return self._root_resolver().expanduser()

    def list_local_datasets(self) -> list[DatasetRef]:
        root = self.root
        if not root.is_dir():
            return []
        refs = [ref for ref in (self._read_local_dataset(entry) for entry in self._iter_dataset_dirs(root)) if ref]
        return sorted(refs, key=lambda ref: ref.id)

    def get_local_dataset(self, dataset_id: str) -> DatasetRef | None:
        target = self.resolve_local_path(dataset_id)
        if not target.is_dir():
            return None
        return self._read_local_dataset(target)

    def require_local_dataset(self, dataset_id: str) -> DatasetRef:
        ref = self.get_local_dataset(dataset_id)
        if ref is None:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        return ref

    def resolve_dataset(self, dataset_id: str) -> DatasetRef:
        local = self.get_local_dataset(dataset_id)
        if local is not None:
            return local
        return self.resolve_remote_dataset(dataset_id)

    def resolve_runtime_dataset(self, runtime_name: str) -> DatasetRef:
        validate_dataset_slug(runtime_name)
        ref = self.get_local_dataset(f"local/{runtime_name}")
        if ref is None or ref.runtime is None:
            raise ValueError(f"Runtime dataset '{runtime_name}' not found")
        return ref

    def prepare_recording_dataset(
        self,
        dataset_name: str = "",
        *,
        prefix: str = "rec",
    ) -> DatasetRef:
        slug = dataset_name.strip() or f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validate_dataset_slug(slug)
        local_path = self.root / "local" / slug
        return DatasetRef(
            id=f"local/{slug}",
            kind="local",
            label=slug,
            slug=slug,
            source_dataset=f"local/{slug}",
            stats=DatasetStats(),
            capabilities=self._local_runtime_capabilities(),
            runtime=DatasetRuntimeRef(name=slug, repo_id=f"local/{slug}", local_path=local_path),
            local_path=local_path,
        )

    def delete_dataset(self, dataset_id: str) -> None:
        ref = self.require_local_dataset(dataset_id)
        if ref.local_path is None:
            raise ValueError(f"Dataset '{dataset_id}' is not a local dataset")
        logger.info("Deleting dataset: {}", ref.local_path)
        shutil.rmtree(ref.local_path)

    def push_dataset(
        self,
        dataset_id: str,
        *,
        repo_id: str,
        token: str = "",
        private: bool = False,
        endpoint: str = "",
        proxy: str = "",
    ) -> str:
        ref = self.require_local_dataset(dataset_id)
        if ref.local_path is None:
            raise ValueError(f"Dataset '{dataset_id}' is not a local dataset")

        from roboclaw.embodied.service.hub.transfer import push_folder

        url = push_folder(
            local_path=ref.local_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            private=private,
            ignore_patterns=["images/"],
            endpoint=endpoint,
            proxy=proxy,
        )
        return f"Dataset '{ref.label}' pushed to {repo_id}\n{url}"

    def pull_dataset(
        self,
        repo_id: str,
        *,
        dataset_id: str = "",
        token: str = "",
        endpoint: str = "",
        proxy: str = "",
        tqdm_class: Any = None,
    ) -> DatasetRef:
        local_id = dataset_id.strip() or repo_id
        local_path = self.resolve_local_path(local_id)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        from roboclaw.embodied.service.hub.transfer import pull_repo

        pull_repo(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_path,
            token=token,
            tqdm_class=tqdm_class,
            endpoint=endpoint,
            proxy=proxy,
        )
        self._stamp_source_dataset(local_path, repo_id)
        return self.require_local_dataset(local_id)

    def queue_import_job(
        self,
        job_id: str,
        *,
        dataset_id: str,
        include_videos: bool,
    ) -> DatasetImportJobRef:
        job = DatasetImportJobRef(
            job_id=job_id,
            dataset_id=dataset_id,
            status="queued",
            include_videos=include_videos,
            message="Queued for import",
        )
        self._import_jobs[job_id] = job
        return job

    def get_import_job(self, job_id: str) -> DatasetImportJobRef | None:
        return self._import_jobs.get(job_id)

    async def run_import_job(
        self,
        job_id: str,
        dataset_id: str,
        *,
        include_videos: bool,
        force: bool,
    ) -> None:
        self._import_jobs[job_id] = DatasetImportJobRef(
            job_id=job_id,
            dataset_id=dataset_id,
            status="running",
            include_videos=include_videos,
            message="Downloading dataset snapshot from Hugging Face",
        )
        try:
            dataset = await asyncio.to_thread(
                self.import_remote_dataset,
                dataset_id,
                include_videos=include_videos,
                force=force,
            )
        except Exception as exc:
            logger.exception("Dataset import failed for {}", dataset_id)
            self._import_jobs[job_id] = DatasetImportJobRef(
                job_id=job_id,
                dataset_id=dataset_id,
                status="error",
                include_videos=include_videos,
                message=str(exc),
            )
            return
        self._import_jobs[job_id] = DatasetImportJobRef(
            job_id=job_id,
            dataset_id=dataset_id,
            status="completed",
            include_videos=include_videos,
            message="Dataset imported",
            dataset=dataset,
        )

    def import_remote_dataset(
        self,
        dataset_id: str,
        *,
        include_videos: bool,
        force: bool,
    ) -> DatasetRef:
        target_dir = self.resolve_local_path(dataset_id)
        target_dir.parent.mkdir(parents=True, exist_ok=True)

        if force and target_dir.exists():
            shutil.rmtree(target_dir)

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=self._import_allow_patterns(include_videos),
        )
        self._stamp_source_dataset(target_dir, dataset_id)
        return self.require_local_dataset(dataset_id)

    def resolve_remote_dataset(self, dataset_id: str) -> DatasetRef:
        from roboclaw.data.explorer.remote import build_remote_dataset_info

        payload = build_remote_dataset_info(dataset_id)
        stats = DatasetStats(
            total_episodes=int(payload.get("total_episodes", 0) or 0),
            total_frames=int(payload.get("total_frames", 0) or 0),
            fps=int(payload.get("fps", 0) or 0),
            robot_type=str(payload.get("robot_type", "")),
            features=tuple(payload.get("features", [])),
            episode_lengths=tuple(int(length) for length in payload.get("episode_lengths", [])),
        )
        return DatasetRef(
            id=str(payload.get("name") or dataset_id),
            kind="remote",
            label=str(payload.get("name") or dataset_id),
            slug=dataset_id.rsplit("/", 1)[-1],
            source_dataset=str(payload.get("source_dataset") or dataset_id),
            stats=stats,
            capabilities=DatasetCapabilities(can_pull=True),
        )

    def resolve_local_path(self, dataset_id: str) -> Path:
        root = self.root.resolve()
        target = (self.root / dataset_id).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Invalid dataset id: {dataset_id!r}")
        return target

    def _iter_dataset_dirs(self, root: Path):
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if self._is_dataset_dir(entry):
                yield entry
                continue
            for child in sorted(entry.iterdir()):
                if child.is_dir() and self._is_dataset_dir(child):
                    yield child

    def _is_dataset_dir(self, dataset_dir: Path) -> bool:
        return (dataset_dir / "meta" / "info.json").is_file()

    def _read_local_dataset(self, dataset_dir: Path) -> DatasetRef | None:
        if not self._is_dataset_dir(dataset_dir):
            return None

        info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        relative = dataset_dir.relative_to(self.root)
        dataset_id = relative.as_posix()
        parent_id = relative.parent.as_posix()
        slug = dataset_dir.name
        label = slug if parent_id in (".", "local") else dataset_id
        runtime = None
        capabilities = self._local_catalog_capabilities()
        if parent_id == "local":
            runtime = DatasetRuntimeRef(name=slug, repo_id=f"local/{slug}", local_path=dataset_dir)
            capabilities = self._local_runtime_capabilities()

        return DatasetRef(
            id=dataset_id,
            kind="local",
            label=label,
            slug=slug,
            source_dataset=str(
                info.get("source_dataset")
                or info.get("repo_id")
                or info.get("dataset_id")
                or dataset_id
            ),
            stats=self._read_stats(dataset_dir, info),
            capabilities=capabilities,
            runtime=runtime,
            local_path=dataset_dir,
        )

    def _read_stats(self, dataset_dir: Path, info: dict[str, Any]) -> DatasetStats:
        episode_lengths: list[int] = []
        episodes_path = dataset_dir / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            for raw_line in episodes_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                episode_lengths.append(int(payload.get("length", 0) or 0))

        return DatasetStats(
            total_episodes=int(info.get("total_episodes", 0) or 0),
            total_frames=int(info.get("total_frames", 0) or 0),
            fps=int(info.get("fps", 0) or 0),
            robot_type=str(info.get("robot_type", "")),
            features=tuple((info.get("features") or {}).keys()),
            episode_lengths=tuple(episode_lengths),
        )

    def _stamp_source_dataset(self, dataset_dir: Path, source_dataset: str) -> None:
        info_path = dataset_dir / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("source_dataset") == source_dataset:
            return
        info["source_dataset"] = source_dataset
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def _import_allow_patterns(self, include_videos: bool) -> list[str]:
        patterns = ["meta/**", "README*"]
        if include_videos:
            patterns.append("videos/**")
        return patterns

    def _local_catalog_capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            can_delete=True,
            can_push=True,
            can_curate=True,
        )

    def _local_runtime_capabilities(self) -> DatasetCapabilities:
        return DatasetCapabilities(
            can_replay=True,
            can_train=True,
            can_delete=True,
            can_push=True,
            can_curate=True,
        )


def list_datasets(root: Path) -> list[dict[str, Any]]:
    """Return list-style dataset summaries for legacy HTTP routes/tests."""
    catalog = DatasetCatalog(root_resolver=lambda: root)
    return [ref.to_dict() for ref in catalog.list_local_datasets()]


def get_dataset_info(root: Path, name: str) -> dict[str, Any] | None:
    """Return a single dataset summary compatible with legacy HTTP routes/tests."""
    if is_session_handle(name):
        try:
            summary = get_dataset_summary(name)
        except FileNotFoundError:
            return None
        return {
            "id": summary["name"],
            "label": summary.get("display_name") or summary["name"],
            "name": summary["name"],
            "display_name": summary.get("display_name"),
            "source_kind": summary.get("source_kind"),
            "kind": "local",
            "slug": summary["name"].split("/")[-1],
            "source_dataset": summary.get("source_dataset") or summary["name"],
            "stats": {
                "total_episodes": summary.get("total_episodes", 0),
                "total_frames": summary.get("total_frames", 0),
                "fps": summary.get("fps", 0),
                "robot_type": summary.get("robot_type", ""),
                "features": summary.get("features", []),
                "episode_lengths": summary.get("episode_lengths", []),
            },
            "capabilities": {
                "can_replay": False,
                "can_train": False,
                "can_delete": False,
                "can_push": False,
                "can_pull": False,
                "can_curate": True,
            },
            "runtime": None,
        }

    catalog = DatasetCatalog(root_resolver=lambda: root)
    try:
        ref = catalog.require_local_dataset(name)
    except ValueError:
        return None
    payload = ref.to_dict()
    payload["name"] = ref.id
    payload["display_name"] = ref.label
    payload["source_kind"] = "workspace"
    return payload


def delete_dataset(root: Path, name: str) -> None:
    """Delete a dataset directory. Raises ValueError if it does not exist."""
    DatasetCatalog(root_resolver=lambda: root).delete_dataset(name)
