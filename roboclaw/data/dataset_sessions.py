"""Dataset session handles for cache-backed remote and uploaded local datasets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from huggingface_hub import snapshot_download

from roboclaw.data.curation.paths import datasets_root
from roboclaw.data.datasets import DatasetCapabilities, DatasetCatalog, DatasetRef, DatasetStats
from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home

SessionKind = Literal["remote", "local_directory"]
SESSION_PREFIX = "session"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _session_root() -> Path:
    return get_roboclaw_home() / "cache" / "dataset-sessions"


def _datasets_root() -> Path:
    return datasets_root()


def _workspace_catalog() -> DatasetCatalog:
    return DatasetCatalog(root_resolver=_datasets_root)


def _session_dir(kind: SessionKind, session_id: str) -> Path:
    return _session_root() / kind / session_id


def _dataset_dir(kind: SessionKind, session_id: str) -> Path:
    return _session_dir(kind, session_id) / "dataset"


def _meta_path(kind: SessionKind, session_id: str) -> Path:
    return _session_dir(kind, session_id) / "session.json"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_session_handle(kind: SessionKind, session_id: str) -> str:
    return f"{SESSION_PREFIX}:{kind}:{session_id}"


def parse_session_handle(handle: str) -> tuple[SessionKind, str] | None:
    parts = handle.split(":", 2)
    if len(parts) != 3 or parts[0] != SESSION_PREFIX:
        return None
    kind = parts[1]
    if kind not in {"remote", "local_directory"}:
        return None
    session_id = parts[2]
    if not _SESSION_ID_RE.fullmatch(session_id):
        return None
    return kind, session_id


def is_session_handle(handle: str) -> bool:
    return parse_session_handle(handle) is not None


def resolve_session_dataset_path(handle: str) -> Path:
    parsed = parse_session_handle(handle)
    if parsed is None:
        raise ValueError(f"Invalid dataset session handle '{handle}'")
    kind, session_id = parsed
    kind_root = (_session_root() / kind).resolve()
    dataset_dir = _dataset_dir(kind, session_id).resolve()
    if not dataset_dir.is_relative_to(kind_root):
        raise ValueError(f"Invalid dataset session handle '{handle}'")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset session '{handle}' not found")
    return dataset_dir


def read_session_metadata(handle: str) -> dict[str, Any]:
    parsed = parse_session_handle(handle)
    if parsed is None:
        raise ValueError(f"Invalid dataset session handle '{handle}'")
    kind, session_id = parsed
    kind_root = (_session_root() / kind).resolve()
    path = _meta_path(kind, session_id).resolve()
    if not path.is_relative_to(kind_root):
        raise ValueError(f"Invalid dataset session handle '{handle}'")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset session metadata for '{handle}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_session_metadata(kind: SessionKind, session_id: str, payload: dict[str, Any]) -> None:
    path = _meta_path(kind, session_id)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_dataset_summary_from_dir(
    *,
    dataset_dir: Path,
    handle: str,
    display_name: str,
    source_kind: str,
    source_dataset: str,
    include_episode_lengths: bool = True,
) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset session '{handle}' is missing meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    episode_lengths: list[int] = []
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if include_episode_lengths and episodes_path.is_file():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            episode_lengths.append(int(entry.get("length", 0) or 0))

    return {
        "name": handle,
        "display_name": display_name,
        "source_kind": source_kind,
        "total_episodes": int(info.get("total_episodes", 0) or 0),
        "total_frames": int(info.get("total_frames", 0) or 0),
        "fps": int(info.get("fps", 0) or 0),
        "robot_type": str(info.get("robot_type", "")),
        "episode_lengths": episode_lengths,
        "features": list((info.get("features") or {}).keys()),
        "source_dataset": source_dataset,
    }


def register_remote_dataset_session(
    dataset_id: str,
    *,
    include_videos: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    session_id = hashlib.sha1(dataset_id.encode("utf-8")).hexdigest()[:16]
    handle = make_session_handle("remote", session_id)
    session_dir = _session_dir("remote", session_id)
    dataset_dir = _dataset_dir("remote", session_id)
    if force and session_dir.exists():
        shutil.rmtree(session_dir)
    created_session = not session_dir.exists()

    try:
        _ensure_dir(dataset_dir)

        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=["meta/**", "README*", *(["videos/**"] if include_videos else [])],
        )

        info_path = dataset_dir / "meta" / "info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("source_dataset") != dataset_id:
                info["source_dataset"] = dataset_id
                info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

        metadata = {
            "handle": handle,
            "kind": "remote",
            "session_id": session_id,
            "display_name": dataset_id,
            "source_dataset": dataset_id,
            "dataset_dir": str(dataset_dir.resolve()),
        }
        _write_session_metadata("remote", session_id, metadata)
        summary = _build_dataset_summary_from_dir(
            dataset_dir=dataset_dir,
            handle=handle,
            display_name=dataset_id,
            source_kind="remote_session",
            source_dataset=dataset_id,
        )
    except Exception:
        if created_session and session_dir.exists():
            shutil.rmtree(session_dir)
        raise
    return {
        "dataset_id": dataset_id,
        "dataset_name": handle,
        "display_name": dataset_id,
        "local_path": str(dataset_dir.resolve()),
        "summary": summary,
    }


def create_uploaded_directory_session(
    *,
    files: list[tuple[str, bytes | Path]],
    display_name: str | None = None,
) -> dict[str, Any]:
    session_id = uuid4().hex[:12]
    handle = make_session_handle("local_directory", session_id)
    dataset_dir = _dataset_dir("local_directory", session_id)
    dataset_root = dataset_dir.resolve()
    write_plan: list[tuple[Path, bytes | Path]] = []

    for relative_path, raw in files:
        target = (dataset_root / relative_path).resolve()
        if target == dataset_root or not target.is_relative_to(dataset_root):
            raise ValueError(f"Invalid uploaded file path '{relative_path}'")
        write_plan.append((target, raw))

    session_dir = _session_dir("local_directory", session_id)
    try:
        _ensure_dir(dataset_dir)
        for target, raw in write_plan:
            _ensure_dir(target.parent)
            if isinstance(raw, Path):
                shutil.copyfile(raw, target)
            else:
                target.write_bytes(raw)

        session_display_name = display_name or dataset_dir.name
        metadata = {
            "handle": handle,
            "kind": "local_directory",
            "session_id": session_id,
            "display_name": session_display_name,
            "source_dataset": session_display_name,
            "dataset_dir": str(dataset_dir.resolve()),
        }
        _write_session_metadata("local_directory", session_id, metadata)
        summary = _build_dataset_summary_from_dir(
            dataset_dir=dataset_dir,
            handle=handle,
            display_name=session_display_name,
            source_kind="local_directory_session",
            source_dataset=session_display_name,
        )
    except Exception:
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise
    return {
        "dataset_name": handle,
        "display_name": session_display_name,
        "local_path": str(dataset_dir.resolve()),
        "summary": summary,
    }


def list_session_dataset_summaries(*, include_remote: bool = True, include_local_directory: bool = True) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    kinds: list[SessionKind] = []
    if include_remote:
        kinds.append("remote")
    if include_local_directory:
        kinds.append("local_directory")

    for kind in kinds:
        kind_root = _session_root() / kind
        if not kind_root.is_dir():
            continue
        for session_dir in sorted(kind_root.iterdir()):
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            if not _SESSION_ID_RE.fullmatch(session_id):
                continue
            handle = make_session_handle(kind, session_id)
            metadata = read_session_metadata(handle)
            summary = _build_dataset_summary_from_dir(
                dataset_dir=_dataset_dir(kind, session_id),
                handle=handle,
                display_name=str(metadata.get("display_name") or handle),
                source_kind="remote_session" if kind == "remote" else "local_directory_session",
                source_dataset=str(
                    metadata.get("source_dataset") or metadata.get("display_name") or handle
                ),
                include_episode_lengths=False,
            )
            results.append(summary)
    return results


def list_curation_dataset_summaries() -> list[dict[str, Any]]:
    workspace_items = [
        {
            **_workspace_dataset_ref_to_summary(ref),
            "source_kind": "workspace",
        }
        for ref in _workspace_catalog().list_local_datasets()
    ]
    return workspace_items + list_session_dataset_summaries(
        include_remote=True,
        include_local_directory=True,
    )


def list_local_dataset_options() -> list[dict[str, Any]]:
    workspace_items = [
        {
            "id": ref.id,
            "label": ref.label,
            "path": str(ref.local_path) if ref.local_path is not None else "",
            "source": "local",
            "source_kind": "workspace",
        }
        for ref in _workspace_catalog().list_local_datasets()
    ]

    session_items = [
        {
            "id": item["name"],
            "label": str(item.get("display_name") or item["name"]),
            "path": "",
            "source": "local",
            "source_kind": item.get("source_kind", "local_directory_session"),
        }
        for item in list_session_dataset_summaries(
            include_remote=False,
            include_local_directory=True,
        )
    ]
    return workspace_items + session_items


def _workspace_dataset_ref_to_summary(ref: DatasetRef) -> dict[str, Any]:
    stats = ref.stats
    return {
        "name": ref.id,
        "display_name": ref.label,
        "total_episodes": stats.total_episodes,
        "total_frames": stats.total_frames,
        "fps": stats.fps,
        "episode_lengths": list(stats.episode_lengths),
        "features": list(stats.features),
        "robot_type": stats.robot_type,
        "source_dataset": ref.source_dataset,
    }


def resolve_dataset_handle_or_workspace(name: str) -> Path:
    if is_session_handle(name):
        return resolve_session_dataset_path(name)

    ref = _get_workspace_dataset(name)
    if ref is not None and ref.local_path is not None:
        return ref.local_path

    raise FileNotFoundError(f"Dataset '{name}' not found")


def get_dataset_summary(name: str) -> dict[str, Any]:
    if is_session_handle(name):
        metadata = read_session_metadata(name)
        dataset_dir = resolve_session_dataset_path(name)
        return _build_dataset_summary_from_dir(
            dataset_dir=dataset_dir,
            handle=name,
            display_name=str(metadata.get("display_name") or name),
            source_kind="remote_session"
            if metadata.get("kind") == "remote"
            else "local_directory_session",
            source_dataset=str(metadata.get("source_dataset") or metadata.get("display_name") or name),
        )

    ref = _get_workspace_dataset(name)
    if ref is not None:
        return {
            **_workspace_dataset_ref_to_summary(ref),
            "source_kind": "workspace",
        }
    raise FileNotFoundError(f"Dataset '{name}' not found")


def session_summary_to_dataset_ref(summary: dict[str, Any]) -> DatasetRef:
    dataset_id = str(summary.get("name") or "")
    label = str(summary.get("display_name") or dataset_id)
    slug = dataset_id.split(":")[-1] if ":" in dataset_id else dataset_id.rsplit("/", 1)[-1]
    stats = DatasetStats(
        total_episodes=int(summary.get("total_episodes", 0) or 0),
        total_frames=int(summary.get("total_frames", 0) or 0),
        fps=int(summary.get("fps", 0) or 0),
        robot_type=str(summary.get("robot_type") or ""),
        features=tuple(summary.get("features") or []),
        episode_lengths=tuple(int(length) for length in summary.get("episode_lengths") or []),
    )
    return DatasetRef(
        id=dataset_id,
        kind="local",
        label=label,
        slug=slug,
        source_dataset=str(summary.get("source_dataset") or dataset_id),
        stats=stats,
        capabilities=DatasetCapabilities(can_curate=True),
    )


def session_summary_to_dataset_dict(summary: dict[str, Any]) -> dict[str, Any]:
    ref = session_summary_to_dataset_ref(summary)
    payload = ref.to_dict()
    payload.update(
        {
            "name": ref.id,
            "display_name": ref.label,
            "source_kind": str(summary.get("source_kind") or "session"),
        }
    )
    return payload


def _get_workspace_dataset(name: str) -> DatasetRef | None:
    catalog = _workspace_catalog()
    direct = catalog.get_local_dataset(name)
    if direct is not None:
        return direct
    runtime = catalog.get_local_dataset(f"local/{name}")
    if runtime is not None:
        return runtime
    return next((ref for ref in catalog.list_local_datasets() if ref.slug == name), None)
