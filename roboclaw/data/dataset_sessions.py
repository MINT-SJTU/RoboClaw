"""Dataset session handles for cache-backed remote and uploaded local datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from huggingface_hub import snapshot_download

from roboclaw.embodied.embodiment.manifest.helpers import get_manifest_path, get_roboclaw_home

SessionKind = Literal["remote", "local_directory"]
SESSION_PREFIX = "session"


def _session_root() -> Path:
    return get_roboclaw_home() / "cache" / "dataset-sessions"


def _datasets_root() -> Path:
    manifest_path = get_manifest_path()
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = data.get("datasets", {}).get("root", "")
        if root:
            return Path(root).expanduser()
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets"


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
    if not _is_safe_session_id(session_id):
        return None
    return kind, session_id


def is_session_handle(handle: str) -> bool:
    return parse_session_handle(handle) is not None


def _is_safe_session_id(session_id: str) -> bool:
    return bool(session_id) and session_id not in {".", ".."} and not any(
        char in session_id
        for char in ("/", "\\", ":")
    )


def resolve_session_dataset_path(handle: str) -> Path:
    parsed = parse_session_handle(handle)
    if parsed is None:
        raise ValueError(f"Invalid dataset session handle '{handle}'")
    kind, session_id = parsed
    dataset_dir = _dataset_dir(kind, session_id)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset session '{handle}' not found")
    return dataset_dir.resolve()


def read_session_metadata(handle: str) -> dict[str, Any]:
    parsed = parse_session_handle(handle)
    if parsed is None:
        raise ValueError(f"Invalid dataset session handle '{handle}'")
    kind, session_id = parsed
    path = _meta_path(kind, session_id)
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
) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset session '{handle}' is missing meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    episode_lengths: list[int] = []
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.is_file():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                episode_lengths.append(int(entry.get("length", 0) or 0))
            except Exception:
                episode_lengths.append(0)

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
    return {
        "dataset_id": dataset_id,
        "dataset_name": handle,
        "display_name": dataset_id,
        "local_path": str(dataset_dir.resolve()),
        "summary": summary,
    }


def create_uploaded_directory_session(
    *,
    files: list[tuple[str, bytes]],
    display_name: str | None = None,
) -> dict[str, Any]:
    session_id = uuid4().hex[:12]
    handle = make_session_handle("local_directory", session_id)
    dataset_dir = _dataset_dir("local_directory", session_id)
    _ensure_dir(dataset_dir)

    for relative_path, raw in files:
        target = (dataset_dir / relative_path).resolve()
        if not str(target).startswith(str(dataset_dir.resolve())):
            raise ValueError(f"Invalid uploaded file path '{relative_path}'")
        _ensure_dir(target.parent)
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
            handle = make_session_handle(kind, session_id)
            try:
                metadata = read_session_metadata(handle)
                summary = _build_dataset_summary_from_dir(
                    dataset_dir=_dataset_dir(kind, session_id),
                    handle=handle,
                    display_name=str(metadata.get("display_name") or handle),
                    source_kind="remote_session" if kind == "remote" else "local_directory_session",
                    source_dataset=str(metadata.get("source_dataset") or metadata.get("display_name") or handle),
                )
            except Exception:
                continue
            results.append(summary)
    return results


def list_curation_dataset_summaries() -> list[dict[str, Any]]:
    workspace_root = _datasets_root()
    workspace_items = [
        {
            **item,
            "display_name": item["name"],
            "source_kind": "workspace",
        }
        for item in _workspace_list_datasets(workspace_root)
    ]
    return workspace_items + list_session_dataset_summaries(
        include_remote=True,
        include_local_directory=True,
    )


def list_local_dataset_options() -> list[dict[str, Any]]:
    workspace_root = _datasets_root()
    workspace_items = [
        {
            "id": item["name"],
            "label": item["name"],
            "path": str((workspace_root / item["name"]).resolve()),
            "source": "local",
            "source_kind": "workspace",
        }
        for item in _workspace_list_datasets(workspace_root)
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


def _workspace_list_datasets(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []

    datasets: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        info = _read_workspace_dataset_info(root, entry)
        if info is not None:
            datasets.append(info)
            continue
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir():
                continue
            info = _read_workspace_dataset_info(root, sub)
            if info is not None:
                datasets.append(info)
    return datasets


def _read_workspace_dataset_info(root: Path, dataset_dir: Path) -> dict[str, Any] | None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return None

    raw = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = raw.get("total_episodes", 0)
    total_frames = raw.get("total_frames", 0)
    fps = raw.get("fps", 0)

    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    episode_lengths: list[int] = []
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            episode_lengths.append(int(entry.get("length", 0) or 0))

    return {
        "name": dataset_dir.relative_to(root).as_posix(),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "episode_lengths": episode_lengths,
        "features": list(raw.get("features", {}).keys()),
        "robot_type": raw.get("robot_type", ""),
        "source_dataset": raw.get("repo_id") or raw.get("dataset_id") or dataset_dir.relative_to(root).as_posix(),
    }


def resolve_dataset_handle_or_workspace(name: str) -> Path:
    if is_session_handle(name):
        return resolve_session_dataset_path(name)

    root = _datasets_root().resolve()
    candidate = (root / name).resolve()
    if candidate.is_dir() and str(candidate).startswith(str(root) + "/"):
        return candidate

    for parent in root.iterdir() if root.exists() else []:
        nested = (parent / name).resolve()
        if parent.is_dir() and nested.is_dir() and str(nested).startswith(str(root) + "/"):
            return nested

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

    from roboclaw.data.datasets import list_datasets

    root = _datasets_root().resolve()
    for item in list_datasets(root):
        if item["name"] == name:
            return {
                **item,
                "display_name": item["name"],
                "source_kind": "workspace",
            }
    raise FileNotFoundError(f"Dataset '{name}' not found")
