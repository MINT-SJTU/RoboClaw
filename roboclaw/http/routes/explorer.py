"""FastAPI routes for remote and local dataset explorer flows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, TypeVar
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub.errors import HFValidationError, HfHubHTTPError, RepositoryNotFoundError
from loguru import logger
from pydantic import BaseModel

from roboclaw.data.curation.features import (
    build_joint_trajectory_payload,
    extract_action_names,
    extract_state_names,
)
from roboclaw.data.curation.paths import datasets_root
from roboclaw.data.curation.serializers import episode_time_bounds
from roboclaw.data.dataset_sessions import (
    create_uploaded_directory_session,
    register_remote_dataset_session,
)
from roboclaw.data.explorer.dual_source import (
    list_local_dataset_options,
    normalize_explorer_source,
    resolve_local_dataset_path,
    resolve_path_dataset,
)
from roboclaw.data.explorer.local import (
    build_explorer_episode_page_from_artifacts,
    build_explorer_overview_from_artifacts,
    build_explorer_summary_from_info,
    load_episodes_list_file,
    load_json_file,
    scan_dataset_siblings,
)
from roboclaw.data.explorer.remote import (
    build_remote_dataset_info,
    build_remote_episode_page,
    build_remote_explorer_details,
    build_remote_explorer_payload,
    build_remote_explorer_summary,
    load_remote_episode_detail,
    search_remote_datasets,
)


class ExplorerPrepareRequest(BaseModel):
    dataset_id: str
    include_videos: bool = False
    force: bool = False


T = TypeVar("T")
_MAX_LOCAL_DIRECTORY_UPLOAD_FILES = 20_000
_MAX_LOCAL_DIRECTORY_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _remote_dataset_not_accessible_detail(dataset_name: str) -> str:
    return f"Remote dataset '{dataset_name}' was not found or is not accessible"


def _remote_dataset_http_exception(dataset_name: str, exc: HfHubHTTPError | httpx.HTTPError) -> HTTPException:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {401, 403, 404}:
        return HTTPException(
            status_code=404,
            detail=_remote_dataset_not_accessible_detail(dataset_name),
        )
    if status_code == 429:
        return HTTPException(
            status_code=503,
            detail=f"Remote dataset '{dataset_name}' is temporarily rate limited by the upstream service",
        )
    return HTTPException(
        status_code=502,
        detail=f"Failed to load remote dataset '{dataset_name}' from the upstream service",
    )


async def _run_remote_dataset_call(
    dataset_name: str,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except RepositoryNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=_remote_dataset_not_accessible_detail(dataset_name),
        ) from exc
    except HFValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HfHubHTTPError as exc:
        raise _remote_dataset_http_exception(dataset_name, exc) from exc
    except httpx.HTTPError as exc:
        raise _remote_dataset_http_exception(dataset_name, exc) from exc


def _normalize_explorer_source_or_http(source: str | None):
    try:
        return normalize_explorer_source(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _local_dataset_name(dataset_path: Path) -> str:
    root = datasets_root().resolve()
    resolved = dataset_path.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return dataset_path.name


def _build_local_explorer_details(dataset_path: Path, dataset_name: str) -> dict[str, Any]:
    info = load_json_file(dataset_path / "meta" / "info.json")
    stats = load_json_file(dataset_path / "meta" / "stats.json")
    siblings = scan_dataset_siblings(dataset_path)
    return build_explorer_overview_from_artifacts(
        dataset_name=dataset_name,
        info=info,
        stats=stats,
        siblings=siblings,
    )


def _build_local_explorer_summary(dataset_path: Path, dataset_name: str) -> dict[str, Any]:
    info = load_json_file(dataset_path / "meta" / "info.json")
    return build_explorer_summary_from_info(dataset_name, info)


def _build_local_episode_page(
    dataset_path: Path,
    dataset_name: str,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    info = load_json_file(dataset_path / "meta" / "info.json")
    episodes_meta = load_episodes_list_file(dataset_path)
    return build_explorer_episode_page_from_artifacts(
        dataset_name=dataset_name,
        info=info,
        episodes_meta=episodes_meta,
        page=page,
        page_size=page_size,
    )


def _serialize_sample_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        serialized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list) and len(value) > 6:
                serialized[key] = value[:4] + ["..."]
            elif hasattr(value, "tolist"):
                lst = value.tolist()
                serialized[key] = lst[:4] + ["..."] if len(lst) > 6 else lst
            else:
                serialized[key] = value
        result.append(serialized)
    return result


def _empty_joint_payload() -> dict[str, Any]:
    return {
        "x_axis_key": "time",
        "x_values": [],
        "time_values": [],
        "frame_values": [],
        "joint_trajectories": [],
        "sampled_points": 0,
        "total_points": 0,
    }


def load_episode_data(dataset_path: Path, episode_index: int) -> dict[str, Any]:
    from roboclaw.data.curation.validators import load_episode_data as _load_episode_data

    return _load_episode_data(dataset_path, episode_index)


def _build_local_episode_payload(
    dataset_path: Path,
    dataset_name: str,
    episode_index: int,
    *,
    preview: bool,
    source: str,
) -> dict[str, Any]:
    data = load_episode_data(dataset_path, episode_index)
    info = data.get("info", {})
    rows = data.get("rows", [])
    action_names = extract_action_names(info)
    state_names = extract_state_names(info)
    start_ts, end_ts = episode_time_bounds(rows)
    duration_s = max(end_ts - start_ts, 0.0) if start_ts is not None and end_ts is not None else 0.0

    videos: list[dict[str, Any]] = []
    for video_path in data.get("video_files", []):
        relative_path = video_path.relative_to(dataset_path).as_posix()
        if source == "path":
            url = (
                f"/api/explorer/local-video/{relative_path}"
                f"?source=path&dataset_path={quote(dataset_path.as_posix(), safe='')}"
            )
        else:
            url = (
                f"/api/explorer/local-video/{relative_path}"
                f"?source=local&dataset={quote(dataset_name, safe='')}"
            )
        videos.append({
            "path": relative_path,
            "url": url,
            "stream": Path(relative_path).stem,
            "from_timestamp": 0,
            "to_timestamp": duration_s if duration_s > 0 else None,
        })

    return {
        "episode_index": episode_index,
        "summary": {
            "row_count": len(rows),
            "fps": info.get("fps", 0),
            "duration_s": round(duration_s, 2),
            "video_count": len(videos),
        },
        "sample_rows": [] if preview else _serialize_sample_rows(rows[:5]),
        "joint_trajectory": _empty_joint_payload()
        if preview
        else build_joint_trajectory_payload(rows, action_names, state_names),
        "videos": videos,
    }


def _resolve_dataset_context(
    *,
    source: str | None,
    dataset: str | None,
    path: str | None,
) -> tuple[str, str | None, Path | None]:
    resolved_source = _normalize_explorer_source_or_http(source)
    if resolved_source == "remote":
        if not dataset or not dataset.strip():
            raise HTTPException(status_code=400, detail="Remote explorer requests require a dataset id")
        return resolved_source, dataset.strip(), None
    if resolved_source == "local":
        if not dataset or not dataset.strip():
            raise HTTPException(
                status_code=400,
                detail="Local explorer requests require a local dataset name",
            )
        try:
            dataset_path = resolve_local_dataset_path(dataset.strip())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return resolved_source, _local_dataset_name(dataset_path), dataset_path

    if not path or not path.strip():
        raise HTTPException(
            status_code=400,
            detail="Path explorer requests require a local dataset path",
        )
    try:
        dataset_path = resolve_path_dataset(path.strip())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset_name = dataset.strip() if dataset and dataset.strip() else dataset_path.name
    return resolved_source, dataset_name, dataset_path


def _validate_upload_relative_path(relative_path: str) -> str:
    value = relative_path.strip()
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or candidate.name in {"", ".", ".."}
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise HTTPException(status_code=400, detail=f"Invalid uploaded file path '{relative_path}'")
    return candidate.as_posix()


async def _spool_upload_to_path(
    upload: UploadFile,
    target: Path,
    *,
    remaining_bytes: int,
) -> int:
    written = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(_UPLOAD_CHUNK_SIZE):
            written += len(chunk)
            if written > remaining_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded dataset directory exceeds the maximum supported size",
                )
            handle.write(chunk)
    return written


def register_explorer_routes(app: FastAPI) -> None:
    """Register all explorer API routes on *app*."""

    @app.get("/api/explorer/datasets")
    async def explorer_datasets(
        query: str = "",
        limit: int = 8,
        source: str = "remote",
    ) -> list[dict]:
        safe_limit = max(1, min(limit, 50))
        resolved_source = _normalize_explorer_source_or_http(source)
        needle = query.strip().lower()
        if resolved_source == "remote":
            if not needle:
                return []
            return await _run_remote_dataset_call(query, search_remote_datasets, query, safe_limit)

        local_items = await asyncio.to_thread(list_local_dataset_options)
        if needle:
            local_items = [
                item
                for item in local_items
                if needle in item["id"].lower() or needle in item["path"].lower()
            ]
        return local_items[:safe_limit]

    @app.get("/api/explorer/dashboard")
    async def explorer_dashboard(
        dataset: str | None = None,
        source: str = "remote",
        path: str | None = None,
    ) -> dict[str, Any]:
        resolved_source, dataset_name, dataset_path = _resolve_dataset_context(
            source=source,
            dataset=dataset,
            path=path,
        )
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(
                dataset_name,
                build_remote_explorer_payload,
                dataset_name,
            )
        else:
            payload = await asyncio.to_thread(_build_local_explorer_details, dataset_path, dataset_name)
        logger.info("Explorer dashboard loaded for '{}' ({})", dataset_name, resolved_source)
        return payload

    @app.get("/api/explorer/summary")
    async def explorer_summary(
        dataset: str | None = None,
        source: str = "remote",
        path: str | None = None,
    ) -> dict[str, Any]:
        resolved_source, dataset_name, dataset_path = _resolve_dataset_context(
            source=source,
            dataset=dataset,
            path=path,
        )
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(
                dataset_name,
                build_remote_explorer_summary,
                dataset_name,
            )
        else:
            payload = await asyncio.to_thread(_build_local_explorer_summary, dataset_path, dataset_name)
        logger.info("Explorer summary loaded for '{}' ({})", dataset_name, resolved_source)
        return payload

    @app.get("/api/explorer/details")
    async def explorer_details(
        dataset: str | None = None,
        source: str = "remote",
        path: str | None = None,
    ) -> dict[str, Any]:
        resolved_source, dataset_name, dataset_path = _resolve_dataset_context(
            source=source,
            dataset=dataset,
            path=path,
        )
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(
                dataset_name,
                build_remote_explorer_details,
                dataset_name,
            )
        else:
            payload = await asyncio.to_thread(_build_local_explorer_details, dataset_path, dataset_name)
        logger.info("Explorer details loaded for '{}' ({})", dataset_name, resolved_source)
        return payload

    @app.get("/api/explorer/episodes")
    async def explorer_episodes(
        dataset: str | None = None,
        source: str = "remote",
        path: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        safe_page_size = max(1, min(page_size, 200))
        resolved_source, dataset_name, dataset_path = _resolve_dataset_context(
            source=source,
            dataset=dataset,
            path=path,
        )
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(
                dataset_name,
                build_remote_episode_page,
                dataset_name,
                page,
                safe_page_size,
            )
        else:
            payload = await asyncio.to_thread(
                _build_local_episode_page,
                dataset_path,
                dataset_name,
                page,
                safe_page_size,
            )
        logger.info(
            "Explorer episode page loaded for '{}' ({}) page {} size {}",
            dataset_name,
            resolved_source,
            payload.get("page"),
            payload.get("page_size"),
        )
        return payload

    @app.get("/api/explorer/episode")
    async def explorer_episode(
        dataset: str | None = None,
        source: str = "remote",
        path: str | None = None,
        episode_index: int = 0,
        preview: bool = False,
        preview_only: bool = False,
    ) -> dict[str, Any]:
        preview_requested = preview or preview_only
        resolved_source, dataset_name, dataset_path = _resolve_dataset_context(
            source=source,
            dataset=dataset,
            path=path,
        )
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(
                dataset_name,
                load_remote_episode_detail,
                dataset_name,
                episode_index,
                preview_only=preview_requested,
            )
        else:
            payload = await asyncio.to_thread(
                _build_local_episode_payload,
                dataset_path,
                dataset_name,
                episode_index,
                preview=preview_requested,
                source=resolved_source,
            )
        logger.info("Explorer episode loaded for '{}' ({}) #{}", dataset_name, resolved_source, episode_index)
        return payload

    @app.get("/api/explorer/dataset-info")
    async def explorer_dataset_info(
        dataset: str | None = None,
        source: str = "remote",
        path: str | None = None,
    ) -> dict[str, Any]:
        resolved_source, dataset_name, dataset_path = _resolve_dataset_context(
            source=source,
            dataset=dataset,
            path=path,
        )
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(
                dataset_name,
                build_remote_dataset_info,
                dataset_name,
            )
        else:
            details = await asyncio.to_thread(_build_local_explorer_summary, dataset_path, dataset_name)
            info = load_json_file(dataset_path / "meta" / "info.json")
            episodes_meta = load_episodes_list_file(dataset_path)
            payload = {
                "name": details["dataset"],
                "total_episodes": details["summary"]["total_episodes"],
                "total_frames": details["summary"]["total_frames"],
                "fps": details["summary"]["fps"],
                "episode_lengths": [
                    int(entry.get("length", 0) or 0)
                    for entry in episodes_meta
                ],
                "features": list((info.get("features") or {}).keys()) if isinstance(info, dict) else [],
                "robot_type": details["summary"]["robot_type"],
                "source_dataset": details["dataset"],
            }
        logger.info("Explorer dataset info loaded for '{}' ({})", dataset_name, resolved_source)
        return payload

    @app.get("/api/explorer/suggest")
    async def explorer_suggest(
        q: str,
        limit: int = 8,
        source: str = "remote",
    ) -> list[dict[str, Any]]:
        resolved_source = _normalize_explorer_source_or_http(source)
        safe_limit = max(1, min(limit, 12))
        if resolved_source == "remote":
            payload = await _run_remote_dataset_call(q, search_remote_datasets, q, safe_limit)
        else:
            needle = q.strip().lower()
            local_items = await asyncio.to_thread(list_local_dataset_options)
            payload = [
                item
                for item in local_items
                if needle in item["id"].lower() or needle in item["path"].lower()
            ][:safe_limit]
        logger.info("Explorer dataset suggestions loaded for '{}' ({})", q, resolved_source)
        return payload

    @app.post("/api/explorer/prepare-remote")
    async def explorer_prepare_remote(body: ExplorerPrepareRequest) -> dict[str, Any]:
        payload = await _run_remote_dataset_call(
            body.dataset_id,
            register_remote_dataset_session,
            body.dataset_id,
            include_videos=body.include_videos,
            force=body.force,
        )
        logger.info("Explorer prepared remote dataset '{}' for workflow", body.dataset_id)
        return payload

    @app.post("/api/explorer/local-directory-session")
    async def explorer_local_directory_session(
        files: list[UploadFile] = File(...),
        relative_paths: list[str] = Form(...),
        display_name: str | None = Form(None),
    ) -> dict[str, Any]:
        if len(files) != len(relative_paths):
            raise HTTPException(status_code=400, detail="files and relative_paths length mismatch")
        if not files:
            raise HTTPException(status_code=400, detail="Uploaded dataset directory is empty")
        if len(files) > _MAX_LOCAL_DIRECTORY_UPLOAD_FILES:
            raise HTTPException(status_code=413, detail="Uploaded dataset directory has too many files")

        validated_paths = [_validate_upload_relative_path(relative_path) for relative_path in relative_paths]
        if len(set(validated_paths)) != len(validated_paths):
            raise HTTPException(status_code=400, detail="Uploaded dataset directory contains duplicate paths")

        with TemporaryDirectory() as temp_dir:
            total_bytes = 0
            file_payloads: list[tuple[str, Path]] = []
            temp_root = Path(temp_dir)
            for index, (upload, relative_path) in enumerate(zip(files, validated_paths)):
                temp_path = temp_root / str(index)
                total_bytes += await _spool_upload_to_path(
                    upload,
                    temp_path,
                    remaining_bytes=_MAX_LOCAL_DIRECTORY_UPLOAD_BYTES - total_bytes,
                )
                file_payloads.append((relative_path, temp_path))

            payload = await asyncio.to_thread(
                create_uploaded_directory_session,
                files=file_payloads,
                display_name=display_name,
            )
        logger.info("Explorer created local directory session '{}'", payload["dataset_name"])
        return payload

    @app.get("/api/explorer/local-video/{path:path}")
    async def explorer_local_video(
        path: str,
        dataset: str | None = None,
        source: str = "local",
        dataset_path: str | None = None,
    ) -> FileResponse:
        resolved_source = _normalize_explorer_source_or_http(source)
        if resolved_source == "path":
            try:
                root = resolve_path_dataset(dataset_path or "")
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            if not dataset:
                raise HTTPException(
                    status_code=400,
                    detail="Local explorer video requests require a dataset name",
                )
            try:
                root = resolve_local_dataset_path(dataset)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        root = root.resolve()
        video_path = (root / path).resolve()
        if not video_path.is_relative_to(root):
            raise HTTPException(status_code=403, detail="Path traversal not allowed")
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail=f"Video file '{video_path}' not found")
        return FileResponse(str(video_path), media_type="video/mp4", filename=video_path.name)
