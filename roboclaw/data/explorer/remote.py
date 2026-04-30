"""Remote-first Hugging Face dataset helpers for the explorer lane.

Uses plain HTTP requests plus the HF Dataset Viewer API for row-level queries.
Metadata parquet files are parsed in-memory from downloaded bytes. No local
dataset workspace or hf_hub_download is used by the explorer lane.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, HFValidationError, RepositoryNotFoundError
from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
from loguru import logger

from roboclaw.data.curation.features import (
    build_joint_trajectory_payload,
    first_present_value,
    resolve_timestamp,
)
from roboclaw.data.datasets import extract_action_names, extract_state_names
from roboclaw.data.explorer.local import (
    build_explorer_episode_page_from_artifacts,
    build_explorer_overview_from_artifacts,
    build_explorer_payload_from_artifacts,
    build_explorer_summary_from_info,
)

_HF_API = HfApi()
_HF_BASE_URL = os.getenv(
    "HF_BASE_URL",
    "https://huggingface.co",
).rstrip("/")
_HF_VIEWER_BASE = os.getenv(
    "HF_DATASET_VIEWER_BASE_URL",
    "https://datasets-server.huggingface.co",
).rstrip("/")
_HF_TOKEN = os.getenv("HF_TOKEN", "")
_VIEWER_TIMEOUT = 30
_VIEWER_PAGE_SIZE = 100
_VIEWER_RETRY_DELAYS = (0.3, 0.8)
_EPISODE_META_INT_FIELDS = {
    "episode_index": ("episode_index", "0"),
    "length": ("length", "9"),
    "data/chunk_index": ("data/chunk_index", "data_chunk_index", "1"),
    "data/file_index": ("data/file_index", "data_file_index", "2"),
    "dataset_from_index": ("dataset_from_index", "3"),
    "dataset_to_index": ("dataset_to_index", "4"),
    "video_chunk_index": ("video_chunk_index", "5"),
    "video_file_index": ("video_file_index", "6"),
}
_EPISODE_META_FLOAT_FIELDS = {
    "video_from_timestamp": ("video_from_timestamp", "7"),
    "video_to_timestamp": ("video_to_timestamp", "8"),
}


def _pyarrow_modules() -> tuple[Any, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def search_remote_datasets(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Return lightweight dataset suggestions for the explorer search box."""
    needle = query.strip()
    if not needle:
        return []

    safe_limit = max(1, min(limit, 12))
    suggestions: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        items = _HF_API.list_datasets(
            search=needle,
            limit=safe_limit,
            full=False,
            token=_HF_TOKEN or None,
        )
        for item in items:
            dataset_id = str(getattr(item, "id", "") or "").strip()
            if not dataset_id or dataset_id in seen:
                continue
            seen.add(dataset_id)
            suggestions.append({"id": dataset_id})
    except (HFValidationError, HfHubHTTPError, RepositoryNotFoundError) as exc:
        logger.warning("HF dataset suggestion lookup failed for '{}': {}", needle, exc)
        return []
    return suggestions


def _repo_file_url(dataset: str, filename: str) -> str:
    return f"{_HF_BASE_URL}/datasets/{quote(dataset, safe='/')}/resolve/main/{quote(filename, safe='/')}"


def _load_json_bytes(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _load_jsonl_bytes(raw: bytes) -> list[dict[str, Any]]:
    if not raw:
        return []
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_repo_path(template: str | None, **values: Any) -> str | None:
    if not isinstance(template, str) or not template.strip():
        return None
    try:
        return template.format(**values)
    except (IndexError, KeyError, ValueError):
        return None


def _video_feature_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return []
    return [
        str(name)
        for name, config in features.items()
        if isinstance(config, dict) and str(config.get("dtype", "")).lower() == "video"
    ]


def _episode_meta_columns(info: dict[str, Any]) -> list[str]:
    columns = [
        "episode_index",
        "length",
        "data/chunk_index",
        "data_chunk_index",
        "data/file_index",
        "data_file_index",
        "dataset_from_index",
        "dataset_to_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
        "video_chunk_index",
        "video_file_index",
        "video_from_timestamp",
        "video_to_timestamp",
        *[str(index) for index in range(10)],
    ]
    for video_key in _video_feature_keys(info):
        prefix = f"videos/{video_key}/"
        columns.extend([
            f"{prefix}chunk_index",
            f"{prefix}file_index",
            f"{prefix}from_timestamp",
            f"{prefix}to_timestamp",
        ])
    return columns


def _episode_data_columns(info: dict[str, Any]) -> list[str]:
    numeric_dtypes = {
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "boolean",
    }
    columns = [
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
        "next.reward",
        "next.done",
        "action",
        "observation.state",
    ]
    features = info.get("features", {})
    if isinstance(features, dict):
        for key, config in features.items():
            if not isinstance(config, dict):
                continue
            dtype = str(config.get("dtype", "")).lower()
            shape = config.get("shape")
            if dtype in numeric_dtypes and isinstance(shape, list) and len(shape) <= 1:
                columns.append(str(key))
    return list(dict.fromkeys(columns))


def _read_parquet_rows_from_bytes(raw: bytes, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if not raw:
        return []
    pa, pq = _pyarrow_modules()
    buffer = pa.BufferReader(raw)
    parquet_file = pq.ParquetFile(buffer)
    available_columns = parquet_file.schema_arrow.names
    selected_columns = columns or available_columns
    valid_columns = [column for column in selected_columns if column in available_columns]
    table = parquet_file.read(columns=valid_columns or None)
    return table.to_pylist()


def _normalize_episode_meta_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalize_episode_meta_row(
    row: dict[str, Any],
    info: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        key: _normalize_episode_meta_value(value)
        for key, value in row.items()
    }

    int_values: dict[str, int | None] = {}
    for field, aliases in _EPISODE_META_INT_FIELDS.items():
        value = _safe_int(first_present_value(normalized, list(aliases)))
        int_values[field] = value
        if value is not None:
            normalized[field] = value

    float_values: dict[str, float | None] = {}
    for field, aliases in _EPISODE_META_FLOAT_FIELDS.items():
        value = _safe_float(first_present_value(normalized, list(aliases)))
        float_values[field] = value
        if value is not None:
            normalized[field] = value

    for video_key in _video_feature_keys(info):
        prefix = f"videos/{video_key}/"
        chunk_key = f"{prefix}chunk_index"
        file_key = f"{prefix}file_index"
        from_key = f"{prefix}from_timestamp"
        to_key = f"{prefix}to_timestamp"

        chunk_index = _safe_int(normalized.get(chunk_key))
        if chunk_index is None:
            chunk_index = int_values.get("video_chunk_index")
        if chunk_index is not None:
            normalized[chunk_key] = chunk_index

        file_index = _safe_int(normalized.get(file_key))
        if file_index is None:
            file_index = int_values.get("video_file_index")
        if file_index is not None:
            normalized[file_key] = file_index

        from_timestamp = _safe_float(normalized.get(from_key))
        if from_timestamp is None:
            from_timestamp = float_values.get("video_from_timestamp")
        if from_timestamp is not None:
            normalized[from_key] = from_timestamp

        to_timestamp = _safe_float(normalized.get(to_key))
        if to_timestamp is None:
            to_timestamp = float_values.get("video_to_timestamp")
        if to_timestamp is not None:
            normalized[to_key] = to_timestamp

    return normalized


def _normalize_episodes_meta(
    rows: list[dict[str, Any]],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized_rows = [
        _normalize_episode_meta_row(row, info)
        for row in rows
        if isinstance(row, dict)
    ]
    return sorted(
        normalized_rows,
        key=lambda row: _safe_int(row.get("episode_index")) or 0,
    )


def _fetch_optional_bytes(url: str) -> bytes | None:
    try:
        resp = httpx.get(url, headers=_viewer_headers(), timeout=_VIEWER_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except httpx.HTTPError:
        return None


@lru_cache(maxsize=32)
def _get_remote_dataset_repo_index(dataset: str) -> dict[str, Any]:
    info = _HF_API.dataset_info(dataset)
    return {
        "dataset": str(info.id or dataset),
        "siblings": [
            {"rfilename": item.rfilename}
            for item in (info.siblings or [])
            if getattr(item, "rfilename", None)
        ],
    }


@lru_cache(maxsize=32)
def _get_remote_info_json(dataset: str) -> dict[str, Any]:
    info_bytes = _fetch_optional_bytes(_repo_file_url(dataset, "meta/info.json"))
    return _load_json_bytes(info_bytes) if info_bytes else {}


@lru_cache(maxsize=32)
def _get_remote_stats_json(dataset: str) -> dict[str, Any]:
    stats_bytes = _fetch_optional_bytes(_repo_file_url(dataset, "meta/stats.json"))
    return _load_json_bytes(stats_bytes) if stats_bytes else {}


def _load_remote_episode_parquet_meta(
    dataset: str,
    siblings: list[dict[str, Any]],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    parquet_files = sorted(
        sibling["rfilename"]
        for sibling in siblings
        if sibling.get("rfilename", "").startswith("meta/episodes/")
        and sibling.get("rfilename", "").endswith(".parquet")
    )
    if not parquet_files:
        return []

    rows: list[dict[str, Any]] = []
    columns = _episode_meta_columns(info)
    for path in parquet_files:
        raw = _fetch_optional_bytes(_repo_file_url(dataset, path))
        rows.extend(_read_parquet_rows_from_bytes(raw or b"", columns=columns))
    return _normalize_episodes_meta(rows, info)


def _get_remote_episodes_meta(
    dataset: str,
    siblings: list[dict[str, Any]],
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    episodes_jsonl_bytes = _fetch_optional_bytes(_repo_file_url(dataset, "meta/episodes.jsonl"))
    if episodes_jsonl_bytes:
        return _normalize_episodes_meta(_load_jsonl_bytes(episodes_jsonl_bytes), info)
    return _load_remote_episode_parquet_meta(dataset, siblings, info)


@lru_cache(maxsize=32)
def get_remote_dataset_artifacts(dataset: str) -> dict[str, Any]:
    repo_index = _get_remote_dataset_repo_index(dataset)

    info_json = _get_remote_info_json(dataset)
    stats_json = _get_remote_stats_json(dataset)
    episodes_meta = _get_remote_episodes_meta(dataset, repo_index["siblings"], info_json)

    return {
        "dataset": repo_index["dataset"],
        "siblings": repo_index["siblings"],
        "info": info_json,
        "stats": stats_json,
        "episodes_meta": episodes_meta,
    }


def build_remote_explorer_payload(dataset: str) -> dict[str, Any]:
    artifacts = get_remote_dataset_artifacts(dataset)
    return build_explorer_payload_from_artifacts(
        dataset_name=artifacts["dataset"],
        info=artifacts["info"],
        stats=artifacts["stats"],
        siblings=artifacts["siblings"],
        episodes_meta=artifacts["episodes_meta"],
    )


def build_remote_explorer_summary(dataset: str) -> dict[str, Any]:
    repo_index = _get_remote_dataset_repo_index(dataset)
    info = _get_remote_info_json(dataset)
    return build_explorer_summary_from_info(repo_index["dataset"], info)


def build_remote_explorer_details(dataset: str) -> dict[str, Any]:
    repo_index = _get_remote_dataset_repo_index(dataset)
    info = _get_remote_info_json(dataset)
    stats = _get_remote_stats_json(dataset)
    return build_explorer_overview_from_artifacts(
        dataset_name=repo_index["dataset"],
        info=info,
        stats=stats,
        siblings=repo_index["siblings"],
    )


def build_remote_episode_page(dataset: str, page: int, page_size: int) -> dict[str, Any]:
    artifacts = get_remote_dataset_artifacts(dataset)
    return build_explorer_episode_page_from_artifacts(
        dataset_name=artifacts["dataset"],
        info=artifacts["info"],
        episodes_meta=artifacts["episodes_meta"],
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# HF Dataset Viewer API
# ---------------------------------------------------------------------------


def _viewer_headers() -> dict[str, str]:
    headers = {"User-Agent": "RoboClaw/1.0"}
    if _HF_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    return headers


def _viewer_fetch_json(url: str) -> dict[str, Any]:
    """Fetch JSON from the HF Dataset Viewer API with retry."""
    last_exc: Exception | None = None
    for attempt in range(1 + len(_VIEWER_RETRY_DELAYS)):
        try:
            resp = httpx.get(url, headers=_viewer_headers(), timeout=_VIEWER_TIMEOUT)
            if resp.status_code in (408, 429, 500, 502, 503, 504) and attempt < len(
                _VIEWER_RETRY_DELAYS
            ):
                time.sleep(_VIEWER_RETRY_DELAYS[attempt])
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < len(_VIEWER_RETRY_DELAYS):
                time.sleep(_VIEWER_RETRY_DELAYS[attempt])
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Viewer API request failed without an HTTP exception")


@lru_cache(maxsize=32)
def _viewer_get_split(dataset: str) -> tuple[str, str]:
    """Return (config, split) for a dataset via the Viewer /splits endpoint."""
    url = f"{_HF_VIEWER_BASE}/splits?dataset={quote(dataset, safe='')}"
    payload = _viewer_fetch_json(url)
    splits = payload.get("splits", [])
    if not splits:
        raise ValueError(f"No splits found for dataset '{dataset}'")
    first = splits[0]
    return first["config"], first["split"]


def _viewer_fetch_episode_rows(
    dataset: str,
    config: str,
    split: str,
    episode_index: int,
    length: int = 500,
) -> list[dict[str, Any]]:
    """Fetch rows for a single episode via the Viewer /filter endpoint."""
    remaining = max(0, int(length or 0))
    if remaining == 0:
        return []

    where = f'"episode_index"={episode_index}'
    rows: list[dict[str, Any]] = []
    offset = 0
    while remaining > 0:
        page_size = min(_VIEWER_PAGE_SIZE, remaining)
        url = (
            f"{_HF_VIEWER_BASE}/filter"
            f"?dataset={quote(dataset, safe='')}"
            f"&config={quote(config, safe='')}"
            f"&split={quote(split, safe='')}"
            f"&where={quote(where, safe='')}"
            f"&offset={offset}&length={page_size}"
        )
        payload = _viewer_fetch_json(url)
        page_rows = [
            entry.get("row", {})
            for entry in payload.get("rows", [])
            if isinstance(entry, dict)
        ]
        if not page_rows:
            break
        rows.extend(row for row in page_rows if isinstance(row, dict))
        fetched = len(page_rows)
        if fetched < page_size:
            break
        offset += fetched
        remaining -= fetched
    return rows


# ---------------------------------------------------------------------------
# Episode detail
# ---------------------------------------------------------------------------


def _build_episode_payload(
    rows: list[dict[str, Any]],
    info: dict[str, Any],
    episode_index: int,
    videos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the episode detail response from rows."""
    action_names = extract_action_names(info)
    state_names = extract_state_names(info)
    joint_trajectory = build_joint_trajectory_payload(rows, action_names, state_names)

    timestamps = [t for row in rows if (t := resolve_timestamp(row)) is not None]
    start_ts = timestamps[0] if timestamps else None
    end_ts = timestamps[-1] if timestamps else None
    duration_s = max(end_ts - start_ts, 0.0) if start_ts is not None and end_ts is not None else 0.0

    return {
        "episode_index": episode_index,
        "summary": {
            "row_count": len(rows),
            "fps": info.get("fps", 0),
            "duration_s": round(duration_s, 2),
            "video_count": len(videos),
        },
        "sample_rows": _serialize_sample_rows(rows[:5]),
        "joint_trajectory": joint_trajectory,
        "videos": videos,
    }


def _build_episode_preview_payload(
    info: dict[str, Any],
    episode_index: int,
    videos: list[dict[str, Any]],
    episode_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    fps = int(info.get("fps", 0) or 0)
    row_count = _safe_int(episode_meta.get("length")) if episode_meta else None
    clip_durations = [
        end - start
        for video in videos
        if (start := _safe_float(video.get("from_timestamp"))) is not None
        and (end := _safe_float(video.get("to_timestamp"))) is not None
        and end >= start
    ]
    duration_s = max(clip_durations) if clip_durations else (
        (row_count / fps) if row_count and fps > 0 else 0.0
    )

    return {
        "episode_index": episode_index,
        "summary": {
            "row_count": row_count or 0,
            "fps": fps,
            "duration_s": round(float(duration_s), 2),
            "video_count": len(videos),
        },
        "sample_rows": [],
        "joint_trajectory": build_joint_trajectory_payload([], [], []),
        "videos": videos,
    }


def _resolve_episode_videos(
    artifacts: dict[str, Any],
    dataset: str,
    episode_index: int,
    info: dict[str, Any],
    episode_meta: dict[str, Any] | None,
    chunks_size: int,
) -> list[dict[str, Any]]:
    shared_videos = _resolve_shared_video_clips(dataset, info, episode_meta)
    if shared_videos:
        return shared_videos

    chunk = f"{episode_index // chunks_size:03d}"
    chunk_prefix = f"videos/chunk-{chunk}/"
    episode_filename = f"episode_{episode_index:06d}.mp4"
    videos = [
        {
            "path": sibling["rfilename"],
            "url": _repo_file_url(dataset, sibling["rfilename"]),
            "stream": Path(sibling["rfilename"]).parent.name,
            "from_timestamp": None,
            "to_timestamp": None,
        }
        for sibling in artifacts["siblings"]
        if sibling["rfilename"].startswith(chunk_prefix)
        and sibling["rfilename"].endswith(episode_filename)
    ]
    return sorted(videos, key=lambda item: item["stream"])


def _episode_meta_entry(
    episodes_meta: list[dict[str, Any]],
    episode_index: int,
) -> dict[str, Any] | None:
    for entry in episodes_meta:
        if _safe_int(entry.get("episode_index")) == episode_index:
            return entry
    return None


def _stream_name_from_video_key(video_key: str) -> str:
    if video_key.startswith("observation.images."):
        return video_key.split("observation.images.", 1)[1]
    if "." in video_key:
        return video_key.rsplit(".", 1)[-1]
    return video_key


def _resolve_shared_video_clips(
    dataset: str,
    info: dict[str, Any],
    episode_meta: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not episode_meta:
        return []

    path_template = info.get("video_path") or DEFAULT_VIDEO_PATH
    videos: list[dict[str, Any]] = []
    for video_key in _video_feature_keys(info):
        prefix = f"videos/{video_key}/"
        chunk_index = _safe_int(episode_meta.get(f"{prefix}chunk_index"))
        if chunk_index is None:
            chunk_index = _safe_int(episode_meta.get("video_chunk_index"))
        file_index = _safe_int(episode_meta.get(f"{prefix}file_index"))
        if file_index is None:
            file_index = _safe_int(episode_meta.get("video_file_index"))
        if chunk_index is None or file_index is None:
            continue

        path = _render_repo_path(
            path_template,
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        if not path:
            path = DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )

        videos.append({
            "path": path,
            "url": _repo_file_url(dataset, path),
            "stream": _stream_name_from_video_key(video_key),
            "from_timestamp": _safe_float(episode_meta.get(f"{prefix}from_timestamp"))
            if episode_meta.get(f"{prefix}from_timestamp") is not None
            else _safe_float(episode_meta.get("video_from_timestamp")),
            "to_timestamp": _safe_float(episode_meta.get(f"{prefix}to_timestamp"))
            if episode_meta.get(f"{prefix}to_timestamp") is not None
            else _safe_float(episode_meta.get("video_to_timestamp")),
        })

    return sorted(videos, key=lambda item: item["stream"])


def _resolve_shared_data_path(info: dict[str, Any], episode_meta: dict[str, Any] | None) -> str | None:
    if not episode_meta:
        return None

    chunk_index = _safe_int(episode_meta.get("data/chunk_index"))
    if chunk_index is None:
        chunk_index = _safe_int(episode_meta.get("data_chunk_index"))
    file_index = _safe_int(episode_meta.get("data/file_index"))
    if file_index is None:
        file_index = _safe_int(episode_meta.get("data_file_index"))
    if chunk_index is None or file_index is None:
        return None

    path = _render_repo_path(
        info.get("data_path") or DEFAULT_DATA_PATH,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    if path:
        return path
    return DEFAULT_DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)


def _select_rows_for_episode(
    rows: list[dict[str, Any]],
    episode_index: int,
    episode_meta: dict[str, Any] | None,
    *,
    allow_unfiltered: bool = False,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if episode_meta:
        start = _safe_int(episode_meta.get("dataset_from_index"))
        stop = _safe_int(episode_meta.get("dataset_to_index"))
        if start is not None and stop is not None:
            first_index = _safe_int(rows[0].get("index"))
            if first_index is not None:
                local_start = max(0, start - first_index)
                local_stop = max(local_start, stop - first_index)
                sliced = rows[local_start:local_stop]
                if sliced:
                    return sliced

            ranged = [
                row for row in rows
                if (index_value := _safe_int(row.get("index"))) is not None
                and start <= index_value < stop
            ]
            if ranged:
                return ranged

    filtered = [row for row in rows if _safe_int(row.get("episode_index")) == episode_index]
    if filtered:
        return filtered

    return rows if allow_unfiltered else []


def load_remote_episode_detail(
    dataset: str,
    episode_index: int,
    *,
    preview_only: bool = False,
) -> dict[str, Any]:
    artifacts = get_remote_dataset_artifacts(dataset)
    info = artifacts["info"]
    episode_meta = _episode_meta_entry(artifacts["episodes_meta"], episode_index)
    chunks_size = int(info.get("chunks_size", 1000) or 1000)
    videos = _resolve_episode_videos(artifacts, dataset, episode_index, info, episode_meta, chunks_size)

    if preview_only:
        return _build_episode_preview_payload(info, episode_index, videos, episode_meta)

    # Try the Viewer API first (zero-download)
    try:
        config, split = _viewer_get_split(dataset)
        expected_length = _safe_int(episode_meta.get("length")) if episode_meta else None
        rows = _viewer_fetch_episode_rows(dataset, config, split, episode_index, expected_length or 500)
        if rows:
            logger.info("Viewer API returned {} rows for {}#{}", len(rows), dataset, episode_index)
            return _build_episode_payload(rows, info, episode_index, videos)
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("Viewer API unavailable for '{}': {}", dataset, exc)

    if _resolve_shared_data_path(info, episode_meta):
        logger.debug(
            "Viewer API unavailable for shared parquet episode '{}#{}'; returning metadata preview",
            dataset,
            episode_index,
        )
        return _build_episode_preview_payload(info, episode_index, videos, episode_meta)

    chunk = f"{episode_index // chunks_size:03d}"
    parquet_path = f"data/chunk-{chunk}/episode_{episode_index:06d}.parquet"
    parquet_bytes = _fetch_optional_bytes(_repo_file_url(dataset, parquet_path))
    rows = _read_parquet_rows_from_bytes(parquet_bytes or b"", columns=_episode_data_columns(info))
    rows = _select_rows_for_episode(rows, episode_index, episode_meta, allow_unfiltered=True)

    return _build_episode_payload(rows, info, episode_index, videos)


def build_remote_dataset_info(dataset: str) -> dict[str, Any]:
    artifacts = get_remote_dataset_artifacts(dataset)
    info = artifacts["info"]
    episodes_meta = artifacts["episodes_meta"]
    episode_lengths: list[int] = []
    if episodes_meta:
        for entry in episodes_meta:
            try:
                episode_lengths.append(int(entry.get("length", 0)))
            except (TypeError, ValueError):
                episode_lengths.append(0)
    else:
        total = int(info.get("total_episodes", 0) or 0)
        episode_lengths = [0] * total

    return {
        "name": artifacts["dataset"],
        "total_episodes": int(info.get("total_episodes", 0) or 0),
        "total_frames": int(info.get("total_frames", 0) or 0),
        "fps": int(info.get("fps", 0) or 0),
        "episode_lengths": episode_lengths,
        "features": list((info.get("features") or {}).keys()),
        "robot_type": str(info.get("robot_type", "")),
        "source_dataset": artifacts["dataset"],
    }


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
