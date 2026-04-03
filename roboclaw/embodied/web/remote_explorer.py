"""Remote-first Hugging Face dataset helpers for the explorer lane.

Uses plain HTTP requests plus the HF Dataset Viewer API for row-level queries.
Metadata parquet files are parsed in-memory from downloaded bytes. No local
dataset workspace or hf_hub_download is used by the explorer lane.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from loguru import logger

from roboclaw.embodied.web.datasets import extract_action_names, extract_state_names
from roboclaw.embodied.web.explorer import build_explorer_payload_from_artifacts
from roboclaw.embodied.workflow.features import build_joint_trajectory_payload, resolve_timestamp

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


def _read_parquet_rows_from_bytes(raw: bytes, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if not raw:
        return []
    buffer = pa.BufferReader(raw)
    parquet_file = pq.ParquetFile(buffer)
    available_columns = parquet_file.schema_arrow.names
    selected_columns = columns or available_columns
    valid_columns = [column for column in selected_columns if column in available_columns]
    table = parquet_file.read(columns=valid_columns or None)
    return table.to_pylist()


def _fetch_optional_bytes(url: str) -> bytes | None:
    try:
        resp = httpx.get(url, headers=_viewer_headers(), timeout=_VIEWER_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


@lru_cache(maxsize=32)
def get_remote_dataset_artifacts(dataset: str) -> dict[str, Any]:
    info = _HF_API.dataset_info(dataset)
    siblings = [
        {"rfilename": item.rfilename}
        for item in (info.siblings or [])
        if getattr(item, "rfilename", None)
    ]

    info_bytes = _fetch_optional_bytes(_repo_file_url(dataset, "meta/info.json"))
    stats_bytes = _fetch_optional_bytes(_repo_file_url(dataset, "meta/stats.json"))
    episodes_jsonl_bytes = _fetch_optional_bytes(_repo_file_url(dataset, "meta/episodes.jsonl"))

    info_json = _load_json_bytes(info_bytes) if info_bytes else {}
    stats_json = _load_json_bytes(stats_bytes) if stats_bytes else {}
    episodes_meta = _load_jsonl_bytes(episodes_jsonl_bytes) if episodes_jsonl_bytes else []

    return {
        "dataset": str(info.id or dataset),
        "siblings": siblings,
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
    retries = [0.3, 0.8]
    last_exc: Exception | None = None
    for attempt in range(1 + len(retries)):
        try:
            resp = httpx.get(url, headers=_viewer_headers(), timeout=_VIEWER_TIMEOUT)
            if resp.status_code in (408, 429, 500, 502, 503, 504) and attempt < len(retries):
                import time
                time.sleep(retries[attempt])
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < len(retries):
                import time
                time.sleep(retries[attempt])
    raise last_exc  # type: ignore[misc]


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
    where = f'"episode_index"={episode_index}'
    url = (
        f"{_HF_VIEWER_BASE}/filter"
        f"?dataset={quote(dataset, safe='')}"
        f"&config={quote(config, safe='')}"
        f"&split={quote(split, safe='')}"
        f"&where={quote(where, safe='')}"
        f"&offset=0&length={length}"
    )
    payload = _viewer_fetch_json(url)
    return [entry.get("row", {}) for entry in payload.get("rows", [])]


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


def _resolve_episode_videos(
    artifacts: dict[str, Any],
    dataset: str,
    episode_index: int,
    chunks_size: int,
) -> list[dict[str, Any]]:
    chunk = f"{episode_index // chunks_size:03d}"
    video_prefix = f"videos/chunk-{chunk}/episode_{episode_index:06d}/"
    return [
        {
            "path": sibling["rfilename"],
            "url": _repo_file_url(dataset, sibling["rfilename"]),
            "stream": Path(sibling["rfilename"]).stem,
        }
        for sibling in artifacts["siblings"]
        if sibling["rfilename"].startswith(video_prefix) and sibling["rfilename"].endswith(".mp4")
    ]


def load_remote_episode_detail(dataset: str, episode_index: int) -> dict[str, Any]:
    artifacts = get_remote_dataset_artifacts(dataset)
    info = artifacts["info"]
    chunks_size = int(info.get("chunks_size", 1000) or 1000)
    videos = _resolve_episode_videos(artifacts, dataset, episode_index, chunks_size)

    # Try the Viewer API first (zero-download)
    try:
        config, split = _viewer_get_split(dataset)
        rows = _viewer_fetch_episode_rows(dataset, config, split, episode_index)
        if rows:
            logger.info("Viewer API returned {} rows for {}#{}", len(rows), dataset, episode_index)
            return _build_episode_payload(rows, info, episode_index, videos)
    except Exception:
        logger.debug("Viewer API unavailable for '{}', falling back to remote parquet bytes", dataset)

    # Fallback: fetch the single episode parquet over HTTP and parse in-memory
    chunk = f"{episode_index // chunks_size:03d}"
    parquet_bytes = _fetch_optional_bytes(
        _repo_file_url(dataset, f"data/chunk-{chunk}/episode_{episode_index:06d}.parquet")
    )
    rows = _read_parquet_rows_from_bytes(parquet_bytes or b"")
    if rows and "episode_index" in rows[0]:
        rows = [row for row in rows if row.get("episode_index") == episode_index]

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
