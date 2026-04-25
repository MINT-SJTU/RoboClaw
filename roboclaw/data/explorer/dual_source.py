"""Helpers for dataset explorer source resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import HTTPException

from roboclaw.data.dataset_sessions import (
    list_local_dataset_options,
    resolve_dataset_handle_or_workspace,
)

ExplorerSource = Literal["remote", "local", "path"]


def normalize_explorer_source(source: str | None) -> ExplorerSource:
    value = (source or "remote").strip().lower()
    if value not in {"remote", "local", "path"}:
        raise HTTPException(status_code=400, detail=f"Unsupported explorer source '{source}'")
    return value  # type: ignore[return-value]

def resolve_local_dataset_path(dataset: str) -> Path:
    try:
        return resolve_dataset_handle_or_workspace(dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def resolve_path_dataset(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset path '{candidate}' does not exist")
    if not (candidate / "meta" / "info.json").is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset path '{candidate}' is missing meta/info.json",
        )
    return candidate
