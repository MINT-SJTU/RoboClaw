"""Helpers for dataset explorer source resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from roboclaw.data.dataset_sessions import (
    list_local_dataset_options,
    resolve_dataset_handle_or_workspace,
)

ExplorerSource = Literal["remote", "local", "path"]


def normalize_explorer_source(source: str | None) -> ExplorerSource:
    value = (source or "remote").strip().lower()
    if value not in {"remote", "local", "path"}:
        raise ValueError(f"Unsupported explorer source '{source}'")
    return cast(ExplorerSource, value)


def resolve_local_dataset_path(dataset: str) -> Path:
    return resolve_dataset_handle_or_workspace(dataset)


def resolve_path_dataset(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"Dataset path '{candidate}' does not exist")
    if not (candidate / "meta" / "info.json").is_file():
        raise ValueError(f"Dataset path '{candidate}' is missing meta/info.json")
    return candidate
