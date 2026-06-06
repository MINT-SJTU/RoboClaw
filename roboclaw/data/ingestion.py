"""Dataset source ingestion helpers.

This module materializes external dataset sources into the local
``DatasetCatalog`` root so the existing curation and training paths can use
them as normal local datasets.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from roboclaw.data.datasets import DatasetCatalog, DatasetRef


@dataclass(frozen=True)
class DatasetIngestSpec:
    dataset_id: str
    source_kind: str
    source_uri: str
    source_auth_ref: str = ""
    include_videos: bool = True
    force: bool = False


def ingest_dataset_source(catalog: DatasetCatalog, spec: DatasetIngestSpec) -> DatasetRef:
    """Materialize *spec* into *catalog* and return the local dataset ref."""
    source_kind = spec.source_kind.strip().lower()
    if source_kind in {"remote_dataset", "huggingface", "hf"}:
        repo_id = _normalize_remote_repo(spec.source_uri)
        return catalog.pull_dataset(
            repo_id,
            dataset_id=spec.dataset_id,
            token=_resolve_auth_token(spec.source_auth_ref),
        )

    if source_kind in {"mounted_path", "local_path"}:
        return _ingest_directory(catalog, spec)

    if source_kind in {"local_upload", "local_archive", "archive"}:
        return _ingest_archive(catalog, spec)

    if source_kind in {"cloud_object", "oss_object", "s3_object", "cos_object", "gcs_object"}:
        raise NotImplementedError(
            "cloud object ingestion requires a configured storage provider "
            "(OSS/S3/COS/GCS) and is not enabled in this backend yet"
        )

    raise ValueError(f"Unsupported dataset source_kind: {spec.source_kind!r}")


def _normalize_remote_repo(source_uri: str) -> str:
    value = source_uri.strip()
    for prefix in ("hf://", "huggingface://"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    if value.startswith("https://huggingface.co/"):
        value = value[len("https://huggingface.co/"):]
    value = value.strip("/")
    if value.startswith("datasets/"):
        value = value[len("datasets/"):]
    if not value or "/" not in value:
        raise ValueError("remote dataset source_uri must be a HuggingFace repo id or hf://owner/name URI")
    return value


def _resolve_auth_token(source_auth_ref: str) -> str:
    ref = source_auth_ref.strip()
    if not ref or ref == "public":
        return ""
    env_key = "ROBOCLAW_DATASET_AUTH_" + re.sub(r"[^A-Za-z0-9]+", "_", ref).upper() + "_TOKEN"
    token = os.environ.get(env_key, "").strip()
    if not token:
        raise ValueError(f"Dataset auth ref {ref!r} is not configured; expected env {env_key}")
    return token


def _ingest_directory(catalog: DatasetCatalog, spec: DatasetIngestSpec) -> DatasetRef:
    source = _resolve_allowed_source_path(catalog, spec.source_uri)
    if not source.is_dir():
        raise ValueError(f"mounted_path source_uri must be a directory: {source}")
    target = catalog.resolve_local_path(spec.dataset_id)
    _prepare_target(target, spec.force)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return catalog.require_local_dataset(spec.dataset_id)


def _ingest_archive(catalog: DatasetCatalog, spec: DatasetIngestSpec) -> DatasetRef:
    source = _resolve_allowed_source_path(catalog, spec.source_uri)
    if not source.is_file():
        raise ValueError(f"archive source_uri must be a file: {source}")
    target = catalog.resolve_local_path(spec.dataset_id)
    _prepare_target(target, spec.force)
    target.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(source), str(target))

    nested = _find_single_nested_dataset(target)
    if nested is not None and nested != target:
        temp = target.with_name(target.name + ".__ingest_tmp__")
        if temp.exists():
            shutil.rmtree(temp)
        nested.rename(temp)
        shutil.rmtree(target)
        temp.rename(target)
    return catalog.require_local_dataset(spec.dataset_id)


def _prepare_target(target: Path, force: bool) -> None:
    if target.exists():
        if not force:
            raise ValueError(f"Dataset target already exists: {target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)


def _find_single_nested_dataset(target: Path) -> Path | None:
    if (target / "meta" / "info.json").is_file():
        return target
    children = [child for child in target.iterdir() if child.is_dir()]
    if len(children) != 1:
        return None
    child = children[0]
    if (child / "meta" / "info.json").is_file():
        return child
    return None


def _resolve_allowed_source_path(catalog: DatasetCatalog, source_uri: str) -> Path:
    parsed = urlparse(source_uri.strip())
    raw_path = parsed.path if parsed.scheme == "file" else source_uri
    source = Path(raw_path).expanduser().resolve()
    allowed_roots = _allowed_ingest_roots(catalog)
    if not any(_is_relative_to(source, root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(
            f"Dataset source path is outside allowed ingest roots: {source}. "
            f"Configure ROBOCLAW_DATASET_INGEST_ROOTS; current roots: {roots}"
        )
    return source


def _allowed_ingest_roots(catalog: DatasetCatalog) -> tuple[Path, ...]:
    configured = os.environ.get("ROBOCLAW_DATASET_INGEST_ROOTS", "").strip()
    roots: list[Path] = []
    if configured:
        roots.extend(Path(value).expanduser().resolve() for value in configured.split(os.pathsep) if value.strip())
    roots.append(catalog.root.resolve())
    return tuple(dict.fromkeys(roots))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
