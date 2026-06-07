"""Dataset list / detail / delete routes."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from roboclaw.data.curation.service import CurationService
from roboclaw.data.ingestion import DatasetIngestSpec, ingest_dataset_source
from roboclaw.data.storage import (
    DataPoolPolicy,
    S3StorageSettings,
    build_private_dataset_layout,
    create_presigned_upload_post,
)
from roboclaw.embodied.service import EmbodiedService
from roboclaw.http.routes.account import get_ledger


class DatasetCompleteUploadRequest(BaseModel):
    dataset_id: str
    username: str = ""
    owner_username: str = ""
    contribution_source: str = "self_collected"
    visibility: str = "private"
    source_kind: str = "local_upload"
    source_uri: str = ""
    source_auth_ref: str = ""
    auto_quality: bool = True
    selected_validators: list[str] = Field(default_factory=lambda: ["metadata", "timing", "action"])
    episode_indices: list[int] | None = None
    threshold_overrides: dict[str, float] | None = None


class DatasetIngestRequest(BaseModel):
    dataset_id: str
    username: str = ""
    owner_username: str = ""
    contribution_source: str = "self_collected"
    source_kind: str
    source_uri: str
    source_auth_ref: str = "public"
    storage_mode: str = "managed"
    include_videos: bool = True
    force: bool = False


class DatasetPublishRequest(BaseModel):
    username: str = ""
    selected_validators: list[str] = Field(default_factory=lambda: ["metadata", "timing", "action"])
    episode_indices: list[int] | None = None
    threshold_overrides: dict[str, float] | None = None


class DatasetRedeemAccessRequest(BaseModel):
    username: str
    price_points: int | None = None


class DatasetUploadUrlRequest(BaseModel):
    dataset_id: str
    username: str = ""
    filename: str
    submission_id: str = ""
    content_type: str = "application/octet-stream"
    max_size_bytes: int = 50 * 1024 * 1024 * 1024


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _is_remote_ingest_kind(source_kind: str) -> bool:
    return source_kind.strip().lower() in {"remote_dataset", "huggingface", "hf"}


def _is_external_reference_kind(source_kind: str) -> bool:
    return source_kind.strip().lower() in {
        "remote_dataset",
        "huggingface",
        "hf",
        "modelscope_dataset",
        "kaggle_dataset",
        "dagshub_dvc",
        "public_http",
        "drive_link",
    }


def _local_remote_ingest_allowed() -> bool:
    return _env_flag("ROBOCLAW_ALLOW_LOCAL_REMOTE_DATASET_INGEST")


def _dataset_owner(info: dict[str, Any]) -> str:
    return str(
        info.get("ownerUsername")
        or info.get("owner_username")
        or info.get("uploaderUsername")
        or info.get("uploadedBy")
        or ""
    ).strip()


def _dataset_visibility(info: dict[str, Any]) -> str:
    return str(info.get("visibility") or info.get("accessLevel") or "private").strip() or "private"


def _is_public_visibility(visibility: str) -> bool:
    return visibility.strip().lower() in {"public", "shared", "open"}


def _require_username(username: str) -> str:
    value = username.strip()
    if not value:
        raise HTTPException(status_code=400, detail="username is required")
    return value


def _resolve_username(request: Request, provided: str = "") -> str:
    header_username = (
        request.headers.get("x-evo-studio-user", "")
        or request.headers.get("x-roboclaw-user", "")
    ).strip()
    auth_mode = os.environ.get("EVO_STUDIO_AUTH_MODE", "dev").strip().lower()
    if auth_mode in {"trusted_header", "proxy", "production"}:
        return _require_username(header_username)
    return _require_username(provided or header_username)


def _optional_username(request: Request, provided: str = "") -> str:
    header_username = (
        request.headers.get("x-evo-studio-user", "")
        or request.headers.get("x-roboclaw-user", "")
    ).strip()
    auth_mode = os.environ.get("EVO_STUDIO_AUTH_MODE", "dev").strip().lower()
    if auth_mode in {"trusted_header", "proxy", "production"}:
        return header_username
    return (provided or header_username).strip()


def _can_read_dataset(info: dict[str, Any], username: str) -> bool:
    return _dataset_owner(info) == username or _is_public_visibility(_dataset_visibility(info))


def _ensure_owner_can_mutate(info: dict[str, Any], username: str) -> None:
    owner = _dataset_owner(info)
    if owner and owner != username:
        raise HTTPException(status_code=403, detail="dataset belongs to another user")


def _dataset_access_price_points(info: dict[str, Any]) -> int:
    configured = info.get("accessPricePoints", info.get("access_price_points"))
    if configured is not None:
        try:
            return max(int(configured), 0)
        except (TypeError, ValueError):
            return 0
    return _env_int("EVO_STUDIO_PUBLIC_DATASET_ACCESS_POINTS", 10)


def _contributor_share_bps() -> int:
    return min(max(_env_int("EVO_STUDIO_DATASET_CONTRIBUTOR_SHARE_BPS", 5000), 0), 10_000)


async def _owner_storage_used_bytes(service: EmbodiedService, username: str) -> int:
    refs = await asyncio.to_thread(service.datasets.list_local_datasets)
    used_bytes = 0
    for ref in refs:
        if ref.local_path is None:
            continue
        try:
            info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
        except ValueError:
            continue
        if _dataset_owner(info) == username:
            used_bytes += int(ref.stats.total_bytes or 0)
    return used_bytes


def _read_dataset_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset metadata is not valid JSON: {info_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {info_path}")
    return payload


def _write_upload_metadata(
    dataset_path: Path,
    *,
    owner_username: str,
    contribution_source: str,
    visibility: str,
    source_kind: str,
    source_uri: str,
    source_auth_ref: str,
    storage_mode: str = "",
    private_retention_days: int | None = None,
    private_expires_at: str = "",
) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = _read_dataset_info(dataset_path)
    existing_owner = _dataset_owner(info)
    if existing_owner and owner_username and existing_owner != owner_username:
        raise ValueError("dataset belongs to another user")
    if owner_username:
        info["ownerUsername"] = owner_username
    if contribution_source:
        info["contributionSource"] = contribution_source
    if visibility:
        info["visibility"] = visibility
    if source_kind:
        info["sourceKind"] = source_kind
    if source_uri:
        info["sourceUri"] = source_uri
    if source_auth_ref:
        info["sourceAuthRef"] = source_auth_ref
    if storage_mode:
        info["storageMode"] = storage_mode
    if private_retention_days is not None:
        info["privateRetentionDays"] = private_retention_days
    if private_expires_at:
        info["privateExpiresAt"] = private_expires_at
    now = datetime.now(timezone.utc).isoformat()
    info.setdefault("createdAt", now)
    info["ingestedAt"] = now
    info["uploadStatus"] = "uploaded"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _write_external_reference_metadata(
    dataset_path: Path,
    *,
    owner_username: str,
    contribution_source: str,
    source_kind: str,
    source_uri: str,
    source_auth_ref: str,
) -> dict[str, Any]:
    if dataset_path.exists() and not (dataset_path / "meta" / "info.json").is_file():
        raise ValueError(f"Dataset target already exists but is not a catalog dataset: {dataset_path}")
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = _read_dataset_info(dataset_path) if info_path.is_file() else {}
    existing_owner = _dataset_owner(info)
    if existing_owner and owner_username and existing_owner != owner_username:
        raise ValueError("dataset belongs to another user")
    now = datetime.now(timezone.utc).isoformat()
    info.update({
        "ownerUsername": owner_username,
        "contributionSource": contribution_source,
        "visibility": "private",
        "sourceKind": source_kind,
        "sourceUri": source_uri,
        "sourceAuthRef": source_auth_ref,
        "storageMode": "external_reference",
        "source_dataset": source_uri,
        "uploadStatus": "referenced",
        "total_episodes": int(info.get("total_episodes", 0) or 0),
        "total_frames": int(info.get("total_frames", 0) or 0),
        "fps": int(info.get("fps", 0) or 0),
    })
    info.setdefault("features", {})
    info.setdefault("createdAt", now)
    info["ingestedAt"] = now
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _write_publish_request_metadata(
    dataset_path: Path,
    *,
    owner_username: str,
) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = _read_dataset_info(dataset_path)
    if owner_username and not info.get("ownerUsername"):
        info["ownerUsername"] = owner_username
    info.setdefault("visibility", "private")
    info["requestedVisibility"] = "public"
    info["publicationStatus"] = "pending_quality"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def register_dataset_routes(
    app: FastAPI,
    service: EmbodiedService,
    *,
    allow_anonymous_list: bool = False,
) -> None:
    curation_service = CurationService()

    @app.get("/api/datasets/storage/status")
    async def datasets_storage_status() -> dict[str, Any]:
        return S3StorageSettings.from_env().to_status_dict()

    @app.post("/api/datasets/upload-url")
    async def dataset_create_upload_url(request: Request, body: DatasetUploadUrlRequest) -> dict[str, Any]:
        username = _resolve_username(request, body.username)
        settings = S3StorageSettings.from_env()
        policy = DataPoolPolicy.from_env()
        if not settings.configured:
            raise HTTPException(
                status_code=501,
                detail={
                    "message": "dataset storage provider is not configured",
                    "missingFields": settings.missing_fields,
                },
            )
        used_bytes = await _owner_storage_used_bytes(service, username)
        quota_bytes = policy.quota_bytes_for(username)
        available_bytes = max(quota_bytes - used_bytes, 0)
        if available_bytes <= 0:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "dataset storage quota exceeded",
                    "username": username,
                    "role": policy.role_for(username),
                    "quotaBytes": quota_bytes,
                    "usedBytes": used_bytes,
                    "availableBytes": available_bytes,
                },
            )
        upload_max_size = min(max(int(body.max_size_bytes), 1), available_bytes)
        layout = build_private_dataset_layout(
            username=username,
            dataset_id=body.dataset_id,
            filename=body.filename,
            prefix=settings.prefix,
        )
        try:
            upload = await asyncio.to_thread(
                create_presigned_upload_post,
                settings,
                key=layout["shardKey"],
                content_type=body.content_type,
                max_size_bytes=upload_max_size,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return {
            "provider": settings.provider,
            "bucket": settings.bucket,
            "datasetId": body.dataset_id,
            "username": username,
            "layout": layout,
            "upload": upload,
            "pendingExpiresAt": policy.pending_expires_at(),
            "policy": policy.to_dict(username=username, used_bytes=used_bytes),
            "warnings": [] if policy.accepts_filename(body.filename) else [
                "Upload datasets as archive/shard files where possible to avoid large numbers of tiny files."
            ],
        }

    @app.get("/api/datasets")
    async def datasets_list_route(request: Request, username: str = "") -> list[dict]:
        username = _optional_username(request, username)
        if not username and not allow_anonymous_list:
            _require_username(username)
        refs = await asyncio.to_thread(service.datasets.list_local_datasets)
        visible: list[dict[str, Any]] = []
        for ref in refs:
            if ref.local_path is None:
                continue
            try:
                info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
            except ValueError:
                continue
            if username:
                can_read = _can_read_dataset(info, username)
            else:
                can_read = _is_public_visibility(_dataset_visibility(info))
            if can_read:
                visible.append(ref.to_dict())
        return visible

    @app.get("/api/datasets/storage-usage")
    async def datasets_storage_usage(request: Request, username: str = "") -> dict[str, Any]:
        username = _resolve_username(request, username)
        policy = DataPoolPolicy.from_env()

        refs = await asyncio.to_thread(service.datasets.list_local_datasets)
        quota_bytes = policy.quota_bytes_for(username)
        dataset_rows: list[dict[str, Any]] = []
        private_bytes = 0
        public_bytes = 0
        used_bytes = 0

        for ref in refs:
            if ref.local_path is None:
                continue
            try:
                info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
            except ValueError:
                continue
            if _dataset_owner(info) != username:
                continue
            visibility = _dataset_visibility(info)
            total_bytes = int(ref.stats.total_bytes or 0)
            used_bytes += total_bytes
            if _is_public_visibility(visibility):
                public_bytes += total_bytes
            else:
                private_bytes += total_bytes
            dataset_rows.append({
                "id": ref.id,
                "label": ref.label,
                "visibility": visibility,
                "totalBytes": total_bytes,
                "sourceKind": str(info.get("sourceKind", "")),
                "sourceUri": str(info.get("sourceUri", "")),
                "storageMode": str(info.get("storageMode", "")),
                "canTrain": bool(ref.capabilities.can_train),
                "createdAt": str(info.get("createdAt", "")),
                "ingestedAt": str(info.get("ingestedAt", "")),
                "publicationStatus": str(info.get("publicationStatus", "")),
                "privateRetentionDays": info.get("privateRetentionDays", ""),
                "privateExpiresAt": str(info.get("privateExpiresAt", "")),
            })

        return {
            "username": username,
            "role": policy.role_for(username),
            "quotaBytes": quota_bytes,
            "usedBytes": used_bytes,
            "availableBytes": max(quota_bytes - used_bytes, 0),
            "datasetCount": len(dataset_rows),
            "privateBytes": private_bytes,
            "publicBytes": public_bytes,
            "datasets": dataset_rows,
            "policy": policy.to_dict(username=username, used_bytes=used_bytes),
        }

    @app.post("/api/datasets/complete-upload")
    async def dataset_complete_upload(request: Request, body: DatasetCompleteUploadRequest) -> dict[str, Any]:
        owner_username = _resolve_username(request, body.owner_username or body.username)
        dataset_path: Path
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, body.dataset_id)
            if ref.local_path is None:
                raise HTTPException(status_code=409, detail=f"Dataset '{body.dataset_id}' has no local workspace")
            dataset_path = ref.local_path
        except ValueError as exc:
            if body.source_kind.strip() == "local_upload" and body.source_uri.strip():
                try:
                    dataset_path = service.datasets.resolve_local_path(body.dataset_id)
                except ValueError as path_exc:
                    raise HTTPException(status_code=400, detail=str(path_exc)) from path_exc
            else:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        policy = DataPoolPolicy.from_env()
        visibility = "private"
        try:
            existing_info = await asyncio.to_thread(_read_dataset_info, dataset_path)
            _ensure_owner_can_mutate(existing_info, owner_username)
            info = await asyncio.to_thread(
                _write_upload_metadata,
                dataset_path,
                owner_username=owner_username,
                contribution_source=body.contribution_source.strip(),
                visibility=visibility,
                source_kind=body.source_kind.strip(),
                source_uri=body.source_uri.strip(),
                source_auth_ref=body.source_auth_ref.strip(),
                storage_mode="managed_upload",
                private_retention_days=policy.private_retention_days_for(owner_username),
                private_expires_at=policy.private_expires_at(owner_username),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        quality: dict[str, Any] = {"autoTriggered": False, "status": "skipped"}

        refreshed = await asyncio.to_thread(service.datasets.require_local_dataset, body.dataset_id)
        return {
            "status": "uploaded",
            "dataset": refreshed.to_dict(),
            "ownerUsername": info.get("ownerUsername", ""),
            "contributionSource": info.get("contributionSource", ""),
            "visibility": info.get("visibility", ""),
            "sourceKind": info.get("sourceKind", ""),
            "sourceUri": info.get("sourceUri", ""),
            "sourceAuthRef": info.get("sourceAuthRef", ""),
            "quality": quality,
        }

    @app.post("/api/datasets/ingest")
    async def dataset_ingest(request: Request, body: DatasetIngestRequest) -> dict[str, Any]:
        owner_username = _resolve_username(request, body.owner_username or body.username)
        dataset_path = service.datasets.resolve_local_path(body.dataset_id)
        if dataset_path.exists() and (dataset_path / "meta" / "info.json").is_file():
            try:
                existing_info = await asyncio.to_thread(_read_dataset_info, dataset_path)
                _ensure_owner_can_mutate(existing_info, owner_username)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        storage_mode = body.storage_mode.strip().lower() or "managed"
        if _is_external_reference_kind(body.source_kind) and storage_mode == "external_reference":
            if dataset_path.exists() and body.force:
                import shutil
                await asyncio.to_thread(shutil.rmtree, dataset_path)
            try:
                info = await asyncio.to_thread(
                    _write_external_reference_metadata,
                    dataset_path,
                    owner_username=owner_username,
                    contribution_source=body.contribution_source.strip(),
                    source_kind=body.source_kind.strip(),
                    source_uri=body.source_uri.strip(),
                    source_auth_ref=body.source_auth_ref.strip(),
                )
                ref = await asyncio.to_thread(service.datasets.require_local_dataset, body.dataset_id)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {
                "status": "referenced",
                "dataset": ref.to_dict(),
                "ownerUsername": info.get("ownerUsername", ""),
                "visibility": info.get("visibility", ""),
                "sourceKind": info.get("sourceKind", ""),
                "sourceUri": info.get("sourceUri", ""),
                "sourceAuthRef": info.get("sourceAuthRef", ""),
                "storageMode": info.get("storageMode", ""),
                "quality": {"autoTriggered": False, "status": "skipped"},
            }

        if _is_remote_ingest_kind(body.source_kind) and not _local_remote_ingest_allowed():
            raise HTTPException(
                status_code=501,
                detail=(
                    "remote dataset ingestion must run in the configured cloud data pool; "
                    "local HuggingFace downloads are disabled by default. Configure the "
                    "Evo Studio cloud ingestion worker/OSS pipeline, or set "
                    "ROBOCLAW_ALLOW_LOCAL_REMOTE_DATASET_INGEST=1 for local development only."
                ),
            )
        try:
            ref = await asyncio.to_thread(
                ingest_dataset_source,
                service.datasets,
                DatasetIngestSpec(
                    dataset_id=body.dataset_id,
                    source_kind=body.source_kind,
                    source_uri=body.source_uri,
                    source_auth_ref=body.source_auth_ref,
                    include_videos=body.include_videos,
                    force=body.force,
                ),
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            policy = DataPoolPolicy.from_env()
            info = await asyncio.to_thread(
                _write_upload_metadata,
                ref.local_path,
                owner_username=owner_username,
                contribution_source=body.contribution_source.strip(),
                visibility="private",
                source_kind=body.source_kind.strip(),
                source_uri=body.source_uri.strip(),
                source_auth_ref=body.source_auth_ref.strip(),
                storage_mode=storage_mode,
                private_retention_days=policy.private_retention_days_for(owner_username),
                private_expires_at=policy.private_expires_at(owner_username),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        refreshed = await asyncio.to_thread(service.datasets.require_local_dataset, body.dataset_id)
        return {
            "status": "ingested",
            "dataset": refreshed.to_dict(),
            "ownerUsername": info.get("ownerUsername", ""),
            "visibility": info.get("visibility", ""),
            "sourceKind": info.get("sourceKind", ""),
            "sourceUri": info.get("sourceUri", ""),
            "sourceAuthRef": info.get("sourceAuthRef", ""),
            "quality": {"autoTriggered": False, "status": "skipped"},
        }

    @app.post("/api/datasets/{dataset_id:path}/publish-request")
    async def dataset_publish_request(dataset_id: str, request: Request, body: DatasetPublishRequest) -> dict[str, Any]:
        username = _resolve_username(request, body.username)
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if ref.local_path is None:
            raise HTTPException(status_code=409, detail=f"Dataset '{dataset_id}' has no local workspace")

        try:
            existing_info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
            _ensure_owner_can_mutate(existing_info, username)
            info = await asyncio.to_thread(
                _write_publish_request_metadata,
                ref.local_path,
                owner_username=username,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        quality = await curation_service.start_quality_run(
            ref.local_path,
            ref.id,
            body.selected_validators,
            body.episode_indices,
            body.threshold_overrides,
            username,
        )
        quality["autoTriggered"] = True

        refreshed = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        return {
            "status": "publish_requested",
            "dataset": refreshed.to_dict(),
            "ownerUsername": info.get("ownerUsername", ""),
            "visibility": info.get("visibility", ""),
            "publicationStatus": info.get("publicationStatus", ""),
            "quality": quality,
        }

    @app.post("/api/datasets/{dataset_id:path}/redeem-access")
    async def dataset_redeem_access(dataset_id: str, request: Request, body: DatasetRedeemAccessRequest) -> dict[str, Any]:
        username = _resolve_username(request, body.username)
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if ref.local_path is None:
            raise HTTPException(status_code=409, detail=f"Dataset '{dataset_id}' has no local workspace")
        try:
            info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        visibility = _dataset_visibility(info)
        if not _is_public_visibility(visibility):
            raise HTTPException(status_code=409, detail="dataset is not public")
        contributor = _dataset_owner(info)
        price_points = body.price_points if body.price_points is not None else _dataset_access_price_points(info)
        try:
            wallet, grant, buyer_record, contributor_record, granted = await asyncio.to_thread(
                get_ledger().redeem_dataset_access,
                username,
                dataset_id,
                price_points,
                contributor_username=contributor,
                contributor_share_bps=_contributor_share_bps(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409 if "insufficient" in str(exc) else 400, detail=str(exc)) from exc

        return {
            "status": "access_granted",
            "granted": granted,
            "dataset": ref.to_dict(),
            "wallet": wallet.to_dict(),
            "accessGrant": grant.to_dict(),
            "buyerRecord": buyer_record.to_dict() if buyer_record else None,
            "contributorRecord": contributor_record.to_dict() if contributor_record else None,
            "pricePoints": price_points,
            "contributorUsername": contributor,
        }

    @app.get("/api/datasets/{dataset_id:path}")
    async def dataset_detail(dataset_id: str, request: Request, username: str = "") -> dict:
        username = _resolve_username(request, username)
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if ref.local_path is None:
            raise HTTPException(status_code=409, detail=f"Dataset '{dataset_id}' has no local workspace")
        try:
            info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not _can_read_dataset(info, username):
            raise HTTPException(status_code=403, detail="dataset is private")
        return ref.to_dict()

    @app.delete("/api/datasets/{dataset_id:path}")
    async def dataset_delete(dataset_id: str, request: Request, username: str = "") -> dict[str, str]:
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        username = _resolve_username(request, username)
        if ref.local_path is None:
            raise HTTPException(status_code=409, detail=f"Dataset '{dataset_id}' has no local workspace")
        try:
            info = await asyncio.to_thread(_read_dataset_info, ref.local_path)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _ensure_owner_can_mutate(info, username)
        try:
            await asyncio.to_thread(service.datasets.delete_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "id": dataset_id}
