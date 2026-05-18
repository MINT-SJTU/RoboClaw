"""Dataset list / detail / delete routes."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from roboclaw.data.curation.service import CurationService
from roboclaw.data.ingestion import DatasetIngestSpec, ingest_dataset_source
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


def _storage_quota_bytes() -> int:
    return _env_int("ROBOCLAW_DATASET_STORAGE_QUOTA_BYTES", 50 * 1024 * 1024 * 1024)


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
) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info = _read_dataset_info(dataset_path)
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
    info["uploadStatus"] = "uploaded"
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
    info["visibility"] = "public"
    info["publicationStatus"] = "pending_quality"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def register_dataset_routes(app: FastAPI, service: EmbodiedService) -> None:
    curation_service = CurationService()

    @app.get("/api/datasets")
    async def datasets_list_route() -> list[dict]:
        refs = await asyncio.to_thread(service.datasets.list_local_datasets)
        return [ref.to_dict() for ref in refs]

    @app.get("/api/datasets/storage-usage")
    async def datasets_storage_usage(username: str = "") -> dict[str, Any]:
        username = username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="username is required")

        refs = await asyncio.to_thread(service.datasets.list_local_datasets)
        quota_bytes = _storage_quota_bytes()
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
            if visibility in {"public", "shared", "open"}:
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
                "canTrain": bool(ref.capabilities.can_train),
            })

        return {
            "username": username,
            "quotaBytes": quota_bytes,
            "usedBytes": used_bytes,
            "availableBytes": max(quota_bytes - used_bytes, 0),
            "datasetCount": len(dataset_rows),
            "privateBytes": private_bytes,
            "publicBytes": public_bytes,
            "datasets": dataset_rows,
        }

    @app.post("/api/datasets/complete-upload")
    async def dataset_complete_upload(body: DatasetCompleteUploadRequest) -> dict[str, Any]:
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, body.dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if ref.local_path is None:
            raise HTTPException(status_code=409, detail=f"Dataset '{body.dataset_id}' has no local workspace")

        owner_username = (body.owner_username or body.username).strip()
        visibility = "private"
        try:
            info = await asyncio.to_thread(
                _write_upload_metadata,
                ref.local_path,
                owner_username=owner_username,
                contribution_source=body.contribution_source.strip(),
                visibility=visibility,
                source_kind=body.source_kind.strip(),
                source_uri=body.source_uri.strip(),
                source_auth_ref=body.source_auth_ref.strip(),
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
    async def dataset_ingest(body: DatasetIngestRequest) -> dict[str, Any]:
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

        owner_username = (body.owner_username or body.username).strip()
        try:
            info = await asyncio.to_thread(
                _write_upload_metadata,
                ref.local_path,
                owner_username=owner_username,
                contribution_source=body.contribution_source.strip(),
                visibility="private",
                source_kind=body.source_kind.strip(),
                source_uri=body.source_uri.strip(),
                source_auth_ref=body.source_auth_ref.strip(),
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
    async def dataset_publish_request(dataset_id: str, body: DatasetPublishRequest) -> dict[str, Any]:
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if ref.local_path is None:
            raise HTTPException(status_code=409, detail=f"Dataset '{dataset_id}' has no local workspace")

        try:
            info = await asyncio.to_thread(
                _write_publish_request_metadata,
                ref.local_path,
                owner_username=body.username.strip(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        quality = await curation_service.start_quality_run(
            ref.local_path,
            ref.id,
            body.selected_validators,
            body.episode_indices,
            body.threshold_overrides,
            "",
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
    async def dataset_redeem_access(dataset_id: str, body: DatasetRedeemAccessRequest) -> dict[str, Any]:
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
        if visibility not in {"public", "shared", "open"}:
            raise HTTPException(status_code=409, detail="dataset is not public")
        contributor = _dataset_owner(info)
        price_points = body.price_points if body.price_points is not None else _dataset_access_price_points(info)
        try:
            wallet, grant, buyer_record, contributor_record, granted = await asyncio.to_thread(
                get_ledger().redeem_dataset_access,
                body.username,
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
    async def dataset_detail(dataset_id: str) -> dict:
        try:
            ref = await asyncio.to_thread(service.datasets.require_local_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ref.to_dict()

    @app.delete("/api/datasets/{dataset_id:path}")
    async def dataset_delete(dataset_id: str) -> dict[str, str]:
        try:
            await asyncio.to_thread(service.datasets.delete_dataset, dataset_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted", "id": dataset_id}
