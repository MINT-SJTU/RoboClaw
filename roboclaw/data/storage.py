"""Cloud dataset storage provider configuration.

The first production-oriented provider is S3-compatible storage. This covers
Cloudflare R2, MinIO, AWS S3, and other services exposing S3 APIs.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._/-]+")
_ENV_FILES_LOADED = False
_GIB = 1024 * 1024 * 1024
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3StorageSettings:
    provider: str = ""
    endpoint: str = ""
    bucket: str = ""
    access_key: str = ""
    secret_key: str = ""
    region: str = "auto"
    prefix: str = "datasets/"
    public_base_url: str = ""

    @classmethod
    def from_env(cls) -> "S3StorageSettings":
        _load_local_env_files_once()
        endpoint = _env_first("EVO_STUDIO_S3_ENDPOINT", "AWS_ENDPOINT_URL_S3")
        access_key = _env_first("EVO_STUDIO_S3_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
        secret_key = _env_first("EVO_STUDIO_S3_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
        provider = _env_first("EVO_STUDIO_STORAGE_PROVIDER").lower()
        if not provider and endpoint and access_key and secret_key:
            provider = "s3"
        return cls(
            provider=provider,
            endpoint=endpoint,
            bucket=_env_first("EVO_STUDIO_S3_BUCKET", "AWS_S3_BUCKET", "AWS_BUCKET_NAME", "S3_BUCKET"),
            access_key=access_key,
            secret_key=secret_key,
            region=_env_first("EVO_STUDIO_S3_REGION", "AWS_REGION", default="auto") or "auto",
            prefix=_env_first("EVO_STUDIO_S3_PREFIX"),
            public_base_url=_env_first("EVO_STUDIO_S3_PUBLIC_BASE_URL"),
        )

    @property
    def configured(self) -> bool:
        if self.provider not in {"s3", "r2", "minio"}:
            return False
        return bool(self.endpoint and self.bucket and self.access_key and self.secret_key)

    @property
    def missing_fields(self) -> list[str]:
        fields: list[str] = []
        if self.provider not in {"s3", "r2", "minio"}:
            fields.append("EVO_STUDIO_STORAGE_PROVIDER=s3|r2|minio")
        if not self.endpoint:
            fields.append("EVO_STUDIO_S3_ENDPOINT or AWS_ENDPOINT_URL_S3")
        if not self.bucket:
            fields.append("EVO_STUDIO_S3_BUCKET")
        if not self.access_key:
            fields.append("EVO_STUDIO_S3_ACCESS_KEY or AWS_ACCESS_KEY_ID")
        if not self.secret_key:
            fields.append("EVO_STUDIO_S3_SECRET_KEY or AWS_SECRET_ACCESS_KEY")
        return fields

    def object_uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def to_status_dict(self) -> dict[str, Any]:
        clean_prefix = self.prefix.strip("/")
        def _prefixed(value: str) -> str:
            return "/".join(part for part in [clean_prefix, value.strip("/")] if part) + "/"

        return {
            "provider": self.provider or "unconfigured",
            "configured": self.configured,
            "endpoint": self.endpoint,
            "bucket": self.bucket,
            "region": self.region,
            "prefix": self.prefix,
            "publicBaseUrl": self.public_base_url,
            "missingFields": self.missing_fields,
            "clientAvailable": _boto3_available(),
            "layout": {
                "pending": _prefixed("pending/submissions/{user_id}/{submission_id}"),
                "private": _prefixed("private/users/{user_id}/datasets/{dataset_id}/v1"),
                "approved": _prefixed("approved/datasets/{dataset_id}/v1"),
                "previews": _prefixed("previews/{dataset_id}"),
                "redemption": _prefixed("redemption/{dataset_id}"),
            },
            "policy": DataPoolPolicy.from_env().to_dict(),
        }


@dataclass(frozen=True)
class DataPoolPolicy:
    pending_retention_days: int = 14
    free_private_retention_days: int = 30
    contributor_private_retention_days: int = 90
    team_private_retention_days: int = 180
    free_quota_bytes: int = 10 * _GIB
    contributor_quota_bytes: int = 50 * _GIB
    team_quota_bytes: int = 200 * _GIB
    billing_alert_usd: float = 20.0
    billing_confirm_usd: float = 50.0
    accepted_upload_extensions: tuple[str, ...] = (
        ".tar",
        ".tar.gz",
        ".tgz",
        ".zip",
        ".hdf5",
        ".h5",
        ".parquet",
        ".jsonl",
    )
    contributor_users: tuple[str, ...] = ()
    team_users: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "DataPoolPolicy":
        _load_local_env_files_once()
        return cls(
            pending_retention_days=_env_int("EVO_STUDIO_PENDING_RETENTION_DAYS", 14),
            free_private_retention_days=_env_int("EVO_STUDIO_FREE_PRIVATE_RETENTION_DAYS", 30),
            contributor_private_retention_days=_env_int("EVO_STUDIO_CONTRIBUTOR_PRIVATE_RETENTION_DAYS", 90),
            team_private_retention_days=_env_int("EVO_STUDIO_TEAM_PRIVATE_RETENTION_DAYS", 180),
            free_quota_bytes=_env_int("EVO_STUDIO_FREE_USER_QUOTA_BYTES", 10 * _GIB),
            contributor_quota_bytes=_env_int("EVO_STUDIO_CONTRIBUTOR_QUOTA_BYTES", 50 * _GIB),
            team_quota_bytes=_env_int("EVO_STUDIO_TEAM_USER_QUOTA_BYTES", 200 * _GIB),
            billing_alert_usd=_env_float("EVO_STUDIO_BILLING_ALERT_USD", 20.0),
            billing_confirm_usd=_env_float("EVO_STUDIO_BILLING_CONFIRM_USD", 50.0),
            accepted_upload_extensions=tuple(
                _env_list(
                    "EVO_STUDIO_ACCEPTED_UPLOAD_EXTENSIONS",
                    ".tar,.tar.gz,.tgz,.zip,.hdf5,.h5,.parquet,.jsonl",
                )
            ),
            contributor_users=tuple(_env_list("EVO_STUDIO_CONTRIBUTOR_USERS", "")),
            team_users=tuple(_env_list("EVO_STUDIO_TEAM_USERS", "")),
        )

    def role_for(self, username: str) -> str:
        value = username.strip()
        if value and value in self.team_users:
            return "team"
        if value and value in self.contributor_users:
            return "contributor"
        return "free"

    def quota_bytes_for(self, username: str) -> int:
        role = self.role_for(username)
        if role == "team":
            return self.team_quota_bytes
        if role == "contributor":
            return self.contributor_quota_bytes
        return self.free_quota_bytes

    def private_retention_days_for(self, username: str) -> int:
        role = self.role_for(username)
        if role == "team":
            return self.team_private_retention_days
        if role == "contributor":
            return self.contributor_private_retention_days
        return self.free_private_retention_days

    def pending_expires_at(self, now: datetime | None = None) -> str:
        current = now or datetime.now(timezone.utc)
        return (current + timedelta(days=max(self.pending_retention_days, 1))).isoformat()

    def private_expires_at(self, username: str, now: datetime | None = None) -> str:
        current = now or datetime.now(timezone.utc)
        return (current + timedelta(days=max(self.private_retention_days_for(username), 1))).isoformat()

    def accepts_filename(self, filename: str) -> bool:
        lowered = filename.strip().lower()
        return any(lowered.endswith(ext.lower()) for ext in self.accepted_upload_extensions)

    def to_dict(self, *, username: str = "", used_bytes: int = 0) -> dict[str, Any]:
        quota = self.quota_bytes_for(username) if username else self.free_quota_bytes
        return {
            "pendingRetentionDays": self.pending_retention_days,
            "privateRetentionDays": self.private_retention_days_for(username) if username else self.free_private_retention_days,
            "billingAlertUsd": self.billing_alert_usd,
            "billingConfirmUsd": self.billing_confirm_usd,
            "acceptedUploadExtensions": list(self.accepted_upload_extensions),
            "recommendedPackaging": "Upload datasets as archive/shard files instead of millions of small files.",
            "privateRetentionByRole": {
                "free": self.free_private_retention_days,
                "contributor": self.contributor_private_retention_days,
                "team": self.team_private_retention_days,
            },
            "publicRetention": "quality-approved public datasets are retained according to platform sharing policy",
            "quotas": {
                "free": self.free_quota_bytes,
                "contributor": self.contributor_quota_bytes,
                "team": self.team_quota_bytes,
            },
            "user": {
                "username": username,
                "role": self.role_for(username) if username else "free",
                "quotaBytes": quota,
                "usedBytes": used_bytes,
                "availableBytes": max(quota - used_bytes, 0),
            },
        }


def build_dataset_object_key(
    *,
    username: str,
    dataset_id: str,
    filename: str,
    prefix: str,
    submission_id: str = "",
) -> str:
    clean_prefix = prefix.strip("/")
    clean_user = _safe_key_part(username.strip() or "anonymous")
    clean_dataset = _safe_key_part(dataset_id.strip())
    clean_name = _safe_key_part(filename.strip() or "dataset.bin")
    clean_submission = _safe_key_part(submission_id.strip() or new_submission_id())
    parts = [
        part for part in [
            clean_prefix,
            "pending",
            "submissions",
            clean_user,
            clean_submission,
            "shards",
            clean_name,
        ] if part
    ]
    return "/".join(parts)


def new_submission_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:10]


def build_submission_layout(
    *,
    username: str,
    dataset_id: str,
    filename: str,
    prefix: str,
    submission_id: str = "",
) -> dict[str, str]:
    clean_prefix = prefix.strip("/")
    clean_user = _safe_key_part(username.strip() or "anonymous")
    clean_dataset = _safe_key_part(dataset_id.strip())
    clean_submission = _safe_key_part(submission_id.strip() or new_submission_id())
    base_parts = [
        part for part in [
            clean_prefix,
            "pending",
            "submissions",
            clean_user,
            clean_submission,
        ] if part
    ]
    base = "/".join(base_parts)
    filename_key = build_dataset_object_key(
        username=clean_user,
        dataset_id=clean_dataset,
        filename=filename,
        prefix=clean_prefix,
        submission_id=clean_submission,
    )
    return {
        "submissionId": clean_submission,
        "submissionPrefix": base + "/",
        "manifestKey": f"{base}/manifest.jsonl",
        "dataCardKey": f"{base}/data_card.md",
        "shardKey": filename_key,
        "approvedPrefix": build_approved_dataset_prefix(
            dataset_id=clean_dataset,
            version="v1",
            prefix=clean_prefix,
        ),
        "previewPrefix": build_preview_prefix(dataset_id=clean_dataset, prefix=clean_prefix),
        "redemptionPrefix": build_redemption_prefix(dataset_id=clean_dataset, prefix=clean_prefix),
    }


def build_private_dataset_layout(
    *,
    username: str,
    dataset_id: str,
    filename: str,
    prefix: str,
    version: str = "v1",
) -> dict[str, str]:
    clean_prefix = prefix.strip("/")
    clean_user = _safe_key_part(username.strip() or "anonymous")
    clean_dataset = _safe_key_part(dataset_id.strip())
    clean_version = _safe_key_part(version or "v1")
    clean_name = _safe_key_part(filename.strip() or "dataset.bin")
    base = "/".join(
        part for part in [
            clean_prefix,
            "private",
            "users",
            clean_user,
            "datasets",
            clean_dataset,
            clean_version,
        ] if part
    )
    return {
        "privatePrefix": base + "/",
        "shardKey": f"{base}/shards/{clean_name}",
        "approvedPrefix": build_approved_dataset_prefix(
            dataset_id=clean_dataset,
            version=clean_version,
            prefix=clean_prefix,
        ),
        "previewPrefix": build_preview_prefix(dataset_id=clean_dataset, prefix=clean_prefix),
        "redemptionPrefix": build_redemption_prefix(dataset_id=clean_dataset, prefix=clean_prefix),
    }


def build_approved_dataset_prefix(*, dataset_id: str, version: str = "v1", prefix: str) -> str:
    clean_prefix = prefix.strip("/")
    clean_dataset = _safe_key_part(dataset_id)
    clean_version = _safe_key_part(version or "v1")
    return "/".join(part for part in [clean_prefix, "approved", "datasets", clean_dataset, clean_version] if part) + "/"


def build_preview_prefix(*, dataset_id: str, prefix: str) -> str:
    clean_prefix = prefix.strip("/")
    clean_dataset = _safe_key_part(dataset_id)
    return "/".join(part for part in [clean_prefix, "previews", clean_dataset] if part) + "/"


def build_redemption_prefix(*, dataset_id: str, prefix: str) -> str:
    clean_prefix = prefix.strip("/")
    clean_dataset = _safe_key_part(dataset_id)
    return "/".join(part for part in [clean_prefix, "redemption", clean_dataset] if part) + "/"


def create_presigned_upload_post(
    settings: S3StorageSettings,
    *,
    key: str,
    content_type: str = "application/octet-stream",
    expires_in: int = 3600,
    max_size_bytes: int = 50 * 1024 * 1024 * 1024,
) -> dict[str, Any]:
    if not settings.configured:
        raise ValueError("dataset storage provider is not configured")

    try:
        import boto3  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("boto3 is required for S3/R2 presigned uploads") from exc

    client = boto3.client(
        "s3",
        endpoint_url=settings.endpoint,
        aws_access_key_id=settings.access_key,
        aws_secret_access_key=settings.secret_key,
        region_name=settings.region,
    )
    fields = {"Content-Type": content_type}
    conditions: list[Any] = [
        {"Content-Type": content_type},
        ["content-length-range", 1, max_size_bytes],
    ]
    post = client.generate_presigned_post(
        Bucket=settings.bucket,
        Key=key,
        Fields=fields,
        Conditions=conditions,
        ExpiresIn=expires_in,
    )
    return {
        "method": "POST",
        "url": post["url"],
        "fields": post["fields"],
        "expiresIn": expires_in,
        "maxSizeBytes": max_size_bytes,
        "objectUri": settings.object_uri(key),
        "objectKey": key,
    }


def _safe_key_part(value: str) -> str:
    value = value.strip().replace("\\", "/").strip("/")
    value = _SAFE_KEY_RE.sub("-", value)
    return value.strip("-._/") or "unnamed"


def _load_local_env_files_once() -> None:
    global _ENV_FILES_LOADED
    if _ENV_FILES_LOADED:
        return
    _ENV_FILES_LOADED = True
    if os.environ.get("EVO_STUDIO_DISABLE_ENV_FILE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    configured = os.environ.get("EVO_STUDIO_ENV_FILE", "").strip()
    candidates = [Path(configured).expanduser()] if configured else [Path.cwd() / ".env.local", Path.cwd() / ".env"]
    for path in candidates:
        if path.is_file():
            _load_env_file(path)


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("\"'")
        os.environ[key] = value


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        _log.warning("Invalid value for env %s=%r, using default %r", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        _log.warning("Invalid value for env %s=%r, using default %r", name, raw, default)
        return default


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _boto3_available() -> bool:
    try:
        import boto3  # type: ignore[import-not-found]  # noqa: F401
    except Exception as exc:
        _log.warning("boto3 is unavailable; S3-compatible storage is disabled: %s", exc)
        return False
    return True
