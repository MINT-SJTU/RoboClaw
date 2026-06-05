"""Server-side training source authorization registry.

This module intentionally stores only connection metadata in the API surface.
Secrets stay in environment variables or a private JSON file on the deployment
server, never in the open-source repository or frontend request body.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboclaw.data.storage import _load_local_env_files_once


_VALID_KINDS = {"data", "model", "both"}
_VALID_VISIBILITY = {"user", "team"}
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._:-]+")
_STORE_LOCK = threading.Lock()
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthConnection:
    id: str
    kind: str = "both"
    provider: str = "custom"
    label: str = ""
    scope: str = ""
    owner: str = ""
    visibility: str = "team"
    source_prefixes: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    required_env: tuple[str, ...] = ()
    secrets: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuthConnection":
        auth_id = str(payload.get("id") or payload.get("authRef") or "").strip()
        kind = str(payload.get("kind") or "both").strip().lower()
        if kind not in _VALID_KINDS:
            kind = "both"
        visibility = str(payload.get("visibility") or "team").strip().lower()
        if visibility not in _VALID_VISIBILITY:
            visibility = "team"
        env_payload = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env = {
            str(runtime_key).strip(): str(source_key).strip()
            for runtime_key, source_key in env_payload.items()
            if str(runtime_key).strip() and str(source_key).strip()
        }
        required_env = tuple(
            str(item).strip()
            for item in payload.get("requiredEnv", payload.get("required_env", [])) or []
            if str(item).strip()
        )
        secrets_payload = payload.get("secrets") if isinstance(payload.get("secrets"), dict) else {}
        secrets = {
            str(secret_key).strip(): str(secret_value)
            for secret_key, secret_value in secrets_payload.items()
            if str(secret_key).strip() and str(secret_value)
        }
        source_prefixes = tuple(
            str(item).strip()
            for item in payload.get("sourcePrefixes", payload.get("source_prefixes", [])) or []
            if str(item).strip()
        )
        return cls(
            id=auth_id,
            kind=kind,
            provider=str(payload.get("provider") or "custom").strip().lower() or "custom",
            label=str(payload.get("label") or auth_id).strip() or auth_id,
            scope=str(payload.get("scope") or "").strip(),
            owner=str(payload.get("owner") or "").strip(),
            visibility=visibility,
            source_prefixes=source_prefixes,
            env=env,
            required_env=required_env,
            secrets=secrets,
            created_at=str(payload.get("createdAt") or payload.get("created_at") or "").strip(),
            updated_at=str(payload.get("updatedAt") or payload.get("updated_at") or "").strip(),
        )

    @property
    def configured(self) -> bool:
        env_names = set(self.required_env)
        env_names.update(name for name in self.env.values() if name)
        env_configured = all(os.environ.get(name, "").strip() for name in env_names)
        if env_names:
            return env_configured
        return bool(self.secrets)

    def supports_kind(self, kind: str) -> bool:
        normalized = kind.strip().lower()
        return self.kind == "both" or self.kind == normalized

    def visible_to(self, username: str) -> bool:
        if self.visibility == "team":
            return True
        return bool(username.strip()) and self.owner == username.strip()

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "provider": self.provider,
            "label": self.label,
            "scope": self.scope,
            "owner": self.owner,
            "visibility": self.visibility,
            "sourcePrefixes": list(self.source_prefixes),
            "configured": self.configured,
            "requiresSecrets": bool(self.env or self.required_env or self.secrets),
            "secretFields": sorted(self.secrets.keys()),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def private_dict(self) -> dict[str, Any]:
        payload = self.public_dict()
        payload["env"] = dict(self.env)
        payload["requiredEnv"] = list(self.required_env)
        payload["secrets"] = dict(self.secrets)
        return payload


def list_auth_connections(kind: str = "", username: str = "") -> list[AuthConnection]:
    _load_local_env_files_once()
    requested = kind.strip().lower()
    user = username.strip()
    by_id: dict[str, AuthConnection] = {}
    for connection in [*_load_file_connections(), *_load_json_connections()]:
        if not connection.id:
            continue
        by_id[connection.id] = connection
    values = sorted(by_id.values(), key=lambda item: (item.kind, item.provider, item.id))
    if requested:
        values = [item for item in values if item.supports_kind(requested)]
    if user:
        values = [item for item in values if item.visible_to(user)]
    else:
        values = [item for item in values if item.visibility == "team"]
    return values


def public_auth_connections(kind: str = "", username: str = "") -> list[dict[str, Any]]:
    return [connection.public_dict() for connection in list_auth_connections(kind, username=username)]


def find_auth_connection(auth_ref: str, kind: str = "", username: str = "") -> AuthConnection | None:
    clean_ref = auth_ref.strip()
    if not clean_ref:
        return None
    for connection in list_auth_connections(kind, username=username):
        if connection.id == clean_ref:
            return connection
    return None


def validate_training_auth_refs(params: dict[str, Any], *, username: str = "") -> list[str]:
    """Return human-readable validation errors for dataset/model auth refs."""

    errors: list[str] = []
    for source_key, kind, label in [
        ("datasetSource", "data", "数据"),
        ("modelSource", "model", "模型"),
    ]:
        source = params.get(source_key)
        if not isinstance(source, dict):
            continue
        auth_ref = str(source.get("authRef") or source.get("credentialRef") or "").strip()
        if not auth_ref:
            continue
        connection = find_auth_connection(auth_ref, kind, username=username)
        if connection is None:
            errors.append(f"{label}授权连接 ID 不存在：{auth_ref}")
            continue
        if not connection.configured:
            errors.append(f"{label}授权连接还没有配置完整密钥：{auth_ref}")
    return errors


def upsert_auth_connection(payload: dict[str, Any], *, username: str) -> AuthConnection:
    _load_local_env_files_once()
    owner = username.strip()
    if not owner:
        raise ValueError("username is required to save a private connection")
    auth_id = _safe_auth_ref(str(payload.get("id") or "").strip() or _default_auth_ref(payload, owner))
    if not auth_id:
        raise ValueError("connection id is required")
    now = datetime.now(timezone.utc).isoformat()
    next_payload = dict(payload)
    next_payload["id"] = auth_id
    next_payload["owner"] = owner
    next_payload["visibility"] = str(payload.get("visibility") or "user").strip().lower() or "user"
    existing = find_auth_connection(auth_id, username=owner)
    next_payload["createdAt"] = existing.created_at if existing else now
    next_payload["updatedAt"] = now
    connection = AuthConnection.from_dict(next_payload)
    if connection.visibility == "user" and connection.owner != owner:
        raise ValueError("user-scoped connection owner mismatch")

    with _STORE_LOCK:
        store_payload = _load_store_payload()
        connections = _connections_from_payload(store_payload)
        kept = [
            item
            for item in connections
            if not (item.id == connection.id and item.owner == connection.owner and item.visibility == connection.visibility)
        ]
        kept.append(connection)
        _write_store_payload({"connections": [item.private_dict() for item in kept]})
    return connection


def delete_auth_connection(auth_ref: str, *, username: str) -> bool:
    owner = username.strip()
    clean_ref = auth_ref.strip()
    if not owner or not clean_ref:
        return False
    with _STORE_LOCK:
        store_payload = _load_store_payload()
        connections = _connections_from_payload(store_payload)
        kept = [
            item
            for item in connections
            if not (item.id == clean_ref and item.visibility == "user" and item.owner == owner)
        ]
        changed = len(kept) != len(connections)
        if changed:
            _write_store_payload({"connections": [item.private_dict() for item in kept]})
        return changed


def _load_file_connections() -> list[AuthConnection]:
    return _connections_from_payload(_load_store_payload())


def _auth_refs_file() -> Path:
    path_value = os.environ.get("EVO_STUDIO_AUTH_REFS_FILE", "").strip()
    if not path_value:
        path_value = str(Path.home() / ".roboclaw" / "auth_refs.json")
    return Path(path_value).expanduser()


def _load_store_payload() -> Any:
    path = _auth_refs_file()
    if not path.is_file():
        return {"connections": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("Failed to load auth connection store %s: %s", path, exc)
        return {"connections": []}


def _write_store_payload(payload: dict[str, Any]) -> None:
    path = _auth_refs_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _load_json_connections() -> list[AuthConnection]:
    raw = os.environ.get("EVO_STUDIO_AUTH_REFS_JSON", "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _log.warning("Invalid EVO_STUDIO_AUTH_REFS_JSON; no env auth connections loaded: %s", exc)
        return []
    return _connections_from_payload(payload)


def _connections_from_payload(payload: Any) -> list[AuthConnection]:
    if isinstance(payload, dict):
        items = payload.get("connections", [])
    else:
        items = payload
    if not isinstance(items, list):
        return []
    connections: list[AuthConnection] = []
    for item in items:
        if isinstance(item, dict):
            connections.append(AuthConnection.from_dict(item))
        else:
            _log.warning("Skipping malformed auth connection entry: %r", item)
    return connections


def _safe_auth_ref(value: str) -> str:
    value = _SAFE_ID_RE.sub("-", value.strip())
    return value.strip("-._:")


def _default_auth_ref(payload: dict[str, Any], owner: str) -> str:
    provider = str(payload.get("provider") or "custom").strip().lower()
    kind = str(payload.get("kind") or "both").strip().lower()
    return f"{owner}-{provider}-{kind}"
