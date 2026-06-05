"""Cloud training start snapshot persistence."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboclaw.embodied.embodiment.manifest import helpers as manifest_helpers

_cloud_start_snapshots: dict[str, dict[str, Any]] = {}
_cloud_start_snapshots_loaded = False
_cloud_start_snapshots_lock = threading.Lock()
_SENSITIVE_SNAPSHOT_KEY_RE = re.compile(r"(token|secret|password|private.?key|access.?key|api.?key)", re.IGNORECASE)
_log = logging.getLogger(__name__)


def _route_module():
    return sys.modules.get("roboclaw.http.routes.train_cloud")


def _sync_loaded_from_route_module() -> None:
    global _cloud_start_snapshots_loaded
    route_module = _route_module()
    if route_module is None:
        return
    route_value = getattr(route_module, "_cloud_start_snapshots_loaded", _cloud_start_snapshots_loaded)
    if route_value != _cloud_start_snapshots_loaded:
        _cloud_start_snapshots_loaded = bool(route_value)


def _publish_loaded_to_route_module() -> None:
    route_module = _route_module()
    if route_module is not None:
        setattr(route_module, "_cloud_start_snapshots_loaded", _cloud_start_snapshots_loaded)


def clear_cloud_start_snapshots_for_tests() -> None:
    global _cloud_start_snapshots_loaded
    with _cloud_start_snapshots_lock:
        _cloud_start_snapshots.clear()
        _cloud_start_snapshots_loaded = True
        _publish_loaded_to_route_module()
        path = _cloud_supervisor_snapshot_path()
        if os.environ.get("EVO_STUDIO_CLOUD_SUPERVISOR_FILE"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

def _cloud_supervisor_snapshot_path():
    configured = os.environ.get("EVO_STUDIO_CLOUD_SUPERVISOR_FILE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return manifest_helpers.get_roboclaw_home() / "workspace" / "embodied" / "cloud_supervisor.json"
def _load_cloud_start_snapshots_unlocked() -> None:
    global _cloud_start_snapshots_loaded
    _sync_loaded_from_route_module()
    if _cloud_start_snapshots_loaded:
        return
    _cloud_start_snapshots_loaded = True
    _publish_loaded_to_route_module()
    path = _cloud_supervisor_snapshot_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except json.JSONDecodeError as exc:
        _log.warning("Cloud start snapshot file corrupted, skipping: %s", exc)
        return
    except OSError:
        return
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else {}
    if not isinstance(snapshots, dict):
        return
    for key, snapshot in snapshots.items():
        if isinstance(key, str) and isinstance(snapshot, dict):
            _cloud_start_snapshots[key] = snapshot
def _save_cloud_start_snapshots_unlocked() -> None:
    path = _cloud_supervisor_snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(_cloud_start_snapshots) >= 200:
        stale_keys = list(_cloud_start_snapshots.keys())[:50]
        for key in stale_keys:
            _cloud_start_snapshots.pop(key, None)
    payload = {
        "kind": "evo_studio_cloud_supervisor_store/v1",
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "snapshots": _cloud_start_snapshots,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
def _snapshot_key(username: str, job_id: str) -> str:
    return f"{username.strip()}::{job_id.strip()}"
def _snapshot_aliases(username: str, payload: dict[str, Any]) -> list[str]:
    values = [
        str(payload.get("job_id") or "").strip(),
        str(payload.get("task_name") or "").strip(),
    ]
    aliases: list[str] = []
    for value in values:
        if not value:
            continue
        aliases.append(_snapshot_key(username, value))
        prefix = f"task:{username.strip()}:"
        if value.startswith(prefix):
            aliases.append(_snapshot_key(username, value[len(prefix):]))
    return aliases
def _redact_snapshot_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key)
            redacted[clean_key] = "***" if _SENSITIVE_SNAPSHOT_KEY_RE.search(clean_key) else _redact_snapshot_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_snapshot_payload(item) for item in value]
    return value
def _remember_cloud_start(username: str, start_payload: dict[str, Any], result_payload: dict[str, Any]) -> None:
    if not username.strip():
        return
    snapshot = {
        "kind": "evo_studio_cloud_start_snapshot/v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "payload": _redact_snapshot_payload(dict(start_payload)),
    }
    with _cloud_start_snapshots_lock:
        _load_cloud_start_snapshots_unlocked()
        for key in _snapshot_aliases(username, result_payload):
            _cloud_start_snapshots[key] = snapshot
        _save_cloud_start_snapshots_unlocked()
def _lookup_cloud_start(username: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    with _cloud_start_snapshots_lock:
        _load_cloud_start_snapshots_unlocked()
        for key in _snapshot_aliases(username, payload):
            snapshot = _cloud_start_snapshots.get(key)
            if snapshot:
                return snapshot
    return None
