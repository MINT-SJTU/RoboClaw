from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Storage layout:
#   {dataset_dir}/.workflow/state.json
#   {dataset_dir}/.workflow/quality/latest.json
#   {dataset_dir}/.workflow/prototypes/latest.json
#   {dataset_dir}/.workflow/annotations/ep_{N}.json
#   {dataset_dir}/.workflow/propagation/latest.json
# ---------------------------------------------------------------------------

_STATE_VERSION = 1


def load_dataset_info(dataset_path: Path) -> dict[str, Any]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def _workflow_dir(dataset_path: Path) -> Path:
    return dataset_path / ".workflow"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------


def init_workflow_state(dataset_path: Path) -> dict[str, Any]:
    """Create a fresh workflow state and persist it."""
    state: dict[str, Any] = {
        "version": _STATE_VERSION,
        "dataset": dataset_path.name,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "stages": {
            "quality_validation": {
                "status": "idle",
                "selected_validators": [],
                "latest_run": None,
                "active_run_id": None,
                "pause_requested": False,
                "summary": None,
            },
            "prototype_discovery": {
                "status": "idle",
                "latest_run": None,
                "summary": None,
                "quality_filter_mode": "passed",
                "selected_episode_indices": [],
            },
            "annotation": {
                "status": "idle",
                "annotated_episodes": [],
                "propagated_source_episodes": [],
                "propagation_run": None,
                "summary": None,
            },
        },
    }
    save_workflow_state(dataset_path, state)
    return state


def load_workflow_state(dataset_path: Path) -> dict[str, Any]:
    path = _workflow_dir(dataset_path) / "state.json"
    data = _read_json(path)
    if data is None:
        return init_workflow_state(dataset_path)
    return _normalize_workflow_state(data)


def _normalize_workflow_state(state: dict[str, Any]) -> dict[str, Any]:
    stages = state.setdefault("stages", {})

    quality_stage = stages.setdefault("quality_validation", {})
    quality_stage.setdefault("status", "idle")
    quality_stage.setdefault("selected_validators", [])
    quality_stage.setdefault("latest_run", None)
    quality_stage.setdefault("active_run_id", None)
    quality_stage.setdefault("pause_requested", False)
    quality_stage.setdefault("summary", None)

    prototype_stage = stages.setdefault("prototype_discovery", {})
    prototype_stage.setdefault("status", "idle")
    prototype_stage.setdefault("latest_run", None)
    prototype_stage.setdefault("summary", None)
    prototype_stage.setdefault("quality_filter_mode", "passed")
    prototype_stage.setdefault("selected_episode_indices", [])

    annotation_stage = stages.setdefault("annotation", {})
    annotation_stage.setdefault("status", "idle")
    annotation_stage.setdefault("annotated_episodes", [])
    annotation_stage.setdefault("propagated_source_episodes", [])
    annotation_stage.setdefault("propagation_run", None)
    annotation_stage.setdefault("summary", None)

    return state


def save_workflow_state(dataset_path: Path, state: dict[str, Any]) -> None:
    _normalize_workflow_state(state)
    state["updated_at"] = _now_iso()
    _write_json(_workflow_dir(dataset_path) / "state.json", state)


def is_stage_pause_requested(dataset_path: Path, stage_key: str) -> bool:
    state = load_workflow_state(dataset_path)
    stage = state.get("stages", {}).get(stage_key, {})
    return bool(stage.get("pause_requested"))


def set_stage_pause_requested(dataset_path: Path, stage_key: str, requested: bool) -> dict[str, Any]:
    state = load_workflow_state(dataset_path)
    state["stages"][stage_key]["pause_requested"] = requested
    save_workflow_state(dataset_path, state)
    return state


# ---------------------------------------------------------------------------
# Quality results
# ---------------------------------------------------------------------------


def load_quality_results(dataset_path: Path) -> dict[str, Any] | None:
    return _read_json(_workflow_dir(dataset_path) / "quality" / "latest.json")


def save_quality_results(dataset_path: Path, results: dict[str, Any]) -> None:
    _write_json(_workflow_dir(dataset_path) / "quality" / "latest.json", results)


# ---------------------------------------------------------------------------
# Prototype results
# ---------------------------------------------------------------------------


def load_prototype_results(dataset_path: Path) -> dict[str, Any] | None:
    return _read_json(_workflow_dir(dataset_path) / "prototypes" / "latest.json")


def save_prototype_results(dataset_path: Path, results: dict[str, Any]) -> None:
    _write_json(_workflow_dir(dataset_path) / "prototypes" / "latest.json", results)


# ---------------------------------------------------------------------------
# Annotations (per-episode)
# ---------------------------------------------------------------------------


def load_annotations(dataset_path: Path, episode_index: int) -> dict[str, Any] | None:
    path = _workflow_dir(dataset_path) / "annotations" / f"ep_{episode_index}.json"
    return _read_json(path)


def save_annotations(dataset_path: Path, episode_index: int, data: dict[str, Any]) -> None:
    path = _workflow_dir(dataset_path) / "annotations" / f"ep_{episode_index}.json"
    existing = _read_json(path) or {}
    created_at = existing.get("created_at") or data.get("created_at") or _now_iso()
    version_number = existing.get("version_number", 0)
    try:
        next_version = int(version_number) + 1
    except (TypeError, ValueError):
        next_version = 1

    payload = {
        **existing,
        **data,
        "episode_index": episode_index,
        "created_at": created_at,
        "updated_at": _now_iso(),
        "version_number": next_version,
    }
    _write_json(path, payload)


# ---------------------------------------------------------------------------
# Propagation results
# ---------------------------------------------------------------------------


def load_propagation_results(dataset_path: Path) -> dict[str, Any] | None:
    return _read_json(_workflow_dir(dataset_path) / "propagation" / "latest.json")


def save_propagation_results(dataset_path: Path, results: dict[str, Any]) -> None:
    _write_json(_workflow_dir(dataset_path) / "propagation" / "latest.json", results)
