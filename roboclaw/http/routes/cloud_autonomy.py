"""Structured autonomy state for cloud training jobs."""

from __future__ import annotations

import re
from typing import Any, Mapping


def build_cloud_autonomy_state(
    payload: Mapping[str, Any],
    supervisor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable Goal -> Observe -> Diagnose -> Repair -> Verify state."""
    supervisor = supervisor or {}
    status = str(payload.get("status") or "").strip()
    status_key = status.lower()
    running = _truthy(payload.get("running"))
    remediation = payload.get("failureRemediation") or payload.get("failure_remediation") or {}
    raw_failure_code = remediation.get("code") if isinstance(remediation, Mapping) else ""
    failure_code = str(raw_failure_code or "").strip()
    supervisor_state = str(supervisor.get("state") or "").strip()
    stage = _stage_from_payload(payload)

    phase = _phase(
        status_key=status_key,
        running=running,
        failure_code=failure_code,
        supervisor_state=supervisor_state,
        runtime_unavailable=bool(supervisor.get("runtimeUnavailable")),
    )
    loop = _loop_for_phase(phase, failure_code=failure_code, supervisor_state=supervisor_state)
    blocked = phase == "blocked"

    return {
        "kind": "evo_studio_autonomy_state/v1",
        "objective": _objective(payload),
        "phase": phase,
        "loop": loop,
        "stage": stage,
        "running": running,
        "blocked": blocked,
        "blockerCode": failure_code if blocked else "",
        "nextAction": supervisor.get("nextAction") or _next_action(phase, failure_code),
        "humanActionRequired": bool(supervisor.get("requiresConfirmation")) or blocked,
        "gates": _gates(payload, supervisor, phase=phase, stage=stage, failure_code=failure_code),
        "trace": {
            "jobId": payload.get("job_id") or payload.get("jobId") or "",
            "taskName": payload.get("task_name") or payload.get("taskName") or "",
            "repairStrategy": supervisor.get("repairStrategy") or "",
            "failureCode": failure_code,
        },
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "running"}
    return bool(value)


def _objective(payload: Mapping[str, Any]) -> str:
    for key in ("task_name", "taskName", "job_id", "jobId"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return "cloud_training_job"


def _stage_from_payload(payload: Mapping[str, Any]) -> str:
    text = "\n".join(str(payload.get(key) or "") for key in ("log_tail", "logTail", "message", "error"))
    failed = re.search(r"__EVO_STAGE_FAILED__=([A-Za-z0-9_.-]+)", text)
    if failed:
        return failed.group(1)
    starts = re.findall(r"__EVO_STAGE_START__=([A-Za-z0-9_.-]+)", text)
    if starts:
        return starts[-1]
    dones = re.findall(r"__EVO_STAGE_DONE__=([A-Za-z0-9_.-]+)", text)
    if dones:
        return dones[-1]
    return ""


def _phase(
    *,
    status_key: str,
    running: bool,
    failure_code: str,
    supervisor_state: str,
    runtime_unavailable: bool,
) -> str:
    if status_key in {"succeeded", "success", "completed", "complete"}:
        return "completed"
    if runtime_unavailable:
        return "blocked"
    if failure_code in {"CLOUD_INSTANCE_UNREACHABLE", "CLOUD_GPU_UNAVAILABLE"}:
        return "blocked"
    if supervisor_state in {"repair_submitted", "repairing", "intervention_submitted"}:
        return "repairing"
    if supervisor_state == "repairable_same_runtime":
        return "repairable"
    if running or status_key in {"running", "submitted", "submitting", "queued", "pending", "starting"}:
        return "running"
    if status_key in {"failed", "stopped", "missing"} or failure_code:
        return "failed"
    return "idle"


def _loop_for_phase(phase: str, *, failure_code: str, supervisor_state: str) -> str:
    if phase == "completed":
        return "report"
    if phase == "running":
        return "observe"
    if phase == "repairing":
        return "verify_repair"
    if phase == "repairable":
        return "repair"
    if phase == "blocked" and failure_code == "CLOUD_INSTANCE_UNREACHABLE":
        return "rebind_runtime"
    if phase == "blocked" and failure_code == "CLOUD_GPU_UNAVAILABLE":
        return "switch_gpu_or_prepare_only"
    if supervisor_state == "needs_review":
        return "diagnose"
    if phase == "failed":
        return "diagnose"
    return "wait"


def _next_action(phase: str, failure_code: str) -> str:
    if phase == "blocked" and failure_code == "CLOUD_INSTANCE_UNREACHABLE":
        return "rebind_runtime"
    if phase == "blocked" and failure_code == "CLOUD_GPU_UNAVAILABLE":
        return "rebind_gpu_runtime_or_prepare_only"
    if phase == "repairable":
        return "auto_retry_same_runtime"
    if phase == "running":
        return "watch_status"
    if phase == "completed":
        return "show_results"
    return "wait"


def _gates(
    payload: Mapping[str, Any],
    supervisor: Mapping[str, Any],
    *,
    phase: str,
    stage: str,
    failure_code: str,
) -> list[dict[str, str]]:
    log_text = "\n".join(str(payload.get(key) or "") for key in ("log_tail", "logTail", "message", "error")).lower()
    source_ready = bool(payload.get("dataset_path") or payload.get("datasetPath"))
    model_ready = bool(payload.get("checkpoint_path") or payload.get("checkpointPath"))
    runtime_unavailable = bool(supervisor.get("runtimeUnavailable"))
    return [
        _gate(
            "runtime",
            "blocked" if runtime_unavailable or failure_code == "CLOUD_INSTANCE_UNREACHABLE" else "ready" if phase != "idle" else "unknown",
            "SSH runtime is not reachable" if runtime_unavailable or failure_code == "CLOUD_INSTANCE_UNREACHABLE" else "SSH runtime is reachable",
        ),
        _gate(
            "gpu",
            "blocked" if failure_code == "CLOUD_GPU_UNAVAILABLE" else "ready" if "cuda" in log_text or phase in {"running", "completed"} else "unknown",
            "GPU is visible" if failure_code != "CLOUD_GPU_UNAVAILABLE" else "GPU is not visible to the job",
        ),
        _gate("sources", "ready" if source_ready and model_ready else "unknown", "dataset/model paths are resolved"),
        _gate("environment", _stage_gate(stage, phase), "runtime environment is being prepared or verified"),
        _gate("execution", "running" if phase == "running" else "done" if phase == "completed" else "waiting", "training/evaluation execution"),
        _gate(
            "results",
            "ready" if phase == "completed" else "waiting",
            "metrics, logs, checkpoints, or rollout summaries",
        ),
    ]


def _stage_gate(stage: str, phase: str) -> str:
    if phase == "completed":
        return "done"
    if phase in {"failed", "blocked"}:
        return "blocked"
    if stage:
        return "running"
    return "unknown"


def _gate(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}
