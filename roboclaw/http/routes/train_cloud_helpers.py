"""Helper functions for cloud training routes."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from . import cloud_snapshot as cloud_snapshot_state
from . import cloud_supervisor as cloud_supervisor_state
from .cloud_snapshot import _lookup_cloud_start, clear_cloud_start_snapshots_for_tests
from .cloud_supervisor import clear_cloud_supervisor_runtime_for_tests

_log = logging.getLogger(__name__)

OFFICIAL_DATASET_SOURCES: dict[str, dict[str, str]] = {
    "libero": {
        "sourceType": "public_reference",
        "datasetId": "libero",
        "uri": "hf://HuggingFaceVLA/libero",
        "format": "lerobot",
        "benchmark": "libero",
    },
}

_TERMINAL_CLOUD_STATUSES = {"failed", "missing", "stopped", "completed", "complete", "succeeded", "success"}
_RUNTIME_BINDING_FAILURE_CODES = {"CLOUD_GPU_UNAVAILABLE", "CLOUD_INSTANCE_UNREACHABLE"}


def _user_facing_runtime_warning(value: Any) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if not text:
        return "云端实例还没有完成连接检查，请重新绑定当前实例。"
    if "error reading ssh protocol banner" in lowered or "ssh protocol layer" in lowered:
        return "当前绑定的云端端口已连接，但没有返回 SSH 登录协议；通常是旧实例端口失效、实例未完成开机，或粘贴的不是最新 SSH 命令。请重新绑定当前实例的最新 SSH 命令。"
    if "configured ssh instance is not reachable" in lowered or "ssh connection failed" in lowered:
        return "当前绑定的云端实例无法建立 SSH 连接；请确认实例已开机，并重新绑定当前实例的最新 SSH 命令。"
    if "connection refused" in lowered:
        return "云端实例拒绝连接；请确认实例已开机、SSH 端口正确，然后重新绑定。"
    if "missing /dev/nvidia0" in lowered or "nvidia-smi has no gpu output" in lowered or "no visible gpu" in lowered:
        return "当前 SSH 实例已连接，但没有可见 GPU；请切到有卡开机，或选择无卡准备环境。"
    return text


def _runtime_configuration_ready(config: dict[str, Any], *, require_gpu: bool = True) -> tuple[bool, str]:
    ready = bool(config.get("configurationReady", config.get("ready", False)))
    warnings = config.get("configurationWarnings", config.get("warnings", []))
    warning_text = " ".join(str(item) for item in warnings if item).lower()
    blockers = (
        "not reachable",
        "unreachable",
        "connection refused",
        "error reading ssh protocol banner",
        "ssh connection failed",
    )
    gpu_blockers = (
        "gpu unavailable",
        "__evo_gpu_unavailable__",
        "no visible gpu",
        "missing /dev/nvidia0",
        "nvidia-smi has no gpu output",
    )
    if any(blocker in warning_text for blocker in blockers):
        return False, _user_facing_runtime_warning(warnings[0] if warnings else "云端实例连接检查未通过")
    if require_gpu and any(blocker in warning_text for blocker in gpu_blockers):
        return False, _user_facing_runtime_warning(warnings[0] if warnings else "云端实例没有可见 GPU")
    if not ready:
        return False, _user_facing_runtime_warning(warnings[0] if warnings else "云端运行环境还没有就绪")
    return True, ""


def _bridge_error_status(exc: RuntimeError) -> int:
    return 503 if "bridge is not enabled" in str(exc).lower() else 502


def clear_cloud_supervisor_snapshots_for_tests() -> None:
    clear_cloud_start_snapshots_for_tests()
    clear_cloud_supervisor_runtime_for_tests()


def _request_username(provided: str = "", header_username: str = "", bearer_token: str = "") -> str:
    from roboclaw.http.auth import resolve_username
    return resolve_username(provided, header_username, bearer_token)


def _cloud_failure_code(payload: dict[str, Any]) -> str:
    remediation = payload.get("failureRemediation") or payload.get("failure_remediation")
    supervisor = payload.get("supervisor")
    if isinstance(remediation, dict):
        code = str(remediation.get("code") or "").strip()
        if code:
            return code.upper()
    if isinstance(supervisor, dict):
        code = str(supervisor.get("failureCode") or "").strip()
        if code:
            return code.upper()
    return ""


def _cloud_failure_fingerprint(payload: dict[str, Any]) -> str:
    text = "\n".join(str(payload.get(key) or "") for key in ("message", "log_tail", "logTail", "error")).lower()
    stage = ""
    marker = "__evo_stage_failed__="
    if marker in text:
        stage = text.split(marker, 1)[1].splitlines()[0].strip()[:80]
    signals = [
        token
        for token in (
            "__evo_gpu_unavailable__",
            "modulenotfounderror",
            "no matching distribution",
            "requires a different python",
            "syntaxerror",
            "killed",
            "terminated",
            "connection refused",
            "error reading ssh protocol banner",
        )
        if token in text
    ]
    return "|".join(
        part for part in (_cloud_failure_code(payload) or "UNKNOWN_CLOUD_FAILURE", stage, ",".join(signals)) if part
    )


def _apply_completed_supervisor_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    supervisor = payload.get("supervisor") if isinstance(payload.get("supervisor"), dict) else {}
    runtime = supervisor.get("runtime") if isinstance(supervisor, dict) else {}
    status_key = str(payload.get("status") or "").strip().lower()
    running = payload.get("running") is True or str(payload.get("running") or "").strip().lower() == "true"
    if not running and status_key in {"succeeded", "success", "completed", "complete"}:
        result = dict(payload)
        patched_supervisor = dict(supervisor)
        patched_supervisor.update({
            "state": "completed",
            "nextAction": "wait",
            "canRetrySameRuntime": False,
            "failureCode": None,
            "repairStrategy": "",
            "requiresConfirmation": False,
        })
        if isinstance(runtime, dict):
            patched_runtime = dict(runtime)
            previous_runtime_state = str(patched_runtime.get("state") or "").strip().lower()
            patched_runtime.pop("failureRemediation", None)
            patched_runtime.pop("failure_remediation", None)
            patched_runtime.update({
                "state": "completed",
                "status": "Succeeded" if status_key in {"succeeded", "success"} else str(payload.get("status") or "Completed"),
                "message": patched_runtime.get("message") if previous_runtime_state == "completed" else "任务已成功完成。",
            })
            patched_supervisor["runtime"] = patched_runtime
        result["supervisor"] = patched_supervisor
        result["status"] = "Succeeded" if status_key in {"succeeded", "success"} else str(payload.get("status") or "Completed")
        result["running"] = False
        result["error"] = ""
        result.pop("failureRemediation", None)
        result.pop("failure_remediation", None)
        message_text = str(result.get("message") or "")
        if "missing required value:" in message_text.lower() or "ssh status check failed" in message_text.lower():
            result["message"] = "任务已成功完成。"
        return result
    if not isinstance(runtime, dict):
        return payload
    runtime_state = str(runtime.get("state") or "").strip().lower()
    runtime_status = str(runtime.get("status") or "").strip()
    runtime_status_key = runtime_status.lower()
    if runtime_state != "completed" or runtime_status_key not in {"succeeded", "success", "completed", "complete"}:
        return payload

    result = dict(payload)
    original_status_key = str(result.get("status") or "").strip().lower()
    patched_supervisor = dict(supervisor)
    patched_supervisor.update({
        "state": "completed",
        "nextAction": "wait",
        "canRetrySameRuntime": False,
        "failureCode": None,
        "repairStrategy": "",
        "requiresConfirmation": False,
    })
    result["supervisor"] = patched_supervisor
    result["status"] = "Succeeded" if runtime_status_key in {"succeeded", "success"} else runtime_status
    result["running"] = False
    result["error"] = ""
    result.pop("failureRemediation", None)
    result.pop("failure_remediation", None)

    stale_status_text = "\n".join(
        str(result.get(key) or "")
        for key in ("message", "log_tail", "logTail", "error")
    ).lower()
    if (
        original_status_key not in {"succeeded", "success", "completed", "complete"}
        or "ssh status check failed" in stale_status_text
        or "error reading ssh protocol banner" in stale_status_text
    ):
        result["message"] = str(runtime.get("message") or "任务已成功完成。")
        result["log_tail"] = ""
        result["logTail"] = ""
    return result


def _is_terminal_cloud_payload(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").strip().lower()
    running = payload.get("running") is True or str(payload.get("running") or "").strip().lower() == "true"
    return not running and status in _TERMINAL_CLOUD_STATUSES


def _runtime_binding_failure_archivable(payload: dict[str, Any]) -> bool:
    if not _is_terminal_cloud_payload(payload):
        return False
    code = _cloud_failure_code(payload)
    if code in _RUNTIME_BINDING_FAILURE_CODES:
        return True
    text = "\n".join(str(payload.get(key) or "") for key in ("message", "log_tail", "logTail", "error")).lower()
    return (
        "__evo_gpu_unavailable__" in text
        or "no visible gpu" in text
        or "missing /dev/nvidia0" in text
        or "nvidia-smi has no gpu output" in text
        or "ssh connection failed" in text
        or "error reading ssh protocol banner" in text
        or "configured ssh instance is not reachable" in text
    )


def _runtime_binding_failure_message(payload: dict[str, Any]) -> str:
    code = _cloud_failure_code(payload)
    if code == "CLOUD_GPU_UNAVAILABLE":
        return "当前云端实例没有可见 GPU，已停止自动重试。请重新绑定有卡实例后重新启动任务，或改成无卡准备。"
    if code == "CLOUD_INSTANCE_UNREACHABLE":
        return "当前云端实例不可达，已停止自动重试。请重新开机或重新绑定实例后再启动任务。"
    return ""


def _cloud_supervisor_max_repairs() -> int:
    raw = os.environ.get("EVO_STUDIO_CLOUD_SUPERVISOR_MAX_REPAIRS", "-1").strip().lower()
    if raw in {"", "-1", "none", "unlimited"}:
        return -1
    try:
        return max(0, int(raw))
    except ValueError:
        _log.warning("Invalid EVO_STUDIO_CLOUD_SUPERVISOR_MAX_REPAIRS=%r; using unlimited repairs", raw)
        return -1


def _archived_current_payload(payload: dict[str, Any], *, message: str) -> dict[str, Any]:
    previous = {
        "job_id": payload.get("job_id"),
        "task_name": payload.get("task_name"),
        "status": payload.get("status"),
        "running": payload.get("running"),
        "provider": payload.get("provider"),
        "failureRemediation": payload.get("failureRemediation"),
        "error": payload.get("error"),
        "log_path": payload.get("log_path"),
    }
    return {
        "job_id": "",
        "task_name": "",
        "status": "Idle",
        "running": False,
        "provider": payload.get("provider") or "",
        "message": message,
        "staleAfterRuntimeRebind": True,
        "archivedPreviousJob": previous,
    }


def _runtime_rebind_restart_request(
    payload: dict[str, Any],
    username: str,
    *,
    deployment_mode: str = "",
    user_guidance: str = "",
) -> dict[str, Any] | None:
    from .cloud_supervisor import _failure_context_payload, _repair_harden_known_training_params

    if not _runtime_binding_failure_archivable(payload):
        return None
    snapshot = _lookup_cloud_start(username, payload)
    if not snapshot:
        return None
    start_payload = dict(snapshot.get("payload") or {})
    params = dict(start_payload.get("params") or {})
    for stale_key in (
        "bootstrapCommands",
        "bootstrapProfileSpec",
        "healthcheckCommands",
        "preflightCommands",
        "setupCommand",
        "sourceResolutions",
        "command",
        "failureContext",
        "failure_context",
        "failureRemediation",
        "failure_remediation",
        "supervisor",
        "repairBootstrapCommands",
        "repairOfJobId",
        "restartOfJobId",
        "repairStrategy",
        "forceRepairBootstrap",
    ):
        params.pop(stale_key, None)
    params.update({
        "restartOfJobId": str(payload.get("job_id") or ""),
        "repairOfJobId": str(payload.get("job_id") or ""),
        "repairStrategy": "restart_after_runtime_rebind",
        "failureRemediation": payload.get("failureRemediation") or payload.get("failure_remediation") or {},
        "failureContext": _failure_context_payload(payload, user_guidance=user_guidance),
        "supervisor": {
            "kind": "evo_studio_job_supervisor/v1",
            "mode": "same_runtime_restart",
            "sameRuntimeOnly": True,
            "inspectLogs": True,
            "retryWithoutUserConfirmation": True,
            "noRuntimeChange": True,
            "noSecretChange": True,
            "noBudgetIncrease": True,
            "userGuidance": user_guidance.strip(),
        },
    })
    params = _repair_harden_known_training_params(params)
    base_task_name = str(start_payload.get("task_name") or start_payload.get("taskName") or "cloud-restart")
    base_task_name = base_task_name.rsplit("-repair-", 1)[0].rsplit("-restart-", 1)[0]
    start_payload.update({
        "username": username,
        "params": params,
        "task_name": f"{base_task_name}-restart-{datetime.now(timezone.utc).strftime('%H%M%S')}",
        "waitForSubmit": True,
    })
    if deployment_mode == "ssh":
        start_payload["hourly_cost_cents"] = 0
        start_payload["hourlyCostCents"] = 0
    return start_payload


def _first_cloud_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _infer_model_source_type(model_source: dict[str, Any], params: dict[str, Any]) -> str:
    explicit = _first_cloud_text(
        model_source.get("sourceType"),
        model_source.get("type"),
        params.get("modelSourceKind"),
    )
    if explicit:
        return explicit

    model_uri = _first_cloud_text(
        model_source.get("uri"),
        model_source.get("modelUri"),
        model_source.get("repoId"),
        model_source.get("repository"),
        model_source.get("checkpoint"),
        model_source.get("path"),
        params.get("checkpointPath"),
        params.get("sftModelPath"),
    ).lower()
    checkpoint_format = _first_cloud_text(params.get("checkpointFormat"), model_source.get("format")).lower()
    config_name = _first_cloud_text(
        params.get("configName"),
        params.get("rlinfConfigName"),
        params.get("builtinTrainingProfile"),
    )

    if checkpoint_format == "rlinf_config" or (config_name and not model_uri):
        return "rlinf_config_default"
    if not model_uri:
        return "builtin_policy"
    if model_uri.startswith(("hf://", "huggingface://", "modelscope://", "https://huggingface.co/", "http://", "https://")):
        return "public_model_repo"
    if model_uri.startswith(("s3://", "oss://", "r2://", "gs://", "cos://")):
        return "user_object_storage"
    if model_uri.startswith("/"):
        return "evo_studio_checkpoint"
    return "public_model_repo"


def _normalize_model_source_contract(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    raw_model_source = normalized.get("modelSource")
    model_source = dict(raw_model_source) if isinstance(raw_model_source, dict) else {}
    if not _first_cloud_text(model_source.get("uri")):
        model_uri = _first_cloud_text(model_source.get("modelUri"), model_source.get("repoId"), model_source.get("repository"))
        if model_uri:
            model_source["uri"] = model_uri
    if not _first_cloud_text(model_source.get("checkpoint")):
        checkpoint = _first_cloud_text(model_source.get("checkpointName"))
        if checkpoint:
            model_source["checkpoint"] = checkpoint
    if not _first_cloud_text(model_source.get("modelFamily")):
        model_family = _first_cloud_text(
            normalized.get("modelFamily"),
            normalized.get("policyType"),
            normalized.get("policyFamily"),
            normalized.get("modelRegistryName"),
        )
        if model_family:
            model_source["modelFamily"] = model_family

    model_source["sourceType"] = _infer_model_source_type(model_source, normalized)
    normalized["modelSource"] = model_source
    normalized["modelSourceKind"] = model_source["sourceType"]

    source_contract = normalized.get("sourceContract")
    if isinstance(source_contract, dict):
        source_contract = dict(source_contract)
        source_contract["modelSource"] = dict(model_source)
        source_contract["modelSourceKind"] = model_source["sourceType"]
        normalized["sourceContract"] = source_contract

    training_contract = normalized.get("trainingContract")
    if isinstance(training_contract, dict):
        training_contract = dict(training_contract)
        sources = training_contract.get("sources")
        if isinstance(sources, dict):
            sources = dict(sources)
            sources["model"] = dict(model_source)
            training_contract["sources"] = sources
            normalized["trainingContract"] = training_contract
    return normalized


def _looks_like_official_libero(benchmark: str, uri: str) -> bool:
    if uri.startswith(("s3://", "oss://", "http://", "https://", "hf://", "huggingface://", "/")):
        return False
    return "libero" in f"{benchmark} {uri}".lower()


def _normalize_cloud_training_params(params: dict[str, Any], *, dataset_name: str = "") -> dict[str, Any]:
    normalized = dict(params or {})
    dataset_source = normalized.get("datasetSource")
    if isinstance(dataset_source, dict):
        dataset_source = dict(dataset_source)
    else:
        dataset_source = {}

    benchmark = str(
        normalized.get("benchmark")
        or normalized.get("suite")
        or dataset_source.get("benchmark")
        or dataset_source.get("datasetId")
        or dataset_name
        or ""
    ).strip().lower()
    dataset_uri = str(
        normalized.get("datasetPath")
        or dataset_source.get("uri")
        or dataset_source.get("datasetId")
        or dataset_name
        or ""
    ).strip()

    if _looks_like_official_libero(benchmark, dataset_uri):
        official = dict(OFFICIAL_DATASET_SOURCES["libero"])
        dataset_source = {**official, **{key: value for key, value in dataset_source.items() if value}}
        dataset_source["sourceType"] = official["sourceType"]
        dataset_source["datasetId"] = official["datasetId"]
        dataset_source["uri"] = official["uri"]
        dataset_source["format"] = dataset_source.get("format") or official["format"]
        normalized["datasetSource"] = dataset_source
        normalized["datasetPath"] = official["uri"]
        normalized["datasetFormat"] = dataset_source["format"]
        normalized.setdefault("benchmark", "libero")
        normalized.setdefault("suite", "libero")
    return _normalize_model_source_contract(normalized)


def _infer_training_benchmark(params: dict[str, Any]) -> str:
    dataset_source = params.get("datasetSource") if isinstance(params.get("datasetSource"), dict) else {}
    text = " ".join(
        str(value or "").lower()
        for value in (
            params.get("benchmark"),
            params.get("suite"),
            params.get("environmentKind"),
            params.get("configName"),
            params.get("rlinfConfigName"),
            params.get("builtinTrainingProfile"),
            dataset_source.get("benchmark"),
            dataset_source.get("datasetId"),
            dataset_source.get("uri"),
        )
    )
    if "maniskill" in text:
        return "maniskill"
    if "libero" in text:
        return "libero"
    if "metaworld" in text:
        return "metaworld"
    if "isaac" in text:
        return "isaaclab"
    return ""


def _model_is_unresolved(params: dict[str, Any], policy_type: str) -> bool:
    model_source = params.get("modelSource") if isinstance(params.get("modelSource"), dict) else {}
    config_name = str(
        params.get("configName")
        or params.get("rlinfConfigName")
        or params.get("builtinTrainingProfile")
        or ""
    ).strip()
    model_text = str(
        model_source.get("uri")
        or model_source.get("checkpoint")
        or model_source.get("modelFamily")
        or params.get("modelFamily")
        or params.get("policyType")
        or policy_type
        or ""
    ).strip().lower()
    return model_text in {"", "auto", "ai_resolved", "builtin_policy", "unknown"}


def _validate_cloud_training_start(params: dict[str, Any], *, policy_type: str) -> None:
    dataset_source = params.get("datasetSource") if isinstance(params.get("datasetSource"), dict) else {}
    benchmark = _infer_training_benchmark(params)
    dataset_text = " ".join(
        str(value or "").lower()
        for value in (
            dataset_source.get("uri"),
            dataset_source.get("datasetId"),
            dataset_source.get("benchmark"),
            params.get("datasetPath"),
            params.get("dataData"),
        )
    )
    if benchmark == "maniskill" and "libero" in dataset_text:
        raise HTTPException(
            status_code=400,
            detail="training source mismatch: benchmark=maniskill cannot launch with a LIBERO dataset source",
        )
    if _model_is_unresolved(params, policy_type):
        raise HTTPException(
            status_code=400,
            detail="model source is unresolved: launch requires a concrete model, checkpoint, or RLinf config",
        )
