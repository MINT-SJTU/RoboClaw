"""Cloud training supervisor state, failure classification, and same-runtime repair."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

from roboclaw.training import TrainingService
from roboclaw.training.rlinf_catalog import apply_rlinf_config_contract

from .cloud_repair_agent import (
    inject_repair_commands as _inject_repair_commands,
    inject_repair_commands_async as _inject_repair_commands_async,
    record_repair_agent_event as _record_repair_agent_event,
    repair_bootstrap_commands_for_failure as _repair_bootstrap_commands_for_failure_text,
)
from .cloud_snapshot import _cloud_supervisor_snapshot_path, _lookup_cloud_start, _snapshot_aliases, _snapshot_key

_cloud_supervisor_tasks: dict[str, asyncio.Task[None]] = {}
_cloud_supervisor_states: dict[str, dict[str, Any]] = {}
_cloud_supervisor_states_loaded = False
_cloud_supervisor_runtime_lock = threading.Lock()
_ROBOCLAW_RLINF_EXT_MODULES = {
    "roboclaw.training.rlinf_registry_hook",
    "roboclaw_vla.rl.registry",
}
_log = logging.getLogger(__name__)


def clear_cloud_supervisor_runtime_for_tests() -> None:
    global _cloud_supervisor_states_loaded
    with _cloud_supervisor_runtime_lock:
        for task in _cloud_supervisor_tasks.values():
            if not task.done():
                task.cancel()
        _cloud_supervisor_tasks.clear()
        _cloud_supervisor_states.clear()
        _cloud_supervisor_states_loaded = True
        if os.environ.get("EVO_STUDIO_CLOUD_SUPERVISOR_FILE"):
            try:
                _cloud_supervisor_runtime_path().unlink()
            except FileNotFoundError:
                pass

def _cloud_supervisor_runtime_path():
    path = _cloud_supervisor_snapshot_path()
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_runtime{suffix}")

def _load_cloud_supervisor_states_unlocked() -> None:
    global _cloud_supervisor_states_loaded
    if _cloud_supervisor_states_loaded:
        return
    _cloud_supervisor_states_loaded = True
    path = _cloud_supervisor_runtime_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except json.JSONDecodeError as exc:
        _log.warning("Cloud supervisor runtime state file corrupted, skipping: %s", exc)
        return
    except OSError as exc:
        _log.warning("Could not load cloud supervisor runtime state: %s", exc)
        return
    states = payload.get("states") if isinstance(payload, dict) else {}
    if not isinstance(states, dict):
        return
    for key, state in states.items():
        if isinstance(key, str) and isinstance(state, dict):
            _cloud_supervisor_states[key] = state

def _save_cloud_supervisor_states_unlocked() -> None:
    path = _cloud_supervisor_runtime_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(_cloud_supervisor_states) >= 200:
        for key in list(_cloud_supervisor_states.keys())[:50]:
            _cloud_supervisor_states.pop(key, None)
    payload = {
        "kind": "evo_studio_cloud_supervisor_runtime_store/v1",
        "updatedAt": _now_iso(),
        "states": _cloud_supervisor_states,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _cloud_supervisor_task_key(username: str, job_id: str) -> str:
    return _snapshot_key(username, job_id)
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
def _supervisor_state_aliases(username: str, payload: dict[str, Any]) -> list[str]:
    aliases = _snapshot_aliases(username, payload)
    supervisor = payload.get("supervisor")
    runtime = supervisor.get("runtime") if isinstance(supervisor, dict) else {}
    if isinstance(runtime, dict):
        for key in ("rootJobId", "currentJobId", "repairOfJobId"):
            value = str(runtime.get(key) or "").strip()
            if value:
                aliases.append(_snapshot_key(username, value))
    return aliases
def _set_cloud_supervisor_state(username: str, root_job_id: str, state: dict[str, Any]) -> None:
    if not username.strip() or not root_job_id.strip():
        return
    state = {
        "kind": "evo_studio_cloud_supervisor_runtime/v1",
        "updatedAt": _now_iso(),
        **state,
    }
    with _cloud_supervisor_runtime_lock:
        _load_cloud_supervisor_states_unlocked()
        _cloud_supervisor_states[_cloud_supervisor_task_key(username, root_job_id)] = state
        for value in (
            str(state.get("currentJobId") or "").strip(),
            str(state.get("repairOfJobId") or "").strip(),
        ):
            if value:
                _cloud_supervisor_states[_cloud_supervisor_task_key(username, value)] = state
        _save_cloud_supervisor_states_unlocked()
def _cloud_supervisor_runtime_state(username: str, payload: dict[str, Any]) -> dict[str, Any]:
    with _cloud_supervisor_runtime_lock:
        _load_cloud_supervisor_states_unlocked()
        for key in _supervisor_state_aliases(username, payload):
            state = _cloud_supervisor_states.get(key)
            if state:
                return dict(state)
    return {}
def _cloud_failure_signal(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").strip().lower()
    remediation = payload.get("failureRemediation") or payload.get("failure_remediation")
    log_text = "\n".join(
        str(payload.get(key) or "")
        for key in ("message", "log_tail", "logTail", "error")
    ).lower()
    return bool(
        status in {"failed", "missing"}
        or (status == "stopped" and _log_tail_has_stage_failure(log_text))
        or payload.get("error")
        or (isinstance(remediation, dict) and remediation.get("code"))
    )
def _log_tail_has_stage_failure(log_text: str) -> bool:
    # Use explicit sentinel tokens or unambiguous Python error patterns only.
    # Avoid broad tokens like "error:" or "terminated" that appear in normal
    # pip/gymnasium output and cause false-positive repair loops.
    exact_tokens = (
        "__evo_stage_failed__",
        "__evo_missing_artifacts__",
        "without required metric artifacts",
        "no matching distribution",
        "requires a different python",
        "importerror:",
        "modulenotfounderror",
        "no valid libero package",
        "no module named 'prismatic'",
        "__evo_libero_runtime_unavailable__",
        "__evo_openvla_oft_runtime_unavailable__",
        "messagefactory' object has no attribute 'getprototype'",
        'messagefactory" object has no attribute "getprototype"',
        "compiled using numpy 1.x cannot be run in",
        "syntaxerror",
    )
    if any(token in log_text for token in exact_tokens):
        return True
    if re.search(r"\bcd:\s+[^\n]+:\s+(?:no such file or directory|没有那个文件或目录)", log_text, flags=re.IGNORECASE):
        return True
    # "killed" and "traceback" are meaningful only outside pip install output.
    # Check they don't appear inside a pip/requirements block.
    pip_context = "requirement already satisfied" in log_text or "successfully installed" in log_text
    if not pip_context:
        if "killed" in log_text or "traceback" in log_text:
            return True
    return False
def _stage_failure_name(log_text: str) -> str:
    match = re.search(r"__EVO_STAGE_FAILED__=([A-Za-z0-9_.-]+)", log_text)
    return match.group(1) if match else ""


def _ray_node_memory_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return bool(
        (
            "ray killed" in lowered
            and "worker" in lowered
            and ("node running low on memory" in lowered or "memory usage threshold" in lowered)
        )
        or "worker(s) were killed due to the node running low on memory" in lowered
        or (
            "memory usage threshold" in lowered
            and "top 10 memory users" in lowered
            and ("ray logs" in lowered or "raylet.out" in lowered)
        )
    )


def _ray_gcs_unavailable(log_text: str) -> bool:
    lowered = log_text.lower()
    return bool(
        "failed to get cluster id from gcs server" in lowered
        or "timed out while waiting for gcs to become available" in lowered
        or "failed to connect to gcs at address" in lowered
    )


def _ray_runtime_reset_command() -> str:
    return (
        "python - <<'PY'\n"
        "import shutil\n"
        "import subprocess\n"
        "\n"
        "subprocess.run(['ray', 'stop', '--force'], check=False)\n"
        "shutil.rmtree('/tmp/ray', ignore_errors=True)\n"
        "print('__EVO_RAY_RUNTIME_RESET__=ok')\n"
        "PY"
    )


def _infer_cloud_failure_remediation(payload: dict[str, Any]) -> dict[str, Any]:
    existing = payload.get("failureRemediation") or payload.get("failure_remediation")
    log_text = "\n".join(
        str(payload.get(key) or "")
        for key in ("message", "log_tail", "logTail", "error")
    )
    lowered = log_text.lower()
    existing_code = str(existing.get("code") if isinstance(existing, dict) else "").strip()
    existing_auto_repair = existing.get("autoRepair") if isinstance(existing, dict) else {}
    existing_safe = bool(isinstance(existing_auto_repair, dict) and existing_auto_repair.get("safe") is True)
    if existing_code and existing_code not in {"UNKNOWN_CLOUD_FAILURE", "CLOUD_STAGE_FAILED"}:
        if existing_code in {"PYTHON_MODULE_MISSING", "PYTHON_IMPORT_MISSING"} and not existing_safe:
            pass
        else:
            return existing
    if not _log_tail_has_stage_failure(lowered):
        return existing if isinstance(existing, dict) and existing_code else {}
    stage = _stage_failure_name(log_text)
    if _ray_node_memory_oom(log_text):
        code = "TRAINING_NODE_MEMORY_OOM"
        summary = "Ray killed rollout workers because node memory usage exceeded the safety threshold."
        strategy = "reduce_rollout_parallelism_and_resume"
    elif re.search(r"\bcd:\s+[^\n]+:\s+(?:no such file or directory|没有那个文件或目录)", log_text, flags=re.IGNORECASE):
        code = "CLOUD_WORKDIR_MISSING"
        summary = "Cloud job tried to enter a work directory that does not exist on the current runtime."
        strategy = "rerun_prepare_code_on_same_runtime"
    elif "terminated" in lowered or "killed" in lowered:
        code = "CLOUD_STAGE_TERMINATED"
        summary = f"Cloud job was terminated during {stage or 'a setup'} stage."
        strategy = "resume_same_runtime_after_termination"
    elif (
        "__evo_missing_artifact__" in lowered
        or "__evo_missing_artifacts__" in lowered
        or "without required metric artifacts" in lowered
    ):
        code = "CLOUD_METRICS_MISSING"
        summary = f"Cloud job finished without required metrics during {stage or 'artifact collection'}."
        strategy = "inspect_outputs_and_retry_metric_collection_or_eval"
    elif "incorrect path_or_model_id" in lowered and "/path/to/model" in lowered:
        code = "MODEL_PATH_PLACEHOLDER"
        summary = "The RLinf config still contains a placeholder model path."
        strategy = "inject_resolved_model_source_and_retry"
    elif (
        "__evo_libero_assets_missing__" in lowered
        or "__evo_libero_assets_unavailable__" in lowered
        or "libero_living_room_tabletop_base_style.xml" in lowered
        or ("/libero/assets/" in lowered and "filenotfounderror" in lowered)
    ):
        code = "LIBERO_ASSETS_MISSING"
        summary = "LIBERO simulation assets are missing from the runtime."
        strategy = "repair_libero_assets_and_retry"
    elif "egl_not_initialized" in lowered or "eglerror" in lowered or "eglmakecurrent" in lowered:
        code = "LIBERO_EGL_CONTEXT_FAILED"
        summary = "LIBERO/robosuite EGL rendering context failed during simulation."
        strategy = "configure_headless_egl_and_retry"
    elif "__evo_gpu_unavailable__" in lowered or "nvidia-smi" in lowered and "/dev/nvidia" in lowered:
        code = "CLOUD_GPU_UNAVAILABLE"
        # Distinguish: PyTorch/CUDA version mismatch (auto-fixable) vs truly no GPU.
        cuda_version_mismatch = (
            "too old" in lowered
            or "cuda driver" in lowered
            or "compiled with your version" in lowered
            or ("found version" in lowered and "update your gpu driver" in lowered)
        )
        if cuda_version_mismatch:
            summary = "PyTorch CUDA version does not match the instance driver; will reinstall matching PyTorch."
            strategy = "reinstall_pytorch_for_driver_cuda_version_and_retry"
            return {
                "code": code,
                "summary": summary,
                "stage": stage,
                "autoRepair": {
                    "safe": True,
                    "strategy": strategy,
                    "changes": [
                        "reinstall PyTorch with the CUDA version matching the instance driver",
                        "reuse existing source/model/data caches",
                        "retry training on the same instance after PyTorch reinstall",
                    ],
                },
                "requiresUserConfirmationBeforeStart": False,
            }
        summary = "Cloud runtime has no visible CUDA GPU for this task."
        strategy = "rebind_gpu_instance_or_run_prepare_only"
        return {
            "code": code,
            "summary": summary,
            "stage": stage,
            "autoRepair": {
                "safe": False,
                "strategy": strategy,
                "changes": [
                    "do not retry GPU training on the same no-GPU runtime",
                    "bind a cloud instance with visible CUDA before launching evaluation/training",
                    "prepare/cache-only jobs may continue without GPU when explicitly requested",
                ],
            },
            "requiresUserConfirmationBeforeStart": True,
        }
    elif "requires a different python" in lowered:
        code = "PYTHON_VERSION_INCOMPATIBLE"
        summary = f"Python version is incompatible during {stage or 'setup'}."
        strategy = "switch_to_python_3_11_runtime_and_retry"
    elif "no matching distribution" in lowered:
        code = "PYTHON_DEPENDENCY_RESOLUTION_FAILED"
        summary = f"Python dependency resolution failed during {stage or 'setup'}."
        strategy = "patch_dependency_constraints_and_retry"
    elif "messagefactory" in lowered and "getprototype" in lowered:
        code = "PYTHON_DEPENDENCY_RESOLUTION_FAILED"
        summary = "Python package dependency versions are incompatible."
        strategy = "patch_dependency_constraints_and_retry"
    elif "compiled using numpy 1.x" in lowered and "numpy 2" in lowered:
        code = "PYTHON_DEPENDENCY_RESOLUTION_FAILED"
        summary = "Python package dependency versions are incompatible."
        strategy = "patch_dependency_constraints_and_retry"
    elif "modulenotfounderror" in lowered or "importerror:" in lowered or "no valid libero package" in lowered:
        code = "PYTHON_IMPORT_MISSING"
        summary = f"Missing Python dependency during {stage or 'runtime'}."
        strategy = "install_missing_dependency_and_retry"
    else:
        code = "CLOUD_STAGE_FAILED"
        summary = f"Cloud job failed during {stage or 'an execution'} stage."
        strategy = "inspect_logs_and_retry_same_runtime"
    return {
        "code": code,
        "summary": summary,
        "stage": stage,
        "autoRepair": {
            "safe": True,
            "strategy": strategy,
            "changes": [
                "reuse the same configured cloud runtime",
                "reuse existing source/model/data caches",
                "retry without changing secrets, provider account, or budget",
            ],
        },
        "requiresUserConfirmationBeforeStart": False,
    }
def _normalize_cloud_failure_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    remediation = _infer_cloud_failure_remediation(result)
    if remediation:
        result["failureRemediation"] = remediation
        if not result.get("error"):
            result["error"] = remediation.get("summary") or remediation.get("code") or ""
        status = str(result.get("status") or "").strip().lower()
        log_text = "\n".join(
            str(result.get(key) or "")
            for key in ("message", "log_tail", "logTail", "error")
        ).lower()
        if status in {"running", "submitting", "submitted", "pending", "queued", "starting", "repairing"} and _log_tail_has_stage_failure(log_text):
            result["status"] = "Failed"
            result["running"] = False
            message = str(result.get("message") or "")
            if "status: Running" in message:
                result["message"] = message.replace("status: Running", "status: Failed").replace("running: True", "running: False")
        if status in {"stopped", "succeeded", "success", "completed", "complete"}:
            result["status"] = "Failed"
            message = str(result.get("message") or "")
            if "status: Stopped" in message:
                result["message"] = message.replace("status: Stopped", "status: Failed")
            elif "status: Succeeded" in message:
                result["message"] = message.replace("status: Succeeded", "status: Failed")
            elif "status: Completed" in message:
                result["message"] = message.replace("status: Completed", "status: Failed")
    return result
def _auto_repair_policy(automation_policy: dict[str, Any]) -> dict[str, Any]:
    raw_mode = str(automation_policy.get("mode") or "").strip()
    mode = raw_mode if raw_mode in {"ask", "safe_auto", "full_auto"} else "safe_auto"
    auto_retry = bool(automation_policy.get("autoRetrySameRuntime", mode in {"safe_auto", "full_auto"}))
    return {
        "mode": mode,
        "autoRetrySameRuntime": auto_retry,
        "allowAgentRepairSameRuntime": bool(automation_policy.get("allowAgentRepairSameRuntime", mode == "full_auto")),
        "paidStartRequiresConfirmation": bool(automation_policy.get("paidStartRequiresConfirmation", mode != "full_auto")),
    }
def _truthy_text(value: Any) -> str:
    return str(value or "").strip()
def _lower_text(value: Any) -> str:
    return _truthy_text(value).lower()
def _is_unresolved_value(value: Any) -> bool:
    return _lower_text(value) in {"", "auto", "ai_resolved", "builtin_policy", "unknown", "none", "null"}
def _training_config_name(params: dict[str, Any]) -> str:
    return _truthy_text(params.get("configName") or params.get("rlinfConfigName") or params.get("builtinTrainingProfile"))
def _looks_like_rlinf_training(params: dict[str, Any]) -> bool:
    backend = _lower_text(params.get("backendKind"))
    workflow = _lower_text(params.get("workflow"))
    repo_url = _lower_text(params.get("repoUrl"))
    workdir = _lower_text(params.get("workdir"))
    config_name = _training_config_name(params)
    return (
        backend == "rlinf"
        or workflow == "rlinf_vla"
        or "github.com/rlinf/rlinf" in repo_url
        or workdir.endswith("/rlinf")
        or bool(config_name and re.search(r"\b(libero|maniskill|metaworld|isaaclab|robotwin|realworld|d4rl|calvin)_", config_name))
    )
def _model_source_uri(model_source: dict[str, Any]) -> str:
    return _truthy_text(
        model_source.get("uri")
        or model_source.get("checkpoint")
        or model_source.get("repo")
        or model_source.get("repoId")
        or model_source.get("path")
    )
def _checkpoint_needs_model_source(checkpoint_path: Any, artifact_path: Any) -> bool:
    checkpoint = _truthy_text(checkpoint_path)
    artifact = _truthy_text(artifact_path)
    if _is_unresolved_value(checkpoint):
        return True
    if artifact and checkpoint == artifact:
        return True
    return bool("/outputs" in checkpoint and not checkpoint.startswith(("hf://", "http://", "https://", "s3://", "oss://", "r2://", "/root/autodl-tmp/evo_studio/cache/models/")))
def _sync_model_contract(params: dict[str, Any]) -> dict[str, Any]:
    hardened = dict(params)
    model_source = hardened.get("modelSource") if isinstance(hardened.get("modelSource"), dict) else {}
    if not model_source:
        return hardened
    model_uri = _model_source_uri(model_source)
    source_type = _truthy_text(model_source.get("sourceType"))
    if source_type:
        hardened["modelSourceKind"] = source_type
    model_format = _truthy_text(model_source.get("format"))
    if model_format and _is_unresolved_value(hardened.get("checkpointFormat")):
        hardened["checkpointFormat"] = model_format
    if model_uri and not _is_unresolved_value(model_uri) and _checkpoint_needs_model_source(
        hardened.get("checkpointPath"),
        hardened.get("artifactPath"),
    ):
        hardened["checkpointPath"] = model_uri
        if model_format:
            hardened["checkpointFormat"] = model_format
        hardened["resolvedModelSource"] = {}
    source_contract = hardened.get("sourceContract") if isinstance(hardened.get("sourceContract"), dict) else {}
    if source_contract:
        source_contract = dict(source_contract)
        source_contract["modelSource"] = dict(model_source)
        if hardened.get("modelSourceKind"):
            source_contract["modelSourceKind"] = hardened["modelSourceKind"]
        if hardened.get("checkpointFormat"):
            source_contract["checkpointFormat"] = hardened["checkpointFormat"]
        hardened["sourceContract"] = source_contract
    training_contract = hardened.get("trainingContract") if isinstance(hardened.get("trainingContract"), dict) else {}
    sources = training_contract.get("sources") if isinstance(training_contract.get("sources"), dict) else {}
    if training_contract and sources:
        training_contract = dict(training_contract)
        sources = dict(sources)
        sources["model"] = dict(model_source)
        training_contract["sources"] = sources
        hardened["trainingContract"] = training_contract
    return hardened
def _repair_harden_known_training_params(params: dict[str, Any]) -> dict[str, Any]:
    """Harden a training contract without baking in one benchmark or model.

    The repair supervisor can receive stale AI output or an older start payload.
    This pass re-materializes framework contracts from capability catalogs and
    synchronizes model/checkpoint fields before a retry is submitted.
    """

    hardened = dict(params)
    config_name = _training_config_name(hardened)
    if _looks_like_rlinf_training(hardened):
        hardened.setdefault("backendKind", "rlinf")
        hardened.setdefault("workflow", "rlinf_vla")
        if config_name:
            context = " ".join(
                _truthy_text(value)
                for value in (
                    config_name,
                    hardened.get("benchmark"),
                    hardened.get("suite"),
                    hardened.get("trainingMode"),
                    hardened.get("policyFamily"),
                    hardened.get("modelFamily"),
                    (hardened.get("datasetSource") or {}).get("uri") if isinstance(hardened.get("datasetSource"), dict) else "",
                    (hardened.get("modelSource") or {}).get("uri") if isinstance(hardened.get("modelSource"), dict) else "",
                )
                if _truthy_text(value)
            )
            hardened = apply_rlinf_config_contract(hardened, config_name=config_name, message=context)
        repo_url = str(hardened.get("repoUrl") or "").strip().lower()
        workdir = str(hardened.get("workdir") or "").strip().lower()
        uses_official_rlinf_repo = "github.com/rlinf/rlinf" in repo_url or workdir.endswith("/rlinf")
        if uses_official_rlinf_repo:
            for field in ("backendExtModule", "rlinfExtModule"):
                if str(hardened.get(field) or "").strip() in _ROBOCLAW_RLINF_EXT_MODULES:
                    hardened[field] = ""
    return _sync_model_contract(hardened)
def _failure_log_context(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("message", "log_tail", "logTail", "error"):
        value = str(payload.get(key) or "").strip()
        if value:
            parts.append(value)
    context = payload.get("failureContext") or payload.get("failure_context")
    if isinstance(context, dict):
        for key in ("message", "logTail", "log_tail", "error", "userGuidance"):
            value = str(context.get(key) or "").strip()
            if value:
                parts.append(value)
    return "\n".join(parts)


def _trim_failure_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _failure_context_payload(payload: dict[str, Any], *, user_guidance: str = "") -> dict[str, str]:
    return {
        "status": _trim_failure_text(payload.get("status"), limit=120),
        "error": _trim_failure_text(payload.get("error"), limit=1200),
        "logTail": _trim_failure_text(payload.get("log_tail") or payload.get("logTail"), limit=1800),
        "message": _trim_failure_text(payload.get("message"), limit=1200),
        "userGuidance": _trim_failure_text(user_guidance, limit=800),
    }


def _repair_bootstrap_commands_for_failure(
    payload: dict[str, Any],
    remediation: dict[str, Any],
) -> list[str]:
    return _repair_bootstrap_commands_for_failure_text(_failure_log_context(payload), remediation)
def _is_prepare_only_params(params: dict[str, Any]) -> bool:
    phase = str(params.get("executionPhase") or params.get("runPhase") or "").strip().lower()
    prepare_only = params.get("prepareOnly")
    if isinstance(prepare_only, str):
        prepare_only_enabled = prepare_only.strip().lower() in {"1", "true", "yes", "on"}
    else:
        prepare_only_enabled = bool(prepare_only)
    return prepare_only_enabled or phase in {"prepare", "prepare_only", "prewarm", "cache", "cache_only"}
def _cloud_training_active(payload: dict[str, Any]) -> bool:
    if _cloud_failure_signal(payload):
        return False
    status = str(payload.get("status") or "").strip().lower()
    running = bool(payload.get("running"))
    if running:
        return True
    return status in {"running", "submitting", "submitted", "pending", "queued", "starting", "repairing"}


def _training_time_intervention(payload: dict[str, Any]) -> dict[str, Any]:
    log_text = _failure_log_context(payload).lower()
    if not log_text:
        return {}
    if _ray_node_memory_oom(log_text):
        return {
            "code": "TRAINING_NODE_MEMORY_OOM",
            "strategy": "reduce_rollout_parallelism_and_resume",
            "summary": "Ray killed rollout workers because node memory usage exceeded the safety threshold.",
        }
    if _ray_gcs_unavailable(log_text):
        return {
            "code": "RAY_GCS_UNAVAILABLE",
            "strategy": "restart_ray_runtime_and_resume",
            "summary": "Ray GCS did not become available during training startup.",
        }
    if "cuda out of memory" in log_text or "outofmemoryerror" in log_text:
        return {
            "code": "TRAINING_OOM",
            "strategy": "halve_batch_size_and_resume",
            "summary": "CUDA out of memory detected during training.",
        }
    step_match = re.search(r"(?:step|global_step)[=:\s]+(\d+)", log_text)
    loss_match = re.search(r"\bloss[=:\s]+([0-9]+(?:\.[0-9]+)?)", log_text)
    if step_match and loss_match and int(step_match.group(1)) > 1000 and float(loss_match.group(1)) > 100:
        return {
            "code": "LOSS_EXPLOSION",
            "strategy": "reduce_learning_rate_and_resume",
            "summary": "Loss explosion detected after warmup.",
        }
    if "nan" in log_text and "grad" in log_text or "inf" in log_text and "grad" in log_text:
        return {
            "code": "NAN_GRADIENTS",
            "strategy": "enable_gradient_clipping_and_resume",
            "summary": "NaN/Inf gradients detected during training.",
        }
    return {}


def _halve_int(value: Any, *, default: int = 1) -> int:
    try:
        return max(1, int(value) // 2)
    except (TypeError, ValueError):
        return default


def _reduce_float(value: Any, *, default: float = 1e-5) -> float:
    try:
        return max(float(value) / 10.0, 1e-8)
    except (TypeError, ValueError):
        return default


def _apply_training_intervention_params(params: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
    patched = dict(params)
    strategy = str(intervention.get("strategy") or "")
    training_contract = dict(patched.get("trainingContract")) if isinstance(patched.get("trainingContract"), dict) else {}
    runner = dict(training_contract.get("runner")) if isinstance(training_contract.get("runner"), dict) else {}
    algorithm = dict(training_contract.get("algorithm")) if isinstance(training_contract.get("algorithm"), dict) else {}
    overrides = list(patched.get("overrides")) if isinstance(patched.get("overrides"), list) else []

    if strategy == "halve_batch_size_and_resume":
        if "batchSize" in patched:
            patched["batchSize"] = _halve_int(patched.get("batchSize"))
        elif "batch_size" in patched:
            patched["batch_size"] = _halve_int(patched.get("batch_size"))
        elif "batch_size" in runner:
            runner["batch_size"] = _halve_int(runner.get("batch_size"))
        else:
            patched["batchSize"] = 1
    elif strategy == "reduce_learning_rate_and_resume":
        if "learningRate" in patched:
            patched["learningRate"] = _reduce_float(patched.get("learningRate"))
        elif "learning_rate" in patched:
            patched["learning_rate"] = _reduce_float(patched.get("learning_rate"))
        elif "learning_rate" in algorithm:
            algorithm["learning_rate"] = _reduce_float(algorithm.get("learning_rate"))
        else:
            patched["learningRate"] = 1e-5
    elif strategy == "enable_gradient_clipping_and_resume":
        patched["gradientClippingMaxNorm"] = 1.0
        algorithm["gradient_clip_max_norm"] = 1.0
    elif strategy == "reduce_rollout_parallelism_and_resume":
        for key in ("numRolloutWorkers", "rolloutWorkers", "numWorkers", "num_workers", "num_envs", "parallelism"):
            if key in patched:
                patched[key] = _halve_int(patched.get(key))
        for key in ("num_workers", "num_envs", "parallelism", "rollout_workers"):
            if key in runner:
                runner[key] = _halve_int(runner.get(key))
        patched.setdefault("numRolloutWorkers", 1)
        runner.setdefault("num_workers", 1)
        runner.setdefault("num_envs", 1)
        for override in (
            "runner.num_workers=1",
            "runner.num_envs=1",
            "rollout.num_workers=1",
            "rollout.num_envs=1",
        ):
            if override not in overrides:
                overrides.append(override)
        env = dict(patched.get("env")) if isinstance(patched.get("env"), dict) else {}
        env.setdefault("RAY_memory_usage_threshold", "0.98")
        patched["env"] = env
    elif strategy == "restart_ray_runtime_and_resume":
        for key in ("numRolloutWorkers", "rolloutWorkers", "numWorkers", "num_workers", "num_envs", "parallelism"):
            if key in patched:
                patched[key] = _halve_int(patched.get(key))
        for key in ("num_workers", "num_envs", "parallelism", "rollout_workers"):
            if key in runner:
                runner[key] = _halve_int(runner.get(key))
        patched.setdefault("numRolloutWorkers", 1)
        runner.setdefault("num_workers", 1)
        runner.setdefault("num_envs", 1)
        for override in (
            "runner.num_workers=1",
            "runner.num_envs=1",
            "rollout.num_workers=1",
            "rollout.num_envs=1",
        ):
            if override not in overrides:
                overrides.append(override)
        env = dict(patched.get("env")) if isinstance(patched.get("env"), dict) else {}
        env.setdefault("RAY_DEDUP_LOGS", "0")
        env.setdefault("RAY_memory_usage_threshold", "0.98")
        patched["env"] = env
        repair_commands = (
            [str(item).strip() for item in patched.get("repairBootstrapCommands", []) if str(item).strip()]
            if isinstance(patched.get("repairBootstrapCommands"), list)
            else []
        )
        reset_command = _ray_runtime_reset_command()
        if reset_command not in repair_commands:
            repair_commands.insert(0, reset_command)
        patched["repairBootstrapCommands"] = repair_commands
        patched["forceRepairBootstrap"] = True
        patched["forceSkipStageCache"] = True

    if runner:
        training_contract["runner"] = runner
    if algorithm:
        training_contract["algorithm"] = algorithm
    if training_contract:
        patched["trainingContract"] = training_contract
    if overrides:
        patched["overrides"] = overrides
    patched["resumeFromCheckpoint"] = True
    patched["repairStrategy"] = strategy
    patched["failureRemediation"] = {
        "code": intervention.get("code") or "TRAINING_INTERVENTION",
        "summary": intervention.get("summary") or "",
        "autoRepair": {
            "safe": True,
            "strategy": strategy,
            "changes": ["resume from last checkpoint", "adjust training hyperparameters in the same runtime"],
        },
    }
    return patched


def _training_intervention_start_request(
    payload: dict[str, Any],
    username: str,
    automation_policy: dict[str, Any],
    intervention: dict[str, Any],
    *,
    deployment_mode: str = "",
) -> dict[str, Any] | None:
    policy = _auto_repair_policy(automation_policy)
    if not (policy["autoRetrySameRuntime"] or policy["allowAgentRepairSameRuntime"] or policy["mode"] == "full_auto"):
        return None
    snapshot = _lookup_cloud_start(username, payload)
    if not snapshot:
        return None
    start_payload = dict(snapshot.get("payload") or {})
    params = dict(start_payload.get("params") or {})
    params = _apply_training_intervention_params(params, intervention)
    params = _repair_harden_known_training_params(params)
    base_task_name = str(start_payload.get("task_name") or start_payload.get("taskName") or "cloud-intervention")
    base_task_name = base_task_name.rsplit("-repair-", 1)[0].rsplit("-intervention-", 1)[0]
    start_payload.update({
        "username": username,
        "params": params,
        "task_name": f"{base_task_name}-intervention-{datetime.now(timezone.utc).strftime('%H%M%S')}",
        "waitForSubmit": True,
    })
    if deployment_mode == "ssh":
        start_payload["hourly_cost_cents"] = 0
        start_payload["hourlyCostCents"] = 0
    return start_payload


def _cloud_supervisor_payload(
    payload: dict[str, Any],
    username: str,
    training: TrainingService,
    *,
    automation_policy: dict[str, Any] | None = None,
    deployment_mode: str = "",
) -> dict[str, Any]:
    policy = _auto_repair_policy(automation_policy or {})
    snapshot = _lookup_cloud_start(username, payload)
    if not deployment_mode:
        bridge = training.cloud_bridge_status()
        deployment_mode = str(bridge.get("deploymentMode") or "").lower()
    else:
        deployment_mode = deployment_mode.lower()
    remediation = payload.get("failureRemediation") or payload.get("failure_remediation") or {}
    auto_repair = remediation.get("autoRepair") if isinstance(remediation, dict) else {}
    safe_repair = bool(isinstance(auto_repair, dict) and auto_repair.get("safe") is True)
    failed = _cloud_failure_signal(payload)
    active = _cloud_training_active(payload)
    same_runtime_available = deployment_mode == "ssh"
    has_snapshot = snapshot is not None
    can_retry_same_runtime = bool(
        failed
        and has_snapshot
        and same_runtime_available
        and (safe_repair or policy["allowAgentRepairSameRuntime"] or policy["mode"] == "full_auto")
    )
    if not failed and not active:
        state = "idle"
        next_action = "wait"
    elif not failed:
        state = "watching"
        next_action = "wait"
    elif can_retry_same_runtime:
        state = "repairable_same_runtime"
        next_action = "auto_retry_same_runtime" if policy["autoRetrySameRuntime"] else "ask_user_to_retry"
    else:
        state = "needs_review"
        next_action = "ask_user"
    supervisor = {
        "kind": "evo_studio_job_supervisor/v1",
        "state": state,
        "nextAction": next_action,
        "hasStartSnapshot": has_snapshot,
        "sameRuntimeAvailable": same_runtime_available,
        "canRetrySameRuntime": can_retry_same_runtime,
        "failureCode": remediation.get("code") if isinstance(remediation, dict) else "",
        "repairStrategy": auto_repair.get("strategy") if isinstance(auto_repair, dict) else "",
        "requiresConfirmation": bool(failed and not (policy["autoRetrySameRuntime"] and can_retry_same_runtime)),
        "guardrails": {
            "noRuntimeChange": True,
            "noSecretChange": True,
            "noBudgetIncrease": True,
        },
    }
    runtime = _cloud_supervisor_runtime_state(username, payload)
    if runtime and (failed or active):
        runtime_state = str(runtime.get("state") or "")
        runtime_current_job_id = str(runtime.get("currentJobId") or "").strip()
        payload_job_id = str(payload.get("job_id") or payload.get("jobId") or "").strip()
        runtime_points_to_newer_job = bool(runtime_current_job_id and runtime_current_job_id != payload_job_id)
        stale_runtime_for_failed_job = bool(
            failed
            and runtime_state in {"watching", "repairing", "repair_submitted"}
            and runtime_current_job_id
            and runtime_current_job_id == payload_job_id
        )
        if not stale_runtime_for_failed_job:
            supervisor["runtime"] = runtime
        if runtime_state in {"watching", "repairing", "repair_submitted"} and (not failed or runtime_points_to_newer_job):
            supervisor["state"] = runtime_state
            supervisor["nextAction"] = "watch_status"
    return supervisor
async def _repair_start_request(
    payload: dict[str, Any],
    username: str,
    training: TrainingService,
    automation_policy: dict[str, Any],
    *,
    deployment_mode: str = "",
    user_guidance: str = "",
    llm_provider: Any | None = None,
) -> dict[str, Any] | None:
    supervisor = _cloud_supervisor_payload(
        payload,
        username,
        training,
        automation_policy=automation_policy,
        deployment_mode=deployment_mode,
    )
    if not supervisor.get("canRetrySameRuntime"):
        return None
    policy = _auto_repair_policy(automation_policy)
    if not policy["autoRetrySameRuntime"]:
        return None
    snapshot = _lookup_cloud_start(username, payload)
    if not snapshot:
        return None
    start_payload = dict(snapshot.get("payload") or {})
    params = dict(start_payload.get("params") or {})
    remediation = payload.get("failureRemediation") or payload.get("failure_remediation") or {}
    auto_repair = remediation.get("autoRepair") if isinstance(remediation, dict) else {}
    repair_strategy = str((auto_repair or {}).get("strategy") or (remediation or {}).get("code") or "same_runtime_repair")
    result_collection_repair = repair_strategy in {
        "inspect_outputs_and_retry_metric_collection_or_eval",
        "CLOUD_METRICS_MISSING",
    }
    if (
        int(start_payload.get("hourly_cost_cents") or start_payload.get("hourlyCostCents") or 0) > 0
        and deployment_mode != "ssh"
        and policy["mode"] != "full_auto"
    ):
        return None
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
        "forceSkipStageCache",
        "resumeFromStage",
    ):
        params.pop(stale_key, None)
    if isinstance(remediation, dict) and str(remediation.get("code") or "").strip().upper() in {
        "TRAINING_OOM",
        "TRAINING_NODE_MEMORY_OOM",
        "LOSS_EXPLOSION",
        "NAN_GRADIENTS",
    }:
        params = _apply_training_intervention_params(
            params,
            {
                "code": remediation.get("code"),
                "strategy": repair_strategy,
                "summary": remediation.get("summary") or "",
            },
        )
    params = await _inject_repair_commands_async(
        params,
        remediation,
        _failure_log_context(payload),
        llm_provider=llm_provider,
    )
    repair_strategy = str(params.get("repairStrategy") or repair_strategy)
    repair_bootstrap_commands = _repair_bootstrap_commands_for_failure(payload, remediation)
    if repair_bootstrap_commands:
        existing_repair_commands = (
            [str(item).strip() for item in params.get("repairBootstrapCommands", []) if str(item).strip()]
            if isinstance(params.get("repairBootstrapCommands"), list)
            else []
        )
        for command in repair_bootstrap_commands:
            if command not in existing_repair_commands:
                existing_repair_commands.append(command)
        params["repairBootstrapCommands"] = existing_repair_commands
        params["forceRepairBootstrap"] = True
        params["forceSkipStageCache"] = True
    if result_collection_repair:
        params["bootstrapProfile"] = ""
        params["bootstrapCommands"] = []
        params["healthcheckCommands"] = []
        params["setupCommand"] = "true"
        params["skipPrepareCode"] = True
        params["skipSetupEnv"] = True
        params["skipSourceResolve"] = True
        params["skipWriteContract"] = True
    params.update({
        "repairOfJobId": str(payload.get("job_id") or ""),
        "repairStrategy": repair_strategy,
        "forceRepairBootstrap": bool(params.get("repairBootstrapCommands")) and not result_collection_repair,
        "failureRemediation": remediation,
        "failureContext": _failure_context_payload(payload, user_guidance=user_guidance),
        "supervisor": {
            "kind": "evo_studio_job_supervisor/v1",
            "mode": policy["mode"],
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
    base_task_name = str(start_payload.get("task_name") or start_payload.get("taskName") or "cloud-repair")
    base_task_name = base_task_name.rsplit("-repair-", 1)[0]
    repair_task_name = f"{base_task_name}-repair-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    _record_repair_agent_event(
        {
            "event": "repair_start_request_built",
            "username": username,
            "repairOfJobId": str(payload.get("job_id") or ""),
            "repairTaskName": repair_task_name,
            "failureCode": remediation.get("code") if isinstance(remediation, dict) else "",
            "repairStrategy": repair_strategy,
            "hasRepairBootstrapCommands": bool(params.get("repairBootstrapCommands")),
            "sameRuntimeOnly": True,
            "noBudgetIncrease": True,
        }
    )
    start_payload.update({
        "username": username,
        "params": params,
        "task_name": repair_task_name,
        "waitForSubmit": True,
    })
    if deployment_mode == "ssh":
        start_payload["hourly_cost_cents"] = 0
        start_payload["hourlyCostCents"] = 0
    return start_payload
