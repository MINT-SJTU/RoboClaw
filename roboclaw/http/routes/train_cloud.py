"""Cloud training routes backed by EVO_Train."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from roboclaw.account import apply_service_fee_cents, estimate_training_hold_cents, hourly_cost_from_params
from roboclaw.data.auth_refs import delete_auth_connection, public_auth_connections, upsert_auth_connection, validate_training_auth_refs
from roboclaw.embodied.service import EmbodiedService
from roboclaw.training import TrainingPlanSpec, TrainingService, TrainingStartSpec, TrainingStopSpec

from . import cloud_snapshot as cloud_snapshot_state
from .cloud_autonomy import build_cloud_autonomy_state
from .cloud_artifacts import _attach_cloud_artifacts, _cloud_metrics_from_payload
from .cloud_billing import _attach_user_wallet, _release_failed_cloud_hold, get_ledger, set_ledger_for_tests
from .cloud_policy import _resolve_automation_policy
from .cloud_snapshot import _lookup_cloud_start, _remember_cloud_start
from .cloud_supervisor import (
    _cloud_failure_signal,
    _cloud_supervisor_payload,
    _cloud_training_active,
    _is_prepare_only_params,
    _normalize_cloud_failure_payload,
    _repair_harden_known_training_params,
    _repair_start_request,
    _set_cloud_supervisor_state,
    clear_cloud_supervisor_runtime_for_tests,
)
from .cloud_ssh import (
    _clear_ssh_runtime_env,
    _listening_on_local_port,
    _local_runtime_bind_enabled,
    _read_remote_text_file,
    _read_ssh_runtime_endpoint,
    _restart_local_evo_train_bridge,
    _ssh_runtime_env_path,
)
from .cloud_runtime_binding import bind_ssh_runtime, unbind_ssh_runtime
from .train_cloud_schema import (
    AuthConnectionSaveRequest,
    CloudResourceCatalogRequest,
    CloudSshRuntimeBindRequest,
    CloudTrainBillingSettleRequest,
    CloudTrainPlanRequest,
    CloudTrainStartRequest,
    CloudTrainStopRequest,
    CloudTrainSupervisorRepairRequest,
    CloudTrainSupervisorWatchRequest,
    ProviderBalanceRequest,
    RuntimeMatchRequest,
    SourcePreflightRequest,
)
from .train_cloud_helpers import (
    OFFICIAL_DATASET_SOURCES,
    _archived_current_payload,
    _apply_completed_supervisor_runtime,
    _bridge_error_status,
    _cloud_failure_code,
    _cloud_failure_fingerprint,
    _cloud_supervisor_max_repairs,
    _first_cloud_text,
    _infer_training_benchmark,
    _is_terminal_cloud_payload,
    _normalize_cloud_training_params,
    _request_username,
    _runtime_rebind_restart_request,
    _runtime_binding_failure_archivable,
    _runtime_binding_failure_message,
    _runtime_configuration_ready,
    _user_facing_runtime_warning,
    _validate_cloud_training_start,
    clear_cloud_supervisor_snapshots_for_tests,
)
from .cloud_watch import schedule_cloud_supervisor

_log = logging.getLogger(__name__)
_cloud_start_snapshots = cloud_snapshot_state._cloud_start_snapshots
_cloud_start_snapshots_lock = cloud_snapshot_state._cloud_start_snapshots_lock
_cloud_start_snapshots_loaded = cloud_snapshot_state._cloud_start_snapshots_loaded

_TERMINAL_CLOUD_STATUSES = {"failed", "missing", "stopped", "completed", "complete", "succeeded", "success"}
_RUNTIME_BINDING_FAILURE_CODES = {"CLOUD_GPU_UNAVAILABLE", "CLOUD_INSTANCE_UNREACHABLE"}
_STALE_SSH_BINDING_MARKERS = (
    "error reading ssh protocol banner",
    "ssh protocol layer",
    "configured ssh instance is not reachable",
    "kex_exchange_identification",
    "connection closed by remote host",
)

def _stale_ssh_binding_detected(warnings: list[str]) -> bool:
    warning_text = " ".join(warnings).lower()
    return any(marker in warning_text for marker in _STALE_SSH_BINDING_MARKERS)


def _ensure_local_evo_train_bridge_env() -> None:
    if not _local_runtime_bind_enabled():
        return
    os.environ.setdefault("ROBOCLAW_EVO_TRAIN_HOST", "127.0.0.1")
    os.environ.setdefault("ROBOCLAW_EVO_TRAIN_PORT", "9000")
    os.environ.setdefault("ROBOCLAW_EVO_TRAIN_PROVIDER", "autodl")
    os.environ.setdefault("ROBOCLAW_EVO_TRAIN_BILLING_MODE", "external")


def register_train_cloud_routes(app: FastAPI, service: EmbodiedService) -> None:
    _ensure_local_evo_train_bridge_env()
    training = TrainingService(service)

    def _attach_autonomy(payload: dict[str, Any]) -> dict[str, Any]:
        supervisor = payload.get("supervisor") if isinstance(payload.get("supervisor"), dict) else {}
        payload["autonomy"] = build_cloud_autonomy_state(payload, supervisor)
        return payload

    def _local_runtime_unbound() -> bool:
        if not _local_runtime_bind_enabled() or _read_ssh_runtime_endpoint().get("endpoint"):
            return False
        if not training.cloud_enabled:
            return True
        settings = getattr(getattr(training, "_bridge", None), "settings", None)
        host = str(getattr(settings, "host", "") or "").strip()
        port = int(getattr(settings, "port", 0) or 0)
        return bool(host in {"127.0.0.1", "localhost"} and port and not _listening_on_local_port(port))

    def _unbound_cloud_current_payload() -> dict[str, Any]:
        return {
            "job_id": "",
            "status": "idle",
            "running": False,
            "message": "还没有绑定云端实例。请粘贴当前实例最新 SSH 命令后连接。",
            "mode": "cloud",
            "pid": None,
            "log_path": "",
            "log_tail": "",
            "task_name": "",
            "checkpoint_path": "",
            "dataset_path": "",
            "provider": "autodl",
            "error": "",
            "failureRemediation": {},
        }

    async def _cloud_deployment_mode(provider: str = "") -> str:
        if _local_runtime_unbound():
            return "ssh"
        try:
            config = await training.configuration_check(provider=provider)
            return str(config.get("mode") or config.get("deploymentMode") or "").lower()
        except RuntimeError:
            status = training.cloud_bridge_status()
            return str(status.get("deploymentMode") or status.get("mode") or "").lower()

    async def _cloud_runtime_ready(provider: str = "") -> bool:
        if _local_runtime_unbound():
            return False
        try:
            config = await training.configuration_check(provider=provider)
            ready, _ = _runtime_configuration_ready(dict(config), require_gpu=True)
            return ready and str(config.get("mode") or config.get("deploymentMode") or "").lower() == "ssh"
        except RuntimeError:
            status = training.cloud_bridge_status()
            return bool(status.get("enabled") and status.get("configurationReady") is True)

    async def _runtime_binding_recovered(payload: dict[str, Any]) -> bool:
        return await _cloud_runtime_ready(provider=str(payload.get("provider") or ""))

    def _runtime_unavailable_supervisor(payload: dict[str, Any], supervisor: dict[str, Any]) -> dict[str, Any]:
        if not _cloud_failure_signal(payload):
            return supervisor
        patched = dict(supervisor)
        patched.update({
            "state": "needs_review",
            "nextAction": "rebind_runtime",
            "sameRuntimeAvailable": False,
            "canRetrySameRuntime": False,
            "requiresConfirmation": True,
            "runtimeUnavailable": True,
        })
        return patched

    def _watch_root_job_id(job_id: str, payload: dict[str, Any]) -> str:
        supervisor = payload.get("supervisor") if isinstance(payload.get("supervisor"), dict) else {}
        runtime = supervisor.get("runtime") if isinstance(supervisor.get("runtime"), dict) else {}
        root_job_id = str(runtime.get("rootJobId") or "").strip()
        if root_job_id:
            return root_job_id
        for marker in ("-intervention-", "-repair-", "-restart-"):
            if marker in job_id:
                return job_id.rsplit(marker, 1)[0]
        return job_id

    def _automation_policy_for_payload(username: str, payload: dict[str, Any]) -> dict[str, Any]:
        snapshot = _lookup_cloud_start(username, payload) if username.strip() else None
        start_payload = snapshot.get("payload") if isinstance(snapshot, dict) else {}
        if not isinstance(start_payload, dict):
            return _resolve_automation_policy(None)
        policy = start_payload.get("automationPolicy") or start_payload.get("automation_policy")
        return _resolve_automation_policy(policy if isinstance(policy, dict) else None)

    async def _cloud_supervisor_for_payload(
        payload: dict[str, Any],
        username: str,
        *,
        automation_policy: dict[str, Any] | None = None,
        deployment_mode: str = "",
    ) -> dict[str, Any]:
        supervisor = _cloud_supervisor_payload(
            payload,
            username,
            training,
            automation_policy=automation_policy,
            deployment_mode=deployment_mode,
        )
        if deployment_mode == "ssh" and not await _cloud_runtime_ready(provider=str(payload.get("provider") or "")):
            return _runtime_unavailable_supervisor(payload, supervisor)
        return supervisor

    async def _supervisor_runtime_job_active(runtime: dict[str, Any], username: str) -> bool:
        current_job_id = str(runtime.get("currentJobId") or "").strip()
        if not current_job_id:
            return False
        try:
            result = await training.status(job_id=current_job_id, username=username)
        except RuntimeError:
            return False
        payload = _normalize_cloud_failure_payload(result.to_dict())
        return _cloud_training_active(payload)

    def _schedule_cloud_supervisor(
        *,
        username: str,
        payload: dict[str, Any],
        automation_policy: dict[str, Any],
    ) -> None:
        async def _start_unscheduled(body: CloudTrainStartRequest) -> dict[str, Any]:
            return await _start_cloud_training(body, schedule_supervisor=False)

        async def _deployment_mode_for_payload(next_payload: dict[str, Any]) -> str:
            return await _cloud_deployment_mode(provider=str(next_payload.get("provider") or ""))

        schedule_cloud_supervisor(
            username=username,
            payload=payload,
            automation_policy=automation_policy,
            training=training,
            start_cloud_training=_start_unscheduled,
            deployment_mode_for_payload=_deployment_mode_for_payload,
            llm_provider=getattr(app.state, "llm_provider", None),
        )

    async def _start_cloud_training(body: CloudTrainStartRequest, *, schedule_supervisor: bool = True) -> dict[str, Any]:
        if not training.cloud_enabled:
            raise HTTPException(status_code=503, detail="EVO_Train bridge is not enabled.")
        username = body.username.strip()
        training_params = _normalize_cloud_training_params(body.params, dataset_name=body.dataset_name)
        training_params = _repair_harden_known_training_params(training_params)
        _validate_cloud_training_start(training_params, policy_type=body.policy_type)
        automation_policy = _resolve_automation_policy(body.automation_policy, body.automation_mode)
        start_snapshot = body.model_dump(by_alias=True)
        start_snapshot["params"] = training_params
        start_snapshot["automationPolicy"] = automation_policy
        auth_errors = validate_training_auth_refs(training_params, username=username)
        if auth_errors:
            raise HTTPException(status_code=400, detail={"code": "training_auth_ref_invalid", "errors": auth_errors})
        prepare_only = _is_prepare_only_params(training_params)
        try:
            runtime_config = dict(await training.configuration_check(provider=body.provider))
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        runtime_ready, runtime_error = _runtime_configuration_ready(runtime_config, require_gpu=not prepare_only)
        if not runtime_ready:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "cloud_runtime_not_ready",
                    "message": runtime_error or "云端运行环境还没有就绪，请先连接云端实例。",
                    "checks": runtime_config.get("checks", []),
                    "missing": runtime_config.get("missing", []),
                    "warnings": runtime_config.get("warnings", runtime_config.get("configurationWarnings", [])),
                },
            )
        hourly_cost_cents = 0 if prepare_only else (body.hourly_cost_cents or hourly_cost_from_params(training_params))
        hold_cents = 0
        freeze_record = None
        if username and hourly_cost_cents:
            try:
                hold_cents = estimate_training_hold_cents(
                    hourly_cost_cents=hourly_cost_cents,
                    service_fee_bps=body.service_fee_bps,
                )
                _wallet, freeze_record = get_ledger().freeze(
                    username,
                    hold_cents,
                    reason="cloud training first-hour hold",
                    task_name=body.task_name or body.dataset_name,
                    job_id=body.task_name or body.dataset_name or "pending-cloud-train",
                )
            except ValueError as exc:
                raise HTTPException(status_code=409 if "insufficient" in str(exc) else 400, detail=str(exc)) from exc
        try:
            result = await training.start(
                TrainingStartSpec(
                    dataset_name=body.dataset_name,
                    policy_type=body.policy_type,
                    steps=body.steps,
                    device=body.device,
                    username=body.username,
                    provider=body.provider,
                    workflow=body.workflow,
                    params=training_params,
                    sku_id=body.sku_id,
                    image_id=body.image_id,
                    task_name=body.task_name,
                    wait_for_submit=body.wait_for_submit,
                )
            )
        except RuntimeError as exc:
            if username and freeze_record is not None:
                try:
                    get_ledger().release_job_hold(
                        username,
                        freeze_record.job_id,
                        reason="release hold after cloud training start failure",
                        task_name=body.task_name or body.dataset_name,
                    )
                except ValueError:
                    pass
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        payload = result.to_dict()
        if body.wait_for_submit and payload.get("job_id"):
            try:
                verified = await training.status(job_id=str(payload["job_id"]), username=username)
                verified_payload = verified.to_dict()
                for key, value in verified_payload.items():
                    if key in {"status", "running"} or (value is not None and value != ""):
                        payload[key] = value
                if str(payload.get("status") or "").strip().lower() == "missing":
                    payload["running"] = False
                    if not payload.get("error"):
                        payload["error"] = "EVO_Train did not create a task record after the start request."
                    if not payload.get("message"):
                        payload["message"] = "云训练启动后没有确认到任务已创建，已按失败处理并交给总控。"
            except RuntimeError as exc:
                payload.update({
                    "status": "failed",
                    "running": False,
                    "error": str(exc),
                    "message": "云训练启动后没有确认到任务状态，已按失败处理并交给总控。",
                })
        payload = _normalize_cloud_failure_payload(payload)
        _remember_cloud_start(username, start_snapshot, payload)
        if username and freeze_record is not None:
            job_id = payload.get("job_id") or freeze_record.job_id
            if job_id != freeze_record.job_id:
                try:
                    freeze_record = get_ledger().reassign_job_hold(
                        username,
                        freeze_record.job_id,
                        str(job_id),
                    )
                except ValueError:
                    pass
            payload["billing"] = {
                "holdCents": hold_cents,
                "hourlyCostCents": hourly_cost_cents,
                "serviceFeeBps": body.service_fee_bps,
                "record": freeze_record.to_dict(),
            }
            payload = _release_failed_cloud_hold(
                payload,
                username,
                reason="release hold after terminal cloud training start result",
            )
        deployment_mode = await _cloud_deployment_mode(provider=body.provider)
        payload["supervisor"] = await _cloud_supervisor_for_payload(
            payload,
            username,
            automation_policy=automation_policy,
            deployment_mode=deployment_mode,
        )
        payload = _attach_autonomy(payload)
        payload = _attach_cloud_artifacts(payload)
        if schedule_supervisor:
            _schedule_cloud_supervisor(
                username=username,
                payload=payload,
                automation_policy=automation_policy,
            )
        payload = _attach_user_wallet(payload, username)
        return payload

    @app.get("/api/train/cloud/bridge")
    async def train_cloud_bridge() -> dict[str, Any]:
        status = dict(training.cloud_bridge_status())
        runtime_endpoint = _read_ssh_runtime_endpoint()
        if runtime_endpoint:
            status["runtimeEndpoint"] = runtime_endpoint.get("endpoint", "")
            status["runtimeHost"] = runtime_endpoint.get("host", "")
            status["runtimePort"] = runtime_endpoint.get("port", "")
            status["runtimeUser"] = runtime_endpoint.get("user", "")
        if _local_runtime_unbound():
            status.update(
                {
                    "enabled": True,
                    "provider": "autodl",
                    "managedBy": "Evo Studio local SSH binding",
                    "deploymentMode": "ssh",
                    "configurationReady": False,
                    "prepareReady": False,
                    "gpuReady": False,
                    "sshGpuReady": False,
                    "sshConnectionReady": False,
                    "sshGpu": "",
                    "configurationWarnings": ["还没有绑定云端实例。请粘贴当前实例最新 SSH 命令后连接。"],
                    "rawConfigurationWarnings": ["ssh runtime is not bound"],
                    "missingDeploymentFields": [],
                    "resourceCatalog": {
                        "skuCount": 0,
                        "readySkuCount": 0,
                        "imageCount": 0,
                        "readyImageCount": 0,
                    },
                    "message": "还没有绑定云端实例。请连接当前 AutoDL / SeetaCloud 实例后再启动任务。",
                    "operatorHint": "",
                }
            )
            return status
        if not training.cloud_enabled:
            if _local_runtime_bind_enabled():
                status.update(
                    {
                        "enabled": True,
                        "provider": "autodl",
                        "managedBy": "Evo Studio local SSH binding",
                        "deploymentMode": "ssh",
                        "configurationReady": False,
                        "prepareReady": False,
                        "gpuReady": False,
                        "sshGpuReady": False,
                        "sshConnectionReady": False,
                        "sshGpu": "",
                        "configurationWarnings": ["还没有绑定云端实例。请粘贴当前实例最新 SSH 命令后连接。"],
                        "rawConfigurationWarnings": ["ssh runtime is not bound"],
                        "missingDeploymentFields": [],
                        "resourceCatalog": {
                            "skuCount": 0,
                            "readySkuCount": 0,
                            "imageCount": 0,
                            "readyImageCount": 0,
                        },
                        "message": "还没有绑定云端实例。请连接当前 AutoDL / SeetaCloud 实例后再启动任务。",
                        "operatorHint": "",
                    }
                )
                return status
            return status
        try:
            config = await training.configuration_check(provider=str(status.get("provider") or ""))
        except RuntimeError as exc:
            status["enabled"] = False
            status["deploymentMode"] = ""
            status["configurationReady"] = False
            status["configurationWarnings"] = [str(exc)]
            status["resourceCatalog"] = {
                "skuCount": 0,
                "readySkuCount": 0,
                "imageCount": 0,
                "readyImageCount": 0,
            }
            status["message"] = "Cloud training bridge is configured but unreachable."
            status["operatorHint"] = "Restart EVO_Train and refresh this page before starting paid training."
            return status
        status["deploymentMode"] = str(config.get("mode") or "")
        runtime_endpoint = _read_ssh_runtime_endpoint()
        if runtime_endpoint:
            status["runtimeEndpoint"] = runtime_endpoint.get("endpoint", "")
            status["runtimeHost"] = runtime_endpoint.get("host", "")
            status["runtimePort"] = runtime_endpoint.get("port", "")
            status["runtimeUser"] = runtime_endpoint.get("user", "")
        raw_warnings = [str(item) for item in config.get("warnings", []) if str(item).strip()]
        if (
            runtime_endpoint.get("endpoint")
            and str(config.get("mode") or "").lower() == "ssh"
            and _stale_ssh_binding_detected(raw_warnings)
        ):
            previous_endpoint = _clear_ssh_runtime_env()
            restart_result = _restart_local_evo_train_bridge(_ssh_runtime_env_path())
            clear_cloud_supervisor_runtime_for_tests()
            return {
                **status,
                "enabled": True,
                "provider": "autodl",
                "managedBy": "Evo Studio local SSH binding",
                "deploymentMode": "ssh",
                "configurationReady": False,
                "prepareReady": False,
                "gpuReady": False,
                "sshGpuReady": False,
                "sshConnectionReady": False,
                "sshGpu": "",
                "runtimeEndpoint": "",
                "runtimeHost": "",
                "runtimePort": "",
                "runtimeUser": "",
                "autoUnboundRuntime": True,
                "previousRuntimeEndpoint": previous_endpoint.get("endpoint", ""),
                "configurationWarnings": [
                    "之前绑定的云端实例已失效，平台已自动解绑。请粘贴当前实例最新 SSH 命令后连接。"
                ],
                "rawConfigurationWarnings": raw_warnings,
                "resourceCatalog": {
                    "skuCount": 0,
                    "readySkuCount": 0,
                    "imageCount": 0,
                    "readyImageCount": 0,
                },
                "bridge": restart_result,
                "message": "旧云端实例绑定已自动清除。请连接当前 AutoDL / SeetaCloud 实例后再启动任务。",
                "operatorHint": "",
            }
        status["configurationReady"] = bool(config.get("ready", False))
        status["prepareReady"] = bool(config.get("prepareReady", config.get("ready", False)))
        status["gpuReady"] = bool(config.get("gpuReady", config.get("sshGpuReady", config.get("ready", False))))
        status["sshGpuReady"] = bool(config.get("sshGpuReady", False))
        status["sshConnectionReady"] = bool(config.get("sshConnectionReady", config.get("sshReady", False)))
        status["sshGpu"] = config.get("sshGpu", "")
        status["configurationWarnings"] = [_user_facing_runtime_warning(item) for item in raw_warnings]
        status["rawConfigurationWarnings"] = raw_warnings
        status["resourceCatalog"] = {
            "skuCount": config.get("skuCount", 0),
            "readySkuCount": config.get("readySkuCount", 0),
            "imageCount": config.get("imageCount", 0),
            "readyImageCount": config.get("readyImageCount", 0),
        }
        status["message"] = _cloud_bridge_message(status)
        return status

    @app.post("/api/train/cloud/dev/rebind-ssh")
    async def train_cloud_dev_rebind_ssh(body: CloudSshRuntimeBindRequest) -> dict[str, Any]:
        return await bind_ssh_runtime(body, training=training)

    @app.post("/api/train/cloud/dev/unbind-ssh")
    async def train_cloud_dev_unbind_ssh() -> dict[str, Any]:
        return await unbind_ssh_runtime()

    def _cloud_bridge_message(status: dict[str, Any]) -> str:
        if status.get("configurationReady") is False:
            return "云端实例未就绪，不能启动 GPU 任务。请重新绑定有卡实例，或选择无卡准备。"
        mode = str(status.get("deploymentMode") or "").lower()
        if mode == "managed":
            return "Cloud training bridge is connected to a managed compute pool."
        if mode == "ssh":
            if status.get("gpuReady") is False:
                return "云端 SSH 已连接，可先无卡准备；GPU 任务需绑定有卡实例。"
            return "云端 SSH 实例已连接。"
        return str(status.get("message") or "Cloud training bridge is connected.")

    @app.get("/api/train/cloud/auth-connections")
    async def train_cloud_auth_connections(
        kind: str = "",
        username: str = "",
        x_evo_studio_user: str = Header(default=""),
    ) -> dict[str, Any]:
        current_username = _request_username(username, x_evo_studio_user)
        connections = public_auth_connections(kind, username=current_username)
        return {
            "configured": bool(connections),
            "connections": connections,
        }

    @app.post("/api/train/cloud/auth-connections")
    async def train_cloud_save_auth_connection(
        body: AuthConnectionSaveRequest,
        x_evo_studio_user: str = Header(default=""),
    ) -> dict[str, Any]:
        username = _request_username(body.username, x_evo_studio_user)
        if not username:
            raise HTTPException(status_code=400, detail="username is required to save a private connection")
        payload = body.model_dump(by_alias=True)
        payload["sourcePrefixes"] = body.source_prefixes
        payload["visibility"] = "user"
        try:
            connection = upsert_auth_connection(payload, username=username)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"connection": connection.public_dict()}

    @app.delete("/api/train/cloud/auth-connections/{auth_ref}")
    async def train_cloud_delete_auth_connection(
        auth_ref: str,
        username: str = "",
        x_evo_studio_user: str = Header(default=""),
    ) -> dict[str, Any]:
        current_username = _request_username(username, x_evo_studio_user)
        if not current_username:
            raise HTTPException(status_code=400, detail="username is required to delete a private connection")
        deleted = delete_auth_connection(auth_ref, username=current_username)
        return {"deleted": deleted}

    @app.post("/api/train/cloud/start")
    async def train_cloud_start(body: CloudTrainStartRequest) -> dict[str, Any]:
        return await _start_cloud_training(body)

    @app.post("/api/train/cloud/billing/settle")
    async def train_cloud_billing_settle(body: CloudTrainBillingSettleRequest) -> dict[str, Any]:
        try:
            charge_cents = apply_service_fee_cents(
                body.provider_cost_cents,
                service_fee_bps=body.service_fee_bps,
            )
            wallet, settle_record, release_record = await asyncio.to_thread(
                get_ledger().settle_job,
                body.username,
                body.job_id,
                charge_cents,
                reason="cloud training final settlement",
                task_name=body.task_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409 if "exceeds" in str(exc) or "no frozen" in str(exc) else 400, detail=str(exc)) from exc
        return {
            "wallet": wallet.to_dict(),
            "chargeCents": charge_cents,
            "providerCostCents": body.provider_cost_cents,
            "serviceFeeBps": body.service_fee_bps,
            "settleRecord": settle_record.to_dict(),
            "releaseRecord": release_record.to_dict() if release_record else None,
        }

    @app.post("/api/train/cloud/stop")
    async def train_cloud_stop(body: CloudTrainStopRequest) -> dict[str, Any]:
        try:
            result = await training.stop(TrainingStopSpec(job_id=body.job_id, username=body.username))
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/api/train/cloud/current")
    async def train_cloud_current(username: str = "") -> dict[str, Any]:
        if _local_runtime_unbound():
            payload = _unbound_cloud_current_payload()
            payload["supervisor"] = await _cloud_supervisor_for_payload(
                payload,
                username,
                deployment_mode="ssh",
            )
            payload = _attach_autonomy(payload)
            return _attach_user_wallet(payload, username)
        try:
            result = await training.current(username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        payload = _release_failed_cloud_hold(
            result.to_dict(),
            username,
            reason="release hold after terminal cloud training status",
        )
        payload = _normalize_cloud_failure_payload(payload)
        deployment_mode = await _cloud_deployment_mode(provider=str(payload.get("provider") or ""))
        payload["supervisor"] = await _cloud_supervisor_for_payload(
            payload,
            username,
            deployment_mode=deployment_mode,
        )
        payload = _apply_completed_supervisor_runtime(payload)
        if _runtime_binding_failure_archivable(payload) and await _runtime_binding_recovered(payload):
            payload = _archived_current_payload(
                payload,
                message="上一任务是在旧实例状态下失败的，已归档；当前云端实例已就绪，可以重新启动。",
            )
        payload = _attach_autonomy(payload)
        payload = _attach_cloud_artifacts(payload)
        if _cloud_training_active(payload):
            _schedule_cloud_supervisor(
                username=username,
                payload=payload,
                automation_policy=_automation_policy_for_payload(username, payload),
            )
        return _attach_user_wallet(payload, username)

    @app.get("/api/train/cloud/status/{job_id}")
    async def train_cloud_status(job_id: str, username: str = "") -> dict[str, Any]:
        try:
            result = await training.status(job_id=job_id, username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        payload = _release_failed_cloud_hold(
            result.to_dict(),
            username,
            reason="release hold after terminal cloud training status",
        )
        payload = _normalize_cloud_failure_payload(payload)
        deployment_mode = await _cloud_deployment_mode(provider=str(payload.get("provider") or ""))
        payload["supervisor"] = await _cloud_supervisor_for_payload(
            payload,
            username,
            deployment_mode=deployment_mode,
        )
        payload = _apply_completed_supervisor_runtime(payload)
        payload = _attach_autonomy(payload)
        payload = _attach_cloud_artifacts(payload)
        if _cloud_training_active(payload):
            _schedule_cloud_supervisor(
                username=username,
                payload=payload,
                automation_policy=_automation_policy_for_payload(username, payload),
            )
        return _attach_user_wallet(payload, username)

    @app.get("/api/train/cloud/artifacts")
    async def train_cloud_artifacts(username: str = "", job_id: str = "") -> dict[str, Any]:
        try:
            if job_id.strip():
                result = await training.status(job_id=job_id.strip(), username=username)
            else:
                result = await training.current(username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        payload = _attach_cloud_artifacts(_normalize_cloud_failure_payload(result.to_dict()))
        artifacts = list(payload.get("artifacts") or [])
        response: dict[str, Any] = {
            "jobId": payload.get("job_id") or job_id,
            "status": payload.get("status") or "",
            "running": payload.get("running") is True,
            "artifacts": artifacts,
        }
        log_metrics = _cloud_metrics_from_payload(payload)
        if log_metrics:
            response["metrics"] = log_metrics
            response["metricsSource"] = "log_tail"
        metrics_artifact = next((item for item in artifacts if item.get("kind") == "metrics"), None)
        if metrics_artifact:
            path = str(metrics_artifact.get("path") or "")
            response["metricsPath"] = path
            try:
                content = await asyncio.to_thread(_read_remote_text_file, path)
                response["metricsRaw"] = content
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        response["metrics"] = parsed
                        response["metricsSource"] = "metrics_file"
                    else:
                        response["metricsParseError"] = "metrics file JSON root is not an object"
                except json.JSONDecodeError:
                    response["metricsParseError"] = "metrics file is not valid JSON"
            except (RuntimeError, ValueError, OSError) as exc:
                response["metricsReadError"] = str(exc)
        return response

    @app.post("/api/train/cloud/supervisor/repair")
    async def train_cloud_supervisor_repair(body: CloudTrainSupervisorRepairRequest) -> dict[str, Any]:
        username = body.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="username is required")
        job_id = body.job_id.strip()
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        try:
            result = await training.status(job_id=job_id, username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        status_payload = _release_failed_cloud_hold(
            result.to_dict(),
            username,
            reason="release hold before same-runtime supervisor repair",
        )
        status_payload = _normalize_cloud_failure_payload(status_payload)
        deployment_mode = await _cloud_deployment_mode(provider=str(status_payload.get("provider") or ""))
        supervisor = await _cloud_supervisor_for_payload(
            status_payload,
            username,
            automation_policy=body.automation_policy,
            deployment_mode=deployment_mode,
        )
        if supervisor.get("runtimeUnavailable"):
            status_payload["supervisor"] = supervisor
            status_payload["message"] = "云端实例当前不可达，不能在同一实例内续跑；请重新绑定最新 SSH 实例。"
            status_payload = _attach_autonomy(status_payload)
            return _attach_user_wallet(status_payload, username)
        runtime = supervisor.get("runtime") if isinstance(supervisor.get("runtime"), dict) else {}
        runtime_state = str(runtime.get("state") or "").strip().lower()
        runtime_current_job_id = str(runtime.get("currentJobId") or "").strip()
        if (
            runtime_state in {"watching", "repairing", "repair_submitted"}
            and runtime_current_job_id
            and await _supervisor_runtime_job_active(runtime, username)
        ):
            status_payload["supervisor"] = supervisor
            status_payload["message"] = "已有后端总控在处理这条任务链，已忽略重复续跑请求。"
            status_payload = _attach_autonomy(status_payload)
            return _attach_user_wallet(status_payload, username)
        repair_payload = None
        if _runtime_binding_failure_archivable(status_payload) and await _cloud_runtime_ready(provider=str(status_payload.get("provider") or "")):
            repair_payload = _runtime_rebind_restart_request(
                status_payload,
                username,
                deployment_mode=deployment_mode,
                user_guidance=body.user_guidance,
            )
        if repair_payload is None:
            repair_payload = await _repair_start_request(
                status_payload,
                username,
                training,
                body.automation_policy,
                deployment_mode=deployment_mode,
                user_guidance=body.user_guidance,
                llm_provider=getattr(app.state, "llm_provider", None),
            )
        if repair_payload is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "supervisor_repair_requires_review",
                    "supervisor": supervisor,
                },
            )
        repair_request = CloudTrainStartRequest(**repair_payload)
        started = await _start_cloud_training(repair_request)
        next_job_id = str(started.get("job_id") or repair_request.task_name or "").strip()
        if next_job_id:
            _set_cloud_supervisor_state(username, next_job_id, {
                "state": "watching",
                "rootJobId": next_job_id,
                "currentJobId": next_job_id,
                "repairOfJobId": job_id,
                "repairCount": 0,
                "maxRepairs": _cloud_supervisor_max_repairs(),
                "status": started.get("status") or "",
                "message": "后端总控正在观察云端任务。",
            })
        fresh_supervisor = await _cloud_supervisor_for_payload(
            started,
            username,
            automation_policy=body.automation_policy,
            deployment_mode=deployment_mode,
        )
        started["supervisor"] = {
            **fresh_supervisor,
            "state": "repair_submitted",
            "nextAction": "watch_status",
            "autoStarted": True,
            "repairOfJobId": job_id,
        }
        started = _attach_autonomy(started)
        return started

    @app.post("/api/train/cloud/supervisor/watch")
    async def train_cloud_supervisor_watch(body: CloudTrainSupervisorWatchRequest) -> dict[str, Any]:
        username = body.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="username is required")
        job_id = body.job_id.strip()
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        try:
            result = await training.status(job_id=job_id, username=username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        payload = _normalize_cloud_failure_payload(result.to_dict())
        deployment_mode = await _cloud_deployment_mode(provider=str(payload.get("provider") or ""))
        payload["supervisor"] = await _cloud_supervisor_for_payload(
            payload,
            username,
            automation_policy=body.automation_policy,
            deployment_mode=deployment_mode,
        )
        payload = _attach_autonomy(payload)
        _schedule_cloud_supervisor(
            username=username,
            payload=payload,
            automation_policy=body.automation_policy,
        )
        return _attach_user_wallet(payload, username)

    @app.post("/api/train/plan")
    async def train_plan(body: CloudTrainPlanRequest) -> dict[str, Any]:
        try:
            result = await training.plan(
                TrainingPlanSpec(
                    username=body.username,
                    message=body.message,
                    workflow=body.workflow,
                    params=body.params,
                    provider=body.provider,
                    sku_id=body.sku_id,
                    image_id=body.image_id,
                )
            )
            return _attach_user_wallet(result, body.username)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.get("/api/train/gpu-skus")
    async def train_gpu_skus(provider: str = "", include_incomplete: bool = False, force_refresh: bool = False) -> dict[str, Any]:
        try:
            return await training.gpu_skus(
                provider=provider,
                include_incomplete=include_incomplete,
                force_refresh=force_refresh,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.get("/api/train/images")
    async def train_images(provider: str = "", include_incomplete: bool = False) -> dict[str, Any]:
        try:
            return await training.images(provider=provider, include_incomplete=include_incomplete)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.get("/api/train/cloud/resources")
    async def train_cloud_resources(
        provider: str = "",
        include_incomplete: bool = False,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        if _local_runtime_unbound():
            return {
                "provider": provider or "autodl",
                "skus": [],
                "images": [],
                "messages": {
                    "skus": "还没有绑定云端实例。连接实例后再刷新资源。",
                    "images": "还没有绑定云端实例。连接实例后再刷新镜像。",
                },
            }
        try:
            # Show live provider inventory even when a SKU is not startable in this
            # deployment; readiness is still exposed per item and checked at start.
            skus = await training.gpu_skus(
                provider=provider,
                include_incomplete=True,
                force_refresh=force_refresh,
            )
            images = await training.images(provider=provider, include_incomplete=include_incomplete)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        selected_provider = provider or str(skus.get("provider") or "")
        return {
            "provider": selected_provider,
            "skus": skus.get("skus", []),
            "images": images.get("images", []),
            "messages": {
                "skus": skus.get("message", ""),
                "images": images.get("message", ""),
            },
        }

    @app.get("/api/train/cloud/provider-balance")
    async def train_cloud_provider_balance(provider: str = "", minimum_assets: int = 0) -> dict[str, Any]:
        try:
            result = await training.provider_balance(provider=provider, minimum_assets=minimum_assets)
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
        return {
            **result,
            "balanceScope": "provider_pool",
            "description": "Operator AutoDL/provider balance for managed compute capacity. User spend is checked with /api/account/balance.",
        }

    @app.post("/api/train/cloud/provider-balance")
    async def train_cloud_provider_balance_post(body: ProviderBalanceRequest) -> dict[str, Any]:
        return await train_cloud_provider_balance(provider=body.provider, minimum_assets=body.minimum_assets)

    @app.post("/api/train/runtime-match")
    async def train_runtime_match(body: RuntimeMatchRequest) -> dict[str, Any]:
        try:
            return await training.runtime_match(
                username=body.username,
                provider=body.provider,
                params=body.params,
                sku_id=body.sku_id,
                image_id=body.image_id,
                force_refresh=body.force_refresh,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc

    @app.post("/api/train/source-preflight")
    async def train_source_preflight(body: SourcePreflightRequest) -> dict[str, Any]:
        try:
            return await training.source_preflight(
                username=body.username,
                provider=body.provider,
                role=body.role,
                source=body.source,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=_bridge_error_status(exc), detail=str(exc)) from exc
