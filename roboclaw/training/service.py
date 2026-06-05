"""Training orchestration across local RoboClaw jobs and remote EVO_Train jobs."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from roboclaw.cloud import EvoTrainBridge
from roboclaw.embodied.board import CH_TRAINING
from roboclaw.training.schema import (
    TrainingJobStatus,
    TrainingPolicyEntry,
    TrainingPlanSpec,
    TrainingStartSpec,
    TrainingStopSpec,
)
from roboclaw.training.rlinf_catalog import set_rlinf_ext_module

if TYPE_CHECKING:
    from roboclaw.embodied.board import Board
    from roboclaw.embodied.service import EmbodiedService


class TrainingService:
    """Single training application service for HTTP routes and future CLI/API entrypoints."""

    def __init__(
        self,
        embodied_service: "EmbodiedService",
        *,
        bridge: EvoTrainBridge | None = None,
        board: "Board | None" = None,
    ) -> None:
        self._embodied_service = embodied_service
        self._bridge = bridge or EvoTrainBridge()
        self._board = board or getattr(embodied_service, "board", None)

    @property
    def cloud_enabled(self) -> bool:
        return self._bridge.enabled

    def cloud_bridge_status(self) -> dict[str, object]:
        settings = self._bridge.settings
        enabled = self._bridge.enabled
        return {
            "enabled": enabled,
            "provider": settings.provider,
            "managedBy": "Evo Studio deployment",
            "userActionRequired": False,
            "message": (
                "Cloud training bridge is connected."
                if enabled
                else "Cloud training bridge is not connected in this backend deployment."
            ),
            "operatorHint": (
                ""
                if enabled
                else "Team operators should configure ROBOCLAW_EVO_TRAIN_HOST and related deployment secrets on the backend server."
            ),
            "missingDeploymentFields": [] if enabled else ["ROBOCLAW_EVO_TRAIN_HOST"],
        }

    async def start(self, spec: TrainingStartSpec) -> TrainingJobStatus:
        if self._use_remote_bridge(spec.username):
            params = dict(spec.params or {})
            if _uses_rlinf_runtime(spec.workflow, params) and _uses_roboclaw_rlinf_adapter(params):
                params.setdefault("rlinfExtModule", set_rlinf_ext_module())
            payload = await asyncio.to_thread(
                self._bridge.start_training,
                self._embodied_service,
                dataset_name=spec.dataset_name,
                policy_type=spec.policy_type,
                steps=spec.steps,
                device=spec.device,
                username=spec.username,
                provider=spec.provider,
                workflow=spec.workflow,
                params=params,
                sku_id=spec.sku_id,
                image_id=spec.image_id,
                task_name=spec.task_name,
                wait_for_submit=spec.wait_for_submit,
            )
            status = TrainingJobStatus.from_payload(payload, mode="cloud")
            await self._emit_training_state(status)
            return status

        status = await self._embodied_service.train.start_job(self._embodied_service.manifest, spec)
        await self._emit_training_state(status)
        return status

    async def plan(self, spec: TrainingPlanSpec) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(
            self._bridge.training_plan,
            username=spec.username,
            message=spec.message,
            workflow=spec.workflow,
            params=dict(spec.params or {}),
            provider=spec.provider,
            sku_id=spec.sku_id,
            image_id=spec.image_id,
        )

    async def gpu_skus(
        self,
        *,
        provider: str = "",
        include_incomplete: bool = False,
        force_refresh: bool = False,
    ) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(
            self._bridge.gpu_skus,
            provider=provider,
            include_incomplete=include_incomplete,
            force_refresh=force_refresh,
        )

    async def images(self, *, provider: str = "", include_incomplete: bool = False) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(self._bridge.images, provider=provider, include_incomplete=include_incomplete)

    async def configuration_check(self, *, provider: str = "") -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(self._bridge.configuration_check, provider=provider)

    async def provider_balance(self, *, provider: str = "", minimum_assets: int = 0) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(
            self._bridge.provider_balance,
            provider=provider,
            minimum_assets=minimum_assets,
        )

    async def runtime_match(
        self,
        *,
        username: str = "",
        provider: str = "",
        params: dict[str, object] | None = None,
        sku_id: str = "",
        image_id: str = "",
        force_refresh: bool = True,
    ) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(
            self._bridge.runtime_match,
            username=username,
            provider=provider,
            params=dict(params or {}),
            sku_id=sku_id,
            image_id=image_id,
            force_refresh=force_refresh,
        )

    async def source_preflight(
        self,
        *,
        username: str = "",
        provider: str = "",
        source: dict[str, object] | None = None,
        role: str = "dataset",
    ) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(
            self._bridge.source_preflight,
            username=username,
            provider=provider,
            source=dict(source or {}),
            role=role,
        )

    async def stop(self, spec: TrainingStopSpec) -> TrainingJobStatus:
        if self._use_remote_bridge(spec.username):
            payload = await asyncio.to_thread(
                self._bridge.stop_training,
                job_id=spec.job_id,
                username=spec.username,
            )
            status = TrainingJobStatus.from_payload(payload, mode="cloud")
            await self._emit_training_state(status)
            return status

        status = await self._embodied_service.train.stop_job_state(spec.job_id)
        await self._emit_training_state(status)
        return status

    async def current(self, *, username: str = "") -> TrainingJobStatus:
        if self._use_remote_bridge(username):
            try:
                payload = await _status_bridge_call(self._bridge.current_task, username=username)
            except asyncio.TimeoutError:
                payload = _cloud_status_timeout_payload(provider=self._bridge.settings.provider)
            return TrainingJobStatus.from_payload(payload, mode="cloud")

        return await self._embodied_service.train.current_job_state()

    async def status(self, *, job_id: str, username: str = "") -> TrainingJobStatus:
        if self._use_remote_bridge(username):
            try:
                payload = await _status_bridge_call(self._bridge.task_status, job_id=job_id, username=username)
            except asyncio.TimeoutError:
                payload = _cloud_status_timeout_payload(job_id=job_id, provider=self._bridge.settings.provider)
            cloud_status = TrainingJobStatus.from_payload(payload, mode="cloud")
            if cloud_status.status != "missing":
                await self._emit_training_state(cloud_status)
                return cloud_status

        status = await self._embodied_service.train.job_status_state(job_id)
        await self._emit_training_state(status)
        return status

    async def list_policies(self, *, username: str = "") -> list[TrainingPolicyEntry]:
        policies = self._embodied_service.train.list_policy_entries(self._embodied_service.manifest)
        if not self._use_remote_bridge(username):
            return policies

        remote_entries = await asyncio.to_thread(self._bridge.list_policy_entries, username=username)
        policies.extend(
            TrainingPolicyEntry.from_payload(entry, source="cloud")
            for entry in remote_entries
        )
        return policies

    def _use_remote_bridge(self, username: str) -> bool:
        if not self._bridge.enabled:
            return False
        return bool((username or self._bridge.settings.username).strip())

    async def _emit_training_state(self, status: TrainingJobStatus) -> None:
        if self._board is None:
            return
        payload = status.to_dict()
        await self._board.emit(
            CH_TRAINING,
            {
                "payload": {
                    "jobId": payload.get("job_id") or payload.get("jobId") or "",
                    "taskName": payload.get("task_name") or payload.get("taskName") or "",
                    "status": payload.get("status") or "",
                    "running": payload.get("running") is True,
                    "mode": payload.get("mode") or "",
                    "provider": payload.get("provider") or "",
                    "message": payload.get("message") or payload.get("error") or "",
                    "raw": payload,
                }
            },
        )


async def _status_bridge_call(func: object, /, **kwargs: object) -> dict[str, object]:
    timeout_s = _status_timeout_s()
    payload = await asyncio.wait_for(asyncio.to_thread(func, **kwargs), timeout=timeout_s)
    return dict(payload)


def _status_timeout_s() -> float:
    raw = os.environ.get("ROBOCLAW_EVO_TRAIN_STATUS_TIMEOUT", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 6.0


def _cloud_status_timeout_payload(*, job_id: str = "", provider: str = "") -> dict[str, object]:
    message = (
        "EVO_Train status query timed out. The configured cloud instance is probably unreachable; "
        "rebind or restart the SSH instance before submitting or repairing jobs."
    )
    return {
        "job_id": job_id,
        "status": "failed" if job_id else "idle",
        "running": False,
        "message": message,
        "error": message,
        "task_name": "",
        "checkpoint_path": "",
        "dataset_path": "",
        "provider": provider,
        "failureRemediation": {
            "code": "CLOUD_INSTANCE_UNREACHABLE",
            "userFacingSummary": "云端实例连接超时，需要重新绑定或重启 SSH 实例。",
            "nextSteps": [
                "确认云端实例正在开机并且 SSH 端口是最新的",
                "在页面重新绑定 SSH 实例",
                "绑定成功后再继续训练或修复任务",
            ],
            "autoRepair": {"safe": False, "reason": "requires updated SSH credentials or instance endpoint"},
        },
    }


def _uses_rlinf_runtime(workflow: str, params: dict[str, object]) -> bool:
    tokens = {
        str(workflow or "").lower(),
        str(params.get("workflow") or "").lower(),
        str(params.get("backendKind") or "").lower(),
        str(params.get("builtinTrainingProfile") or "").lower(),
        str(params.get("repoUrl") or "").lower(),
    }
    return any("rlinf" in token or "github.com/rlinf/rlinf" in token for token in tokens)


def _uses_roboclaw_rlinf_adapter(params: dict[str, object]) -> bool:
    repo_url = str(params.get("repoUrl") or "").lower()
    workdir = str(params.get("workdir") or "").lower()
    launcher_module = str(params.get("launcherModule") or "").lower()
    return (
        "roboclaw" in repo_url
        or "roboclaw" in workdir
        or launcher_module.startswith("roboclaw_vla.")
    )
