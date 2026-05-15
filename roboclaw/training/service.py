"""Training orchestration across local RoboClaw jobs and remote EVO_Train jobs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from roboclaw.cloud import EvoTrainBridge
from roboclaw.training.schema import (
    TrainingJobStatus,
    TrainingPolicyEntry,
    TrainingPlanSpec,
    TrainingStartSpec,
    TrainingStopSpec,
)

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


class TrainingService:
    """Single training application service for HTTP routes and future CLI/API entrypoints."""

    def __init__(
        self,
        embodied_service: "EmbodiedService",
        *,
        bridge: EvoTrainBridge | None = None,
    ) -> None:
        self._embodied_service = embodied_service
        self._bridge = bridge or EvoTrainBridge()

    async def start(self, spec: TrainingStartSpec) -> TrainingJobStatus:
        if self._use_remote_bridge(spec.username):
            payload = await asyncio.to_thread(
                self._bridge.start_training,
                self._embodied_service,
                dataset_name=spec.dataset_name,
                policy_type=spec.policy_type,
                steps=spec.steps,
                device=spec.device,
                username=spec.username,
                workflow=spec.workflow,
                params=dict(spec.params or {}),
                sku_id=spec.sku_id,
                image_id=spec.image_id,
                task_name=spec.task_name,
            )
            return TrainingJobStatus.from_payload(payload, mode="cloud")

        return await self._embodied_service.train.start_job(self._embodied_service.manifest, spec)

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

    async def gpu_skus(self, *, provider: str = "", include_incomplete: bool = False) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(
            self._bridge.gpu_skus,
            provider=provider,
            include_incomplete=include_incomplete,
        )

    async def images(self, *, include_incomplete: bool = False) -> dict[str, object]:
        if not self._bridge.enabled:
            raise RuntimeError("EVO_Train bridge is not enabled.")
        return await asyncio.to_thread(self._bridge.images, include_incomplete=include_incomplete)

    async def runtime_match(
        self,
        *,
        username: str = "",
        provider: str = "",
        params: dict[str, object] | None = None,
        sku_id: str = "",
        image_id: str = "",
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
        )

    async def stop(self, spec: TrainingStopSpec) -> TrainingJobStatus:
        if self._use_remote_bridge(spec.username):
            payload = await asyncio.to_thread(
                self._bridge.stop_training,
                job_id=spec.job_id,
                username=spec.username,
            )
            return TrainingJobStatus.from_payload(payload, mode="cloud")

        return await self._embodied_service.train.stop_job_state(spec.job_id)

    async def current(self, *, username: str = "") -> TrainingJobStatus:
        if self._use_remote_bridge(username):
            payload = await asyncio.to_thread(self._bridge.current_task, username=username)
            return TrainingJobStatus.from_payload(payload, mode="cloud")

        return await self._embodied_service.train.current_job_state()

    async def status(self, *, job_id: str, username: str = "") -> TrainingJobStatus:
        if self._use_remote_bridge(username):
            payload = await asyncio.to_thread(self._bridge.task_status, job_id=job_id, username=username)
            cloud_status = TrainingJobStatus.from_payload(payload, mode="cloud")
            if cloud_status.status != "missing":
                return cloud_status

        return await self._embodied_service.train.job_status_state(job_id)

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
