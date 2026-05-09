"""Unified embodied workflow specification and planning helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from roboclaw.data.datasets import DatasetCatalog, DatasetRuntimeRef
from roboclaw.embodied.command import ActionError, CommandBuilder

WorkflowStageName = Literal["record", "train", "infer"]


class WorkflowModel(BaseModel):
    """Base workflow model with camelCase compatibility for JSON APIs."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class WorkflowHardwareSpec(WorkflowModel):
    """Hardware-facing options shared across workflow stages."""

    arms: str = ""
    use_cameras: bool = True


class RecordWorkflowSpec(WorkflowModel):
    enabled: bool = False
    task: str = ""
    dataset_name: str = ""
    num_episodes: int = 10
    fps: int = 30
    episode_time_s: int = 300
    reset_time_s: int = 10


class TrainWorkflowSpec(WorkflowModel):
    enabled: bool = False
    dataset_name: str = ""
    policy_type: str = "act"
    steps: int = 100_000
    device: str = "cuda"


class InferWorkflowSpec(WorkflowModel):
    enabled: bool = False
    checkpoint_path: str = ""
    source_dataset: str = ""
    dataset_name: str = ""
    task: str = "eval"
    num_episodes: int = 1
    episode_time_s: int = 60


class WorkflowSpec(WorkflowModel):
    """Single spec that describes an embodied data/train/infer workflow."""

    name: str = ""
    hardware: WorkflowHardwareSpec = Field(default_factory=WorkflowHardwareSpec)
    record: RecordWorkflowSpec = Field(default_factory=RecordWorkflowSpec)
    train: TrainWorkflowSpec = Field(default_factory=TrainWorkflowSpec)
    infer: InferWorkflowSpec = Field(default_factory=InferWorkflowSpec)


class WorkflowIssue(WorkflowModel):
    """Validation issue found while compiling or checking a workflow."""

    stage: WorkflowStageName
    code: str
    message: str
    field: str = ""


class WorkflowStagePlan(WorkflowModel):
    """Planner output for one workflow stage."""

    stage: WorkflowStageName
    enabled: bool = False
    ready: bool = False
    owner: str = ""
    capability: str = ""
    dataset_name: str = ""
    source_dataset: str = ""
    checkpoint_path: str = ""
    output_path: str = ""
    command: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    issues: list[WorkflowIssue] = Field(default_factory=list)


class WorkflowPlan(WorkflowModel):
    """Complete compiled view of a workflow spec."""

    name: str = ""
    ok: bool = False
    stages: list[WorkflowStagePlan] = Field(default_factory=list)
    issues: list[WorkflowIssue] = Field(default_factory=list)

    def stage(self, stage_name: WorkflowStageName) -> WorkflowStagePlan:
        for stage in self.stages:
            if stage.stage == stage_name:
                return stage
        raise KeyError(stage_name)


class WorkflowPlanner:
    """Compile a workflow spec into concrete stage plans and validations."""

    def __init__(self, manifest: Any, datasets: DatasetCatalog) -> None:
        self._manifest = manifest
        self._datasets = datasets

    def plan(self, spec: WorkflowSpec | dict[str, Any]) -> WorkflowPlan:
        workflow = spec if isinstance(spec, WorkflowSpec) else WorkflowSpec.model_validate(spec)
        stages: list[WorkflowStagePlan] = []

        record_plan = self._plan_record(workflow)
        stages.append(record_plan)

        train_plan = self._plan_train(workflow, record_plan)
        stages.append(train_plan)

        infer_plan = self._plan_infer(workflow, record_plan, train_plan)
        stages.append(infer_plan)

        issues = [issue for stage in stages for issue in stage.issues]
        if not any(stage.enabled for stage in stages):
            issues.append(WorkflowIssue(
                stage="record",
                code="empty_workflow",
                message="Enable at least one workflow stage.",
            ))

        return WorkflowPlan(
            name=workflow.name,
            ok=not issues,
            stages=stages,
            issues=issues,
        )

    def _plan_record(self, spec: WorkflowSpec) -> WorkflowStagePlan:
        stage = WorkflowStagePlan(
            stage="record",
            enabled=spec.record.enabled,
            owner="recording",
            capability="record" if spec.hardware.use_cameras else "record_without_cameras",
        )
        if not spec.record.enabled:
            return stage

        dataset = self._datasets.prepare_recording_dataset(spec.record.dataset_name, prefix="rec")
        stage.dataset_name = dataset.runtime.name
        stage.output_path = str(dataset.runtime.local_path)
        if not spec.record.dataset_name.strip():
            stage.notes.append("dataset_name is omitted and will be auto-generated at runtime.")
        if not spec.record.task.strip():
            stage.issues.append(WorkflowIssue(
                stage="record",
                code="missing_task",
                field="record.task",
                message="record.task is required when the record stage is enabled.",
            ))

        if stage.issues:
            return stage

        try:
            stage.command = CommandBuilder.record(
                self._manifest,
                dataset=dataset.runtime,
                task=spec.record.task,
                num_episodes=spec.record.num_episodes,
                fps=spec.record.fps,
                episode_time_s=spec.record.episode_time_s,
                reset_time_s=spec.record.reset_time_s,
                arms=spec.hardware.arms,
                use_cameras=spec.hardware.use_cameras,
            )
        except (ActionError, ValueError) as exc:
            stage.issues.append(WorkflowIssue(
                stage="record",
                code="compile_error",
                message=str(exc),
            ))
            return stage

        stage.ready = True
        return stage

    def _plan_train(self, spec: WorkflowSpec, record_plan: WorkflowStagePlan) -> WorkflowStagePlan:
        stage = WorkflowStagePlan(stage="train", enabled=spec.train.enabled, owner="training")
        if not spec.train.enabled:
            return stage

        dataset_name = spec.train.dataset_name.strip() or record_plan.dataset_name
        stage.dataset_name = dataset_name
        if not dataset_name:
            stage.issues.append(WorkflowIssue(
                stage="train",
                code="missing_dataset",
                field="train.dataset_name",
                message="train.dataset_name is required unless the record stage feeds training.",
            ))
            return stage

        dataset_runtime = self._resolve_runtime_dataset(
            dataset_name,
            allow_planned=record_plan.enabled and dataset_name == record_plan.dataset_name,
        )
        if dataset_runtime is None:
            stage.issues.append(WorkflowIssue(
                stage="train",
                code="dataset_not_found",
                field="train.dataset_name",
                message=f"Runtime dataset '{dataset_name}' was not found.",
            ))
            return stage

        try:
            stage.command = CommandBuilder.train(
                self._manifest,
                dataset=dataset_runtime,
                policy_type=spec.train.policy_type,
                steps=spec.train.steps,
                device=spec.train.device,
            )
        except (ActionError, ValueError) as exc:
            stage.issues.append(WorkflowIssue(
                stage="train",
                code="compile_error",
                message=str(exc),
            ))
            return stage

        output_dir = _argv_value(stage.command, "--output_dir=")
        if output_dir:
            stage.output_path = output_dir
            stage.checkpoint_path = str(Path(output_dir) / "checkpoints" / "last" / "pretrained_model")

        if not spec.train.dataset_name.strip() and record_plan.enabled:
            stage.notes.append("train.dataset_name is inherited from the record stage output.")

        stage.ready = True
        return stage

    def _plan_infer(
        self,
        spec: WorkflowSpec,
        record_plan: WorkflowStagePlan,
        train_plan: WorkflowStagePlan,
    ) -> WorkflowStagePlan:
        stage = WorkflowStagePlan(
            stage="infer",
            enabled=spec.infer.enabled,
            owner="inferring",
            capability="infer" if spec.hardware.use_cameras else "infer_without_cameras",
        )
        if not spec.infer.enabled:
            return stage

        output_dataset = self._datasets.prepare_recording_dataset(spec.infer.dataset_name, prefix="eval")
        stage.dataset_name = output_dataset.runtime.name
        stage.output_path = str(output_dataset.runtime.local_path)
        if not spec.infer.dataset_name.strip():
            stage.notes.append("infer.dataset_name is omitted and will be auto-generated at runtime.")

        source_dataset_name = (
            spec.infer.source_dataset.strip()
            or train_plan.dataset_name
            or record_plan.dataset_name
        )
        stage.source_dataset = source_dataset_name

        checkpoint_path = spec.infer.checkpoint_path.strip()
        stage.checkpoint_path = checkpoint_path
        if not checkpoint_path and not source_dataset_name:
            stage.issues.append(WorkflowIssue(
                stage="infer",
                code="missing_checkpoint_source",
                field="infer.checkpoint_path",
                message="Provide infer.checkpoint_path or a source dataset/train stage for checkpoint resolution.",
            ))
            return stage

        source_dataset = None
        if source_dataset_name:
            source_dataset = self._resolve_runtime_dataset(
                source_dataset_name,
                allow_planned=(
                    (record_plan.enabled and source_dataset_name == record_plan.dataset_name)
                    or (train_plan.enabled and source_dataset_name == train_plan.dataset_name)
                ),
            )
            if source_dataset is None and not checkpoint_path:
                stage.issues.append(WorkflowIssue(
                    stage="infer",
                    code="source_dataset_not_found",
                    field="infer.source_dataset",
                    message=f"Runtime dataset '{source_dataset_name}' was not found for checkpoint resolution.",
                ))
                return stage

        try:
            stage.command = CommandBuilder.infer(
                self._manifest,
                dataset=output_dataset.runtime,
                checkpoint_path=checkpoint_path,
                source_dataset=source_dataset,
                task=spec.infer.task,
                num_episodes=spec.infer.num_episodes,
                episode_time_s=spec.infer.episode_time_s,
                arms=spec.hardware.arms,
                use_cameras=spec.hardware.use_cameras,
            )
        except (ActionError, ValueError) as exc:
            stage.issues.append(WorkflowIssue(
                stage="infer",
                code="compile_error",
                message=str(exc),
            ))
            return stage

        resolved_checkpoint = _argv_value(stage.command, "--policy.path=")
        if resolved_checkpoint:
            stage.checkpoint_path = resolved_checkpoint
        if not checkpoint_path and train_plan.enabled and train_plan.checkpoint_path:
            stage.notes.append("infer.checkpoint_path is derived from the planned training output.")
        elif not checkpoint_path and source_dataset_name:
            stage.notes.append("infer.checkpoint_path is derived from the source dataset policy directory.")

        stage.ready = True
        return stage

    def _resolve_runtime_dataset(self, name: str, *, allow_planned: bool) -> DatasetRuntimeRef | None:
        try:
            return self._datasets.resolve_runtime_dataset(name).runtime
        except ValueError:
            if not allow_planned:
                return None
            return DatasetRuntimeRef(
                name=name,
                repo_id=f"local/{name}",
                local_path=self._datasets.root / "local" / name,
            )


def _argv_value(argv: list[str], prefix: str) -> str:
    for item in argv:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return ""
