"""Unified embodied workflow specification and planning helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from roboclaw.data.datasets import DatasetCatalog, DatasetRuntimeRef, validate_dataset_slug
from roboclaw.embodied.command import ActionError, CommandBuilder

WorkflowStageName = Literal["record", "train", "infer"]
_CHECKPOINT_CONFIG_FILES = (
    "config.json",
    "train_config.json",
    "policy_config.json",
    "preprocessor_config.json",
)
_CHECKPOINT_WEIGHT_PATTERNS = (
    "model.safetensors",
    "*.safetensors",
    "*.pt",
    "*.pth",
    "*.bin",
)


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
    blocked_by: list[WorkflowStageName] = Field(default_factory=list)
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

        dataset = self._prepare_output_dataset(
            spec,
            dataset_name=spec.record.dataset_name,
            prefix="rec",
            payload={
                "name": spec.name,
                "hardware": spec.hardware.model_dump(mode="json", by_alias=False),
                "record": spec.record.model_dump(mode="json", by_alias=False),
            },
        )
        stage.dataset_name = dataset.runtime.name
        stage.output_path = str(dataset.runtime.local_path)
        if not spec.record.dataset_name.strip():
            stage.notes.append(
                f"record.dataset_name is omitted and resolves deterministically to '{stage.dataset_name}'."
            )
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
        stage = WorkflowStagePlan(
            stage="train",
            enabled=spec.train.enabled,
            owner="training",
            capability="train",
        )
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

        dataset_runtime = self._resolve_runtime_dataset(dataset_name)
        inherited_from_record = (
            not spec.train.dataset_name.strip()
            and record_plan.enabled
            and dataset_name == record_plan.dataset_name
        )
        if dataset_runtime is None:
            if inherited_from_record:
                dataset_runtime = self._planned_runtime_dataset(dataset_name)
                stage.blocked_by.append("record")
                stage.notes.append("train.dataset_name is inherited from the record stage output.")
                stage.notes.append("train cannot run until the record stage materializes its dataset.")
            else:
                stage.issues.append(WorkflowIssue(
                    stage="train",
                    code="dataset_not_found",
                    field="train.dataset_name",
                    message=f"Runtime dataset '{dataset_name}' was not found.",
                ))
                return stage
        elif inherited_from_record:
            stage.notes.append("train.dataset_name is inherited from the record stage output.")

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

        stage.ready = not stage.blocked_by
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

        output_dataset = self._prepare_output_dataset(
            spec,
            dataset_name=spec.infer.dataset_name,
            prefix="eval",
            payload={
                "name": spec.name,
                "hardware": spec.hardware.model_dump(mode="json", by_alias=False),
                "infer": spec.infer.model_dump(mode="json", by_alias=False),
            },
        )
        stage.dataset_name = output_dataset.runtime.name
        stage.output_path = str(output_dataset.runtime.local_path)
        checkpoint_path = spec.infer.checkpoint_path.strip()
        stage.checkpoint_path = checkpoint_path
        if not spec.infer.dataset_name.strip():
            stage.notes.append(
                f"infer.dataset_name is omitted and resolves deterministically to '{stage.dataset_name}'."
            )
        if checkpoint_path:
            checkpoint_issue = _checkpoint_validation_message(checkpoint_path)
            if checkpoint_issue:
                stage.issues.append(WorkflowIssue(
                    stage="infer",
                    code="invalid_checkpoint",
                    field="infer.checkpoint_path",
                    message=checkpoint_issue,
                ))
                return stage

        source_dataset_name = spec.infer.source_dataset.strip()
        derived_from_train = False
        if not checkpoint_path and not source_dataset_name and train_plan.enabled:
            source_dataset_name = train_plan.dataset_name
            derived_from_train = True
        elif not checkpoint_path and not source_dataset_name:
            source_dataset_name = record_plan.dataset_name
        stage.source_dataset = source_dataset_name

        if not checkpoint_path and not source_dataset_name:
            stage.issues.append(WorkflowIssue(
                stage="infer",
                code="missing_checkpoint_source",
                field="infer.checkpoint_path",
                message="Provide infer.checkpoint_path or a source dataset/train stage for checkpoint resolution.",
            ))
            return stage

        source_dataset = None
        if not checkpoint_path and source_dataset_name:
            source_dataset = self._resolve_runtime_dataset(source_dataset_name)
            if source_dataset is None and derived_from_train:
                source_dataset = self._planned_runtime_dataset(source_dataset_name)
                stage.blocked_by.append("train")
            elif source_dataset is None:
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
        if not checkpoint_path and derived_from_train:
            stage.notes.append("infer.checkpoint_path is derived from the planned training output.")
            if stage.checkpoint_path and not Path(stage.checkpoint_path).expanduser().exists():
                if "train" not in stage.blocked_by:
                    stage.blocked_by.append("train")
                stage.notes.append("infer cannot run until the train stage produces the checkpoint.")
            else:
                checkpoint_issue = _checkpoint_validation_message(stage.checkpoint_path)
                if checkpoint_issue:
                    stage.issues.append(WorkflowIssue(
                        stage="infer",
                        code="invalid_checkpoint",
                        field="infer.checkpoint_path",
                        message=checkpoint_issue,
                    ))
                    return stage
        elif not checkpoint_path and source_dataset_name:
            stage.notes.append("infer.checkpoint_path is derived from the source dataset policy directory.")
            checkpoint_issue = _checkpoint_validation_message(stage.checkpoint_path)
            if checkpoint_issue:
                stage.issues.append(WorkflowIssue(
                    stage="infer",
                    code="invalid_checkpoint",
                    field="infer.checkpoint_path",
                    message=checkpoint_issue,
                ))
                return stage

        stage.ready = not stage.blocked_by
        return stage

    def _resolve_runtime_dataset(self, name: str) -> DatasetRuntimeRef | None:
        try:
            return self._datasets.resolve_runtime_dataset(name).runtime
        except ValueError:
            return None

    def _planned_runtime_dataset(self, name: str) -> DatasetRuntimeRef:
        return DatasetRuntimeRef(
            name=name,
            repo_id=f"local/{name}",
            local_path=self._datasets.root / "local" / name,
        )

    def _prepare_output_dataset(
        self,
        spec: WorkflowSpec,
        *,
        dataset_name: str,
        prefix: str,
        payload: dict[str, Any],
    ) -> Any:
        resolved_name = dataset_name.strip() or _default_dataset_name(spec, prefix=prefix, payload=payload)
        return self._datasets.prepare_recording_dataset(resolved_name, prefix=prefix)


def _argv_value(argv: list[str], prefix: str) -> str:
    for item in argv:
        if item.startswith(prefix):
            return item.split("=", 1)[1]
    return ""


def _default_dataset_name(spec: WorkflowSpec, *, prefix: str, payload: dict[str, Any]) -> str:
    slug = _slugify_name(spec.name)
    if slug:
        candidate = f"{prefix}_{slug}"
    else:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        candidate = f"{prefix}_{hashlib.sha1(encoded.encode('utf-8')).hexdigest()[:10]}"
    validate_dataset_slug(candidate)
    return candidate


def _slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower()).strip("_-")
    return slug[:48]


def _checkpoint_validation_message(raw_path: str) -> str:
    if not raw_path or _looks_like_remote_policy_id(raw_path):
        return ""

    path = Path(raw_path).expanduser()
    if not path.exists():
        return f"Resolved checkpoint '{path}' was not found."
    if not path.is_dir():
        return f"Resolved checkpoint '{path}' must be a directory."
    if not any((path / name).is_file() for name in _CHECKPOINT_CONFIG_FILES):
        joined = ", ".join(_CHECKPOINT_CONFIG_FILES)
        return f"Resolved checkpoint '{path}' is missing a recognized config file ({joined})."
    if not any(any(path.glob(pattern)) for pattern in _CHECKPOINT_WEIGHT_PATTERNS):
        joined = ", ".join(_CHECKPOINT_WEIGHT_PATTERNS)
        return f"Resolved checkpoint '{path}' is missing model weights ({joined})."
    return ""


def _looks_like_remote_policy_id(raw_path: str) -> bool:
    path = Path(raw_path).expanduser()
    if path.exists() or path.is_absolute():
        return False
    if raw_path.startswith(("~", ".", "/")):
        return False
    parts = raw_path.split("/")
    return len(parts) == 2 and all(parts) and not any(part in {".", ".."} for part in parts)
