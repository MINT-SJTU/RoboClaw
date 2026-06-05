"""Pydantic request/response models for cloud training routes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CloudTrainStartRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_name: str = ""
    policy_type: str = "act"
    steps: int = 100_000
    device: str = "cuda"
    username: str = ""
    provider: str = ""
    workflow: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    sku_id: str = ""
    image_id: str = ""
    task_name: str = ""
    wait_for_submit: bool = Field(
        default=True,
        alias="waitForSubmit",
        description="Wait for provider submission before returning so startup failures surface before the page marks a job running.",
    )
    hourly_cost_cents: int = Field(
        default=0,
        description="Provider hourly compute cost in cents before service fee.",
    )
    service_fee_bps: int = Field(default=1_000, description="Service fee in basis points. 1000 = 10%.")
    confirmed: bool = Field(default=False, description="Skip user confirmation gate for full_auto mode.")
    automation_mode: str = Field(default="safe_auto", alias="automationMode")
    automation_policy: dict[str, Any] = Field(default_factory=dict, alias="automationPolicy")


class CloudTrainStopRequest(BaseModel):
    job_id: str
    username: str = ""


class CloudTrainBillingSettleRequest(BaseModel):
    username: str
    job_id: str
    provider_cost_cents: int = Field(..., description="Actual provider compute cost in cents before service fee.")
    service_fee_bps: int = Field(default=1_000, description="Service fee in basis points. 1000 = 10%.")
    task_name: str = ""


class CloudTrainPlanRequest(BaseModel):
    username: str = ""
    message: str = ""
    workflow: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    provider: str = ""
    sku_id: str = ""
    image_id: str = ""


class RuntimeMatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = ""
    provider: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    sku_id: str = ""
    image_id: str = ""
    force_refresh: bool = Field(default=True, alias="forceRefresh")


class SourcePreflightRequest(BaseModel):
    username: str = ""
    provider: str = ""
    role: str = "dataset"
    source: dict[str, Any] = Field(default_factory=dict)


class CloudResourceCatalogRequest(BaseModel):
    provider: str = ""
    include_incomplete: bool = False


class ProviderBalanceRequest(BaseModel):
    provider: str = ""
    minimum_assets: int = Field(default=0, alias="minimumAssets")


class AuthConnectionSaveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = ""
    id: str = ""
    kind: str = "both"
    provider: str = "custom"
    label: str = ""
    scope: str = ""
    visibility: str = "user"
    source_prefixes: list[str] = Field(default_factory=list, alias="sourcePrefixes")
    secrets: dict[str, str] = Field(default_factory=dict)


class CloudTrainSupervisorRepairRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = ""
    job_id: str = Field(default="", alias="jobId")
    automation_policy: dict[str, Any] = Field(default_factory=dict, alias="automationPolicy")
    user_guidance: str = Field(default="", alias="userGuidance")


class CloudTrainSupervisorWatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = ""
    job_id: str = Field(default="", alias="jobId")
    automation_policy: dict[str, Any] = Field(default_factory=dict, alias="automationPolicy")


class CloudSshRuntimeBindRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ssh_command: str = Field(default="", alias="sshCommand")
    password: str = ""
    key_path: str = Field(default="", alias="keyPath")
    restart_bridge: bool = Field(default=True, alias="restartBridge")
