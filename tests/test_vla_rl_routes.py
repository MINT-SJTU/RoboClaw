from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from roboclaw.embodied.board import Board
from roboclaw.embodied.embodiment.hardware.monitor import HardwareMonitor
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.service import EmbodiedService
from roboclaw.http.routes.vla_rl import register_vla_rl_routes


class StubBridge:
    enabled = True
    settings = SimpleNamespace(username="default-user")

    def __init__(self) -> None:
        self.plan_calls: list[dict[str, object]] = []

    def training_plan(self, **kwargs: object) -> dict[str, object]:
        self.plan_calls.append(kwargs)
        params = dict(kwargs.get("params") or {})
        return {
            "message": "plan generated",
            "plan": {
                "workflow": kwargs.get("workflow") or "rlinf_vla",
                "params": params,
                "readyToStart": True,
                "missingFields": [],
                "warnings": [],
            },
        }


@pytest.fixture()
def route_app(tmp_path: Path):
    app = FastAPI()
    board = Board()
    manifest = Manifest(path=tmp_path / "manifest.json", board=board)
    hw_monitor = HardwareMonitor(board=board, manifest=manifest)
    service = EmbodiedService(hardware_monitor=hw_monitor, board=board, manifest=manifest)
    return app, service


def test_vla_rl_plan_normalizes_capabilities_before_evo_train(route_app):
    app, service = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_vla_rl_routes(app, service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/vla-rl/plan",
        json={
            "username": "pearl",
            "message": "用 Pi0.5 co-training 在 XLeRobot 上做 action expert 和 LLM 联合优化，Blackwell 镜像",
            "params": {"configName": "xlerobot_pi05_cotrain"},
            "provider": "autodl",
            "sku_id": "autodl-4090d",
            "image_id": "blackwell-vla",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    params = bridge.plan_calls[0]["params"]
    assert bridge.plan_calls[0]["workflow"] == "rlinf_vla"
    assert params["launchMode"] == "project_backend"
    assert params["modelFamily"] == "pi0.5"
    assert "builtinTrainingProfile" not in params
    assert params["trainingMode"] == "co_training"
    assert params["coTrainingTargets"] == ["action_expert", "llm"]
    assert params["robotAdapter"] == "xlerobot"
    assert params["imageProfile"] == "blackwell"
    assert data["vlaPlan"]["workflow"] == "rlinf_vla"
    assert data["vlaPlan"]["deployabilityHints"]


def test_vla_rl_profiles_expose_policy_registry_capabilities(route_app):
    app, service = route_app
    register_vla_rl_routes(app, service)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/vla-rl/profiles")

    assert resp.status_code == 200
    data = resp.json()
    assert data["workflow"] == "vla_rl_backend"
    assert "rlinf_vla" in data["compatibleWorkflows"]
    assert "pi0" in data["supportedPolicyTypes"]
    assert "pi05" in data["supportedPolicyTypes"]
    assert "groot" in data["supportedPolicyTypes"]
    dm0_profile = next(item for item in data["profiles"] if item["id"] == "dexbotic_dm0_rlinf")
    assert dm0_profile["backendKind"] == "rlinf"
    assert dm0_profile["requiredParams"]
    rlinf_profile = next(item for item in data["profiles"] if item["id"] == "roboclaw_rlinf_backend")
    assert "workdir" in rlinf_profile["requiredParams"]
    grpo_profile = next(item for item in data["profiles"] if item["id"] == "roboclaw_grpo_backend")
    assert grpo_profile["algorithm"] == "grpo"
    assert grpo_profile["groupSize"] == 8
    assert grpo_profile["placementStrategy"] == "single_node"
    assert "configPath" in grpo_profile["requiredParams"]
    lerobot_profile = next(item for item in data["profiles"] if item["id"] == "roboclaw_lerobot_backend")
    assert lerobot_profile["backendKind"] == "lerobot"
    assert lerobot_profile["availableInPolicyRegistry"] is True
    backend_kinds = {item["backendKind"] for item in data["profiles"]}
    assert {"rlinf", "lerobot", "dexbotic", "custom"} <= backend_kinds
    assert data["backendKindExtensible"] is True
    assert "knownBackendKinds" not in data
    assert "knownBenchmarks" not in data
    assert "knownCapabilities" not in data
    rlinf_interface = data["backendInterfaces"]["rlinf"]
    assert rlinf_interface["workflow"] == "rlinf_vla"
    assert rlinf_interface["registryInjection"]["env"] == "RLINF_EXT_MODULE"
    assert "import rlinf" in rlinf_interface["preflightChecks"]
    assert "VLA_RL_CONTRACT_PATH" in rlinf_interface["envExports"]
    assert rlinf_interface["artifactContract"]["contractFile"] == "run_contract.json"
    assert rlinf_interface["algorithmToLauncherKind"]["grpo"] == "python_module"
    for backend_kind, import_check in (
        ("lerobot", "import lerobot"),
        ("dexbotic", "Dexbotic project path is importable"),
        ("custom", "import launcherModule"),
    ):
        interface = data["backendInterfaces"][backend_kind]
        assert interface["workflow"] == "vla_rl_backend"
        assert interface["registryInjection"]["env"] == "VLA_RL_BACKEND_EXT_MODULE"
        assert import_check in interface["preflightChecks"]
        assert interface["artifactContract"]["contractFile"] == "run_contract.json"
    assert data["backendInterfaceConfigurable"] is True
    assert "ROBOCLAW_VLA_BACKEND_INTERFACES_JSON" in data["backendInterfaceConfigSources"]


def test_vla_rl_playground_exposes_guided_training_flow(route_app):
    app, service = route_app
    register_vla_rl_routes(app, service)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.get("/api/vla-rl/playground")

    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "roboclaw_vla_training_playground/v1"
    assert data["entrypoints"]["plan"] == "/api/vla-rl/plan"
    assert data["entrypoints"]["runtimeMatch"] == "/api/train/runtime-match"
    assert data["entrypoints"]["start"] == "/api/train/cloud/start"
    stage_ids = [stage["id"] for stage in data["stages"]]
    assert stage_ids == ["intent", "plan", "runtime_match", "confirm", "execute", "review", "deployability"]
    assert "providerToken" in data["inputSchema"]["adminOnlyFields"]
    assert any("Do not start paid compute" in item for item in data["guardrails"])
    assert "rlinf" in data["backendInterfaces"]
    assert any(profile["id"] == "roboclaw_grpo_backend" for profile in data["profiles"])
    assert data["recommendedSmokeParams"]["params"]["builtinTrainingProfile"] == "roboclaw_grpo_backend"


def test_roboclaw_vla_rlinf_adapter_modules_are_importable():
    import roboclaw_vla.rl.adapters  # noqa: F401
    import roboclaw_vla.rl.evaluate  # noqa: F401
    import roboclaw_vla.rl.launcher  # noqa: F401
    from roboclaw_vla.rl import registry

    assert registry.register() is False


def test_roboclaw_grpo_hydra_template_uses_rlinf_style_sections():
    config_path = Path("roboclaw_vla/config/rl/libero_10_grpo_roboclaw.yaml")
    config = config_path.read_text(encoding="utf-8")

    for relative_path in (
        "env/libero_10.yaml",
        "model/roboclaw_pi0.yaml",
        "training_backend/fsdp.yaml",
    ):
        assert (config_path.parent / relative_path).is_file()
    assert "env/libero_10@env.train" in config
    assert "env/libero_10@env.eval" in config
    assert "model/roboclaw_pi0@actor.model" in config
    assert "training_backend/fsdp@actor.fsdp_config" in config
    assert "cluster:" in config
    assert "component_placement:" in config
    assert "actor,env,rollout: all" in config
    assert "model_path: path/to/roboclaw_pi0_checkpoint" in config
    assert "optim:" in config
    assert "enable_offload: true" in config
    model_config = (config_path.parent / "model" / "roboclaw_pi0.yaml").read_text(encoding="utf-8")
    fsdp_config = (config_path.parent / "training_backend" / "fsdp.yaml").read_text(encoding="utf-8")
    env_config = (config_path.parent / "env" / "libero_10.yaml").read_text(encoding="utf-8")
    assert "add_value_head: true" in model_config
    assert "precision: bfloat16" in model_config
    assert "torch_dtype" not in model_config
    assert "enable_gradient_accumulation: true" in fsdp_config
    assert "mixed_precision:" in fsdp_config
    assert "use_orig_params: false" in fsdp_config
    assert "total_num_envs: 16" in env_config
    assert "is_eval: true" in env_config


def test_vla_rl_profiles_merge_configured_backend_interfaces(route_app):
    app, service = route_app
    configured = {
        "mybackend": {
            "interfaceVersion": "vla-rl-backend/v1",
            "workflow": "vla_rl_backend",
            "requiredParams": ["repoUrl", "launcherModule", "artifactPath"],
            "preflightChecks": ["import mybackend"],
            "artifactContract": {"contractFile": "run_contract.json"},
        },
        "lerobot": {
            "preflightChecks": ["import lerobot", "custom dataset resolver"],
        },
    }

    with patch.dict("os.environ", {"ROBOCLAW_VLA_BACKEND_INTERFACES_JSON": json.dumps(configured)}, clear=False):
        register_vla_rl_routes(app, service)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/vla-rl/profiles")

    assert resp.status_code == 200
    data = resp.json()
    assert data["backendInterfaces"]["mybackend"]["preflightChecks"] == ["import mybackend"]
    assert data["backendInterfaces"]["lerobot"]["workflow"] == "vla_rl_backend"
    assert data["backendInterfaces"]["lerobot"]["preflightChecks"] == ["import lerobot", "custom dataset resolver"]


def test_vla_rl_plan_can_target_generic_backend_workflow(route_app):
    app, service = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_vla_rl_routes(app, service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/vla-rl/plan",
        json={
            "username": "pearl",
            "workflow": "vla_rl_backend",
            "message": "用 LeRobot backend 微调 SO-101 的 pi0 policy",
            "params": {"backendKind": "lerobot", "configName": "so101_pi0_lora"},
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert bridge.plan_calls[0]["workflow"] == "vla_rl_backend"
    assert bridge.plan_calls[0]["params"]["backendKind"] == "lerobot"
    assert data["vlaPlan"]["workflow"] == "vla_rl_backend"


@pytest.mark.parametrize(
    "backend_kind",
    [
        "lerobot",
        "dexbotic",
        "custom",
    ],
)
def test_vla_rl_plan_forwards_supported_backend_kinds(route_app, backend_kind):
    app, service = route_app
    bridge = StubBridge()

    with patch("roboclaw.training.service.EvoTrainBridge", return_value=bridge):
        register_vla_rl_routes(app, service)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/vla-rl/plan",
        json={
            "username": "pearl",
            "workflow": "vla_rl_backend",
            "message": f"use {backend_kind} backend",
            "params": {
                "backendKind": backend_kind,
                "configName": f"{backend_kind}_smoke",
                "launcherModule": f"project.{backend_kind}.launch",
            },
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert bridge.plan_calls[0]["workflow"] == "vla_rl_backend"
    assert bridge.plan_calls[0]["params"]["backendKind"] == backend_kind
    assert data["vlaPlan"]["workflow"] == "vla_rl_backend"


def test_vla_rl_artifact_review_reads_contract_and_metrics(route_app, tmp_path: Path):
    app, service = route_app
    register_vla_rl_routes(app, service)
    artifact_dir = tmp_path / "artifacts"
    checkpoint_dir = tmp_path / "checkpoint"
    artifact_dir.mkdir()
    checkpoint_dir.mkdir()
    metrics_path = artifact_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"success_rate": 0.82}), encoding="utf-8")
    contract_path = artifact_dir / "run_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "modelFamily": "pi0.5",
                "checkpointPath": str(checkpoint_dir),
                "artifactPath": str(artifact_dir),
                "metricPaths": [str(metrics_path)],
                "successMetric": "success_rate",
                "robotEmbodiment": "xlerobot",
                "observationSchema": "rgb+state",
                "actionSchema": "joint_delta",
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/vla-rl/artifact-review", json={"contract_path": str(contract_path)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["successValue"] == 0.82
    assert data["checkpointExists"] is True
    assert data["artifactExists"] is True
    assert data["deployability"]["deployable"] is True


def test_vla_rl_deployability_gate_rejects_schema_mismatch(route_app):
    app, service = route_app
    register_vla_rl_routes(app, service)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post(
        "/api/vla-rl/deployability",
        json={
            "contract": {
                "modelFamily": "gr00tn1",
                "checkpointPath": "/tmp/checkpoint",
                "artifactPath": "/tmp/artifacts",
                "robotEmbodiment": "so-101",
                "observationSchema": "rgb",
                "actionSchema": "joint_delta",
            },
            "robot_embodiment": "xlerobot",
            "observation_schema": "rgb",
            "action_schema": "joint_delta",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["deployable"] is False
    assert any("robotEmbodiment mismatch" in item for item in data["blockingReasons"])
