"""RLinf capability catalog and launch contract helpers."""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any, Mapping


RLINF_REPO_URL = "https://github.com/RLinf/RLinf.git"
RLINF_WORKDIR = "/root/autodl-tmp/RLinf"
ROBOCLAW_RLINF_EXT_MODULE = "roboclaw.training.rlinf_registry_hook"
_log = logging.getLogger(__name__)

_CONFIG_ROOTS = {
    "embodiment": Path("examples/embodiment/config"),
    "sft": Path("examples/sft/config"),
}

_ALGORITHMS = (
    "crossq",
    "dagger",
    "dsrl",
    "grpo",
    "iql",
    "nft",
    "ppo",
    "rlpd",
    "sac",
    "sft",
)

_MODEL_TOKENS = (
    ("dexbotic_dm0", "dexbotic_dm0"),
    ("dexbotic_pi0", "dexbotic_pi0"),
    ("lingbotvla", "lingbotvla"),
    ("openvlaoft", "openvla-oft"),
    ("openvla", "openvla"),
    ("openpi_pi05", "openpi_pi05"),
    ("openpi", "openpi"),
    ("dreamzero", "dreamzero"),
    ("starvla", "starvla"),
    ("gr00t", "gr00t"),
    ("qwentrend", "qwentrend"),
    ("resnet", "resnet"),
    ("flow", "flow"),
    ("cnn", "cnn"),
    ("mlp", "mlp"),
)

_BENCHMARK_PREFIXES = (
    "behavior",
    "calvin",
    "d4rl",
    "dosw1",
    "embodichain",
    "frankasim",
    "gsenv",
    "isaaclab",
    "libero",
    "maniskill",
    "metaworld",
    "opensora",
    "realworld",
    "robocasa",
    "robotwin",
    "roboverse",
    "wan",
)

_DEFAULT_MODEL_SOURCES: dict[tuple[str, str], dict[str, str]] = {
    (
        "libero",
        "openvla-oft",
    ): {
        "sourceType": "public_model_repo",
        "modelFamily": "openvla-oft",
        "uri": "hf://moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "format": "huggingface_transformers",
    },
}


def default_rlinf_repo_path() -> Path | None:
    """Return the first configured RLinf checkout path, if present."""

    candidates = [
        os.environ.get("ROBOCLAW_RLINF_REPO_PATH", "").strip(),
        os.environ.get("RLINF_REPO_PATH", "").strip(),
        RLINF_WORKDIR,
        "/opt/RLinf",
        "/private/tmp/RLinf-inspect",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "examples").exists():
            return path
    return None


def set_rlinf_ext_module() -> str:
    """Let RLinf worker processes load RoboClaw model registrations."""

    os.environ.setdefault("RLINF_EXT_MODULE", ROBOCLAW_RLINF_EXT_MODULE)
    return os.environ["RLINF_EXT_MODULE"]


def register_all_rlinf_models() -> bool:
    """Register RoboClaw-owned RLinf model adapters inside a worker process."""

    from roboclaw_vla.rl.registry import register_all

    return register_all()


def discover_rlinf_catalog(repo_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Discover RLinf example configs from a checkout.

    The catalog intentionally uses file names and launch scripts instead of
    importing RLinf. This keeps the control plane lightweight and lets the cloud
    runtime own the actual RLinf dependency stack.
    """

    root = Path(repo_path).expanduser() if repo_path else default_rlinf_repo_path()
    configs: list[dict[str, Any]] = []
    if root is None or not (root / "examples").exists():
        _log.warning(
            "RLinf repo not found at %s; no training recipes available. "
            "Clone https://github.com/RLinf/RLinf.git or set ROBOCLAW_RLINF_REPO_PATH.",
            root or repo_path or RLINF_WORKDIR,
        )
        return {
            "repoUrl": RLINF_REPO_URL,
            "repoPath": str(root) if root else "",
            "configured": False,
            "configCount": 0,
            "benchmarks": [],
            "algorithms": [],
            "modelFamilies": [],
            "domains": sorted(_CONFIG_ROOTS),
            "scripts": {},
            "configs": [],
        }
    if root is not None:
        for domain, rel_root in _CONFIG_ROOTS.items():
            config_root = root / rel_root
            if not config_root.exists():
                continue
            for path in sorted(config_root.glob("*.yaml")):
                configs.append(infer_rlinf_config(path.stem, domain=domain, config_path=str(rel_root / path.name)))

    benchmarks = sorted({str(item.get("benchmark")) for item in configs if item.get("benchmark")})
    algorithms = sorted({str(item.get("algorithm")) for item in configs if item.get("algorithm")})
    models = sorted({str(item.get("modelFamily")) for item in configs if item.get("modelFamily")})
    return {
        "repoUrl": RLINF_REPO_URL,
        "repoPath": str(root) if root else "",
        "configured": root is not None,
        "configCount": len(configs),
        "benchmarks": benchmarks,
        "algorithms": algorithms,
        "modelFamilies": models,
        "domains": sorted(_CONFIG_ROOTS),
        "scripts": {
            "embodimentTrain": "examples/embodiment/run_embodiment.sh",
            "embodimentEval": "examples/embodiment/eval_embodiment.sh",
            "offlineRl": "examples/embodiment/run_offline_rl.sh",
            "realworldTrain": "examples/embodiment/run_realworld.sh",
            "realworldEval": "examples/embodiment/run_realworld_eval.sh",
            "vlaSft": "examples/sft/run_vla_sft.sh",
        },
        "configs": configs,
    }


def match_rlinf_config_name(message: str, *, catalog: Mapping[str, Any] | None = None) -> str:
    """Find an RLinf config name explicitly mentioned in text."""

    text = message.lower()
    source = catalog or discover_rlinf_catalog()
    configs = source.get("configs") if isinstance(source, Mapping) else []
    names = sorted(
        (str(item.get("configName") or "").lower() for item in configs if isinstance(item, Mapping)),
        key=len,
        reverse=True,
    )
    for name in names:
        if name and name in text:
            return name
    match = re.search(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}\b", text)
    return match.group(0) if match else ""


def infer_rlinf_config(config_name: str, *, domain: str = "embodiment", config_path: str = "") -> dict[str, Any]:
    """Infer launch metadata from an RLinf config file name."""

    name = config_name.removesuffix(".yaml")
    benchmark = _infer_benchmark(name)
    algorithm = _infer_algorithm(name)
    model_family = _infer_model_family(name)
    is_eval = name.endswith("_eval") or "_eval_" in name
    if domain == "sft":
        training_mode = "supervised_finetune"
        script_path = "examples/sft/run_vla_sft.sh"
        entrypoint = "examples/sft/train_vla_sft.py"
    elif "collect_data" in name:
        training_mode = "data_collection"
        script_path = "examples/embodiment/collect_data.sh"
        entrypoint = "examples/embodiment/collect_real_data.py"
    elif benchmark == "d4rl":
        training_mode = "offline_rl"
        script_path = "examples/embodiment/run_offline_rl.sh"
        entrypoint = "examples/embodiment/train_offline_rl.py"
    elif benchmark == "realworld":
        training_mode = "real_robot_eval" if is_eval else "real_robot_rl"
        script_path = "examples/embodiment/run_realworld_eval.sh" if is_eval else "examples/embodiment/run_realworld.sh"
        entrypoint = "examples/embodiment/eval_embodied_agent.py" if is_eval else "examples/embodiment/train_embodied_agent.py"
    else:
        training_mode = "simulation_eval" if is_eval else "rl_post_train"
        script_path = "examples/embodiment/eval_embodiment.sh" if is_eval else "examples/embodiment/run_embodiment.sh"
        entrypoint = "examples/embodiment/eval_embodied_agent.py" if is_eval else "examples/embodiment/train_embodied_agent.py"
    return {
        "configName": name,
        "configPath": config_path or str(_CONFIG_ROOTS.get(domain, _CONFIG_ROOTS["embodiment"]) / f"{name}.yaml"),
        "domain": domain,
        "benchmark": benchmark,
        "algorithm": algorithm,
        "modelFamily": model_family,
        "trainingMode": training_mode,
        "scriptPath": script_path,
        "entrypoint": entrypoint,
        "launcherKind": "python_script",
        "robotPlatform": _robot_platform_for_benchmark(benchmark),
        "liberoTypeSupport": ["standard", "pro", "plus"] if benchmark == "libero" else [],
        "status": "discovered",
    }


def apply_rlinf_config_contract(
    params: Mapping[str, Any],
    *,
    config_name: str,
    message: str = "",
    libero_type: str = "",
) -> dict[str, Any]:
    """Apply a discovered RLinf config as an Evo Studio executable contract."""

    catalog = discover_rlinf_catalog()
    entry = _find_catalog_entry(config_name, catalog) or infer_rlinf_config(config_name)
    config = str(entry["configName"])
    inferred_libero_type = libero_type or _infer_libero_type(message)
    artifact_path = str(params.get("artifactPath") or f"{RLINF_WORKDIR}/outputs/{config}")
    if inferred_libero_type and entry.get("benchmark") == "libero":
        artifact_path = str(params.get("artifactPath") or f"{RLINF_WORKDIR}/outputs/{config}_{inferred_libero_type}")
    direct_eval = str(entry.get("trainingMode") or "") in {"simulation_eval", "real_robot_eval"}
    entrypoint = str(entry.get("entrypoint") or "")
    script_path = entrypoint if direct_eval and entrypoint else str(entry.get("scriptPath") or "examples/embodiment/run_embodiment.sh")

    enriched = dict(params)
    enriched.update(
        {
            "backendKind": "rlinf",
            "workflow": "rlinf_vla",
            "repoUrl": RLINF_REPO_URL,
            "branch": "main",
            "workdir": RLINF_WORKDIR,
            "launchMode": "project_backend",
            "launcherKind": entry.get("launcherKind") or "python_script",
            "scriptPath": script_path,
            "entrypoint": entrypoint,
            "configName": config,
            "configPath": entry.get("configPath") or f"examples/embodiment/config/{config}.yaml",
            "artifactPath": artifact_path,
            "contractPath": f"{artifact_path}/run_contract.json",
            "metricPaths": [f"{artifact_path}/metrics.json", f"{artifact_path}/rollout_summary.json"],
            "resultFiles": ["run_contract.json", "metrics.json", "rollout_summary.json", "logs"],
            "successMetric": "success_rate",
            "backendInterface": rlinf_shell_backend_interface(
                libero_type=inferred_libero_type,
                robot_platform=str(entry.get("robotPlatform") or ""),
                direct_python=direct_eval,
            ),
            "rlinfConfig": entry,
            "trainingContract": {
                "interfaceKind": "rlinf_runner",
                "framework": "rlinf",
                "sources": {
                    "code": {"uri": RLINF_REPO_URL},
                    "dataset": enriched.get("datasetSource", {}),
                    "model": enriched.get("modelSource", {}),
                },
                "runner": "EmbodiedRunner" if entry.get("domain") == "embodiment" else "SFT",
                "algorithm": {"name": entry.get("algorithm") or enriched.get("algorithm") or "auto"},
                "env": {
                    "benchmark": entry.get("benchmark") or enriched.get("benchmark") or "",
                    "liberoType": inferred_libero_type,
                },
                "runtime": {"placementStrategy": enriched.get("placementStrategy", "single_node")},
                "artifacts": {"path": artifact_path},
            },
        }
    )
    for key in ("benchmark", "algorithm", "modelFamily", "trainingMode"):
        if entry.get(key):
            enriched[key] = entry[key]
    if entry.get("benchmark") == "libero":
        enriched.setdefault(
            "datasetSource",
            {
                "sourceType": "public_reference",
                "datasetId": "libero",
                "uri": "hf://HuggingFaceVLA/libero",
                "format": "libero",
                "benchmark": "libero",
            },
        )
        enriched.setdefault("datasetFormat", "libero")
    model_source = enriched.get("modelSource") if isinstance(enriched.get("modelSource"), dict) else {}
    model_uri = str(model_source.get("uri") or model_source.get("checkpoint") or "").strip()
    model_source_type = str(model_source.get("sourceType") or "").strip().lower()
    model_family = str(model_source.get("modelFamily") or "").strip().lower()
    entry_model_family = str(entry.get("modelFamily") or "").strip()
    if entry_model_family and (
        not model_uri
        or model_uri.lower() in {"auto", "unknown"}
        or (model_source_type == "builtin_policy" and model_family in {"", "auto", "unknown"})
    ):
        default_source = _default_model_source(entry.get("benchmark"), entry_model_family)
        if default_source:
            enriched["modelSource"] = dict(default_source)
            enriched["checkpointPath"] = default_source["uri"]
            enriched["checkpointFormat"] = default_source["format"]
        else:
            enriched["modelSource"] = {
                "sourceType": "rlinf_config_default",
                "modelFamily": entry_model_family,
                "format": "rlinf_config",
            }
            enriched.setdefault("checkpointFormat", "rlinf_config")
        _sync_model_source_contract(enriched)
    return enriched


def _default_model_source(benchmark: object, model_family: object) -> dict[str, str]:
    key = (str(benchmark or "").strip().lower(), str(model_family or "").strip().lower())
    return dict(_DEFAULT_MODEL_SOURCES.get(key) or {})


def _sync_model_source_contract(params: dict[str, Any]) -> None:
    model_source = params.get("modelSource") if isinstance(params.get("modelSource"), dict) else {}
    if not model_source:
        return
    source_contract = params.get("sourceContract")
    if isinstance(source_contract, dict):
        source_contract = dict(source_contract)
        source_contract["modelSource"] = dict(model_source)
        source_contract["modelSourceKind"] = str(model_source.get("sourceType") or "")
        source_contract["checkpointFormat"] = str(model_source.get("format") or source_contract.get("checkpointFormat") or "")
        params["sourceContract"] = source_contract
    training_contract = params.get("trainingContract")
    if isinstance(training_contract, dict):
        training_contract = dict(training_contract)
        sources = training_contract.get("sources")
        if isinstance(sources, dict):
            sources = dict(sources)
            sources["model"] = dict(model_source)
            training_contract["sources"] = sources
        params["trainingContract"] = training_contract


def rlinf_shell_backend_interface(
    *,
    libero_type: str = "",
    robot_platform: str = "",
    direct_python: bool = False,
) -> dict[str, Any]:
    env_exports: dict[str, str] = {
        "VLA_RL_CONTRACT_PATH": "contractPath",
        "VLA_RL_MODEL_FAMILY": "modelFamily",
        "VLA_RL_TRAINING_MODE": "trainingMode",
    }
    if libero_type:
        env_exports["LIBERO_TYPE"] = f"literal:{libero_type}"
    if robot_platform:
        env_exports["ROBOT_PLATFORM"] = f"literal:{robot_platform}"
    if direct_python:
        launcher_command = (
            "python {entrypoint} --config-path config "
            "--config-name {configName} runner.logger.log_path={artifactPath}/logs"
        )
    else:
        launcher_command = "bash {scriptPath} {configName}"
    if robot_platform and not direct_python:
        launcher_command = f"{launcher_command} {robot_platform}"
    return {
        "interfaceVersion": "vla-rl-backend/v1",
        "backendKind": "rlinf",
        "workflow": "rlinf_vla",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_script"],
        "requiredParams": ["repoUrl", "workdir", "scriptPath", "configName", "artifactPath"],
        "registryInjection": {"field": "rlinfExtModule", "env": "RLINF_EXT_MODULE", "preflightImport": False},
        "envExports": env_exports,
        "preflightImports": ["rlinf"],
        "preflightCommands": [
            "test -f {scriptPath}",
            "test -f {configPath}",
        ],
        "usePreflightCommands": True,
        "useLauncherContract": True,
        "launcherContract": {
            "python_script": launcher_command,
        },
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "modelFamily", "artifactPath", "metricPaths"],
            "successMetricField": "success_rate",
        },
    }


def _find_catalog_entry(config_name: str, catalog: Mapping[str, Any]) -> dict[str, Any] | None:
    for item in catalog.get("configs") or []:
        if isinstance(item, Mapping) and str(item.get("configName") or "") == config_name:
            return dict(item)
    return None


def _infer_benchmark(name: str) -> str:
    for prefix in _BENCHMARK_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return name.split("_", 1)[0] if "_" in name else ""


def _infer_algorithm(name: str) -> str:
    tokens = set(name.split("_"))
    for algorithm in _ALGORITHMS:
        if algorithm in tokens:
            return algorithm
    return "sft" if "_sft_" in name or name.endswith("_sft") else ""


def _infer_model_family(name: str) -> str:
    for token, family in _MODEL_TOKENS:
        if token in name:
            return family
    return ""


def _robot_platform_for_benchmark(benchmark: str) -> str:
    if benchmark == "libero":
        return "LIBERO"
    if benchmark == "robotwin":
        return "ROBOTWIN"
    if benchmark == "realworld":
        return "REALWORLD"
    return ""


def _infer_libero_type(message: str) -> str:
    lowered = message.lower()
    if any(token in lowered for token in ("libero plus", "libero-plus", "liberoplus")):
        return "plus"
    if any(token in lowered for token in ("libero pro", "libero-pro", "liberopro")):
        return "pro"
    return ""
