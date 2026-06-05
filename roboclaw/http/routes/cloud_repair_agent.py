"""Autonomous same-runtime cloud repair decisions.

This module is intentionally separate from cloud_supervisor.py: the supervisor
owns job state, while this repair agent owns diagnosis-to-command decisions,
LLM fallback, command safety checks, and audit logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any

from roboclaw.embodied.embodiment.manifest import helpers as manifest_helpers

_log = logging.getLogger(__name__)

_TENSORFLOW_PYTHON_CONFLICT_RE = re.compile(
    r"(?:could not find a version that satisfies the requirement|no matching distribution found for)\s+"
    r"tensorflow(?:-cpu|-gpu)?(?:[<>=!~]=?[^\s),;]*)?"
    r"|package\s+['\"]tensorflow(?:-cpu|-gpu)?['\"]\s+requires a different python",
    flags=re.IGNORECASE,
)


def _stage_failure_name(log_text: str) -> str:
    match = re.search(r"__EVO_STAGE_FAILED__=([A-Za-z0-9_.-]+)", log_text)
    return match.group(1) if match else ""


def _trim_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _extract_missing_python_module(log_text: str) -> str:
    patterns = (
        r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]([^'\"]+)['\"]",
        r"\bNo module named ['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, log_text, flags=re.IGNORECASE)
        if match:
            module = match.group(1).strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", module):
                return module
    return ""


def _python_module_to_pip_package(module: str) -> str:
    module = module.strip()
    if not module:
        return ""
    top_level = module.split(".", 1)[0]
    blocked = {"experiments", "roboclaw", "roboclaw_vla", "train", "scripts"}
    if top_level in blocked:
        return ""
    module_packages = {
        "PIL": "pillow",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
        "ruamel": "ruamel.yaml",
    }
    return module_packages.get(top_level, top_level)


def _openvla_oft_runtime_repair_command() -> str:
    return (
        "python -m pip install --retries 5 --timeout 60 --upgrade "
        "'transformers==4.40.2' 'peft==0.11.1' 'sentencepiece>=0.2.0' 'tokenizers==0.19.1' "
        "&& echo __EVO_OPENVLA_OFT_RUNTIME_REPAIR__=1"
    )


def _libero_runtime_repair_command() -> str:
    return (
        "python -m pip install --retries 5 --timeout 60 --upgrade "
        "'gym==0.26.2' 'gymnasium>=0.29,<1.0' libero robosuite mujoco "
        "&& echo __EVO_LIBERO_RUNTIME_REPAIR__=1"
    )


def _torch_cuda_repair_command() -> str:
    return (
        "python - <<'PY'\n"
        "import re, shutil, subprocess, sys\n"
        "def out(args):\n"
        "    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)\n"
        "nvidia_smi = shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi'\n"
        "text = out([nvidia_smi])\n"
        "m = re.search(r'CUDA Version:\\s*([0-9.]+)', text)\n"
        "version = tuple(int(x) for x in (m.group(1) if m else '0.0').split('.')[:2])\n"
        "tag = 'cu128' if version >= (12,8) else 'cu126' if version >= (12,6) else 'cu124' if version >= (12,4) else 'cu121' if version >= (12,1) else 'cu118'\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--progress-bar', 'off', '--timeout', '120', '--retries', '5', 'torch', 'torchvision', 'torchaudio', '--index-url', f'https://download.pytorch.org/whl/{tag}'])\n"
        "print(f'__EVO_TORCH_CUDA_REPAIR__={tag}')\n"
        "PY"
    )


def _tensorflow_python_compat_repair_command(log_text: str) -> str:
    if not _TENSORFLOW_PYTHON_CONFLICT_RE.search(log_text):
        return ""
    return (
        "python - <<'PY'\n"
        "import re, subprocess, sys\n"
        "from pathlib import Path\n"
        "if sys.version_info >= (3, 12):\n"
        "    tensorflow_spec = 'tensorflow>=2.16,<2.22'\n"
        "elif sys.version_info >= (3, 10):\n"
        "    tensorflow_spec = 'tensorflow>=2.15,<2.22'\n"
        "else:\n"
        "    raise SystemExit(f'Unsupported Python for TensorFlow repair: {sys.version.split()[0]}')\n"
        "dependency_pattern = re.compile(\n"
        "    r\"tensorflow(?:-cpu|-gpu)?\\s*(?:==|~=|>=|<=|>|<|!=)\\s*[^'\\\"\\],;\\s]+\",\n"
        "    flags=re.IGNORECASE,\n"
        ")\n"
        "targets = [Path('pyproject.toml'), Path('setup.py'), Path('setup.cfg')]\n"
        "targets.extend(Path('.').glob('requirements*.txt'))\n"
        "patched = []\n"
        "for path in targets:\n"
        "    if not path.exists() or not path.is_file():\n"
        "        continue\n"
        "    text = path.read_text(encoding='utf-8')\n"
        "    updated = dependency_pattern.sub(tensorflow_spec, text)\n"
        "    if updated != text:\n"
        "        path.write_text(updated, encoding='utf-8')\n"
        "        patched.append(str(path))\n"
        "subprocess.check_call([\n"
        "    sys.executable,\n"
        "    '-m',\n"
        "    'pip',\n"
        "    'install',\n"
        "    '--retries',\n"
        "    '5',\n"
        "    '--timeout',\n"
        "    '60',\n"
        "    '--upgrade',\n"
        "    tensorflow_spec,\n"
        "])\n"
        "print(f'__EVO_PYTHON_DEPENDENCY_REPAIR__=tensorflow_python_compat:{tensorflow_spec}')\n"
        "if patched:\n"
        "    print('__EVO_TENSORFLOW_REQUIREMENTS_PATCHED__=' + ','.join(patched))\n"
        "PY"
    )


def _dependency_conflict_repair_command(log_text: str) -> str:
    tensorflow_command = _tensorflow_python_compat_repair_command(log_text)
    if tensorflow_command:
        return tensorflow_command
    lowered = log_text.lower()
    rules: tuple[tuple[tuple[str, ...], str, str], ...] = (
        (
            ("messagefactory", "getprototype"),
            "python -m pip install --retries 5 --timeout 60 --upgrade 'protobuf>=3.20.3,<5'",
            "__EVO_PYTHON_DEPENDENCY_REPAIR__=protobuf_compat",
        ),
        (
            ("compiled using numpy 1.x", "numpy 2"),
            "python -m pip install --retries 5 --timeout 60 --upgrade 'numpy<2'",
            "__EVO_PYTHON_DEPENDENCY_REPAIR__=numpy_abi",
        ),
    )
    for tokens, command, marker in rules:
        if all(token in lowered for token in tokens):
            return f"{command} && echo {marker}"
    return ""


def _python_dependency_repair_command(log_text: str) -> str:
    conflict_command = _dependency_conflict_repair_command(log_text)
    if conflict_command:
        return conflict_command
    module = _extract_missing_python_module(log_text)
    package = _python_module_to_pip_package(module)
    if package:
        return f'python -m pip install --retries 5 --timeout 60 --upgrade "{package}"'
    return ""


def _unknown_failure_repair_command(log_text: str) -> str:
    configured = os.environ.get("EVO_STUDIO_UNKNOWN_FAILURE_REPAIR_COMMAND", "").strip()
    if configured:
        return configured
    summary = _trim_text(log_text, limit=900).replace("'", "'\"'\"'")
    return f"printf '%s\\n' '{summary}' > /tmp/evo_unknown_failure.log"


def _normalize_repair_command(command: str) -> str:
    command = command.strip()
    if command.startswith("```"):
        command = command.strip("`").split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
    if "\n" in command:
        command = " && ".join(line.strip() for line in command.splitlines() if line.strip())
    return command.strip()


def _repair_command_is_safe(command: str) -> bool:
    command = command.strip()
    if not command:
        return False
    if re.match(r"^(?:&&|\|\||[;|&])", command) or re.search(r"(?:&&|\|\||[;|&])\s*$", command):
        return False
    first_word = command.split(None, 1)[0].strip()
    allowed_first_words = {
        "python",
        "python3",
        "pip",
        "pip3",
        "conda",
        "mamba",
        "micromamba",
        "uv",
        "export",
        "unset",
        "printf",
        "echo",
        "test",
        "mkdir",
        "cd",
        "git",
        "sed",
        "perl",
        "find",
        "touch",
        "chmod",
        "ln",
        "cp",
        "mv",
    }
    if "=" not in first_word and first_word not in allowed_first_words:
        return False
    lowered = command.lower()
    blocked = (
        "rm -rf /",
        "mkfs",
        "shutdown",
        "reboot",
        "poweroff",
        "curl | sh",
        "wget | sh",
        "chmod -r 777 /",
        "dd if=",
    )
    return not any(token in lowered for token in blocked)


def repair_command_from_llm_content(content: str | None) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        text = str(parsed.get("command") or "").strip()
    command = _normalize_repair_command(text)
    if not _repair_command_is_safe(command):
        _log.warning("Rejected unsafe or empty LLM repair command: %r", command[:160])
        return ""
    return command


async def _configured_provider_unknown_failure_repair_command(log_text: str, llm_provider: Any | None) -> str:
    if llm_provider is None:
        return ""
    response = await llm_provider.chat_with_retry(
        [
            {
                "role": "system",
                "content": (
                    "You are RoboClaw's cloud training repair agent. Inspect the failed ML training log and "
                    "return one safe same-runtime POSIX shell command as JSON: {\"command\":\"...\"}. "
                    "Only fix Python packages, conda/pip environment, cache markers, or runtime env vars. "
                    "Do not change machines, secrets, account credentials, budget, datasets, checkpoints, "
                    "or delete user data."
                ),
            },
            {"role": "user", "content": _trim_text(log_text, limit=6000)},
        ],
        max_tokens=700,
        temperature=0,
    )
    if response.finish_reason == "error":
        _log.warning("Configured LLM repair provider failed: %s", response.content)
        return ""
    return repair_command_from_llm_content(response.content)


def _env_llm_unknown_failure_repair_command(log_text: str) -> str:
    base_url = os.environ.get("EVO_STUDIO_REPAIR_LLM_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("EVO_STUDIO_REPAIR_LLM_MODEL", "").strip()
    if not base_url or not model:
        return ""
    api_key = os.environ.get("EVO_STUDIO_REPAIR_LLM_API_KEY", "").strip()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You repair failed Linux cloud ML training jobs. Return only one safe POSIX shell command "
                    "that can run inside the existing project environment. Do not change machines, secrets, "
                    "budgets, or delete user data. Prefer pip/conda/env fixes and verification."
                ),
            },
            {"role": "user", "content": _trim_text(log_text, limit=4000)},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(
        request,
        timeout=float(os.environ.get("EVO_STUDIO_REPAIR_LLM_TIMEOUT", "20") or "20"),
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return repair_command_from_llm_content(str(payload["choices"][0]["message"]["content"]))


async def llm_unknown_failure_repair_command(log_text: str, llm_provider: Any | None = None) -> str:
    command = await _configured_provider_unknown_failure_repair_command(log_text, llm_provider)
    if command:
        return command
    return await asyncio.to_thread(_env_llm_unknown_failure_repair_command, log_text)


async def _unknown_failure_repair_command_async(log_text: str, llm_provider: Any | None = None) -> str:
    configured = os.environ.get("EVO_STUDIO_UNKNOWN_FAILURE_REPAIR_COMMAND", "").strip()
    if configured:
        return configured
    generated = await llm_unknown_failure_repair_command(log_text, llm_provider)
    if generated:
        return generated
    return _unknown_failure_repair_command(log_text)


def inject_repair_commands(params: dict[str, Any], remediation: dict[str, Any], log_text: str) -> dict[str, Any]:
    return _inject_repair_commands_with_commands(params, remediation, log_text, unknown_repair_command=None)


async def inject_repair_commands_async(
    params: dict[str, Any],
    remediation: dict[str, Any],
    log_text: str,
    *,
    llm_provider: Any | None = None,
) -> dict[str, Any]:
    code = str(remediation.get("code") or "").strip().upper()
    unknown_command = ""
    if code in {"PYTHON_DEPENDENCY_RESOLUTION_FAILED", "UNKNOWN_CLOUD_FAILURE", "CLOUD_STAGE_FAILED"}:
        unknown_command = await _unknown_failure_repair_command_async(log_text, llm_provider)
    return _inject_repair_commands_with_commands(
        params,
        remediation,
        log_text,
        unknown_repair_command=unknown_command,
    )


def _inject_repair_commands_with_commands(
    params: dict[str, Any],
    remediation: dict[str, Any],
    log_text: str,
    *,
    unknown_repair_command: str | None,
) -> dict[str, Any]:
    repaired = dict(params)
    code = str(remediation.get("code") or "").strip().upper()
    auto_repair = remediation.get("autoRepair") if isinstance(remediation.get("autoRepair"), dict) else {}
    strategy = str(auto_repair.get("strategy") or remediation.get("strategy") or code or "").strip()
    stage = str(remediation.get("stage") or _stage_failure_name(log_text) or "").strip()
    lowered = log_text.lower()
    commands: list[str] = []

    if code in {"PYTHON_IMPORT_MISSING", "PYTHON_MODULE_MISSING"}:
        if "__evo_openvla_oft_runtime_unavailable__" in lowered or "openvla" in lowered:
            commands.append(_openvla_oft_runtime_repair_command())
        elif "libero" in lowered:
            commands.append(_libero_runtime_repair_command())
        else:
            commands.append(_python_dependency_repair_command(log_text))
    elif code == "LIBERO_EGL_CONTEXT_FAILED" or (
        "egl_not_initialized" in lowered or "eglerror" in lowered or "eglmakecurrent" in lowered
    ):
        repaired["mujocoGl"] = "egl"
        repaired["pyopenglPlatform"] = "egl"
        repaired["eglDeviceId"] = "0"
        strategy = strategy or "configure_headless_egl_and_retry"
    elif code == "CLOUD_GPU_UNAVAILABLE" and (
        "too old" in lowered
        or "cuda driver" in lowered
        or "compiled with your version" in lowered
        or "torch_cuda" in lowered
        or "cuda_version_mismatch" in lowered
    ):
        commands.append(_torch_cuda_repair_command())
        strategy = strategy or "reinstall_pytorch_for_driver_cuda_version_and_retry"
    elif code == "PYTHON_DEPENDENCY_RESOLUTION_FAILED":
        commands.append(_python_dependency_repair_command(log_text) or unknown_repair_command or "")
    elif code == "CLOUD_STAGE_TERMINATED":
        if stage:
            repaired["resumeFromStage"] = stage
    elif code == "CLOUD_WORKDIR_MISSING":
        repaired["forceSkipStageCache"] = True
        repaired["forceRepairBootstrap"] = True
        repaired["resumeFromStage"] = "prepare_code"
        repaired["skipPrepareCode"] = False
        strategy = strategy or "rerun_prepare_code_on_same_runtime"
    elif code in {"UNKNOWN_CLOUD_FAILURE", "CLOUD_STAGE_FAILED"}:
        commands.append(unknown_repair_command or _unknown_failure_repair_command(log_text))

    if commands:
        existing = (
            [str(item).strip() for item in repaired.get("repairBootstrapCommands", []) if str(item).strip()]
            if isinstance(repaired.get("repairBootstrapCommands"), list)
            else []
        )
        seen = set(existing)
        for command in commands:
            command = command.strip()
            if command and command not in seen:
                existing.append(command)
                seen.add(command)
        repaired["repairBootstrapCommands"] = existing
        repaired["forceRepairBootstrap"] = True
        repaired["forceSkipStageCache"] = True

    if strategy:
        repaired["repairStrategy"] = strategy
    return repaired


def repair_bootstrap_commands_for_failure(log_text: str, remediation: dict[str, Any]) -> list[str]:
    auto_repair = remediation.get("autoRepair") if isinstance(remediation, dict) else {}
    code = str(remediation.get("code") or "").strip() if isinstance(remediation, dict) else ""
    strategy = str(auto_repair.get("strategy") or "").strip() if isinstance(auto_repair, dict) else ""
    if code not in {"PYTHON_MODULE_MISSING", "PYTHON_IMPORT_MISSING"} and strategy != "install_missing_dependency_and_retry":
        return []
    lowered = log_text.lower()
    if (
        "__evo_openvla_oft_runtime_unavailable__" in lowered
        and "pretrainedmodel" in lowered
        and ("could not import module" in lowered or "modulenotfounderror" in lowered or "importerror" in lowered)
    ):
        return ['python -m pip install --upgrade "transformers==4.40.2"']
    module = _extract_missing_python_module(log_text)
    package = _python_module_to_pip_package(module)
    if not package:
        return []
    return [f'python -m pip install --upgrade "{package}"']


def _audit_path() -> str:
    configured = os.environ.get("EVO_STUDIO_CLOUD_REPAIR_AUDIT_FILE", "").strip()
    if configured:
        return configured
    return str(manifest_helpers.get_roboclaw_home() / "workspace" / "embodied" / "cloud_repair_agent.jsonl")


def record_repair_agent_event(event: dict[str, Any]) -> None:
    payload = {
        "kind": "evo_studio_cloud_repair_agent_event/v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    path = _audit_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        _log.warning("Could not write cloud repair agent audit event: %s", exc)
