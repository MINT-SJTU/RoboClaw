"""Local SSH runtime binding helpers for cloud training development deployments."""

from __future__ import annotations

import os
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

def _local_runtime_bind_enabled() -> bool:
    value = os.environ.get("EVO_STUDIO_ENABLE_LOCAL_RUNTIME_BIND", "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get("EVO_STUDIO_AUTH_MODE", "dev").strip().lower() == "dev"
def _parse_ssh_command(command: str) -> tuple[str, str, str]:
    text = command.strip()
    if not text:
        raise ValueError("ssh command is required")
    parts = shlex.split(text)
    if not parts or parts[0] != "ssh":
        raise ValueError("Cannot parse SSH command. Expected: ssh -p <port> <user>@<host>")
    port = ""
    target = ""
    index = 1
    while index < len(parts):
        item = parts[index]
        if item == "-p" and index + 1 < len(parts):
            port = parts[index + 1]
            index += 2
            continue
        if item.startswith("-"):
            index += 2 if index + 1 < len(parts) and not parts[index + 1].startswith("-") else 1
            continue
        target = item
        index += 1
    if not port or "@" not in target:
        raise ValueError("Cannot parse SSH command. Expected: ssh -p <port> <user>@<host>")
    user, host = target.split("@", 1)
    if not port.isdigit() or not user.strip() or not host.strip():
        raise ValueError("Cannot parse SSH command. Expected: ssh -p <port> <user>@<host>")
    return host.strip(), port.strip(), user.strip()
def _set_env_export(text: str, key: str, value: str) -> str:
    line = f"export {key}={shlex.quote(value)}"
    pattern = re.compile(rf"^export\s+{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(text):
        replaced = False
        lines: list[str] = []
        for existing in text.splitlines():
            if pattern.match(existing):
                if not replaced:
                    lines.append(line)
                    replaced = True
                continue
            lines.append(existing)
        return "\n".join(lines).rstrip() + "\n"
    suffix = "\n" if text.endswith("\n") else "\n\n"
    return f"{text}{suffix}{line}\n"
def _ssh_runtime_env_path() -> Path:
    return Path(os.environ.get("EVO_TRAIN_SEETACLOUD_ENV_FILE", "/private/tmp/evo_train_seetacloud_env.sh")).expanduser()


def _read_ssh_runtime_credentials() -> dict[str, str]:
    path = _ssh_runtime_env_path()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("export "):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) < 2 or "=" not in parts[1]:
            continue
        key, value = parts[1].split("=", 1)
        if key in {"AUTODL_HOST", "AUTODL_PORT", "AUTODL_USER", "AUTODL_PASSWORD", "AUTODL_KEY_PATH"}:
            values[key] = value
    host = values.get("AUTODL_HOST", "").strip()
    port = values.get("AUTODL_PORT", "").strip()
    user = values.get("AUTODL_USER", "").strip()
    endpoint = f"{user}@{host}:{port}" if host and port and user else ""
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": values.get("AUTODL_PASSWORD", "").strip(),
        "keyPath": values.get("AUTODL_KEY_PATH", "").strip(),
        "endpoint": endpoint,
        "envPath": str(path),
    }


def _read_ssh_runtime_endpoint() -> dict[str, str]:
    credentials = _read_ssh_runtime_credentials()
    return {
        "host": credentials.get("host", ""),
        "port": credentials.get("port", ""),
        "user": credentials.get("user", ""),
        "endpoint": credentials.get("endpoint", ""),
        "envPath": credentials.get("envPath", ""),
    }


def _read_remote_text_file(remote_path: str, *, max_bytes: int = 2_000_000) -> str:
    path = remote_path.strip()
    if not path.startswith(("/root/autodl-tmp/", "/workspace/", "/tmp/")):
        raise ValueError("remote artifact path is outside allowed cloud output directories")
    credentials = _read_ssh_runtime_credentials()
    host = credentials.get("host", "")
    port = credentials.get("port", "")
    user = credentials.get("user", "")
    if not host or not port or not user:
        raise RuntimeError("cloud SSH runtime is not bound")
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("Missing SSH dependency. Install it with: python3 -m pip install paramiko") from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = credentials.get("keyPath", "")
    password = credentials.get("password", "")
    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": int(port),
        "username": user,
        "timeout": 8,
        "banner_timeout": 8,
        "auth_timeout": 8,
    }
    if key_path:
        connect_kwargs["key_filename"] = str(Path(key_path).expanduser())
    if password:
        connect_kwargs["password"] = password
    try:
        client.connect(**connect_kwargs)
        with client.open_sftp() as sftp:
            size = int(sftp.stat(path).st_size)
            if size > max_bytes:
                raise RuntimeError(f"remote artifact is too large to preview: {size} bytes")
            with sftp.open(path, "r") as handle:
                raw = handle.read(max_bytes + 1)
    finally:
        client.close()
    if isinstance(raw, str):
        return raw
    return bytes(raw).decode("utf-8", errors="replace")


def _clear_ssh_runtime_env() -> dict[str, str]:
    path = _ssh_runtime_env_path()
    previous = _read_ssh_runtime_endpoint()
    if not path.exists():
        return previous
    cleared_keys = {
        "AUTODL_HOST",
        "AUTODL_PORT",
        "AUTODL_USER",
        "AUTODL_PASSWORD",
        "AUTODL_KEY_PATH",
    }
    kept_lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = []
            if len(parts) >= 2 and "=" in parts[1]:
                key, _ = parts[1].split("=", 1)
                if key in cleared_keys:
                    continue
        kept_lines.append(raw_line)
    path.write_text("\n".join(kept_lines).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return previous


def _probe_ssh_banner(
    host: str,
    port: str,
    *,
    timeout: float = 5.0,
    attempts: int = 1,
    interval_s: float = 0.0,
) -> tuple[bool, str]:
    last_error = ""
    total_attempts = max(1, attempts)
    for attempt in range(1, total_attempts + 1):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout) as sock:
                sock.settimeout(timeout)
                banner = sock.recv(128)
        except OSError as exc:
            last_error = f"无法连接 SSH 端口：{exc}"
        else:
            if banner.startswith(b"SSH-"):
                return True, ""
            if not banner:
                last_error = "端口已连接，但远端没有返回 SSH 登录协议。"
            else:
                preview = banner[:80].decode("utf-8", errors="replace").strip()
                last_error = f"端口已连接，但返回的不是 SSH 登录协议：{preview}"
        if attempt < total_attempts and interval_s > 0:
            time.sleep(interval_s)
    if total_attempts > 1:
        return False, f"{last_error} 已自动等待并重试 {total_attempts} 次。"
    return False, last_error


def _write_ssh_runtime_env(*, host: str, port: str, user: str, password: str, key_path: str) -> Path:
    path = _ssh_runtime_env_path()
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = "#!/usr/bin/env zsh\nexport TRAIN_PLATFORM=autodl\n"
    values = {
        "TRAIN_PLATFORM": "autodl",
        "AUTODL_HOST": host,
        "AUTODL_PORT": port,
        "AUTODL_USER": user,
        "AUTODL_PASSWORD": password,
        "AUTODL_KEY_PATH": key_path,
        "AUTODL_WORKDIR": "/root/autodl-tmp/evo_train",
        "AUTODL_JOB_ROOT": "/root/autodl-tmp/evo_train/jobs",
        "EVO_TRAIN_TASK_DB": os.environ.get("EVO_TRAIN_TASK_DB", "/private/tmp/evo_train_seetacloud_tasks.sqlite3"),
        "EVO_TRAIN_ALLOW_RAW_COMMAND": "false",
    }
    for key, value in values.items():
        text = _set_env_export(text, key, str(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path
def _listening_on_local_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0
def _restart_local_evo_train_bridge(env_path: Path) -> dict[str, Any]:
    for raw_pid in subprocess.run(
        ["lsof", "-tiTCP:9000", "-sTCP:LISTEN"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines():
        try:
            os.kill(int(raw_pid.strip()), 15)
        except (OSError, ValueError):
            pass
    time.sleep(0.8)
    for raw_pid in subprocess.run(
        ["lsof", "-tiTCP:9000", "-sTCP:LISTEN"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines():
        try:
            os.kill(int(raw_pid.strip()), 9)
        except (OSError, ValueError):
            pass
    log_path = Path(os.environ.get("EVO_TRAIN_SEETACLOUD_LOG_FILE", "/private/tmp/evo_train_seetacloud_9000.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(os.environ.get("EVO_TRAIN_REPO_DIR", "/Users/pearl/Documents/codex/tmp/EVO_Train"))
    python_bin = os.environ.get("EVO_TRAIN_PYTHON_BIN", "/Users/pearl/anaconda3/bin/python3")
    command = (
        f"source {shlex.quote(str(env_path))}; "
        f"cd {shlex.quote(str(repo_dir))}; "
        "export AUTODL_SSH_CONNECT_RETRIES=${AUTODL_SSH_CONNECT_RETRIES:-2}; "
        "export AUTODL_SSH_BANNER_TIMEOUT=${AUTODL_SSH_BANNER_TIMEOUT:-12}; "
        "export AUTODL_SSH_AUTH_TIMEOUT=${AUTODL_SSH_AUTH_TIMEOUT:-12}; "
        "export EVO_TRAIN_BACKGROUND_SUBMIT_RETRIES=${EVO_TRAIN_BACKGROUND_SUBMIT_RETRIES:-2}; "
        f"exec {shlex.quote(python_bin)} server_tcp/server_connection.py --host 127.0.0.1 --port 9000 --workers 4 --disable-billing-scheduler"
    )
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            ["zsh", "-lc", command],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.time() + 5
    while time.time() < deadline:
        if _listening_on_local_port(9000):
            return {"restarted": True, "pid": process.pid, "listening": True, "logPath": str(log_path)}
        if process.poll() is not None:
            break
        time.sleep(0.25)
    return {"restarted": True, "pid": process.pid, "listening": False, "logPath": str(log_path)}
