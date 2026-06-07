#!/usr/bin/env python3
"""Update local EVO_Train SSH runtime env without printing secrets."""

from __future__ import annotations

import getpass
import os
import re
from pathlib import Path


ENV_PATH = Path("/private/tmp/evo_train_seetacloud_env.sh")


def main() -> int:
    ssh_line = input("SSH command, e.g. ssh -p 42552 root@connect.cqa1.seetacloud.com: ").strip()
    parsed = _parse_ssh_command(ssh_line)
    if parsed is None:
        print("Could not parse the SSH command automatically. Please fill the fields below.")
        host = input("SSH host, e.g. connect.cqa1.seetacloud.com: ").strip()
        port = input("SSH port, e.g. 42552: ").strip()
        user = input("SSH user (default root): ").strip() or "root"
        if not host or not port.isdigit() or not user:
            print("Host, numeric port, and user are required.")
            return 2
    else:
        port, user, host = parsed
    auth_mode = _prompt_auth_mode()

    password = ""
    key_path = ""
    if auth_mode == "password":
        password = getpass.getpass("SSH password (hidden, not printed): ")
        if not password:
            print("Password is empty; cancelled.")
            return 2
    else:
        key_path = input("Private key path on this Mac: ").strip()
        if not key_path:
            print("Key path is empty; cancelled.")
            return 2
        if not Path(key_path).expanduser().is_file():
            print(f"Key path does not exist: {key_path}")
            return 2

    existing = ENV_PATH.read_text() if ENV_PATH.is_file() else ""
    sku_json = _extract_export(existing, "AUTODL_GPU_SKUS_JSON") or (
        '[{"skuId":"seetacloud-4090-1x","provider":"autodl","displayName":"SeetaCloud RTX 4090 · 1卡",'
        '"gpuSpec":"RTX 4090","gpuCount":1,"costHourlyCents":1000,"hourlyPriceCents":1000,'
        '"autodlGpuSpecUuid":"ssh-existing","requiresImage":false,"gpuMemoryGb":24,'
        '"supportedBackends":["lerobot","rlinf","tinyvla","openvla_oft"],'
        '"supportedModels":["act","smolvla","tinyvla","rynnvla","pi0","openvla","openvla-oft","oft"],'
        '"supportedBenchmarks":["libero"],"capabilities":["ssh-existing","smoke","vla"]}]'
    )
    images_json = _extract_export(existing, "AUTODL_IMAGES_JSON") or (
        '[{"imageId":"seetacloud-current-vla","provider":"autodl","displayName":"SeetaCloud 当前 VLA 环境",'
        '"autodlImageUuid":"ssh-current","cudaVFrom":121,'
        '"supportedBackends":["lerobot","rlinf","tinyvla","openvla_oft"],'
        '"supportedModels":["act","smolvla","tinyvla","rynnvla","pi0","openvla","openvla-oft","oft"],'
        '"supportedBenchmarks":["libero"],"capabilities":["ssh-existing","vla"]}]'
    )

    lines = [
        "#!/usr/bin/env zsh",
        "export TRAIN_PLATFORM=autodl",
        f"export AUTODL_HOST={shell_quote(host)}",
        f"export AUTODL_PORT={shell_quote(port)}",
        f"export AUTODL_USER={shell_quote(user)}",
        f"export AUTODL_PASSWORD={shell_quote(password)}",
        "# If this instance uses a private key instead of a password, set AUTODL_KEY_PATH and leave AUTODL_PASSWORD empty.",
        f"export AUTODL_KEY_PATH={shell_quote(key_path)}",
        "export AUTODL_WORKDIR=/root/autodl-tmp/evo_train",
        "export AUTODL_JOB_ROOT=/root/autodl-tmp/evo_train/jobs",
        "export EVO_TRAIN_TASK_DB=/private/tmp/evo_train_seetacloud_tasks.sqlite3",
        "export EVO_TRAIN_ALLOW_RAW_COMMAND=false",
        f"export AUTODL_GPU_SKUS_JSON={shell_quote(sku_json)}",
        f"export AUTODL_IMAGES_JSON={shell_quote(images_json)}",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines))
    os.chmod(ENV_PATH, 0o600)
    print(f"Updated {ENV_PATH}")
    print(f"SSH target: {user}@{host}:{port}")
    print(f"Auth mode: {auth_mode}")
    return 0


def _extract_export(text: str, name: str) -> str:
    pattern = re.compile(rf"^export\s+{re.escape(name)}=(.*)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return ""
    raw = match.group(1).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return raw


def _parse_ssh_command(raw: str) -> tuple[str, str, str] | None:
    line = raw.strip().replace("\u00a0", " ").replace("：", ":")
    if not line:
        return None
    # Strip common terminal prompts and trailing comments.
    if "ssh " in line:
        line = line[line.index("ssh "):]
    line = line.split("#", 1)[0].strip()

    patterns = [
        # ssh -p 42552 root@connect.cqa1.seetacloud.com
        r"\bssh\b.*?(?:^|\s)-p\s+(\d+).*?\s([^@\s]+)@([^\s:]+)",
        # ssh root@connect.cqa1.seetacloud.com -p 42552
        r"\bssh\b\s+([^@\s]+)@([^\s:]+).*?(?:^|\s)-p\s+(\d+)",
        # root@connect.cqa1.seetacloud.com:42552
        r"\b([^@\s]+)@([A-Za-z0-9_.-]+):(\d+)\b",
    ]
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, line)
        if not match:
            continue
        if index == 0:
            port, user, host = match.group(1), match.group(2), match.group(3)
        else:
            user, host, port = match.group(1), match.group(2), match.group(3)
        return port, user, host
    return None


def _prompt_auth_mode() -> str:
    password_aliases = {"", "p", "pass", "password", "pwd", "密码", "口令"}
    key_aliases = {"k", "key", "private_key", "private-key", "私钥", "密钥"}
    while True:
        raw = input("Auth mode [password/key] (press Enter for password): ").strip().lower()
        if raw in password_aliases:
            return "password"
        if raw in key_aliases:
            return "key"
        print("Please enter password/key, or press Enter for password.")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
