"""Shared helpers for cloud/local training backends."""

from __future__ import annotations

import shlex
import tarfile
import time
from pathlib import Path
from typing import Sequence

from roboclaw.embodied.training.types import TrainingRequest


def repo_root() -> Path:
    """Return the repository root inferred from the installed package path."""
    return Path(__file__).resolve().parents[3]


def arg_value(argv: Sequence[str], prefix: str) -> str:
    """Return the value of the first arg that matches *prefix*."""
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return ""


def rewrite_train_argv(
    argv: Sequence[str],
    *,
    dataset_root: str,
    output_dir: str,
    strip_resume: bool = True,
) -> list[str]:
    """Rewrite local LeRobot train argv for a remote execution environment."""
    rewritten: list[str] = []
    for index, arg in enumerate(argv):
        if index == 0 and Path(arg).name == "lerobot-train":
            rewritten.append("lerobot-train")
            continue
        if arg.startswith("--dataset.root="):
            rewritten.append(f"--dataset.root={dataset_root}")
            continue
        if arg.startswith("--output_dir="):
            rewritten.append(f"--output_dir={output_dir}")
            continue
        if strip_resume and (arg == "--resume=true" or arg.startswith("--config_path=")):
            continue
        rewritten.append(arg)
    return rewritten


def shell_join(argv: Sequence[str]) -> str:
    """Quote argv as a shell-safe command string."""
    return " ".join(shlex.quote(part) for part in argv)


def remote_entrypoint_for_request(
    request: TrainingRequest,
    *,
    dataset_root: str,
    output_dir: str,
) -> str:
    """Return the remote shell command used by cloud backends."""
    if request.entrypoint.strip():
        return request.entrypoint.strip()
    rewritten = rewrite_train_argv(
        request.train_argv,
        dataset_root=dataset_root,
        output_dir=output_dir,
        strip_resume=True,
    )
    return shell_join(rewritten)


def make_tarball(src_dir: Path, out_path: Path) -> None:
    """Create a tarball suitable for code or dataset staging."""
    with tarfile.open(out_path, "w:gz") as tar:
        for item in sorted(src_dir.rglob("*")):
            if any(
                part in {"__pycache__", ".git", ".venv", "node_modules", ".pytest_cache"}
                for part in item.parts
            ):
                continue
            arcname = item.relative_to(src_dir).as_posix()
            tar.add(item, arcname=arcname, recursive=False)


def timestamp_slug() -> str:
    """Return a sortable UTC timestamp slug for staging keys and job names."""
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())
