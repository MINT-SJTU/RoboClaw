"""AutoDL backend implemented over SSH-accessible training instances."""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from roboclaw.embodied.command import ActionError
from roboclaw.embodied.executor import SubprocessExecutor
from roboclaw.embodied.training.backend import BaseTrainingBackend
from roboclaw.embodied.training.common import (
    make_tarball,
    remote_entrypoint_for_request,
)
from roboclaw.embodied.training.types import (
    TrainingJobRecord,
    TrainingJobState,
    TrainingJobStatus,
    TrainingProvider,
    TrainingRequest,
    TrainingSubmitResult,
)


@dataclass
class AutoDLTrainingConfig:
    """Connection details for a running AutoDL instance."""

    host: str
    user: str = "root"
    port: int = 22
    key_path: str = ""
    workdir: str = "/root/autodl-tmp/roboclaw-training"
    activate: str = ""
    strict_host_key_checking: bool = False

    @classmethod
    def from_env(cls, prefix: str = "ROBOCLAW_AUTODL_") -> "AutoDLTrainingConfig":
        host = os.environ.get(f"{prefix}HOST", "").strip()
        if not host:
            raise ValueError(f"Missing required env var: {prefix}HOST")
        return cls(
            host=host,
            user=os.environ.get(f"{prefix}USER", "root"),
            port=int(os.environ.get(f"{prefix}PORT", "22")),
            key_path=os.environ.get(f"{prefix}KEY_PATH", ""),
            workdir=os.environ.get(f"{prefix}WORKDIR", "/root/autodl-tmp/roboclaw-training"),
            activate=os.environ.get(f"{prefix}ACTIVATE", ""),
            strict_host_key_checking=os.environ.get(
                f"{prefix}STRICT_HOST_KEY_CHECKING", "false"
            ).lower() in {"1", "true", "yes"},
        )


class AutoDLTrainingBackend(BaseTrainingBackend):
    """Run training on an SSH-accessible AutoDL instance."""

    provider = TrainingProvider.AUTODL

    def __init__(self, config: AutoDLTrainingConfig | None = None) -> None:
        self._config = config

    @property
    def config(self) -> AutoDLTrainingConfig:
        if self._config is None:
            self._config = AutoDLTrainingConfig.from_env()
        return self._config

    async def submit(self, request: TrainingRequest) -> TrainingSubmitResult:
        local_job_id = uuid.uuid4().hex[:12]
        remote_root = (request.remote_workdir.strip() or "").rstrip("/") or (
            f"{self.config.workdir.rstrip('/')}/{local_job_id}"
        )
        remote_code_archive = f"{remote_root}/code.tar.gz"
        remote_dataset_archive = f"{remote_root}/dataset.tar.gz"
        remote_code_dir = f"{remote_root}/code"
        remote_dataset_dir = f"{remote_root}/dataset"
        remote_output_dir = f"{remote_root}/artifacts"
        remote_log_dir = f"{remote_root}/logs"
        remote_log_path = f"{remote_log_dir}/train.log"
        remote_exit_code = f"{remote_log_dir}/exit_code"

        entrypoint = remote_entrypoint_for_request(
            request,
            dataset_root=remote_dataset_dir,
            output_dir=remote_output_dir,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            code_archive = tmp_dir / "code.tar.gz"
            dataset_archive = tmp_dir / "dataset.tar.gz"
            make_tarball(request.code_dir, code_archive)
            make_tarball(request.dataset_local_path, dataset_archive)
            await self._checked_run(self._ssh_argv("mkdir", "-p", remote_root))
            await self._checked_run(self._scp_upload_argv(code_archive, remote_code_archive))
            await self._checked_run(self._scp_upload_argv(dataset_archive, remote_dataset_archive))

        submit_script = _compose_submit_script(
            remote_root=remote_root,
            remote_code_archive=remote_code_archive,
            remote_dataset_archive=remote_dataset_archive,
            remote_code_dir=remote_code_dir,
            remote_dataset_dir=remote_dataset_dir,
            remote_output_dir=remote_output_dir,
            remote_log_dir=remote_log_dir,
            remote_log_path=remote_log_path,
            remote_exit_code=remote_exit_code,
            activate=self.config.activate,
            user_entrypoint=entrypoint,
        )
        _, stdout, _ = await self._checked_run(self._ssh_argv("sh", "-lc", submit_script))
        remote_pid = stdout.strip().splitlines()[-1].strip()
        if not remote_pid:
            raise ActionError("AutoDL backend did not return a remote pid.")

        return TrainingSubmitResult(
            job_id=local_job_id,
            provider=self.provider,
            message=(
                f"AutoDL training submitted. Job ID: {local_job_id}\n"
                f"Remote PID: {remote_pid}"
            ),
            remote_job_id=remote_pid,
            provider_data={
                "remote_workdir": remote_root,
                "remote_log_path": remote_log_path,
            },
        )

    async def status(self, record: TrainingJobRecord) -> TrainingJobStatus:
        remote_pid = record.remote_job_id
        remote_workdir = str(record.provider_data.get("remote_workdir", ""))
        remote_log_path = str(record.provider_data.get("remote_log_path", ""))
        if not remote_pid or not remote_workdir:
            return TrainingJobStatus(
                job_id=record.job_id,
                provider=self.provider,
                state=TrainingJobState.MISSING,
                running=False,
                message="Missing AutoDL remote metadata.",
                output_dir=record.output_dir,
            )

        exit_code_path = f"{remote_workdir}/logs/exit_code"
        script = _compose_status_script(
            remote_pid=remote_pid,
            exit_code_path=exit_code_path,
            remote_log_path=remote_log_path,
        )
        _, stdout, _ = await self._checked_run(self._ssh_argv("sh", "-lc", script))
        state, log_tail = _parse_status_output(stdout)
        return TrainingJobStatus(
            job_id=record.job_id,
            provider=self.provider,
            state=state,
            running=state is TrainingJobState.RUNNING,
            message=state.value,
            remote_job_id=remote_pid,
            log_path=remote_log_path,
            log_tail=log_tail,
            output_dir=record.output_dir,
            provider_data={"remote_workdir": remote_workdir},
        )

    async def stop(self, record: TrainingJobRecord) -> TrainingJobStatus:
        remote_pid = record.remote_job_id
        remote_workdir = str(record.provider_data.get("remote_workdir", ""))
        if not remote_pid or not remote_workdir:
            return TrainingJobStatus(
                job_id=record.job_id,
                provider=self.provider,
                state=TrainingJobState.MISSING,
                running=False,
                message="Missing AutoDL remote metadata.",
                output_dir=record.output_dir,
            )
        script = _compose_stop_script(remote_pid)
        await self._checked_run(self._ssh_argv("sh", "-lc", script))
        return TrainingJobStatus(
            job_id=record.job_id,
            provider=self.provider,
            state=TrainingJobState.STOPPED,
            running=False,
            message="stop_requested",
            remote_job_id=remote_pid,
            output_dir=record.output_dir,
            provider_data={"remote_workdir": remote_workdir},
        )

    async def collect(
        self,
        record: TrainingJobRecord,
        *,
        output_dir: str | None = None,
    ) -> list[str]:
        remote_workdir = str(record.provider_data.get("remote_workdir", ""))
        if not remote_workdir:
            return []
        target = Path(output_dir or record.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        artifact_source = f"{self._ssh_target()}:{remote_workdir}/artifacts/."
        await self._checked_run(self._scp_download_argv(artifact_source, target))
        remote_log_path = str(record.provider_data.get("remote_log_path", ""))
        if remote_log_path:
            logs_target = target / "logs"
            logs_target.mkdir(parents=True, exist_ok=True)
            await self._checked_run(
                self._scp_download_argv(f"{self._ssh_target()}:{remote_log_path}", logs_target / "train.log")
            )
        return [str(target)]

    async def _checked_run(self, argv: list[str]) -> tuple[int, str, str]:
        code, stdout, stderr = await SubprocessExecutor().run(argv=argv, timeout=1800)
        if code != 0:
            raise ActionError(stderr.strip() or stdout.strip() or f"Command failed: {' '.join(argv)}")
        return code, stdout, stderr

    def _ssh_target(self) -> str:
        return f"{self.config.user}@{self.config.host}"

    def _ssh_common(self) -> list[str]:
        argv = ["ssh", "-p", str(self.config.port)]
        if self.config.key_path:
            argv.extend(["-i", self.config.key_path])
        if not self.config.strict_host_key_checking:
            argv.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        return argv

    def _ssh_argv(self, *remote_argv: str) -> list[str]:
        return [*self._ssh_common(), self._ssh_target(), *remote_argv]

    def _scp_upload_argv(self, local_path: Path, remote_path: str) -> list[str]:
        argv = ["scp", "-P", str(self.config.port)]
        if self.config.key_path:
            argv.extend(["-i", self.config.key_path])
        if not self.config.strict_host_key_checking:
            argv.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        argv.extend([str(local_path), f"{self._ssh_target()}:{remote_path}"])
        return argv

    def _scp_download_argv(self, remote_path: str, local_path: Path) -> list[str]:
        argv = ["scp", "-P", str(self.config.port)]
        if self.config.key_path:
            argv.extend(["-i", self.config.key_path])
        if not self.config.strict_host_key_checking:
            argv.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        argv.extend(["-r", remote_path, str(local_path)])
        return argv


def _compose_submit_script(
    *,
    remote_root: str,
    remote_code_archive: str,
    remote_dataset_archive: str,
    remote_code_dir: str,
    remote_dataset_dir: str,
    remote_output_dir: str,
    remote_log_dir: str,
    remote_log_path: str,
    remote_exit_code: str,
    activate: str,
    user_entrypoint: str,
) -> str:
    activation = f". {shlex.quote(activate)} && " if activate else ""
    detached_command = (
        f"{activation}{user_entrypoint}; "
        f"rc=$?; echo $rc > {shlex.quote(remote_exit_code)}"
    )
    return (
        "set -e && "
        f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(remote_code_dir)} "
        f"{shlex.quote(remote_dataset_dir)} {shlex.quote(remote_output_dir)} "
        f"{shlex.quote(remote_log_dir)} && "
        f"tar -xzf {shlex.quote(remote_code_archive)} -C {shlex.quote(remote_code_dir)} && "
        f"tar -xzf {shlex.quote(remote_dataset_archive)} -C {shlex.quote(remote_dataset_dir)} && "
        f"cd {shlex.quote(remote_code_dir)} && "
        f"nohup sh -lc {shlex.quote(detached_command)} > {shlex.quote(remote_log_path)} 2>&1 "
        "< /dev/null & echo $!"
    )


def _compose_status_script(
    *,
    remote_pid: str,
    exit_code_path: str,
    remote_log_path: str,
) -> str:
    return (
        f"if kill -0 {shlex.quote(remote_pid)} 2>/dev/null; then "
        "state=running; "
        f"elif [ -f {shlex.quote(exit_code_path)} ]; then "
        f"rc=$(cat {shlex.quote(exit_code_path)}); "
        'if [ "$rc" = "0" ]; then state=succeeded; else state=failed; fi; '
        "else state=missing; fi; "
        'printf "__STATE__:%s\\n" "$state"; '
        'printf "__TAIL__\\n"; '
        f"if [ -f {shlex.quote(remote_log_path)} ]; then tail -n 40 {shlex.quote(remote_log_path)}; fi"
    )


def _compose_stop_script(remote_pid: str) -> str:
    return (
        f"kill -INT {shlex.quote(remote_pid)} 2>/dev/null || true; "
        "sleep 1; "
        f"kill -0 {shlex.quote(remote_pid)} 2>/dev/null && "
        f"kill -KILL {shlex.quote(remote_pid)} 2>/dev/null || true"
    )


def _parse_status_output(stdout: str) -> tuple[TrainingJobState, str]:
    lines = stdout.splitlines()
    state = TrainingJobState.UNKNOWN
    tail_lines: list[str] = []
    tail_mode = False
    for line in lines:
        if line.startswith("__STATE__:"):
            state_name = line.split(":", 1)[1].strip()
            state = TrainingJobState(state_name)
            continue
        if line == "__TAIL__":
            tail_mode = True
            continue
        if tail_mode:
            tail_lines.append(line)
    return state, "\n".join(tail_lines).strip()

