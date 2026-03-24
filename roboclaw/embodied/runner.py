"""Local runner for LeRobot CLI commands."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from pathlib import Path
from uuid import uuid4


class LocalLeRobotRunner:
    """Runs LeRobot CLI commands via subprocess."""

    async def run(self, argv: list[str], timeout: int = 300) -> tuple[int, str, str]:
        """Run command, return (returncode, stdout, stderr)."""
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise

        return (
            process.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    async def run_interactive(self, argv: list[str]) -> int:
        """Run command with inherited TTY (user sees output directly). Returns exit code."""
        process = await asyncio.create_subprocess_exec(*argv)
        return await process.wait()

    async def run_detached(self, argv: list[str], log_dir: Path) -> str:
        """Run command in background, return job_id (uuid). Save pid and log path."""
        log_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid4())
        log_path = log_dir / f"{job_id}.log"
        pid_path = log_dir / f"{job_id}.pid"
        log_path_file = log_dir / f"{job_id}.logpath"

        with log_path.open("ab") as log_file:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

        pid_path.write_text(str(process.pid), encoding="utf-8")
        log_path_file.write_text(str(log_path), encoding="utf-8")
        return job_id

    async def job_status(self, job_id: str, log_dir: Path) -> dict[str, str | int | bool | None]:
        """Check if detached job is still running, return status + last N lines of log."""
        pid_path = log_dir / f"{job_id}.pid"
        log_path = self._job_log_path(job_id, log_dir)

        if not pid_path.exists():
            return {
                "job_id": job_id,
                "status": "missing",
                "running": False,
                "pid": None,
                "log_path": str(log_path),
                "log_tail": self._tail_text(log_path),
            }

        pid = int(pid_path.read_text(encoding="utf-8").strip())
        running = self._is_running(pid)
        return {
            "job_id": job_id,
            "status": "running" if running else "finished",
            "running": running,
            "pid": pid,
            "log_path": str(log_path),
            "log_tail": self._tail_text(log_path),
        }

    def _is_running(self, pid: int) -> bool:
        """Return whether the process is still alive."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _job_log_path(self, job_id: str, log_dir: Path) -> Path:
        """Return the persisted log path for a detached job."""
        log_path_file = log_dir / f"{job_id}.logpath"
        if not log_path_file.exists():
            return log_dir / f"{job_id}.log"
        return Path(log_path_file.read_text(encoding="utf-8").strip())

    def _tail_text(self, path: Path, lines: int = 20) -> str:
        """Return the last few log lines."""
        if not path.exists():
            return ""

        tail = deque(maxlen=lines)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                tail.append(line.rstrip("\n"))
        return "\n".join(tail)
